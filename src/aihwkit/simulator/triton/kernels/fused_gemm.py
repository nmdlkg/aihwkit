# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Fused GEMM + noise + clamp Triton kernel.

Computes: out[m, n] = clamp(sum_k(x[m,k] * W[n,k]) + noise_std * N(0,1), -bound, bound)
in ONE kernel launch, fusing matmul, noise injection, and output clamping.

This eliminates the separate GEMM + noise + clamp kernel launches required by the
approach in forward.py, reducing memory bandwidth and kernel launch overhead.

Key function:
    triton_fused_gemm(x, weights, noise_std, bound, seed) -> Tensor
    triton_fused_gemm_backward(d, weights, noise_std, bound, seed) -> Tensor
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton Kernel: Fused GEMM + noise + clamp
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
                num_warps=4,
                num_stages=2,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
                num_warps=4,
                num_stages=2,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
                num_warps=8,
                num_stages=3,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2
            ),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _fused_gemm_noise_clamp_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,  # W is [N, K]: W[n, k] = w_ptr + stride_wn*n + stride_wk*k
        stride_om,
        stride_on,
        noise_std: tl.constexpr,
        bound: tl.constexpr,
        seed: tl.int64,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused GEMM: out = clamp(x @ W.T + noise*noise_std, -bound, bound).

        Each program instance computes a BLOCK_M × BLOCK_N output tile.
        W is stored as [N, K] (already "transposed" w.r.t. the multiplication),
        so x @ W.T reads x[m, k] and W[n, k] for the same k-index.

        Noise uses Box-Muller transform with per-tile unique seeds for
        reproducibility. noise_std and bound are constexpr to allow the compiler
        to eliminate the noise/clamp branches when they are disabled.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Row offsets for x (output rows) and column offsets for W (output cols)
        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Accumulate the dot product in float32
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Iterate over K dimension in tiles of BLOCK_K
        for k_start in range(0, tl.cdiv(K, BLOCK_K)):
            k_offs = k_start * BLOCK_K + tl.arange(0, BLOCK_K)

            # Load x[m_offs, k_offs]: shape [BLOCK_M, BLOCK_K]
            x_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
            x_tile = tl.load(
                x_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk,
                mask=x_mask,
                other=0.0,
            )

            # Load W[n_offs, k_offs] (W is [N, K]): shape [BLOCK_N, BLOCK_K]
            w_mask = (n_offs[:, None] < N) & (k_offs[None, :] < K)
            w_tile = tl.load(
                w_ptr + n_offs[:, None] * stride_wn + k_offs[None, :] * stride_wk,
                mask=w_mask,
                other=0.0,
            )

            # acc += x_tile @ w_tile.T: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
            # allow_tf32=False ensures full float32 precision (TF32 causes ~1e-2 error)
            acc = acc + tl.dot(x_tile, tl.trans(w_tile), allow_tf32=False)

        # -----------------------------------------------------------------
        # Optionally add Gaussian noise (Box-Muller transform)
        # noise_std is constexpr: branch is compiled away when noise_std == 0.0
        # -----------------------------------------------------------------
        if noise_std > 0.0:
            # Unique per-tile seed to ensure independent noise across tiles
            # Using a simple hash to combine seed, pid_m, pid_n
            num_pid_n = tl.num_programs(1)
            block_seed = seed + pid_m * num_pid_n + pid_n

            # Intra-tile element offsets (0..BLOCK_M*BLOCK_N-1) for tl.rand
            intra_m = tl.arange(0, BLOCK_M)
            intra_n = tl.arange(0, BLOCK_N)
            intra_offs = intra_m[:, None] * BLOCK_N + intra_n[None, :]

            # Box-Muller: two uniform streams → standard normal
            u1 = tl.rand(block_seed, intra_offs)
            u2 = tl.rand(block_seed + 1, intra_offs)
            # Z ~ N(0,1) via Box-Muller
            z = tl.sqrt(-2.0 * tl.log(u1 + 1e-7)) * tl.cos(2.0 * 3.14159265358979 * u2)

            acc = acc + noise_std * z

        # -----------------------------------------------------------------
        # Optionally clamp to [-bound, bound]
        # bound is constexpr: branch compiled away when bound >= 1e9 (sentinel for inf)
        # -----------------------------------------------------------------
        if bound < 1e9:
            acc = tl.maximum(acc, -bound)
            acc = tl.minimum(acc, bound)

        # Store output tile
        out_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
        tl.store(
            out_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on,
            acc.to(tl.float32),
            mask=out_mask,
        )


# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------


def _cpu_fused_gemm(
    x: torch.Tensor,
    weights: torch.Tensor,
    noise_std: float,
    bound: float,
    seed: int,
) -> torch.Tensor:
    """CPU fallback: pure torch operations for fused GEMM + noise + clamp."""
    out = torch.mm(x, weights.T)

    if noise_std > 0.0:
        gen = torch.Generator()
        gen.manual_seed(int(seed) & 0xFFFFFFFF)
        noise = torch.randn(
            out.shape, generator=gen, dtype=out.dtype, device=out.device
        )
        out = out + noise * noise_std

    if bound < float("inf"):
        out = out.clamp(-bound, bound)

    return out


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------


def triton_fused_gemm(
    x: torch.Tensor,
    weights: torch.Tensor,
    noise_std: float = 0.0,
    bound: float = float("inf"),
    seed: int = 0,
) -> torch.Tensor:
    """Fused GEMM: out = clamp(x @ weights.T + noise, -bound, bound).

    Computes the matrix product x @ weights.T with optional Gaussian noise
    injection and symmetric clamping, all fused into a single Triton kernel
    (on CUDA) or via torch operations (CPU fallback).

    Args:
        x:         Input tensor of shape [M, K].  Also accepts:
                   - 1D [K]: treated as [1, K], output squeezed back to [N].
                   - 3D [B, P, K]: reshaped to [B*P, K], output reshaped back.
        weights:   Weight matrix of shape [N, K].
        noise_std: Gaussian noise standard deviation (0.0 = no noise).
        bound:     Symmetric clamp bound; ``inf`` disables clamping.
        seed:      Integer RNG seed for reproducible noise.

    Returns:
        Float tensor of shape [M, N] (or [N] / [B, P, N] matching x's shape).
    """
    # --- Shape normalisation ------------------------------------------------
    squeeze = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze = True

    orig_3d_shape = None
    if x.dim() == 3:
        orig_3d_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

    assert weights.dim() == 2, f"weights must be 2D [N, K], got shape {weights.shape}"
    assert x.dim() == 2, f"x must be 2D [M, K] after normalisation, got {x.dim()}D"

    # --- CPU / no-Triton fallback -------------------------------------------
    if not x.is_cuda or not HAS_TRITON:
        out = _cpu_fused_gemm(x, weights, noise_std, bound, seed)
    else:
        # Cast to float32 (kernel only supports float32)
        x_f32 = x.contiguous().float()
        w_f32 = weights.contiguous().float()

        M, K = x_f32.shape
        N = w_f32.shape[0]

        out = torch.empty((M, N), device=x_f32.device, dtype=torch.float32)

        # Sentinel: pass 1e10 for bound=inf so the constexpr check (< 1e9) fails
        kernel_bound = float(bound) if bound < float("inf") else 1e10

        grid = lambda meta: (  # noqa: E731
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )
        _fused_gemm_noise_clamp_kernel[grid](
            x_f32,
            w_f32,
            out,
            M,
            N,
            K,
            x_f32.stride(0),
            x_f32.stride(1),
            w_f32.stride(0),
            w_f32.stride(1),
            out.stride(0),
            out.stride(1),
            float(noise_std),
            kernel_bound,
            int(seed),
        )

    # --- Restore original shape ---------------------------------------------
    if orig_3d_shape is not None:
        out = out.reshape(orig_3d_shape[0], orig_3d_shape[1], -1)
    if squeeze:
        out = out.squeeze(0)

    return out


def triton_fused_gemm_backward(
    d: torch.Tensor,
    weights: torch.Tensor,
    noise_std: float = 0.0,
    bound: float = float("inf"),
    seed: int = 0,
) -> torch.Tensor:
    """Backward fused GEMM: dx = clamp(d @ weights + noise, -bound, bound).

    Computes the input gradient for a linear layer that used ``triton_fused_gemm``
    in the forward pass.  The gradient w.r.t. the input is ``d @ W`` (not
    ``d @ W.T``), which is equivalent to ``triton_fused_gemm(d, W.T)``.

    Args:
        d:         Upstream gradient of shape [M, N].
        weights:   Weight matrix of shape [N, K] (same as used in forward).
        noise_std: Optional noise injected into the backward pass.
        bound:     Symmetric clamp bound for the backward pass.
        seed:      RNG seed (should differ from forward seed to avoid correlation).

    Returns:
        Float tensor of shape [M, K] — gradient w.r.t. x.
    """
    # dx = d @ W  (shape: [M, N] @ [N, K] → [M, K])
    # This equals triton_fused_gemm(d, W.T) where W.T is [K, N]:
    #   d @ (W.T).T = d @ W  ✓
    return triton_fused_gemm(
        d,
        weights.T.contiguous(),
        noise_std=noise_std,
        bound=bound,
        seed=seed,
    )
