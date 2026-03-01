# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Fused forward-pass MVM kernel: GEMM + noise injection + clamp.

Computes:
    out = clamp(weights @ x + noise * noise_std, -bound, bound)

Strategy:
- GEMM: delegated to torch.mm (well-optimized cuBLAS/cuDNN path)
- Post-processing: Triton fused kernel for noise injection + clamping

This separation avoids reimplementing GEMM while still fusing the
noise and clamp operations into a single Triton pass (one read + one write).

Key function:
    triton_forward_mvm(weights, x, noise_std, bound, is_test, seed) -> Tensor
"""

from __future__ import annotations

from typing import Union

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton Kernel: Fused noise add + clamp
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 256}),
            triton.Config({"BLOCK": 512}),
            triton.Config({"BLOCK": 1024}),
        ],
        key=["n"],
    )
    @triton.jit
    def _add_noise_clamp_kernel(
        y_ptr,  # input: base GEMM output (float32)
        noise_ptr,  # input: raw Gaussian noise N(0,1) from RNG kernel
        out_ptr,  # output tensor
        n,  # number of elements
        noise_std: tl.float32,  # noise scale
        bound: tl.float32,  # clamp bound (clamp to [-bound, bound])
        do_clamp: tl.constexpr,  # whether to apply clamping
        BLOCK: tl.constexpr,
    ):
        """Fused: result = clamp(y + noise * noise_std, -bound, bound).

        Each program instance handles BLOCK consecutive elements.
        do_clamp is a constexpr so the clamp branch is compiled away when False.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n

        y = tl.load(y_ptr + offs, mask=mask)
        noise = tl.load(noise_ptr + offs, mask=mask)

        result = y + noise * noise_std

        if do_clamp:
            result = tl.maximum(result, -bound)
            result = tl.minimum(result, bound)

        tl.store(out_ptr + offs, result, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 256}),
            triton.Config({"BLOCK": 512}),
            triton.Config({"BLOCK": 1024}),
        ],
        key=["n"],
    )
    @triton.jit
    def _clamp_only_kernel(
        y_ptr,  # input tensor
        out_ptr,  # output tensor
        n,  # number of elements
        bound: tl.float32,  # clamp to [-bound, bound]
        BLOCK: tl.constexpr,
    ):
        """Clamp elements to [-bound, bound] without noise."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n

        y = tl.load(y_ptr + offs, mask=mask)
        result = tl.maximum(tl.minimum(y, bound), -bound)
        tl.store(out_ptr + offs, result, mask=mask)


# ---------------------------------------------------------------------------
# CPU fallbacks
# ---------------------------------------------------------------------------


def _cpu_forward_mvm(
    weights: torch.Tensor,
    x: torch.Tensor,
    noise_std: float,
    bound: float,
    is_test: bool,
    seed: int,
) -> torch.Tensor:
    """CPU fallback: pure torch operations."""
    if x.dim() == 1:
        # [in_size] → [out_size]
        y = weights @ x
    else:
        # weights: [out_size, in_size], x: [batch, in_size] → y: [batch, out_size]
        y = x @ weights.T

    if not is_test and noise_std > 0.0:
        # Use torch for reproducible noise on CPU
        gen = torch.Generator()
        gen.manual_seed(int(seed) & 0xFFFFFFFF)
        noise = torch.randn(y.shape, generator=gen, dtype=y.dtype, device=y.device)
        y = y + noise * noise_std

    if bound < float("inf"):
        y = y.clamp(-bound, bound)

    return y


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------


def triton_forward_mvm(
    weights: torch.Tensor,
    x: torch.Tensor,
    noise_std: float = 0.0,
    bound: float = float("inf"),
    is_test: bool = False,
    seed: int = 0,
) -> torch.Tensor:
    """Fused forward matrix-vector multiply with optional noise and clamping.

    Computes: ``out = clamp(weights @ x + noise * noise_std, -bound, bound)``

    The GEMM is handled by torch.mm (cuBLAS) for correctness and performance.
    Noise injection and clamping are fused in a single Triton kernel pass.

    Args:
        weights:   Weight matrix of shape ``[out_size, in_size]``.
        x:         Input tensor of shape ``[in_size]`` (vector) or
                   ``[batch, in_size]`` (batch of row vectors).
        noise_std: Standard deviation for Gaussian output noise.
                   Pass 0.0 or set ``is_test=True`` to disable noise.
        bound:     Symmetric clamp bound.  Output is clamped to
                   ``[-bound, bound]`` when bound < inf.
        is_test:   When True, noise is suppressed (inference mode).
        seed:      Integer seed for the Triton RNG kernel.

    Returns:
        Float tensor of shape:
        - ``[out_size]``        when x is 1-D
        - ``[batch, out_size]`` when x is 2-D  (matches x @ weights.T layout)

    Note:
        The function preserves the input dtype.  Internally the Triton kernel
        operates on float32; a cast back to the original dtype is applied
        when necessary.
    """
    # -----------------------------------------------------------------------
    # CPU fallback (no Triton / not on CUDA)
    # -----------------------------------------------------------------------
    if not HAS_TRITON or not weights.is_cuda or not x.is_cuda:
        return _cpu_forward_mvm(weights, x, noise_std, bound, is_test, seed)

    original_dtype = x.dtype
    is_noisy = (not is_test) and (noise_std > 0.0)
    do_clamp = bound < float("inf")

    # -----------------------------------------------------------------------
    # Step 1: GEMM via torch.mm (cuBLAS path — no need to reimplement)
    # -----------------------------------------------------------------------
    if x.dim() == 1:
        # weights: [out, in], x: [in] → y: [out]
        y = weights @ x  # uses torch.mv internally
    else:
        # weights: [out, in], x: [batch, in] → y: [batch, out]
        # Equivalent to torch.mm(x, weights.T)
        y = torch.mm(x, weights.T)

    # -----------------------------------------------------------------------
    # Fast path: no noise, no clamp
    # -----------------------------------------------------------------------
    if not is_noisy and not do_clamp:
        return y

    # Convert to float32 for the Triton kernel (matches noise kernel dtype)
    y_f32 = y.to(torch.float32).contiguous()
    n = y_f32.numel()
    out = torch.empty_like(y_f32)

    # -----------------------------------------------------------------------
    # Fast path: clamp only (no noise) — use simpler kernel
    # -----------------------------------------------------------------------
    if not is_noisy:
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
        _clamp_only_kernel[grid](y_f32, out, n, float(bound))
        return out.to(original_dtype).reshape(y.shape)

    # -----------------------------------------------------------------------
    # Step 2: Generate Gaussian noise via the RNG kernel
    # -----------------------------------------------------------------------
    from aihwkit.simulator.triton.kernels.rng import gaussian_noise  # noqa: PLC0415

    noise = gaussian_noise(
        n_elements=n,
        seed=seed,
        mean=0.0,
        std=1.0,  # raw N(0,1); scaled inside the kernel by noise_std
        device=str(weights.device),
    )

    # -----------------------------------------------------------------------
    # Step 3: Fused noise-add + clamp kernel
    # -----------------------------------------------------------------------
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
    _add_noise_clamp_kernel[grid](
        y_f32,
        noise,
        out,
        n,
        float(noise_std),
        float(bound) if do_clamp else 0.0,
        do_clamp,
    )

    # Restore original dtype and shape
    return out.to(original_dtype).reshape(y.shape)
