# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Fused Bernoulli-coincidence weight update kernel.

Single Triton kernel that fuses:
  1. Per-sample Bernoulli pulse generation (BL rounds per sample)
  2. Coincidence detection via outer product
  3. Batch accumulation in registers (no intermediate tensors)
  4. Weight update with optional multiplicative noise
  5. Weight clamping

This replaces the 3-step pipeline (triton_get_counts + triton_pulsed_update)
and the memory-hungry PyTorch binomial path, eliminating:
  - All intermediate [B, out_size, in_size] tensor allocations
  - Multiple separate kernel launches
  - Memory round-trips between operations

Algorithm per weight element (i, j):
    signed_acc = 0
    for b in range(B):
        for bl in range(BL):
            d_fired = Bernoulli(d_prob[b, i])   # via tl.rand
            x_fired = Bernoulli(x_prob[b, j])   # via tl.rand
            if d_fired and x_fired:             # coincidence
                signed_acc += d_sign[b,i] * x_sign[b,j]
    delta_w = dw_min * signed_acc + noise_term
    w[i,j] = clamp(w[i,j] + delta_w, w_min, w_max)

RNG consistency:
    d samples use tl.rand(seed, b*BL*M + bl*M + m_offsets) — same across all k-tiles.
    x samples use tl.rand(seed+1, b*BL*K + bl*K + k_offsets) — same across all m-tiles.
    Using offset-based uniqueness within a single seed per stream ensures identical
    Bernoulli outcomes for the same (b, bl, element) across weight tiles.

Equivalence to CUDA:
    The CUDA backend generates one Bernoulli per (row, round) and one per
    (column, round), then checks coincidence.  This kernel does exactly that:
    d_fire[m] is shared across all columns (same seed per m-tile), and
    x_fire[k] is shared across all rows (same seed per k-tile).  The
    resulting coincidence distribution is Binomial(BL, p_d[b,i] * p_x[b,j]),
    identical to the CUDA reference implementation.

NOTE: This kernel performs in-place weight updates.  ``restore_value=['weights_ptr']``
    is passed to ``@triton.autotune`` so that the weight tensor is saved/restored
    between benchmarking iterations, preventing corruption during autotuning.
Key function:
    triton_fused_bernoulli_update(weights, d_sp, x_sp, dw_min, dw_min_std,
                                  bl, bound_lower, bound_upper, seed) -> None
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
# Triton Kernel (autotune with restore_value for in-place safety)
# ---------------------------------------------------------------------------
if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=1),
        ],
        key=["M", "K"],
        restore_value=["weights_ptr"],
    )
    @triton.jit
    def _fused_bernoulli_update_kernel(
        weights_ptr,  # [M, K] float32, in/out
        d_sp_ptr,  # [B, M] float32, signed prob: (-scale_d * d).clamp(-1,1)
        x_sp_ptr,  # [B, K] float32, signed prob: (scale_x * x).clamp(-1,1)
        M,  # out_size (int)
        K,  # in_size (int)
        B,  # batch_size (int)
        BL,  # bit lines / pulse rounds (int)
        dw_min: tl.constexpr,
        dw_min_std: tl.constexpr,
        bound_lower: tl.constexpr,
        bound_upper: tl.constexpr,
        seed: tl.int64,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused Bernoulli sampling + coincidence accumulation + weight update.

        Each program computes a [BLOCK_M, BLOCK_K] tile of the weight matrix.
        Iterates over B batch elements x BL pulse rounds, sampling Bernoulli
        pulses inline and accumulating signed coincidence counts in registers.
        """
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        m_mask = m_offs < M
        k_mask = k_offs < K

        # Accumulators — signed coincidence sum and unsigned total
        signed_acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        coinc_total = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        for b in range(B):
            # Load signed probabilities: sp = prob * sign, values in [-1, 1]
            dsp = tl.load(d_sp_ptr + b * M + m_offs, mask=m_mask, other=0.0)
            xsp = tl.load(x_sp_ptr + b * K + k_offs, mask=k_mask, other=0.0)

            # Extract probability magnitude and sign direction
            dp = tl.abs(dsp)  # [BLOCK_M]
            xp = tl.abs(xsp)  # [BLOCK_K]
            d_dir = tl.where(dsp >= 0.0, 1.0, -1.0)  # [BLOCK_M]
            x_dir = tl.where(xsp >= 0.0, 1.0, -1.0)  # [BLOCK_K]

            for bl in range(BL):
                # Inline Bernoulli: unique offset per (b, bl, element)
                # d: tl.rand(seed, b*BL*M + bl*M + m_offs) — same d pulse across k-tiles
                # x: tl.rand(seed+1, b*BL*K + bl*K + k_offs) — same x pulse across m-tiles
                d_off = b * BL * M + bl * M + m_offs
                x_off = b * BL * K + bl * K + k_offs
                d_rand = tl.rand(seed, d_off)
                x_rand = tl.rand(seed + 1, x_off)

                d_fire = (d_rand < dp).to(tl.float32)  # [BLOCK_M]
                x_fire = (x_rand < xp).to(tl.float32)  # [BLOCK_K]

                # Signed pulse: {-1, 0, +1}
                d_pulse = d_fire * d_dir  # [BLOCK_M]
                x_pulse = x_fire * x_dir  # [BLOCK_K]

                # Coincidence outer product and accumulate
                signed_coinc = d_pulse[:, None] * x_pulse[None, :]
                signed_acc += signed_coinc

                if dw_min_std > 0.0:
                    # Track unsigned coincidence count for noise scaling
                    coinc_total += tl.abs(signed_coinc)

        # ----- Weight update -----
        delta_w = dw_min * signed_acc

        if dw_min_std > 0.0:
            # Derive pos/neg counts: pos = (total + signed) / 2
            pos_counts = (coinc_total + signed_acc) * 0.5
            neg_counts = (coinc_total - signed_acc) * 0.5

            # Box-Muller for two independent N(0,1) noise samples
            w_seed_offs = m_offs[:, None] * K + k_offs[None, :]
            u1 = tl.rand(seed + 2, w_seed_offs)
            u2 = tl.rand(seed + 3, w_seed_offs)
            u3 = tl.rand(seed + 4, w_seed_offs)
            u4 = tl.rand(seed + 5, w_seed_offs)
            z1 = tl.sqrt(-2.0 * tl.log(u1 + 1e-7)) * tl.cos(6.283185307179586 * u2)
            z2 = tl.sqrt(-2.0 * tl.log(u3 + 1e-7)) * tl.cos(6.283185307179586 * u4)

            delta_w += (
                dw_min
                * dw_min_std
                * (tl.sqrt(pos_counts + 1e-10) * z1 - tl.sqrt(neg_counts + 1e-10) * z2)
            )

        # Load weights, apply update, clamp, store
        w_mask = m_mask[:, None] & k_mask[None, :]
        w_offs = m_offs[:, None] * K + k_offs[None, :]
        w = tl.load(weights_ptr + w_offs, mask=w_mask, other=0.0)
        w_new = w + delta_w
        w_new = tl.minimum(tl.maximum(w_new, bound_lower), bound_upper)
        tl.store(weights_ptr + w_offs, w_new, mask=w_mask)


# ---------------------------------------------------------------------------
# PyTorch fallback (CPU / no-Triton)
# ---------------------------------------------------------------------------


def _torch_fused_bernoulli_update(
    weights: torch.Tensor,
    d_sp: torch.Tensor,
    x_sp: torch.Tensor,
    dw_min: float,
    dw_min_std: float,
    bl: int,
    bound_lower: float,
    bound_upper: float,
) -> None:
    """PyTorch fallback using BL rounds of independent Bernoulli + matmul.

    Matches the Triton kernel semantics exactly: independent row/column pulse
    generation per round with coincidence via outer product.
    """
    B, M = d_sp.shape
    _, K = x_sp.shape

    d_prob = d_sp.abs()  # [B, M]
    x_prob = x_sp.abs()  # [B, K]
    d_sign = d_sp.sign()  # [B, M]
    x_sign = x_sp.sign()  # [B, K]

    signed_acc = torch.zeros(M, K, device=weights.device, dtype=weights.dtype)
    coinc_total = torch.zeros_like(signed_acc) if dw_min_std > 0.0 else None

    for _ in range(bl):
        # Independent Bernoulli per row (d) and column (x) per round
        d_fire = torch.bernoulli(d_prob)  # [B, M], {0, 1}
        x_fire = torch.bernoulli(x_prob)  # [B, K], {0, 1}
        d_pulse = d_fire * d_sign  # [B, M], {-1, 0, +1}
        x_pulse = x_fire * x_sign  # [B, K], {-1, 0, +1}

        # Signed coincidence via matmul: sum over batch
        signed_acc += d_pulse.T @ x_pulse  # [M, K]

        if coinc_total is not None:
            coinc_total += d_fire.T @ x_fire  # unsigned coincidence

    delta_w = dw_min * signed_acc

    if dw_min_std > 0.0 and coinc_total is not None:
        pos_counts = (coinc_total + signed_acc) * 0.5
        neg_counts = (coinc_total - signed_acc) * 0.5
        z1 = torch.randn_like(weights)
        z2 = torch.randn_like(weights)
        delta_w += (
            dw_min * dw_min_std * (pos_counts.sqrt() * z1 - neg_counts.sqrt() * z2)
        )

    weights.add_(delta_w)
    weights.clamp_(bound_lower, bound_upper)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def triton_fused_bernoulli_update(
    weights: torch.Tensor,
    d_sp: torch.Tensor,
    x_sp: torch.Tensor,
    dw_min: float = 0.01,
    dw_min_std: float = 0.0,
    bl: int = 31,
    bound_lower: float = -0.6,
    bound_upper: float = 0.6,
    seed: int | None = None,
) -> None:
    """Fused Bernoulli-coincidence weight update (in-place).

    Replaces the separate Bernoulli + matmul + update pipeline with a single
    Triton kernel that fuses BL rounds of per-sample Bernoulli pulse generation,
    coincidence counting via outer product, and in-place weight update.

    Statistically equivalent to the CUDA backend: generates independent row
    and column pulses per round, then detects coincidences.  The resulting
    per-element coincidence count follows Binomial(BL, p_d * p_x).

    Args:
        weights:     [M, K] float32 tensor, modified in-place.
        d_sp:        [B, M] float32 — signed probability for d (output):
                     ``(-scale_d * d_all).clamp(-1, 1)``.  Sign is pre-negated
                     for gradient descent.
        x_sp:        [B, K] float32 — signed probability for x (input):
                     ``(scale_x * x_all).clamp(-1, 1)``.
        dw_min:      Minimum weight change per coincident pulse.
        dw_min_std:  Multiplicative noise std (0.0 = no noise).
        bl:          Number of bit lines / pulse rounds per sample.
        bound_lower: Lower weight clamp bound.
        bound_upper: Upper weight clamp bound.
        seed:        RNG seed; None = random per call.

    Returns:
        None — weights tensor is modified in-place.
    """
    assert weights.ndim == 2, f"weights must be 2D, got shape {weights.shape}"
    assert d_sp.ndim == 2, f"d_sp must be 2D, got shape {d_sp.shape}"
    assert x_sp.ndim == 2, f"x_sp must be 2D, got shape {x_sp.shape}"

    M, K = weights.shape
    B = d_sp.shape[0]

    assert d_sp.shape == (B, M), f"d_sp shape {d_sp.shape} != ({B}, {M})"
    assert x_sp.shape == (B, K), f"x_sp shape {x_sp.shape} != ({B}, {K})"

    if not HAS_TRITON or not weights.is_cuda:
        _torch_fused_bernoulli_update(
            weights,
            d_sp,
            x_sp,
            dw_min,
            dw_min_std,
            bl,
            bound_lower,
            bound_upper,
        )
        return

    if seed is None:
        seed = torch.randint(0, 2**30, (1,), dtype=torch.int64).item()

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(K, meta["BLOCK_K"]),
    )

    _fused_bernoulli_update_kernel[grid](
        weights,
        d_sp.contiguous(),
        x_sp.contiguous(),
        M,
        K,
        B,
        bl,
        float(dw_min),
        float(dw_min_std),
        float(bound_lower),
        float(bound_upper),
        int(seed),
    )
