# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Pulsed weight update kernel for analog RPU devices.

Implements coincidence-based pulsed weight updates for 3 device models:
  - IDEAL: dw = dw_min * sign(x) * sign(d) for coincident pulses
  - CONSTANT_STEP: IDEAL + optional multiplicative noise + bounds
  - EXP_STEP: exponential dependence on current weight value

Core concept — coincident pulses:
    For each weight element (i, j):
    - If x_counts[j] != 0 AND d_counts[i] != 0 → apply device update rule
    - Otherwise → no change

Reference implementations:
    - CUDA: src/rpucuda/cuda/pwu_kernel.h
    - ConstantStep: src/rpucuda/rpu_constantstep_device.h
    - ExpStep: src/rpucuda/rpu_expstep_device.h

Key function:
    triton_pulsed_update(weights, x_counts, d_counts, dw_min, functor, params,
                         bound_lower, bound_upper) -> None  (in-place)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton Kernels (preferred path)
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 16, "BLOCK_K": 16}),
            triton.Config({"BLOCK_M": 32, "BLOCK_K": 16}),
            triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}),
            triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}),
        ],
        key=["M", "K"],
    )
    @triton.jit
    def _pulsed_update_ideal_kernel(
        weights_ptr,
        x_counts_ptr,
        d_counts_ptr,
        M,
        K,
        dw_min: tl.constexpr,
        bound_lower: tl.constexpr,
        bound_upper: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Ideal pulsed update: dw = dw_min * sign(x) * sign(d) for coincident pulses."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

        m_mask = m_offs < M
        k_mask = k_offs < K

        # Load pulse counts
        d_c = tl.load(d_counts_ptr + m_offs, mask=m_mask, other=0)  # [BLOCK_M]
        x_c = tl.load(x_counts_ptr + k_offs, mask=k_mask, other=0)  # [BLOCK_K]

        # Coincidence detection: both non-zero → outer product
        d_nonzero = (d_c != 0).to(tl.int32)  # [BLOCK_M]
        x_nonzero = (x_c != 0).to(tl.int32)  # [BLOCK_K]
        coincident = d_nonzero[:, None] * x_nonzero[None, :]  # [BLOCK_M, BLOCK_K]

        # Sign-based update direction
        d_sign = tl.where(d_c > 0, 1.0, -1.0).to(tl.float32)  # [BLOCK_M]
        x_sign = tl.where(x_c > 0, 1.0, -1.0).to(tl.float32)  # [BLOCK_K]
        dw_outer = d_sign[:, None] * x_sign[None, :]  # [BLOCK_M, BLOCK_K]
        dw_outer = dw_outer * coincident.to(tl.float32)  # Zero out non-coincident

        # Load, update, clamp, store
        w_mask = m_mask[:, None] & k_mask[None, :]
        w_offs = m_offs[:, None] * K + k_offs[None, :]

        w = tl.load(weights_ptr + w_offs, mask=w_mask, other=0.0)
        w_new = w + dw_min * dw_outer
        w_new = tl.minimum(tl.maximum(w_new, bound_lower), bound_upper)
        tl.store(weights_ptr + w_offs, w_new, mask=w_mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 16, "BLOCK_K": 16}),
            triton.Config({"BLOCK_M": 32, "BLOCK_K": 16}),
            triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}),
            triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}),
        ],
        key=["M", "K"],
    )
    @triton.jit
    def _pulsed_update_constant_step_kernel(
        weights_ptr,
        x_counts_ptr,
        d_counts_ptr,
        noise_ptr,  # pre-generated noise tensor, shape [M, K]
        M,
        K,
        dw_min: tl.constexpr,
        dw_min_std: tl.constexpr,
        bound_lower: tl.constexpr,
        bound_upper: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """ConstantStep pulsed update: IDEAL + multiplicative noise.

        dw = dw_min * (1 + dw_min_std * randn) * sign(x) * sign(d) for coincident pulses.
        """
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

        m_mask = m_offs < M
        k_mask = k_offs < K

        d_c = tl.load(d_counts_ptr + m_offs, mask=m_mask, other=0)
        x_c = tl.load(x_counts_ptr + k_offs, mask=k_mask, other=0)

        d_nonzero = (d_c != 0).to(tl.int32)
        x_nonzero = (x_c != 0).to(tl.int32)
        coincident = d_nonzero[:, None] * x_nonzero[None, :]

        d_sign = tl.where(d_c > 0, 1.0, -1.0).to(tl.float32)
        x_sign = tl.where(x_c > 0, 1.0, -1.0).to(tl.float32)
        dw_outer = d_sign[:, None] * x_sign[None, :]
        dw_outer = dw_outer * coincident.to(tl.float32)

        w_mask = m_mask[:, None] & k_mask[None, :]
        w_offs = m_offs[:, None] * K + k_offs[None, :]

        # Load noise (pre-generated on host) and apply multiplicative noise factor
        noise = tl.load(noise_ptr + w_offs, mask=w_mask, other=0.0)
        noise_factor = 1.0 + dw_min_std * noise

        w = tl.load(weights_ptr + w_offs, mask=w_mask, other=0.0)
        w_new = w + dw_min * noise_factor * dw_outer
        w_new = tl.minimum(tl.maximum(w_new, bound_lower), bound_upper)
        tl.store(weights_ptr + w_offs, w_new, mask=w_mask)


# ---------------------------------------------------------------------------
# PyTorch reference / fallback implementations
# ---------------------------------------------------------------------------


def _pulsed_update_torch(
    weights: torch.Tensor,
    x_counts: torch.Tensor,
    d_counts: torch.Tensor,
    dw_min: float,
    functor: str,
    params: Dict,
    bound_lower: float,
    bound_upper: float,
) -> None:
    """PyTorch reference implementation (correct on CPU and CUDA).

    Handles IDEAL, CONSTANT_STEP, EXP_STEP functors.
    """
    # Coincidence mask: outer product of non-zero indicators
    x_nonzero = (x_counts != 0).float()  # [in_size]
    d_nonzero = (d_counts != 0).float()  # [out_size]
    coincident = d_nonzero.unsqueeze(1) * x_nonzero.unsqueeze(0)  # [out_size, in_size]

    x_sign = torch.sign(x_counts.float())  # [in_size]
    d_sign = torch.sign(d_counts.float())  # [out_size]
    update_sign = (
        d_sign.unsqueeze(1) * x_sign.unsqueeze(0)
    ) * coincident  # [out_size, in_size]

    if functor == "IDEAL":
        dw = dw_min * update_sign

    elif functor == "CONSTANT_STEP":
        dw_min_std = params.get("dw_min_std", 0.0)
        if dw_min_std > 0.0:
            noise = 1.0 + dw_min_std * torch.randn_like(weights)
        else:
            noise = 1.0
        dw = dw_min * noise * update_sign

    elif functor == "EXP_STEP":
        # ExpStep parameters from rpu_expstep_device.h
        # dw = scale * MAX(1 - A * exp(gamma * z), 0)
        # where z = 2*w / (w_max - w_min) * es_a + es_b
        A_up = params.get("es_A_up", 0.00081)
        A_down = params.get("es_A_down", 0.36833)
        gamma_up = params.get("es_gamma_up", 12.44625)
        gamma_down = params.get("es_gamma_down", 12.78785)
        es_a = params.get("es_a", 0.244)
        es_b = params.get("es_b", 0.2425)
        w_max = params.get("w_max", bound_upper)
        w_min = params.get("w_min", bound_lower)
        up_down = params.get("up_down", 0.0)

        # Bias for asymmetric up/down scaling
        up_bias = 0.0 if up_down > 0.0 else up_down
        down_bias = -up_down if up_down > 0.0 else 0.0
        scale_up = (1.0 + up_bias) * dw_min
        scale_down = (1.0 + down_bias) * dw_min

        # Normalized weight position z
        w_range = max(w_max - w_min, 1e-6)
        z = 2.0 * weights / w_range * es_a + es_b

        # dw magnitudes: MAX(1 - A*exp(gamma*z), 0)
        dw_up = scale_up * torch.clamp(1.0 - A_up * torch.exp(gamma_up * z), min=0.0)
        dw_down = scale_down * torch.clamp(
            1.0 - A_down * torch.exp(gamma_down * (-z)), min=0.0
        )

        dw = torch.where(
            update_sign > 0,
            dw_up,
            torch.where(update_sign < 0, -dw_down, torch.zeros_like(weights)),
        )

    else:
        raise ValueError(
            f"Unknown functor '{functor}'. Supported: 'IDEAL', 'CONSTANT_STEP', 'EXP_STEP'"
        )

    weights.add_(dw)
    weights.clamp_(bound_lower, bound_upper)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def triton_pulsed_update(
    weights: torch.Tensor,
    x_counts: torch.Tensor,
    d_counts: torch.Tensor,
    dw_min: float = 0.01,
    functor: str = "IDEAL",
    params: Optional[Dict] = None,
    bound_lower: float = -1.0,
    bound_upper: float = 1.0,
) -> None:
    """Apply pulsed weight update based on coincident pulses (in-place).

    For each weight element (i, j):
      - If x_counts[j] != 0 AND d_counts[i] != 0 → apply device update rule
      - Otherwise → no change

    Args:
        weights:     [out_size, in_size] float32 tensor, modified in place.
        x_counts:    [in_size]  int32 tensor — input (column) pulse counts.
        d_counts:    [out_size] int32 tensor — output (row) pulse counts.
        dw_min:      Minimum weight change per coincident pulse pair.
        functor:     Device model: 'IDEAL', 'CONSTANT_STEP', or 'EXP_STEP'.
        params:      Device-specific parameter dict (optional).
                     CONSTANT_STEP: {'dw_min_std': float}
                     EXP_STEP: {'es_A_up', 'es_A_down', 'es_gamma_up',
                                'es_gamma_down', 'es_a', 'es_b',
                                'w_max', 'w_min', 'up_down'}
        bound_lower: Lower weight clamp bound.
        bound_upper: Upper weight clamp bound.

    Returns:
        None — weights tensor is modified in-place.
    """
    if params is None:
        params = {}

    assert weights.ndim == 2, f"weights must be 2D, got shape {weights.shape}"
    assert x_counts.ndim == 1, f"x_counts must be 1D, got shape {x_counts.shape}"
    assert d_counts.ndim == 1, f"d_counts must be 1D, got shape {d_counts.shape}"
    assert weights.dtype == torch.float32, (
        f"weights must be float32, got {weights.dtype}"
    )

    M, K = weights.shape
    assert x_counts.shape[0] == K, f"x_counts size {x_counts.shape[0]} != in_size {K}"
    assert d_counts.shape[0] == M, f"d_counts size {d_counts.shape[0]} != out_size {M}"

    # Use Triton kernel for IDEAL (GPU-only) when available
    if HAS_TRITON and weights.is_cuda and functor == "IDEAL":
        grid = lambda meta: (  # noqa: E731
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(K, meta["BLOCK_K"]),
        )
        _pulsed_update_ideal_kernel[grid](
            weights,
            x_counts.to(torch.int32),
            d_counts.to(torch.int32),
            M,
            K,
            dw_min,
            bound_lower,
            bound_upper,
        )
        return

    # Use Triton kernel for CONSTANT_STEP when available
    if HAS_TRITON and weights.is_cuda and functor == "CONSTANT_STEP":
        dw_min_std = float(params.get("dw_min_std", 0.0))
        noise = torch.randn(M, K, device=weights.device, dtype=torch.float32)

        grid = lambda meta: (  # noqa: E731
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(K, meta["BLOCK_K"]),
        )
        _pulsed_update_constant_step_kernel[grid](
            weights,
            x_counts.to(torch.int32),
            d_counts.to(torch.int32),
            noise,
            M,
            K,
            dw_min,
            dw_min_std,
            bound_lower,
            bound_upper,
        )
        return

    # Fallback: PyTorch implementation (correct for all functors, CPU + CUDA)
    _pulsed_update_torch(
        weights, x_counts, d_counts, dw_min, functor, params, bound_lower, bound_upper
    )
