# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Triton kernel for bit line maker (pulse count generation).

Converts scaled analog input values in [-1, 1] to digital pulse counts
in [0, bl_count]. This is the quantization step that maps continuous
analog signals to discrete pulse trains for the crossbar array.

Algorithm:
    count = round((x + 1.0) / 2.0 * bl_count)
    count = clamp(count, 0, bl_count)

Examples:
    x = -1.0 → 0
    x =  0.0 → bl_count / 2
    x =  0.5 → 75  (for bl_count=100)
    x =  1.0 → bl_count
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["n_elements"],
)
@triton.jit
def _get_counts_kernel(
    x_ptr,  # float32* — input values in [-1, 1]
    out_ptr,  # int32* — output pulse counts
    n_elements,
    bl_count: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel: convert analog values to pulse counts.

    Maps x ∈ [-1, 1] → count ∈ [0, bl_count] via:
        normalized = (x + 1) * 0.5       # map to [0, 1]
        count = round(normalized * bl_count)
        count = clamp(count, 0, bl_count)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Convert to pulse count: [-1, 1] -> [0, 1] -> [0, bl_count]
    normalized = (x + 1.0) * 0.5
    counts_f = normalized * bl_count

    # Round to nearest integer: floor(x + 0.5) == round(x) for positive values
    # Since counts_f is in [0, bl_count] (non-negative after normalization),
    # floor(x + 0.5) gives correct rounding behavior.
    counts = (counts_f + 0.5).to(tl.int32)

    # Clamp to valid range [0, bl_count]
    zeros = tl.zeros_like(counts)
    bl_max = zeros + bl_count
    counts = tl.maximum(counts, zeros)
    counts = tl.minimum(counts, bl_max)

    tl.store(out_ptr + offs, counts, mask=mask)


def triton_get_counts(x_scaled: torch.Tensor, bl_count: int) -> torch.Tensor:
    """Convert scaled input [-1, 1] to pulse counts [0, bl_count].

    Deterministic rounding: count = round((x + 1) / 2 * bl_count)

    Args:
        x_scaled: Input tensor with values in [-1, 1]. Any shape.
        bl_count: Number of bit lines (pulse resolution).

    Returns:
        Tensor of int32 pulse counts in [0, bl_count], same shape as input.
    """
    if not x_scaled.is_cuda:
        # CPU fallback — no Triton needed
        normalized = (x_scaled + 1.0) * 0.5
        counts = (normalized * bl_count).round().long().clamp(0, bl_count)
        return counts.to(torch.int32)

    out = torch.empty_like(x_scaled, dtype=torch.int32)
    n = x_scaled.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _get_counts_kernel[grid](
        x_scaled.contiguous().view(-1),
        out.view(-1),
        n,
        bl_count,
    )
    return out.reshape(x_scaled.shape)


def triton_get_counts_stochastic(
    x_scaled: torch.Tensor, bl_count: int, seed: int = 0
) -> torch.Tensor:
    """Stochastic rounding variant: add uniform noise before quantizing.

    Adds uniform noise in [-0.5, 0.5] scaled to one count step before
    deterministic rounding. This provides unbiased quantization on average.

    Args:
        x_scaled: Input tensor with values in [-1, 1]. Any shape.
        bl_count: Number of bit lines (pulse resolution).
        seed: Random seed (used via torch generator for reproducibility).

    Returns:
        Tensor of int32 pulse counts in [0, bl_count], same shape as input.
    """
    if seed != 0:
        gen = torch.Generator(device=x_scaled.device)
        gen.manual_seed(seed)
        noise = (
            torch.rand(
                x_scaled.shape,
                device=x_scaled.device,
                dtype=x_scaled.dtype,
                generator=gen,
            )
            - 0.5
        )
    else:
        noise = torch.rand_like(x_scaled) - 0.5

    # Noise scaled to one count step in the input domain:
    # One count step in normalized space = 1/bl_count
    # In input space [-1, 1] (range 2): step = 2/bl_count
    x_noisy = x_scaled + noise * 2.0 / bl_count
    return triton_get_counts(x_noisy, bl_count)
