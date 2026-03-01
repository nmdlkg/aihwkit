# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Triton-based backward MVM (matrix-vector multiply) kernel.

Computes the backward pass: x_grad = clamp(d @ W + noise, -bound, bound)
where W: [out_size, in_size], d: [batch, out_size] → result: [batch, in_size].

In column-major notation this is W.T @ d_col, but using PyTorch row convention:
    x_grad = d @ W  (shape: [batch, in_size])
"""

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton kernel: element-wise add-noise-clamp
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _add_noise_clamp_kernel(
        out_ptr,
        y_ptr,
        noise_ptr,
        n_elements: tl.constexpr,
        noise_std: tl.float32,
        bound: tl.float32,
        use_bound: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Element-wise: out = clamp(y + noise * noise_std, -bound, bound).

        When use_bound is False the clamp is skipped for performance.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        noise = tl.load(noise_ptr + offsets, mask=mask, other=0.0)

        result = y + noise * noise_std

        if use_bound:
            result = tl.minimum(tl.maximum(result, -bound), bound)

        tl.store(out_ptr + offsets, result, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper for add_noise_clamp
# ---------------------------------------------------------------------------


def _add_noise_clamp(
    y: torch.Tensor,
    noise: torch.Tensor,
    noise_std: float,
    bound: float,
) -> torch.Tensor:
    """Apply noise and optional clamp to a flat or batched tensor.

    Args:
        y: Base tensor (any shape, float32).
        noise: Noise tensor same shape as y.
        noise_std: Noise scale factor.
        bound: Clamp bound (inf to disable).

    Returns:
        Tensor of same shape: clamp(y + noise * noise_std, -bound, bound).
    """
    if not y.is_cuda or not HAS_TRITON:
        result = y + noise * noise_std
        if bound < float("inf"):
            result = result.clamp(-bound, bound)
        return result

    flat_y = y.contiguous().reshape(-1)
    flat_noise = noise.contiguous().reshape(-1)
    n = flat_y.numel()
    out = torch.empty_like(flat_y)
    use_bound = bound < float("inf")

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)  # noqa: E731
    _add_noise_clamp_kernel[grid](
        out,
        flat_y,
        flat_noise,
        n,
        noise_std,
        bound if use_bound else 1.0,  # dummy value when not used
        use_bound,
    )
    return out.reshape(y.shape)


# ---------------------------------------------------------------------------
# Public API: triton_backward_mvm
# ---------------------------------------------------------------------------


def triton_backward_mvm(
    weights: torch.Tensor,
    d: torch.Tensor,
    noise_std: float = 0.0,
    bound: float = float("inf"),
    seed: int = 0,
) -> torch.Tensor:
    """Backward pass MVM: x_grad = clamp(d @ weights + noise, -bound, bound).

    Computes the gradient of the input given upstream gradient *d* and the
    weight matrix *weights*.  In column notation this is W^T @ d_col; in
    PyTorch row convention it is ``d @ weights``.

    Args:
        weights: Weight matrix of shape [out_size, in_size].
        d: Upstream gradient of shape [batch, out_size].
        noise_std: Standard deviation of additive Gaussian noise.  Pass 0.0
            (default) for a noiseless, purely deterministic result.
        bound: Symmetric clamp bound applied after noise injection.  Use
            ``float('inf')`` (default) to disable clamping.
        seed: Integer seed for the Triton noise kernel.  The same seed always
            produces the same noise for a given tensor size.

    Returns:
        Gradient tensor of shape [batch, in_size] on the same device as
        *weights* / *d*.
    """
    # Transposed MVM: d @ W  ≡  W^T @ d (column notation)
    # weights: [out_size, in_size], d: [batch, out_size]
    # y: [batch, in_size]
    y = d @ weights  # [batch, in_size]

    if noise_std == 0.0:
        if bound < float("inf"):
            return y.clamp(-bound, bound)
        return y

    # Generate Gaussian noise matching output shape
    from aihwkit.simulator.triton.kernels.rng import gaussian_noise  # noqa: PLC0415

    device_str = y.device.type
    noise = gaussian_noise(y.numel(), seed, 0.0, 1.0, device_str)
    noise = noise.reshape(y.shape)

    if y.is_cuda and HAS_TRITON:
        return _add_noise_clamp(y, noise, noise_std, bound)

    # CPU fallback
    result = y + noise * noise_std
    if bound < float("inf"):
        result = result.clamp(-bound, bound)
    return result
