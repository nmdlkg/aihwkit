"""Triton math utility kernels: reductions and element-wise operations.

Provides standalone Triton kernels ported from:
- rpucuda/cuda/cuda_math_util.cu (elemscale, aclip, elemabs)
- rpucuda/cuda/maximizer.cu (atomicMaxFP reduction)

Functions:
    triton_abs_max: absolute maximum reduction
    triton_clamp: element-wise clamp
    triton_elem_scale: element-wise scalar multiply
    triton_decay: weight decay (x *= 1 - rate)

Each kernel-backed operation has CPU fallback via torch operations.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = [
    "triton_abs_max",
    "triton_clamp",
    "triton_elem_scale",
    "triton_decay",
]

# ---------------------------------------------------------------------------
# Autotuning configurations
# ---------------------------------------------------------------------------
_REDUCTION_CONFIGS = [triton.Config({"BLOCK_SIZE": bs}) for bs in [256, 512, 1024]]

_ELEMENTWISE_CONFIGS = [triton.Config({"BLOCK_SIZE": bs}) for bs in [256, 512, 1024]]


# ===========================================================================
# Kernel 1: Absolute Max Reduction
# Ported from maximizer.cu: atomicMaxFP + kernelMaximizeBatchTrans
# ===========================================================================
@triton.autotune(configs=_REDUCTION_CONFIGS, key=["n_elements"])
@triton.jit
def _abs_max_kernel(
    x_ptr,
    partial_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute per-block absolute max, write to partial results buffer.

    Each program instance reduces BLOCK_SIZE elements to a single abs-max
    value and stores it at ``partial_out_ptr[program_id]``.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    abs_x = tl.abs(x)
    block_max = tl.max(abs_x, axis=0)
    tl.store(partial_out_ptr + pid, block_max)


def triton_abs_max(x: torch.Tensor) -> torch.Tensor:
    """Return scalar tensor with the absolute maximum of *x*.

    Uses a two-pass approach: a Triton kernel computes per-block absolute
    max values, then ``torch.max`` reduces across blocks.

    CPU fallback: ``x.abs().max()``
    """
    if not x.is_cuda:
        return x.abs().max()

    n = x.numel()
    if n == 0:
        return torch.tensor(0.0, device=x.device, dtype=torch.float32)

    x_flat = x.contiguous().reshape(-1).to(torch.float32)
    # Allocate for worst-case block count (smallest BLOCK_SIZE in configs).
    max_blocks = triton.cdiv(n, 256)
    # Zeros are safe: abs values >= 0, so unused slots don't affect max.
    partial = torch.zeros(max_blocks, device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _abs_max_kernel[grid](x_flat, partial, n)

    return partial.max()


# ===========================================================================
# Kernel 2: Element-wise Clamp
# Ported from cuda_math_util.cu: kernelAClip / kernelElemSat
# ===========================================================================
@triton.autotune(configs=_ELEMENTWISE_CONFIGS, key=["n_elements"])
@triton.jit
def _clamp_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    """Clamp each element to [min_val, max_val]."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    clamped = tl.maximum(tl.minimum(x, max_val), min_val)
    tl.store(out_ptr + offsets, clamped, mask=mask)


def triton_clamp(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """Element-wise clamp to ``[min_val, max_val]``.

    CPU fallback: ``torch.clamp(x, min_val, max_val)``
    """
    if not x.is_cuda:
        return x.clamp(min_val, max_val)

    x_flat = x.contiguous().reshape(-1)
    out = torch.empty_like(x_flat)
    n = x_flat.numel()

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _clamp_kernel[grid](x_flat, out, n, min_val, max_val)

    return out.reshape(x.shape)


# ===========================================================================
# Kernel 3: Element-wise Scale
# Ported from cuda_math_util.cu: kernelElemScale (scalar alpha variant)
# ===========================================================================
@triton.autotune(configs=_ELEMENTWISE_CONFIGS, key=["n_elements"])
@triton.jit
def _elem_scale_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    """Multiply each element by a scalar."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * scale, mask=mask)


def triton_elem_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Element-wise multiply by scalar.

    CPU fallback: ``x * scale``
    """
    if not x.is_cuda:
        return x * scale

    x_flat = x.contiguous().reshape(-1)
    out = torch.empty_like(x_flat)
    n = x_flat.numel()

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _elem_scale_kernel[grid](x_flat, out, n, scale)

    return out.reshape(x.shape)


# ===========================================================================
# Kernel 4: Weight Decay
# Decay: x * (1 - decay_rate), delegates to triton_elem_scale.
# ===========================================================================
def triton_decay(x: torch.Tensor, decay_rate: float) -> torch.Tensor:
    """Apply weight decay: ``x * (1 - decay_rate)``. Returns a new tensor.

    Internally delegates to :func:`triton_elem_scale` with
    ``scale = 1.0 - decay_rate``.

    CPU fallback: ``x * (1.0 - decay_rate)``
    """
    if not x.is_cuda:
        return x * (1.0 - decay_rate)

    return triton_elem_scale(x, 1.0 - decay_rate)
