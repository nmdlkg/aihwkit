"""Triton kernels for per-row absolute max finding and row scaling operations."""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({"BLOCK_N": b}) for b in [64, 128, 256, 512]], key=["N"]
)
@triton.jit
def _row_abs_max_kernel(
    x_ptr,  # float32* [M, N]
    out_ptr,  # float32* [M]
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    """Per-row absolute max reduction kernel.

    Computes the absolute maximum value for each row of a 2D tensor.
    One program per row, processes columns in BLOCK_N chunks.
    """
    row = tl.program_id(0)  # one program per row
    row_start = row * N
    
    # Initialize max to -inf (will be updated with abs values)
    row_max = float('-inf')
    
    # Iterate over column chunks
    offs = tl.arange(0, BLOCK_N)
    for k in range(0, tl.cdiv(N, BLOCK_N)):
        col = k * BLOCK_N + offs
        mask = col < N
        x = tl.load(x_ptr + row_start + col, mask=mask, other=0.0)
        abs_x = tl.abs(x)
        block_max = tl.max(abs_x)
        row_max = tl.maximum(row_max, block_max)
    
    tl.store(out_ptr + row, row_max)


def triton_row_max(x: torch.Tensor) -> torch.Tensor:
    """Compute per-row absolute max of 2D tensor.

    Uses Triton kernel for GPU acceleration when available.
    Falls back to torch for CPU tensors.

    Args:
        x: [M, N] tensor (float32)

    Returns:
        [M] tensor with absolute max of each row
    """
    if not x.is_cuda:
        # CPU fallback
        return x.abs().max(dim=1).values

    M, N = x.shape
    out = torch.empty(M, dtype=x.dtype, device=x.device)

    # Launch kernel: one block per row
    grid = (M,)
    _row_abs_max_kernel[grid](
        x,
        out,
        M,
        N,
    )

    return out


@triton.autotune(
    configs=[triton.Config({"BLOCK_N": b}) for b in [64, 128, 256, 512]], key=["N"]
)
@triton.jit
def _scale_rows_kernel(
    x_ptr,  # float32* [M, N]
    row_max_ptr,  # float32* [M]
    out_ptr,  # float32* [M, N]
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    """Row-wise scaling kernel.

    Divides each row by its max value (element-wise division).
    One program per row, processes columns in BLOCK_N chunks.
    """
    row = tl.program_id(0)
    row_start = row * N

    # Load the max value for this row
    row_max = tl.load(row_max_ptr + row)
    # Clamp to avoid division by zero
    safe_max = tl.maximum(row_max, 1e-7)

    # Process columns in chunks
    offs = tl.arange(0, BLOCK_N)
    for k in range(0, tl.cdiv(N, BLOCK_N)):
        col = k * BLOCK_N + offs
        mask = col < N
        x = tl.load(x_ptr + row_start + col, mask=mask, other=0.0)
        scaled = x / safe_max
        tl.store(out_ptr + row_start + col, scaled, mask=mask)


def triton_scale_rows(x: torch.Tensor, row_max: torch.Tensor) -> torch.Tensor:
    """Divide each row by its max value (normalize rows to [-1, 1]).

    Uses Triton kernel for GPU acceleration when available.
    Falls back to torch for CPU tensors.

    Args:
        x: [M, N] tensor (float32)
        row_max: [M] per-row max values (float32)

    Returns:
        [M, N] tensor with rows normalized by their max values
    """
    if not x.is_cuda:
        # CPU fallback
        safe_max = row_max.clamp(min=1e-7)
        return x / safe_max.unsqueeze(1)

    M, N = x.shape
    out = torch.empty_like(x)

    # Launch kernel: one block per row
    grid = (M,)
    _scale_rows_kernel[grid](
        x,
        row_max,
        out,
        M,
        N,
    )

    return out
