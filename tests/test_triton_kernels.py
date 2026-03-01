# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Tests for Triton kernel-level wrappers (public API only).

Covers:
  - math_utils: triton_abs_max, triton_clamp, triton_elem_scale
  - maximizer: triton_row_max, triton_scale_rows
  - bit_line_maker: triton_get_counts
  - pulsed_update: triton_pulsed_update
  - fused_gemm: triton_fused_gemm, triton_fused_gemm_backward
"""

import importlib.util
import os

import pytest
import torch

# Skip entire module if triton is not available
triton = pytest.importorskip("triton")

# Skip entire module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _import_kernel(module_name: str):
    """Import a kernel module directly from the worktree src, bypassing sys.path."""
    src_root = os.path.join(os.path.dirname(__file__), '..', 'src')
    rel_path = module_name.replace('.', os.sep) + '.py'
    full_path = os.path.join(src_root, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_math_utils = _import_kernel('aihwkit.simulator.triton.kernels.math_utils')
triton_abs_max = _math_utils.triton_abs_max
triton_clamp = _math_utils.triton_clamp
triton_elem_scale = _math_utils.triton_elem_scale

_maximizer = _import_kernel('aihwkit.simulator.triton.kernels.maximizer')
triton_row_max = _maximizer.triton_row_max
triton_scale_rows = _maximizer.triton_scale_rows

_bit_line_maker = _import_kernel('aihwkit.simulator.triton.kernels.bit_line_maker')
triton_get_counts = _bit_line_maker.triton_get_counts

_pulsed_update = _import_kernel('aihwkit.simulator.triton.kernels.pulsed_update')
triton_pulsed_update = _pulsed_update.triton_pulsed_update

_fused_gemm = _import_kernel('aihwkit.simulator.triton.kernels.fused_gemm')
triton_fused_gemm = _fused_gemm.triton_fused_gemm
triton_fused_gemm_backward = _fused_gemm.triton_fused_gemm_backward


# ===========================================================================
# math_utils: triton_abs_max
# ===========================================================================


class TestTritonAbsMax:
    """Tests for triton_abs_max kernel wrapper."""

    @pytest.mark.parametrize("size", [64, 256, 1024])
    def test_abs_max_cuda(self, size):
        """CUDA kernel should match x.abs().max() for random tensors."""
        x = torch.randn(size, device="cuda")
        result = triton_abs_max(x)
        expected = x.abs().max()
        assert torch.allclose(result, expected, atol=1e-5), (
            f"size={size}: got {result.item()}, expected {expected.item()}"
        )

    def test_abs_max_negative_dominant(self):
        """Abs max should find the largest-magnitude negative element."""
        x = torch.tensor([0.1, -5.0, 2.0, -0.3], device="cuda")
        result = triton_abs_max(x)
        assert torch.allclose(result, torch.tensor(5.0, device="cuda"), atol=1e-5)

    def test_abs_max_cpu_fallback(self):
        """CPU tensors use torch fallback path."""
        x = torch.randn(128)
        result = triton_abs_max(x)
        expected = x.abs().max()
        assert torch.allclose(result, expected, atol=1e-6)


# ===========================================================================
# math_utils: triton_clamp
# ===========================================================================


class TestTritonClamp:
    """Tests for triton_clamp kernel wrapper."""

    def test_clamp_cuda(self):
        """CUDA kernel should match torch.clamp for random tensors."""
        x = torch.randn(512, device="cuda")
        lo, hi = -0.5, 0.5
        result = triton_clamp(x, lo, hi)
        expected = torch.clamp(x, lo, hi)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_clamp_boundary_values(self):
        """Values exactly at / beyond boundaries should be clamped correctly."""
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], device="cuda")
        result = triton_clamp(x, -0.5, 0.5)
        expected = torch.tensor([-0.5, -0.5, 0.0, 0.5, 0.5], device="cuda")
        assert torch.allclose(result, expected, atol=1e-6)

    def test_clamp_cpu_fallback(self):
        """CPU tensors use torch fallback path."""
        x = torch.randn(256)
        result = triton_clamp(x, -1.0, 1.0)
        expected = torch.clamp(x, -1.0, 1.0)
        assert torch.allclose(result, expected, atol=1e-6)


# ===========================================================================
# math_utils: triton_elem_scale
# ===========================================================================


class TestTritonElemScale:
    """Tests for triton_elem_scale kernel wrapper."""

    def test_elem_scale_cuda(self):
        """CUDA kernel should match x * scalar."""
        x = torch.randn(512, device="cuda")
        scale = 3.14
        result = triton_elem_scale(x, scale)
        expected = x * scale
        assert torch.allclose(result, expected, atol=1e-5)

    def test_elem_scale_cpu_fallback(self):
        """CPU tensors use torch fallback path."""
        x = torch.randn(256)
        result = triton_elem_scale(x, 0.5)
        expected = x * 0.5
        assert torch.allclose(result, expected, atol=1e-6)


# ===========================================================================
# maximizer: triton_row_max
# ===========================================================================


class TestTritonRowMax:
    """Tests for triton_row_max kernel wrapper."""

    def test_row_max_cuda(self):
        """CUDA kernel should match x.abs().max(dim=1).values."""
        x = torch.randn(32, 128, device="cuda")
        result = triton_row_max(x)
        expected = x.abs().max(dim=1).values
        assert torch.allclose(result, expected, atol=1e-5), (
            f"max diff: {(result - expected).abs().max().item()}"
        )

    def test_row_max_single_row(self):
        """Should work with a single-row matrix."""
        x = torch.tensor([[1.0, -3.0, 2.0]], device="cuda")
        result = triton_row_max(x)
        assert torch.allclose(result, torch.tensor([3.0], device="cuda"), atol=1e-5)

    def test_row_max_cpu_fallback(self):
        """CPU tensors use torch fallback path."""
        x = torch.randn(16, 64)
        result = triton_row_max(x)
        expected = x.abs().max(dim=1).values
        assert torch.allclose(result, expected, atol=1e-6)


# ===========================================================================
# maximizer: triton_scale_rows
# ===========================================================================


class TestTritonScaleRows:
    """Tests for triton_scale_rows kernel wrapper."""

    def test_scale_rows_normalizes(self):
        """After scaling, each row's abs max should be ≈ 1.0."""
        x = torch.randn(16, 64, device="cuda")
        row_max = triton_row_max(x)
        scaled = triton_scale_rows(x, row_max)
        row_abs_max = scaled.abs().max(dim=1).values
        assert torch.allclose(row_abs_max, torch.ones_like(row_abs_max), atol=1e-5), (
            f"max deviation: {(row_abs_max - 1.0).abs().max().item()}"
        )

    def test_scale_rows_cpu_fallback(self):
        """CPU tensors use torch fallback path."""
        x = torch.randn(8, 32)
        row_max = x.abs().max(dim=1).values
        result = triton_scale_rows(x, row_max)
        safe_max = row_max.clamp(min=1e-7)
        expected = x / safe_max.unsqueeze(1)
        assert torch.allclose(result, expected, atol=1e-6)


# ===========================================================================
# bit_line_maker: triton_get_counts
# ===========================================================================


class TestTritonGetCounts:
    """Tests for triton_get_counts kernel wrapper."""

    def test_get_counts_deterministic(self):
        """Known input values should produce exact pulse counts.

        Formula: count = round((x + 1) / 2 * bl_count)
          x=-1 → 0, x=0 → 50, x=0.5 → 75, x=1 → 100
        """
        bl = 100
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0], device="cuda")
        counts = triton_get_counts(x, bl)
        expected = torch.tensor([0, 50, 75, 100], device="cuda", dtype=torch.int32)
        assert torch.equal(counts, expected), f"got {counts}, expected {expected}"

    def test_get_counts_range(self):
        """Output counts should all be within [0, bl_count]."""
        bl = 64
        x = torch.randn(256, device="cuda").clamp(-1, 1)
        counts = triton_get_counts(x, bl)
        assert counts.min() >= 0
        assert counts.max() <= bl
        assert counts.dtype == torch.int32

    def test_get_counts_cpu_fallback(self):
        """CPU tensors use torch fallback path."""
        bl = 100
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0])
        counts = triton_get_counts(x, bl)
        expected = torch.tensor([0, 50, 75, 100], dtype=torch.int32)
        assert torch.equal(counts, expected)


# ===========================================================================
# pulsed_update: triton_pulsed_update
# ===========================================================================


class TestTritonPulsedUpdate:
    """Tests for triton_pulsed_update kernel wrapper."""

    def test_coincidence_detection(self):
        """Only positions where BOTH x!=0 AND d!=0 should update."""
        out_size, in_size = 4, 4
        weights = torch.zeros(out_size, in_size, device="cuda", dtype=torch.float32)
        # x_counts: only indices 0,2 are non-zero
        x_counts = torch.tensor([1, 0, 1, 0], device="cuda", dtype=torch.int32)
        # d_counts: only indices 0,1 are non-zero
        d_counts = torch.tensor([1, 1, 0, 0], device="cuda", dtype=torch.int32)
        dw_min = 0.01

        triton_pulsed_update(weights, x_counts, d_counts, dw_min, functor="IDEAL")

        # Coincident positions: (0,0), (0,2), (1,0), (1,2)
        # Non-coincident: all others should remain zero
        assert weights[2, 0].item() == 0.0, "row 2 (d=0) should not update"
        assert weights[3, 1].item() == 0.0, "row 3 (d=0), col 1 (x=0) should not update"
        assert weights[0, 1].item() == 0.0, "col 1 (x=0) should not update"
        # Coincident positions should have changed
        assert weights[0, 0].item() != 0.0, "coincident (0,0) should update"
        assert weights[1, 2].item() != 0.0, "coincident (1,2) should update"

    def test_update_respects_bounds(self):
        """Weights should stay within [bound_lower, bound_upper] after update."""
        out_size, in_size = 8, 8
        weights = torch.full(
            (out_size, in_size), 0.9, device="cuda", dtype=torch.float32
        )
        x_counts = torch.ones(in_size, device="cuda", dtype=torch.int32)
        d_counts = torch.ones(out_size, device="cuda", dtype=torch.int32)

        for _ in range(100):
            triton_pulsed_update(
                weights,
                x_counts,
                d_counts,
                dw_min=0.1,
                functor="IDEAL",
                bound_lower=-1.0,
                bound_upper=1.0,
            )

        assert weights.max().item() <= 1.0 + 1e-6
        assert weights.min().item() >= -1.0 - 1e-6

    def test_pulsed_update_cpu_fallback(self):
        """CPU tensors use PyTorch fallback path."""
        out_size, in_size = 4, 4
        weights = torch.zeros(out_size, in_size, dtype=torch.float32)
        x_counts = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
        d_counts = torch.tensor([1, 1, 0, 0], dtype=torch.int32)

        triton_pulsed_update(weights, x_counts, d_counts, 0.01, functor="IDEAL")

        # Same coincidence checks as CUDA test
        assert weights[2, 0].item() == 0.0
        assert weights[0, 0].item() != 0.0


# ===========================================================================
# fused_gemm: triton_fused_gemm
# ===========================================================================


class TestTritonFusedGemm:
    """Tests for triton_fused_gemm kernel wrapper."""

    def test_fused_gemm_standard_shapes(self):
        """Compare with torch.mm for standard shapes (32,128)x(256,128)."""
        M, K, N = 32, 128, 256
        x = torch.randn(M, K, device="cuda", dtype=torch.float32)
        W = torch.randn(N, K, device="cuda", dtype=torch.float32)
        result = triton_fused_gemm(x, W)
        expected = torch.mm(x, W.T)
        assert torch.allclose(result, expected, atol=1e-4), (
            f"max diff: {(result - expected).abs().max().item()}"
        )

    def test_fused_gemm_non_power_of_two(self):
        """Non-power-of-2 shapes (32,513)x(255,513) should work."""
        M, K, N = 32, 513, 255
        x = torch.randn(M, K, device="cuda", dtype=torch.float32)
        W = torch.randn(N, K, device="cuda", dtype=torch.float32)
        result = triton_fused_gemm(x, W)
        expected = torch.mm(x, W.T)
        assert torch.allclose(result, expected, atol=1e-4), (
            f"max diff: {(result - expected).abs().max().item()}"
        )

    def test_fused_gemm_cpu_fallback(self):
        """CPU tensors use torch.mm fallback."""
        M, K, N = 16, 64, 32
        x = torch.randn(M, K, dtype=torch.float32)
        W = torch.randn(N, K, dtype=torch.float32)
        result = triton_fused_gemm(x, W)
        expected = torch.mm(x, W.T)
        assert torch.allclose(result, expected, atol=1e-6)


# ===========================================================================
# fused_gemm: triton_fused_gemm_backward
# ===========================================================================


class TestTritonFusedGemmBackward:
    """Tests for triton_fused_gemm_backward kernel wrapper."""

    def test_backward_standard_shapes(self):
        """Compare with torch.mm(d, W) for shapes (32,256)x(256,128) → (32,128)."""
        M, N, K = 32, 256, 128
        d = torch.randn(M, N, device="cuda", dtype=torch.float32)
        W = torch.randn(N, K, device="cuda", dtype=torch.float32)
        result = triton_fused_gemm_backward(d, W)
        expected = torch.mm(d, W)
        assert result.shape == (M, K)
        assert torch.allclose(result, expected, atol=1e-4), (
            f"max diff: {(result - expected).abs().max().item()}"
        )

    def test_backward_cpu_fallback(self):
        """CPU tensors use torch.mm fallback."""
        M, N, K = 16, 32, 64
        d = torch.randn(M, N, dtype=torch.float32)
        W = torch.randn(N, K, dtype=torch.float32)
        result = triton_fused_gemm_backward(d, W)
        expected = torch.mm(d, W)
        assert torch.allclose(result, expected, atol=1e-6)
