#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Benchmark: Triton fused GEMM vs cuBLAS for TritonAnalogTile operations.

Compares wall-clock time of:
  - Forward pass:  cuBLAS (default) vs Triton fused GEMM (use_triton_gemm=True)
  - Backward pass: cuBLAS (default) vs Triton fused GEMM (use_triton_gemm=True)
  - Update:        ConstantStepTritonTile pulsed update vs ideal outer-product update

Matrix sizes: 64x64, 256x256, 512x512, 1024x1024

Usage::

    python benchmarks/triton_vs_cublas.py
"""

import sys
import time
from types import SimpleNamespace

import torch

from aihwkit.simulator.triton.tiles.analog import TritonAnalogTile
from aihwkit.simulator.triton.tiles.constant_step import ConstantStepTritonTile

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MATRIX_SIZES = [(64, 64), (256, 256), (512, 512), (1024, 1024)]
BATCH_SIZE = 128
WARMUP_ITERS = 10
TIMING_ITERS = 100
DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_constant_step_config():
    """Create a minimal config namespace for ConstantStepTritonTile."""
    return SimpleNamespace(
        forward=None,
        backward=None,
        use_triton_gemm=False,
        device=None,
        update=SimpleNamespace(desired_bl=31),
        use_triton=True,
    )


def _time_fn(fn, warmup=WARMUP_ITERS, iters=TIMING_ITERS):
    """Time a callable on GPU with proper synchronization.

    Returns:
        Mean wall-clock time in milliseconds.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iters
    return elapsed_ms


def _speedup(base_ms, fast_ms):
    """Compute speedup ratio (>1 means fast_ms is faster)."""
    if fast_ms <= 0:
        return float("inf")
    return base_ms / fast_ms


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------


def bench_forward(out_size, in_size):
    """Benchmark forward pass: cuBLAS vs Triton fused GEMM."""
    # cuBLAS tile (default: _use_triton_gemm=False)
    tile_cublas = TritonAnalogTile(out_size, in_size, rpu_config=None).cuda()
    tile_cublas.set_weights_uniform_random(-0.5, 0.5)

    # Triton fused GEMM tile
    tile_triton = TritonAnalogTile(out_size, in_size, rpu_config=None).cuda()
    tile_triton.set_weights(tile_cublas.get_weights()[0])
    tile_triton._use_triton_gemm = True

    x = torch.randn(BATCH_SIZE, in_size, device=DEVICE)

    t_cublas = _time_fn(lambda: tile_cublas._forward_impl(x, is_test=True))
    t_triton = _time_fn(lambda: tile_triton._forward_impl(x, is_test=True))

    return t_cublas, t_triton


def bench_backward(out_size, in_size):
    """Benchmark backward pass: cuBLAS vs Triton fused GEMM."""
    tile_cublas = TritonAnalogTile(out_size, in_size, rpu_config=None).cuda()
    tile_cublas.set_weights_uniform_random(-0.5, 0.5)

    tile_triton = TritonAnalogTile(out_size, in_size, rpu_config=None).cuda()
    tile_triton.set_weights(tile_cublas.get_weights()[0])
    tile_triton._use_triton_gemm = True

    d = torch.randn(BATCH_SIZE, out_size, device=DEVICE)

    t_cublas = _time_fn(lambda: tile_cublas._backward_impl(d))
    t_triton = _time_fn(lambda: tile_triton._backward_impl(d))

    return t_cublas, t_triton


def bench_update(out_size, in_size):
    """Benchmark pulsed update: absolute time for Triton kernel pipeline."""
    cfg = _make_constant_step_config()
    tile = ConstantStepTritonTile(out_size, in_size, cfg).cuda()
    tile.set_weights_uniform_random(-0.5, 0.5)
    tile.set_learning_rate(0.01)

    x = torch.randn(BATCH_SIZE, in_size, device=DEVICE)
    d = torch.randn(BATCH_SIZE, out_size, device=DEVICE)

    t_update = _time_fn(lambda: tile.update(x, d))
    return t_update


def bench_update_vs_ideal(out_size, in_size):
    """Compare ConstantStep pulsed update (Triton) vs ideal (outer-product) update."""
    # Pulsed update (Triton kernel pipeline)
    cfg_pulsed = _make_constant_step_config()
    tile_pulsed = ConstantStepTritonTile(out_size, in_size, cfg_pulsed).cuda()
    tile_pulsed.set_weights_uniform_random(-0.5, 0.5)
    tile_pulsed.set_learning_rate(0.01)

    # Ideal update (outer-product, base TritonAnalogTile)
    tile_ideal = TritonAnalogTile(out_size, in_size, rpu_config=None).cuda()
    tile_ideal.set_weights_uniform_random(-0.5, 0.5)
    tile_ideal.set_learning_rate(0.01)

    x = torch.randn(BATCH_SIZE, in_size, device=DEVICE)
    d = torch.randn(BATCH_SIZE, out_size, device=DEVICE)

    t_pulsed = _time_fn(lambda: tile_pulsed.update(x, d))
    t_ideal = _time_fn(lambda: tile_ideal.update(x, d))

    return t_ideal, t_pulsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Benchmark requires GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 80)
    print(f"  Triton vs cuBLAS Benchmark — {gpu_name}")
    print(
        f"  Batch size: {BATCH_SIZE}  |  Warmup: {WARMUP_ITERS}  |  Iters: {TIMING_ITERS}"
    )
    print("=" * 80)

    # ---- Forward ----
    print("\n--- Forward Pass (cuBLAS vs Triton Fused GEMM) ---")
    print(f"{'Size':>12s}  {'cuBLAS (ms)':>12s}  {'Triton (ms)':>12s}  {'Speedup':>8s}")
    print("-" * 50)
    for out_sz, in_sz in MATRIX_SIZES:
        t_c, t_t = bench_forward(out_sz, in_sz)
        sp = _speedup(t_c, t_t)
        print(f"  {out_sz:>4d}x{in_sz:<4d}  {t_c:>12.4f}  {t_t:>12.4f}  {sp:>7.2f}x")

    # ---- Backward ----
    print("\n--- Backward Pass (cuBLAS vs Triton Fused GEMM) ---")
    print(f"{'Size':>12s}  {'cuBLAS (ms)':>12s}  {'Triton (ms)':>12s}  {'Speedup':>8s}")
    print("-" * 50)
    for out_sz, in_sz in MATRIX_SIZES:
        t_c, t_t = bench_backward(out_sz, in_sz)
        sp = _speedup(t_c, t_t)
        print(f"  {out_sz:>4d}x{in_sz:<4d}  {t_c:>12.4f}  {t_t:>12.4f}  {sp:>7.2f}x")

    # ---- Update: Pulsed vs Ideal ----
    print("\n--- Update: Pulsed (ConstantStep Triton) vs Ideal (outer-product) ---")
    print(f"{'Size':>12s}  {'Ideal (ms)':>12s}  {'Pulsed (ms)':>12s}  {'Ratio':>8s}")
    print("-" * 50)
    for out_sz, in_sz in MATRIX_SIZES:
        t_ideal, t_pulsed = bench_update_vs_ideal(out_sz, in_sz)
        ratio = _speedup(t_ideal, t_pulsed)
        print(
            f"  {out_sz:>4d}x{in_sz:<4d}  {t_ideal:>12.4f}  {t_pulsed:>12.4f}  {ratio:>7.2f}x"
        )

    # ---- Update: Absolute Triton pipeline time ----
    print("\n--- Update: ConstantStep Triton Pipeline (absolute) ---")
    print(f"{'Size':>12s}  {'Time (ms)':>12s}")
    print("-" * 30)
    for out_sz, in_sz in MATRIX_SIZES:
        t = bench_update(out_sz, in_sz)
        print(f"  {out_sz:>4d}x{in_sz:<4d}  {t:>12.4f}")

    print("\n" + "=" * 80)
    print("  Benchmark complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
