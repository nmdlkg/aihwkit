# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Triton-based random number generation kernels.

Replaces cuRAND for the aihwkit Triton backend.
Provides Gaussian and uniform noise generation via Triton JIT kernels.
"""

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton Kernels
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
    def triton_gaussian_noise_kernel(
        out_ptr,
        n_elements: tl.constexpr,
        seed: tl.int64,
        mean: tl.float32,
        std: tl.float32,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Generate Gaussian noise via Box-Muller transform.

        Uses two independent uniform streams from tl.rand and transforms
        them into standard-normal samples via Box-Muller.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Two independent uniform [0,1) streams
        u1 = tl.rand(seed, offsets)
        u2 = tl.rand(seed + 1, offsets)

        # Box-Muller transform: Z ~ N(0,1)
        # Add small epsilon to avoid log(0)
        z = tl.sqrt(-2.0 * tl.log(u1 + 1e-7)) * tl.cos(2.0 * 3.14159265358979 * u2)

        result = z * std + mean
        tl.store(out_ptr + offsets, result, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def triton_uniform_noise_kernel(
        out_ptr,
        n_elements: tl.constexpr,
        seed: tl.int64,
        low: tl.float32,
        high: tl.float32,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Generate uniform noise in [low, high).

        Uses tl.rand directly and scales to the desired range.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Uniform [0,1) → [low, high)
        result = low + (high - low) * tl.rand(seed, offsets)
        tl.store(out_ptr + offsets, result, mask=mask)


# ---------------------------------------------------------------------------
# CPU fallbacks (always available)
# ---------------------------------------------------------------------------


def _cpu_gaussian_noise(n: int, mean: float, std: float) -> torch.Tensor:
    """CPU fallback for Gaussian noise generation."""
    return torch.randn(n, dtype=torch.float32) * std + mean


def _cpu_uniform_noise(n: int, low: float, high: float) -> torch.Tensor:
    """CPU fallback for uniform noise generation."""
    return torch.empty(n, dtype=torch.float32).uniform_(low, high)


# ---------------------------------------------------------------------------
# Python-level wrappers (device-aware)
# ---------------------------------------------------------------------------


def gaussian_noise(
    n_elements: int,
    seed: int,
    mean: float = 0.0,
    std: float = 1.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate Gaussian-distributed noise.

    Args:
        n_elements: Number of random values to generate.
        seed: RNG seed (integer). Deterministic for same seed+shape.
        mean: Distribution mean.
        std: Distribution standard deviation.
        device: Target device ('cuda' or 'cpu').

    Returns:
        Float32 tensor of shape (n_elements,) on the specified device.
    """
    if (
        not HAS_TRITON
        or not str(device).startswith("cuda")
        or not torch.cuda.is_available()
    ):
        return _cpu_gaussian_noise(n_elements, mean, std).to(device)

    out = torch.empty(n_elements, device=device, dtype=torch.float32)
    # Grid is determined by autotune; use a lambda so autotune can inject BLOCK_SIZE
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
    triton_gaussian_noise_kernel[grid](
        out,
        n_elements,
        seed,
        mean,
        std,
    )
    return out


def uniform_noise(
    n_elements: int,
    seed: int,
    low: float = 0.0,
    high: float = 1.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate uniformly-distributed noise in [low, high).

    Args:
        n_elements: Number of random values to generate.
        seed: RNG seed (integer). Deterministic for same seed+shape.
        low: Lower bound (inclusive).
        high: Upper bound (exclusive).
        device: Target device ('cuda' or 'cpu').

    Returns:
        Float32 tensor of shape (n_elements,) on the specified device.
    """
    if (
        not HAS_TRITON
        or not str(device).startswith("cuda")
        or not torch.cuda.is_available()
    ):
        return _cpu_uniform_noise(n_elements, low, high).to(device)

    out = torch.empty(n_elements, device=device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
    triton_uniform_noise_kernel[grid](
        out,
        n_elements,
        seed,
        low,
        high,
    )
    return out


# ---------------------------------------------------------------------------
# TritonNoiseManager
# ---------------------------------------------------------------------------


class TritonNoiseManager:
    """Manages seeded Triton-based noise generation for analog tile simulation.

    Provides stateful RNG via an internal counter so each call produces
    statistically independent (but deterministic) noise given the same
    initial seed.

    Example::

        mgr = TritonNoiseManager(seed=42)
        w_noise = mgr.gaussian(shape=(256, 256), std=0.1, device='cuda')
        inp_noise = mgr.uniform(shape=(32,), low=-0.5, high=0.5, device='cuda')
    """

    def __init__(self, seed: int = 0) -> None:
        """Initialize the noise manager.

        Args:
            seed: Base seed for all noise generation calls.
        """
        self.seed = seed
        self._counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def gaussian(
        self,
        shape: Tuple[int, ...],
        mean: float = 0.0,
        std: float = 1.0,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Generate Gaussian noise of the requested shape.

        Each call uses a unique seed derived from (self.seed + counter) so
        successive calls produce independent noise streams.

        Args:
            shape: Output tensor shape.
            mean: Gaussian mean.
            std: Gaussian standard deviation.
            device: Target device.

        Returns:
            Float32 tensor of given shape.
        """
        n = 1
        for d in shape:
            n *= d
        seed = self.seed + self._counter
        self._counter += 1
        noise = gaussian_noise(n, seed, mean, std, device)
        return noise.reshape(shape)

    def uniform(
        self,
        shape: Tuple[int, ...],
        low: float = 0.0,
        high: float = 1.0,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Generate uniform noise of the requested shape.

        Args:
            shape: Output tensor shape.
            low: Lower bound (inclusive).
            high: Upper bound (exclusive).
            device: Target device.

        Returns:
            Float32 tensor of given shape.
        """
        n = 1
        for d in shape:
            n *= d
        seed = self.seed + self._counter
        self._counter += 1
        noise = uniform_noise(n, seed, low, high, device)
        return noise.reshape(shape)

    def reset(self) -> None:
        """Reset the internal counter (reuse the same seed sequence)."""
        self._counter = 0

    def __repr__(self) -> str:
        return f"TritonNoiseManager(seed={self.seed}, counter={self._counter})"
