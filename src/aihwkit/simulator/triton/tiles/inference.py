# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Triton-based inference tile with programming noise, drift, and read noise."""

from typing import Optional, Any

import torch
from torch import Tensor

from aihwkit.simulator.triton.tiles.analog import TritonAnalogTile


class InferenceTritonTile(TritonAnalogTile):
    """Triton-accelerated tile for analog inference with programming noise and drift.

    Extends :class:`TritonAnalogTile` with inference-specific functionality:

    - **Programming noise**: Gaussian noise applied to weights via
      :meth:`program_weights` to simulate conductance write variability.
    - **Power-law drift**: Weights drift over time via :meth:`drift_weights`
      using ``w *= (t / t0)^(-nu)`` to model conductance relaxation.
    - **Read noise**: Optional Gaussian noise added to forward-pass output
      during inference to model ADC/sensor noise.

    This tile is intended for inference only. The :meth:`update` method
    raises ``NotImplementedError`` since training should be done with
    :class:`TritonAnalogTile` and weights transferred for inference.

    Args:
        out_size: Output (row) dimension of the weight matrix.
        in_size: Input (column) dimension of the weight matrix.
        rpu_config: RPU config object (typically ``InferenceRPUConfig``).
        bias: Unused flag kept for API compatibility.
    """

    def __init__(
        self,
        out_size: int,
        in_size: int,
        rpu_config: Any = None,
        bias: bool = False,
    ) -> None:
        super().__init__(out_size, in_size, rpu_config, bias)

        # Inference state
        self._programmed_weights: Optional[Tensor] = None
        self._drift_noise_parameters: Optional[Tensor] = None
        self._t_inference: float = 0.0

        # Extract noise/drift parameters from config (duck-typed)
        noise_model = getattr(rpu_config, "noise_model", None) if rpu_config else None
        self._prog_noise_scale: float = (
            float(getattr(noise_model, "prog_noise_scale", 0.06))
            if noise_model
            else 0.06
        )
        self._read_noise_scale: float = (
            float(getattr(noise_model, "read_noise_scale", 0.01))
            if noise_model
            else 0.01
        )
        self._drift_nu: float = (
            float(getattr(noise_model, "drift_nu", 0.06)) if noise_model else 0.06
        )
        self._drift_nu_std: float = (
            float(getattr(noise_model, "drift_nu_std", 0.003)) if noise_model else 0.003
        )
        self._t0: float = 1.0  # reference time (seconds)

    # ------------------------------------------------------------------
    # Programming noise
    # ------------------------------------------------------------------

    @torch.no_grad()
    def program_weights(self, from_reference: bool = True) -> None:
        """Apply programming noise to weights.

        Simulates the imprecision of writing conductance values to analog
        devices. Adds Gaussian noise proportional to ``prog_noise_scale``
        and the absolute weight magnitude.

        After programming, the noisy weights are stored as the baseline
        for subsequent drift experiments.

        Args:
            from_reference: If True and programmed weights already exist,
                re-program from the original (pre-noise) weights.
        """
        if from_reference and self._programmed_weights is not None:
            # Reset to original weights before re-programming
            self.weight.data.copy_(self._programmed_weights)

        if self._prog_noise_scale > 0:
            # Multiplicative Gaussian noise: w += scale * |w| * N(0,1)
            noise = (
                self._prog_noise_scale
                * self.weight.data.abs()
                * torch.randn_like(self.weight.data)
            )
            self.weight.data.add_(noise)

        # Store programmed weights as baseline for drift
        self._programmed_weights = self.weight.data.clone()

        # Generate per-weight drift exponents: nu ~ N(drift_nu, drift_nu_std)
        self._drift_noise_parameters = (
            self._drift_nu + self._drift_nu_std * torch.randn_like(self.weight.data)
        ).clamp(min=0.0)

    # ------------------------------------------------------------------
    # Power-law drift
    # ------------------------------------------------------------------

    @torch.no_grad()
    def drift_weights(self, t_inference: float = 0.0) -> None:
        """Apply power-law drift to weights.

        Models conductance relaxation over time using the power-law:
        ``w(t) = w_programmed * (t / t0)^(-nu)``

        where ``nu`` is a per-weight drift exponent drawn during
        :meth:`program_weights`.

        Args:
            t_inference: Time (in seconds) since programming.
                Programming ends at t=0. Drift is only applied when
                ``t_inference > 0``.
        """
        if self._programmed_weights is None:
            self.program_weights()

        self._t_inference = t_inference

        if t_inference > 0 and self._drift_noise_parameters is not None:
            # Power-law drift: w *= (t / t0)^(-nu)
            drift_factor = (t_inference / self._t0) ** (-self._drift_noise_parameters)
            self.weight.data.copy_(self._programmed_weights * drift_factor)
        else:
            # No drift at t=0: just use programmed weights
            self.weight.data.copy_(self._programmed_weights)

    # ------------------------------------------------------------------
    # Forward with read noise
    # ------------------------------------------------------------------

    def _forward_impl(self, x: Tensor, is_test: bool = False) -> Tensor:
        """Forward MVM with optional read noise during inference.

        Calls the parent Triton forward kernel, then adds Gaussian
        read noise to the output when in test/inference mode.
        """
        y = super()._forward_impl(x, is_test=is_test)

        # Add read noise during inference
        if is_test and self._read_noise_scale > 0:
            y = y + self._read_noise_scale * torch.randn_like(y)

        return y

    # ------------------------------------------------------------------
    # Update (disabled for inference tile)
    # ------------------------------------------------------------------

    def update(self, x_input: Tensor, d_input: Tensor, **kwargs) -> Tensor:
        """Weight update is not supported on inference tiles.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            "InferenceTritonTile is for inference only. "
            "Train with TritonAnalogTile, then transfer weights."
        )

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_brief_info(self) -> str:
        """One-line tile description."""
        return (
            f"InferenceTritonTile(out_size={self.out_size}, in_size={self.in_size}, "
            f"prog_noise={self._prog_noise_scale:.3f}, "
            f"drift_nu={self._drift_nu:.3f})"
        )


__all__ = ["InferenceTritonTile"]
