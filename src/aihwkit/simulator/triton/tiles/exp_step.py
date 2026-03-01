# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""ExpStep Triton tile: pulsed weight updates with exponential saturation."""

import torch
from torch import Tensor

from aihwkit.simulator.triton.tiles.analog import TritonAnalogTile
from aihwkit.simulator.triton.devices.exp_step import TritonExpStepDevice


class ExpStepTritonTile(TritonAnalogTile):
    """Analog tile with ExpStep pulsed weight updates.

    Overrides the base ``update()`` to use stochastic coincidence-based pulsed
    updates with an exponential weight-dependent step size (ExpStep model),
    following the CUDA reference formula from rpu_pulsed_meta_parameter.cpp.

    The pulsed update flow:
        1. Compute scale factor ``A = sqrt(lr / (dw_min * BL))``
        2. Generate Bernoulli samples: ``pulse_x[j] ~ Bernoulli(A * |x[j]|)``
           and ``pulse_d[i] ~ Bernoulli(A * |d[i]|)``
        3. Coincident pulse at (i,j) = pulse_x[j] AND pulse_d[i]
        4. ExpStep update applied for coincident pulses
        5. Weights are clamped to ``[w_min, w_max]`` in-place

    The ExpStep model causes updates to shrink as |w| increases,
    producing exponential saturation behavior near the weight bounds.

    Args:
        out_size: Output (row) dimension of the weight matrix.
        in_size: Input (column) dimension of the weight matrix.
        rpu_config: RPU config object; reads ``device`` sub-config for
            ExpStep parameters (duck-typed).
        bias: Unused flag kept for API compatibility.
    """

    def __init__(self, out_size, in_size, rpu_config=None, bias=False):
        super().__init__(out_size, in_size, rpu_config, bias)
        device_config = getattr(rpu_config, "device", None)
        self._tile_device = TritonExpStepDevice(device_config)
        # Get bl_count from update config
        update_config = getattr(rpu_config, "update", None)
        self.bl_count = int(getattr(update_config, "desired_bl", 31))

    def update(
        self,
        x_input,
        d_input,
        bias=False,
        in_trans=False,
        out_trans=False,
        non_blocking=False,
    ):
        """Stochastic pulsed weight update via coincidence-based Bernoulli sampling.

        Processes each batch element independently, matching the CUDA reference:
            A = sqrt(lr / (dw_min * BL))
            For each sample b:
                x_pulse[b,j] ~ Bernoulli(A * |x[b,j]|) * sign(x[b,j])
                d_pulse[b,i] ~ Bernoulli(A * |d[b,i]|) * (-sign(d[b,i]))  [negated for descent]
                Coincident: both x_pulse[b,j] != 0 AND d_pulse[b,i] != 0
            Accumulate: coincidence_sum[i,j] = sum_b d_pulse[b,i] * x_pulse[b,j]
            Apply ExpStep magnitude to each coincident update.

        Args:
            x_input: Input activations ``[batch, in_size]`` (or transposed).
            d_input: Upstream gradients ``[batch, out_size]`` (or transposed).
            bias: Unused; kept for API compatibility.
            in_trans: If True, ``x_input`` is ``[in_size, batch]``.
            out_trans: If True, ``d_input`` is ``[out_size, batch]``.
            non_blocking: Unused; kept for API compatibility.

        Returns:
            Updated weight tensor (same object as ``self.weight.data``).
        """
        x = x_input.T if in_trans else x_input
        d = d_input.T if out_trans else d_input

        # Ensure 2D: [batch, size]
        x_all = x if x.dim() > 1 else x.unsqueeze(0)  # [B, in_size]
        d_all = d if d.dim() > 1 else d.unsqueeze(0)  # [B, out_size]

        lr = abs(self._learning_rate)  # stored as negative internally
        dw_min = self._tile_device.dw_min
        bl_count = self.bl_count

        # CUDA reference update management (rpu_pulsed_meta_parameter.cpp):
        # scale_x = base_A * sqrt(d_amax / x_amax)  [CUDA's B, for x pulses]
        # scale_d = base_A * sqrt(x_amax / d_amax)  [CUDA's A, for d pulses]
        # This ensures prob_x_max = prob_d_max = base_A * sqrt(x_amax * d_amax)
        x_amax = x_all.abs().max().clamp(min=1e-7)
        d_amax = d_all.abs().max().clamp(min=1e-7)

        # Dynamic BL adjustment (update_bl_management)
        k_val = lr * x_amax * d_amax / dw_min
        BL = min(max(1, int(k_val.ceil().item())), bl_count)

        # Adaptive scale factors (update_management)
        base_A = (lr / (dw_min * BL)) ** 0.5
        scale_x = base_A * (d_amax / x_amax) ** 0.5  # for x pulses (CUDA's B)
        scale_d = base_A * (x_amax / d_amax) ** 0.5  # for d pulses (CUDA's A)

        # Stochastic pulse generation per batch element (vectorized)
        x_prob = (scale_x * x_all.abs()).clamp(0.0, 1.0)  # [B, in_size]
        d_prob = (scale_d * d_all.abs()).clamp(0.0, 1.0)  # [B, out_size]

        # Signed pulses: +/-1 (direction) or 0 (no pulse)
        x_pulses = torch.bernoulli(x_prob) * x_all.sign()  # [B, in_size]
        d_pulses = torch.bernoulli(d_prob) * (-d_all.sign())  # [B, out_side], negated for descent


        # Coincident pulse sum over batch: direction and count encoded
        # coincidence_sum[i,j] = sum_b d_pulse[b,i] * x_pulse[b,j]
        coincidence_sum = d_pulses.T @ x_pulses  # [out_size, in_size]

        # ExpStep: weight-dependent magnitude per coincident pulse
        params = self._tile_device.get_params_dict()
        es_A_up = params.get("es_A_up", 0.00081)
        es_A_down = params.get("es_A_down", 0.36833)
        es_gamma_up = params.get("es_gamma_up", 12.44625)
        es_gamma_down = params.get("es_gamma_down", 12.78785)
        es_a = params.get("es_a", 0.244)
        es_b = params.get("es_b", 0.2425)
        w_max = self._tile_device.w_max
        w_min = self._tile_device.w_min
        w_range = max(w_max - w_min, 1e-6)

        with torch.no_grad():
            z = 2.0 * self.weight.data / w_range * es_a + es_b
            dw_up = dw_min * (1.0 - es_A_up * torch.exp(es_gamma_up * z)).clamp(min=0.0)
            dw_down = dw_min * (1.0 - es_A_down * torch.exp(es_gamma_down * (-z))).clamp(min=0.0)

            # Apply per-coincidence ExpStep update scaled by coincidence magnitude
            delta_w = torch.where(
                coincidence_sum > 0,
                dw_up * coincidence_sum,
                torch.where(coincidence_sum < 0, -dw_down * coincidence_sum.abs(),
                            torch.zeros_like(self.weight.data)),
            )
            self.weight.data += delta_w
            self.weight.data.clamp_(w_min, w_max)
        return self.weight.data

    def get_brief_info(self) -> str:
        """One-line tile description."""
        return (
            f"ExpStepTritonTile(out_size={self.out_size}, in_size={self.in_size}, "
            f"dw_min={self._tile_device.dw_min}, bl_count={self.bl_count})"
        )


__all__ = ["ExpStepTritonTile"]
