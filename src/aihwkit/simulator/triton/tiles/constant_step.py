# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""ConstantStep Triton tile: pulsed weight updates with bounds enforcement."""

import torch
from torch import Tensor

from aihwkit.simulator.triton.tiles.analog import TritonAnalogTile
from aihwkit.simulator.triton.devices.constant_step import TritonConstantStepDevice


class ConstantStepTritonTile(TritonAnalogTile):
    """Analog tile with ConstantStep pulsed weight updates.

    Overrides the base ``update()`` to use stochastic coincidence-based pulsed
    updates following the CUDA reference formula from rpu_pulsed_meta_parameter.cpp.

    Two update paths are available, controlled by ``use_triton_update`` in the config:

    **PyTorch path** (default, ``use_triton_update=False``):
        Per-sample Binomial coincidence counting over ``BL`` pulses.
        Matches CUDA pulse-train coincidence semantics and applies
        ``dw_min_std`` cycle noise with ``sqrt(n)`` scaling.

    **Triton kernel path** (``use_triton_update=True``):
        Batch-reduced Triton pipeline (triton_get_counts + triton_pulsed_update).
        Faster on GPU due to fused kernel execution, but uses batch-mean
        approximation for pulse counts.

    The pulsed update flow:
        1. Compute scale factor ``A = sqrt(lr / (dw_min * BL))``
        2. Generate Bernoulli samples: ``pulse_x[j] ~ Bernoulli(A * |x[j]|)``
           and ``pulse_d[i] ~ Bernoulli(A * |d[i]|)``
        3. Coincident pulse at (i,j) = pulse_x[j] AND pulse_d[i]
        4. Apply ``dw_min * sign(d) * sign(x)`` for coincident pulses
        5. Weights are clamped to ``[w_min, w_max]`` in-place

    Args:
        out_size: Output (row) dimension of the weight matrix.
        in_size: Input (column) dimension of the weight matrix.
        rpu_config: RPU config object; reads ``device`` sub-config for
            ConstantStep parameters (duck-typed).
        bias: Unused flag kept for API compatibility.
    """

    def __init__(self, out_size, in_size, rpu_config=None, bias=False):
        super().__init__(out_size, in_size, rpu_config, bias)
        device_config = getattr(rpu_config, "device", None)
        self._tile_device = TritonConstantStepDevice(device_config)
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
        """Pulsed weight update — Triton kernel pipeline on GPU (opt-in), PyTorch fallback.

        GPU Triton path (``use_triton_update=True``):
            1. Normalize x, d to [-1, 1] using CUDA reference scale factors.
            2. Reduce over batch: x_1d = mean(x_norm, dim=0).
            3. x_counts = triton_get_counts(|x_1d|, BL) * sign(x_1d)
               triton_get_counts maps [0,1] -> [BL/2, BL]; multiplying by sign
               ensures exactly-zero inputs produce count=0 (no pulse).
            4. d_counts negated for gradient descent direction.
            5. triton_pulsed_update with CONSTANT_STEP functor updates weights
               in-place and clamps to [w_min, w_max].

        PyTorch path (default, ``use_triton_update=False``):
            Per-sample Binomial coincidence counting over ``BL`` pulses.

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

        # Flatten 3D inputs (from convolution unfold) to 2D for weight update
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])  # [batch*patches, in_size]
        if d.dim() == 3:
            d = d.reshape(-1, d.shape[-1])  # [batch*patches, out_size]

        # Ensure 2D: [batch, size]
        x_all = x if x.dim() > 1 else x.unsqueeze(0)  # [B, in_size]
        d_all = d if d.dim() > 1 else d.unsqueeze(0)  # [B, out_size]

        lr = abs(self._learning_rate)  # stored as negative internally
        dw_min = self._tile_device.dw_min
        bl_count = self.bl_count

        # CUDA reference update management (rpu_pulsed_meta_parameter.cpp):
        # k_val = lr * x_amax * d_amax / dw_min
        # BL_adj = min(ceil(k_val), desired_bl)  [update_bl_management]
        # base_A = sqrt(lr / (dw_min * BL_adj))
        # scale_x = base_A * sqrt(d_amax / x_amax)  [B in CUDA, for x pulses]
        # scale_d = base_A * sqrt(x_amax / d_amax)  [A in CUDA, for d pulses]
        x_amax = x_all.abs().max().clamp(min=1e-7)
        d_amax = d_all.abs().max().clamp(min=1e-7)

        # Dynamic BL adjustment (update_bl_management): reduces BL for weak gradients,
        # increasing base_A to maintain update efficiency
        k_val = lr * x_amax * d_amax / dw_min
        BL = min(max(1, int(k_val.ceil().item())), bl_count)

        # Adaptive scale factors (update_management): ensures max x and max d
        # have equal pulse probability = base_A * sqrt(x_amax * d_amax)
        base_A = (lr / (dw_min * BL)) ** 0.5
        scale_x = base_A * (d_amax / x_amax) ** 0.5  # for x pulses (CUDA's B)
        scale_d = base_A * (x_amax / d_amax) ** 0.5  # for d pulses (CUDA's A)

        # Read toggle from config (default=False — PyTorch Bernoulli path)
        use_triton_update = bool(getattr(self.rpu_config, "use_triton_update", False))

        with torch.no_grad():
            if x_all.is_cuda and self.weight.data.is_cuda and use_triton_update:
                # ------------------------------------------------------------------
                # GPU path: Triton kernel pipeline
                # ------------------------------------------------------------------
                from aihwkit.simulator.triton.kernels.pulsed_update import (
                    triton_pulsed_update,
                )
                from aihwkit.simulator.triton.kernels.bit_line_maker import (
                    triton_get_counts,
                )

                # Step 1: Normalize inputs to [-1, 1] using computed scale factors
                x_norm = (x_all * scale_x).clamp(-1.0, 1.0)  # [B, in_size]
                d_norm = (d_all * scale_d).clamp(-1.0, 1.0)  # [B, out_size]

                # Step 2: Reduce over batch dimension (approximate multi-sample as mean)
                x_1d = x_norm.mean(0)  # [in_size] in [-1, 1]
                d_1d = d_norm.mean(0)  # [out_size] in [-1, 1]

                # Step 3: Build signed pulse counts for triton_pulsed_update
                # triton_get_counts(|x|, BL) maps [0,1] -> [BL/2, BL] (always non-zero
                # for non-zero |x|). Multiplying by sign(x) produces signed counts where
                # sign(0) = 0 ensures exactly-zero inputs give count=0 (no pulse).
                # The pulsed_update kernel uses only sign(count) for update direction.
                x_sign = x_1d.sign().to(torch.int32)
                d_sign = d_1d.sign().to(torch.int32)

                x_counts_raw = triton_get_counts(x_1d.abs().float(), BL)  # [in_size]
                d_counts_raw = triton_get_counts(d_1d.abs().float(), BL)  # [out_size]

                x_counts = x_counts_raw * x_sign  # signed; 0 where x_1d == 0
                d_counts = d_counts_raw * (-d_sign)  # negated for gradient descent

                # Step 4: Apply pulsed update with CONSTANT_STEP functor (clamp enforced
                # internally by triton_pulsed_update)
                triton_pulsed_update(
                    self.weight.data,
                    x_counts,
                    d_counts,
                    dw_min=dw_min,
                    functor="CONSTANT_STEP",
                    params={"dw_min_std": self._tile_device.dw_min_std},
                    bound_lower=self._tile_device.w_min,
                    bound_upper=self._tile_device.w_max,
                )

            else:
                x_prob = (scale_x * x_all.abs()).clamp(0.0, 1.0)  # [B, in_size]
                d_prob = (scale_d * d_all.abs()).clamp(0.0, 1.0)  # [B, out_size]
                x_sign = x_all.sign()  # [B, in_size]
                d_sign = -d_all.sign()  # [B, out_size]

                out_size = d_prob.shape[1]
                in_size = x_prob.shape[1]

                pos_counts = torch.zeros(
                    (out_size, in_size),
                    device=self.weight.data.device,
                    dtype=self.weight.data.dtype,
                )
                neg_counts = torch.zeros_like(pos_counts)

                max_elements = 8_000_000
                denom = max(1, out_size * in_size)
                chunk_size = max(1, max_elements // denom)

                for start in range(0, x_prob.shape[0], chunk_size):
                    end = min(start + chunk_size, x_prob.shape[0])

                    x_prob_chunk = x_prob[start:end]
                    d_prob_chunk = d_prob[start:end]
                    x_sign_chunk = x_sign[start:end]
                    d_sign_chunk = d_sign[start:end]

                    coincidence_prob = d_prob_chunk.unsqueeze(
                        2
                    ) * x_prob_chunk.unsqueeze(1)
                    n_trials = torch.full_like(coincidence_prob, float(BL))
                    coincidence_counts = torch.binomial(n_trials, coincidence_prob)

                    coincidence_sign = d_sign_chunk.unsqueeze(
                        2
                    ) * x_sign_chunk.unsqueeze(1)
                    pos_counts += (coincidence_counts * (coincidence_sign > 0)).sum(
                        dim=0
                    )
                    neg_counts += (coincidence_counts * (coincidence_sign < 0)).sum(
                        dim=0
                    )

                coincidence_sum = pos_counts - neg_counts

                dw_min_std = self._tile_device.dw_min_std
                if dw_min_std > 0.0:
                    pos_noise = torch.randn_like(pos_counts)
                    neg_noise = torch.randn_like(neg_counts)
                    delta_w = dw_min * coincidence_sum
                    delta_w += (
                        dw_min
                        * dw_min_std
                        * (
                            pos_counts.sqrt() * pos_noise
                            - neg_counts.sqrt() * neg_noise
                        )
                    )
                else:
                    delta_w = dw_min * coincidence_sum

                self.weight.data += delta_w
                self.weight.data.clamp_(
                    self._tile_device.w_min, self._tile_device.w_max
                )

        return self.weight.data

    def get_brief_info(self) -> str:
        """One-line tile description."""
        return (
            f"ConstantStepTritonTile(out_size={self.out_size}, in_size={self.in_size}, "
            f"dw_min={self._tile_device.dw_min}, bl_count={self.bl_count})"
        )


__all__ = ["ConstantStepTritonTile"]
