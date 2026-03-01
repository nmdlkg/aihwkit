# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024, 2025 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""I/O Manager for Triton-based analog tiles.

Handles input/output scaling and noise management for analog matrix-vector
multiplications. Mirrors the logic from ``analog_mvm.py`` /
``io_manager.cu`` in a pure Python implementation compatible with the
Triton backend.
"""

import torch
from typing import Tuple, Optional, Union
from dataclasses import dataclass

from .kernels.math_utils import triton_abs_max, triton_clamp, triton_elem_scale

@dataclass
class IOScaleResult:
    """Holds the result of input management including scaled tensor and scale factor.

    Attributes:
        x_scaled: Input tensor after noise, scaling, and clamping.
        scale: Scale factor applied (needed for output rescaling).
    """

    x_scaled: torch.Tensor
    scale: Union[float, torch.Tensor]


class TritonIOManager:
    """Manages input/output scaling and noise for analog tile MVMs.

    Implements the I/O management logic from ``io_manager.cu`` /
    ``AnalogMVM._compute_noise_management`` in pure Python for use with
    the Triton backend.

    All parameter access uses ``getattr(io_pars, field, default)`` so that
    any duck-typed object (not just ``IOParameters``) can be passed.
    """

    def manage_input(self, x: torch.Tensor, io_pars: object) -> IOScaleResult:
        """Pre-process input for analog MVM.

        Applies:
            1. Input noise (``inp_noise``)
            2. Input scaling based on ``noise_management`` type
            3. Input bound clamping

        Args:
            x: Input activation tensor, shape ``(*, in_size)``.
            io_pars: Parameter object with IO configuration fields.
                Accepts ``IOParameters`` or any duck-typed object.

        Returns:
            ``IOScaleResult`` with scaled/clamped input and the scale factor.
        """
        inp_bound: float = getattr(io_pars, "inp_bound", 1.0)
        inp_noise: float = getattr(io_pars, "inp_noise", 0.0)
        noise_mgmt = getattr(io_pars, "noise_management", "ABS_MAX")

        # Normalise enum → string for comparison
        noise_mgmt_str = self._enum_to_str(noise_mgmt)
        use_cuda = x.is_cuda

        # 1. Add input noise if configured
        x_proc = x
        if inp_noise > 0.0:
            x_proc = x + torch.randn_like(x) * inp_noise

        # 2. Compute noise-management scale
        nm_scale = self._compute_noise_management_scale(x_proc, noise_mgmt_str, io_pars)

        # Derive input scale: maps nm_scale → inp_bound range
        if nm_scale > 0:
            scale = inp_bound / nm_scale
        else:
            scale = 1.0

        # Convert to float for Triton kernel scalar args
        scale_f = float(scale) if isinstance(scale, torch.Tensor) else scale

        # 3. Scale input
        if use_cuda:
            x_scaled = triton_elem_scale(x_proc, scale_f)
        else:
            x_scaled = x_proc * scale_f

        # 4. Clamp to input bound
        if inp_bound > 0:
            if use_cuda:
                x_scaled = triton_clamp(x_scaled, -inp_bound, inp_bound)
            else:
                x_scaled = x_scaled.clamp(-inp_bound, inp_bound)

        return IOScaleResult(x_scaled=x_scaled, scale=scale_f)

    def manage_output(
        self,
        y: torch.Tensor,
        io_pars: object,
        input_scale: float = 1.0,
    ) -> torch.Tensor:
        """Post-process output of analog MVM.

        Applies:
            1. Output noise (``out_noise``)
            2. Output bound clamping
            3. Rescale by ``1/input_scale`` to undo input scaling

        Args:
            y: Raw output tensor from analog MVM, shape ``(*, out_size)``.
            io_pars: Parameter object with IO configuration fields.
            input_scale: Scale factor returned from ``manage_input``.

        Returns:
            Post-processed output tensor.
        """
        out_bound: float = getattr(io_pars, "out_bound", 0.0)
        out_noise: float = getattr(io_pars, "out_noise", 0.0)
        out_scale: float = getattr(io_pars, "out_scale", 1.0)

        y_out = y

        # 1. Add output noise (before clamping, matching C++ reference)
        if out_noise > 0.0:
            out_noise_values = getattr(io_pars, "_out_noise_values", None)
            if out_noise_values is not None:
                # Per-output systematic noise variation
                y_out = y_out + torch.randn_like(y_out) * out_noise_values.view(
                    1, -1, *((1,) * (y_out.ndim - 2))
                )
            else:
                y_out = y_out + torch.randn_like(y_out) * out_noise

        # 2. Apply output bound
        use_cuda = y.is_cuda
        if out_bound > 0:
            if use_cuda:
                y_out = triton_clamp(y_out, -out_bound, out_bound)
            else:
                y_out = y_out.clamp(-out_bound, out_bound)

        # 3. Rescale: undo input scaling and apply out_scale
        input_scale_f = float(input_scale) if isinstance(input_scale, torch.Tensor) else input_scale
        if input_scale_f != 0 and input_scale_f != 1.0:
            combined = out_scale / input_scale_f
            if use_cuda:
                y_out = triton_elem_scale(y_out, combined)
            else:
                y_out = y_out * combined
        elif out_scale != 1.0:
            if use_cuda:
                y_out = triton_elem_scale(y_out, out_scale)
            else:
                y_out = y_out * out_scale

        return y_out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _enum_to_str(value: object) -> str:
        """Convert an enum value or string to a canonical uppercase string."""
        if hasattr(value, "name"):
            # Standard Python Enum
            return str(value.name).upper()
        s = str(value).upper()
        # Strip prefix like "NOISEMANAGEMENTTYPE."
        if "." in s:
            s = s.rsplit(".", 1)[-1]
        return s

    @staticmethod
    def _compute_noise_management_scale(
        x: torch.Tensor, nm_type_str: str, io_pars: object
    ) -> Union[float, torch.Tensor]:
        """Compute the noise-management scale (raw, not inverted).

        This mirrors ``AnalogMVM._compute_noise_management`` but returns
        a scalar float (CPU) or scalar tensor (GPU) and operates globally
        over the tensor (suitable for single-sample or batched usage
        without per-row distinction).

        Args:
            x: Input tensor.
            nm_type_str: Canonical uppercase noise management type string.
            io_pars: IO parameters for thresholds etc.

        Returns:
            Raw scale value (e.g. ``abs_max(x)`` for ABS_MAX mode).
        """
        nm_thres: float = getattr(io_pars, "nm_thres", 0.0)
        use_cuda = x.is_cuda

        if nm_type_str == "ABS_MAX":
            if use_cuda:
                abs_max = triton_abs_max(x)  # scalar tensor, fused abs+max
                if nm_thres > 0.0:
                    abs_max = abs_max.clamp(max=nm_thres)
                return abs_max
            else:
                abs_max = x.abs().max().item()
                if nm_thres > 0.0:
                    abs_max = min(abs_max, nm_thres)
                return abs_max

        if nm_type_str == "MAX":
            if use_cuda:
                max_val = x.max()  # scalar tensor
                if nm_thres > 0.0:
                    max_val = max_val.clamp(max=nm_thres)
                return max_val
            else:
                max_val = x.max().item()
                if nm_thres > 0.0:
                    max_val = min(max_val, nm_thres)
                return max_val

        if nm_type_str == "CONSTANT":
            return nm_thres if nm_thres > 0.0 else 1.0

        # NONE or unknown \u2192 no scaling
        return 1.0
