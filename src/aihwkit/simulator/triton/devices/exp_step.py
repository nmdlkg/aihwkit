# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""ExpStep device model: pulsed update with exponential weight-dependent step size."""

from typing import Dict, Any

from torch import Tensor

from aihwkit.simulator.triton.devices import BaseTritonDevice


class TritonExpStepDevice(BaseTritonDevice):
    """ExpStep device: pulsed update with exponential weight-dependent step size.

    Each coincident pulse produces a weight change that depends exponentially
    on the current weight value:
        z = 2*w / (w_max - w_min) * a + b
        dw_up   = scale_up   * max(1 - A_up   * exp(gamma_up   * z), 0)
        dw_down = scale_down * max(1 - A_down  * exp(gamma_down * (-z)), 0)

    As |w| increases, the update magnitude decreases → saturation behavior.

    Args:
        device_config: Optional RPU device config object; reads ``A_up``,
            ``A_down``, ``gamma_up``, ``gamma_down``, ``a``, ``b``,
            ``dw_min``, ``w_max``, ``w_min`` attributes.
    """

    def __init__(self, device_config=None):
        self.es_A_up = float(getattr(device_config, "A_up", 0.00081))
        self.es_A_down = float(getattr(device_config, "A_down", 0.36833))
        self.es_gamma_up = float(getattr(device_config, "gamma_up", 12.44625))
        self.es_gamma_down = float(getattr(device_config, "gamma_down", 12.78785))
        self.es_a = float(getattr(device_config, "a", 0.244))
        self.es_b = float(getattr(device_config, "b", 0.2425))
        self.dw_min = float(getattr(device_config, "dw_min", 0.001))
        self.w_max = float(getattr(device_config, "w_max", 0.6))
        self.w_min = float(getattr(device_config, "w_min", -0.6))

    def apply_update(
        self,
        weights: Tensor,
        delta_w: Tensor,
        params: Dict[str, Any] = None,
    ) -> Tensor:
        """Apply weight update (not used directly by ExpStepTritonTile).

        The tile's ``update()`` method calls ``triton_pulsed_update`` directly
        with the EXP_STEP functor for full pulsed behavior.

        Args:
            weights: Current weight tensor [out_size, in_size].
            delta_w: Weight delta tensor [out_size, in_size].
            params: Device parameters dictionary.

        Returns:
            Updated weight tensor [out_size, in_size].
        """
        return weights + delta_w

    def get_hidden_parameters(self) -> Dict[str, Tensor]:
        """No hidden parameters for ExpStep device.

        Returns:
            Empty dict.
        """
        return {}

    def set_hidden_parameters(self, params: Dict[str, Tensor]) -> None:
        """No-op: ExpStep device has no hidden parameters.

        Args:
            params: Ignored.
        """
        pass

    def get_params_dict(self) -> Dict[str, Any]:
        """Get device parameters for ``triton_pulsed_update``.

        Returns:
            Dict with ExpStep parameters for the EXP_STEP functor.
        """
        return {
            "es_A_up": self.es_A_up,
            "es_A_down": self.es_A_down,
            "es_gamma_up": self.es_gamma_up,
            "es_gamma_down": self.es_gamma_down,
            "es_a": self.es_a,
            "es_b": self.es_b,
            "w_max": self.w_max,
            "w_min": self.w_min,
        }


__all__ = ["TritonExpStepDevice"]
