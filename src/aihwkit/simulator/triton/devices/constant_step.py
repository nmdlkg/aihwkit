# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""ConstantStep device model: pulsed update with optional multiplicative noise."""

from typing import Dict, Any

from torch import Tensor

from aihwkit.simulator.triton.devices import BaseTritonDevice


class TritonConstantStepDevice(BaseTritonDevice):
    """ConstantStep device: pulsed update with optional multiplicative noise.

    Each coincident pulse produces a weight change of:
        dw = dw_min * (1 + dw_min_std * randn) * sign(x) * sign(d)

    Weights are clamped to [w_min, w_max] after each update.

    Args:
        device_config: Optional RPU device config object; reads ``dw_min``,
            ``dw_min_std``, ``w_max``, ``w_min`` attributes.
    """

    def __init__(self, device_config=None):
        self.dw_min = float(getattr(device_config, 'dw_min', 0.01))
        self.dw_min_std = float(getattr(device_config, 'dw_min_std', 0.0))
        self.w_max = float(getattr(device_config, 'w_max', 0.6))
        self.w_min = float(getattr(device_config, 'w_min', -0.6))

    def apply_update(
        self,
        weights: Tensor,
        delta_w: Tensor,
        params: Dict[str, Any] = None,
    ) -> Tensor:
        """Apply weight update (not used directly by ConstantStepTritonTile).

        The tile's ``update()`` method calls ``triton_pulsed_update`` directly
        with the CONSTANT_STEP functor for full pulsed behavior.

        Args:
            weights: Current weight tensor [out_size, in_size].
            delta_w: Weight delta tensor [out_size, in_size].
            params: Device parameters dictionary.

        Returns:
            Updated weight tensor [out_size, in_size].
        """
        return weights + delta_w

    def get_hidden_parameters(self) -> Dict[str, Tensor]:
        """No hidden parameters for ConstantStep device.

        Returns:
            Empty dict.
        """
        return {}

    def set_hidden_parameters(self, params: Dict[str, Tensor]) -> None:
        """No-op: ConstantStep device has no hidden parameters.

        Args:
            params: Ignored.
        """
        pass

    def get_params_dict(self) -> Dict[str, Any]:
        """Get device parameters for ``triton_pulsed_update``.

        Returns:
            Dict with ``dw_min_std`` for the CONSTANT_STEP functor.
        """
        return {'dw_min_std': self.dw_min_std}


__all__ = ["TritonConstantStepDevice"]
