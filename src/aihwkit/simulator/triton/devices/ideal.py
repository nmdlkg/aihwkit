# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Ideal device model: perfect weight updates, no noise."""

from typing import Dict, Any

import torch
from torch import Tensor

from aihwkit.simulator.triton.devices import BaseTritonDevice


class TritonIdealDevice(BaseTritonDevice):
    """Ideal analog device: simple gradient descent, no noise or non-idealities.

    Weight update: w += learning_rate * delta_w

    Learning rate is read from params dict key 'learning_rate' (default 1.0).
    """

    def apply_update(
        self,
        weights: Tensor,
        delta_w: Tensor,
        params: Dict[str, Any] = None,
    ) -> Tensor:
        """Simple gradient update: w += lr * delta_w.

        Args:
            weights: Current weight tensor [out_size, in_size].
            delta_w: Weight delta tensor [out_size, in_size].
            params: Parameter dict; reads 'learning_rate' (default 1.0).

        Returns:
            Updated weight tensor [out_size, in_size].
        """
        lr = 1.0
        if params:
            lr = float(params.get("learning_rate", 1.0))
        return weights + lr * delta_w

    def get_hidden_parameters(self) -> Dict[str, Tensor]:
        """No hidden parameters for ideal device.

        Returns:
            Empty dict.
        """
        return {}

    def set_hidden_parameters(self, params: Dict[str, Tensor]) -> None:
        """No-op: ideal device has no hidden parameters.

        Args:
            params: Ignored.
        """
        pass  # No hidden params for ideal device


__all__ = ["TritonIdealDevice"]
