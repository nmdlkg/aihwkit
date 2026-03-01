# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Device models for Triton backend."""

from abc import ABC, abstractmethod
from typing import Dict, Any

from torch import Tensor


class BaseTritonDevice(ABC):
    """Abstract base class for Triton device models.

    Device models define how weights are updated and how hidden parameters
    are managed during training.
    """

    @abstractmethod
    def apply_update(
        self,
        weights: Tensor,
        delta_w: Tensor,
        params: Dict[str, Any],
    ) -> Tensor:
        """Apply weight update based on device model.

        Args:
            weights: Current weight tensor [out_size, in_size]
            delta_w: Weight delta tensor [out_size, in_size]
            params: Device parameters dictionary

        Returns:
            Updated weight tensor [out_size, in_size]
        """

    @abstractmethod
    def get_hidden_parameters(self) -> Dict[str, Tensor]:
        """Get hidden parameters of the device.

        Returns:
            Dictionary mapping parameter names to tensors
        """

    @abstractmethod
    def set_hidden_parameters(self, params: Dict[str, Tensor]) -> None:
        """Set hidden parameters of the device.

        Args:
            params: Dictionary mapping parameter names to tensors
        """


__all__ = [
    "BaseTritonDevice",
]
