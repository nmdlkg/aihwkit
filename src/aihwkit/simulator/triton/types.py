# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Type definitions and dataclasses for Triton backend."""

from dataclasses import dataclass
from typing import Dict

import torch

# Type aliases
RPUTensorT = torch.Tensor


@dataclass
class TritonWeightState:
    """State container for Triton tile weights and hidden parameters.

    Attributes:
        weights: Weight matrix tensor [out_size, in_size]
        hidden_params: Dictionary of hidden parameters
    """

    weights: torch.Tensor
    hidden_params: Dict
