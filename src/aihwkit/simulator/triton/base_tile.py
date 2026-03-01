# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Abstract base class for Triton-based tiles."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from torch import Tensor
import torch


class TritonBaseTile(ABC):
    """Abstract base class for Triton-based analog tiles.

    Defines the interface that all Triton tile implementations must follow.
    Subclasses must implement all abstract methods.
    """

    @abstractmethod
    def forward(
        self,
        x_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        is_test: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """Perform forward pass through the tile.

        Args:
            x_input: Input tensor [N, in_size] or [in_size, N] if in_trans
            bias: Whether to apply bias
            in_trans: Whether input is transposed
            out_trans: Whether output should be transposed
            is_test: Whether in test mode
            non_blocking: Whether to use non-blocking operations

        Returns:
            Output tensor [N, out_size] or [out_size, N] if out_trans
        """

    @abstractmethod
    def backward(
        self,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """Perform backward pass through the tile.

        Args:
            d_input: Gradient tensor [N, out_size] or [out_size, N] if out_trans
            bias: Whether to apply bias
            in_trans: Whether input is transposed
            out_trans: Whether output is transposed
            non_blocking: Whether to use non-blocking operations

        Returns:
            Gradient tensor [N, in_size] or [in_size, N] if in_trans
        """

    @abstractmethod
    def update(
        self,
        x_input: Tensor,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """Perform weight update pass.

        Args:
            x_input: Input tensor [N, in_size] or [in_size, N] if in_trans
            d_input: Gradient tensor [N, out_size] or [out_size, N] if out_trans
            bias: Whether to apply bias
            in_trans: Whether input is transposed
            out_trans: Whether output is transposed
            non_blocking: Whether to use non-blocking operations

        Returns:
            Updated weight tensor
        """

    @abstractmethod
    def get_brief_info(self) -> str:
        """Get brief information about the tile.

        Returns:
            String description of tile configuration
        """

    @abstractmethod
    def get_weights(self) -> tuple:
        """Get the analog weights.

        Returns:
            Tuple (weight_tensor, bias_or_None) where weight is [out_size, in_size]
        """

    @abstractmethod
    def set_weights(self, weight: Tensor, bias=None) -> None:
        """Set the analog weights.

        Args:
            weight: Weight tensor [out_size, in_size]
            bias: Optional bias tensor (ignored, kept for API compatibility)
        """

    @abstractmethod
    def get_x_size(self) -> int:
        """Get input size of tile.

        Returns:
            Input dimension
        """

    @abstractmethod
    def get_d_size(self) -> int:
        """Get output size of tile.

        Returns:
            Output dimension
        """

    @abstractmethod
    def get_hidden_parameters(self) -> Tensor:
        """Get the hidden parameters of the tile.

        Returns:
            Hidden parameter tensor
        """

    @abstractmethod
    def get_hidden_parameter_names(self) -> List[str]:
        """Get the names of hidden parameters.

        Each name corresponds to a slice in the hidden parameters tensor.

        Returns:
            List of parameter names
        """

    @abstractmethod
    def set_hidden_parameters(self, params: Tensor) -> None:
        """Set the hidden parameters of the tile.

        Args:
            params: Hidden parameter tensor
        """

    @abstractmethod
    def get_learning_rate(self) -> Optional[float]:
        """Get the learning rate of the tile.

        Returns:
            Learning rate if exists, None otherwise
        """

    @abstractmethod
    def set_learning_rate(self, learning_rate: Optional[float]) -> None:
        """Set the learning rate of the tile.

        Args:
            learning_rate: Learning rate to set
        """

    @abstractmethod
    def dump_extra(self) -> Optional[Dict]:
        """Dump any extra states for checkpointing.

        Returns:
            Dictionary of extra states or None
        """

    @abstractmethod
    def load_extra(self, extra: Dict, strict: bool = False) -> None:
        """Load extra states from checkpoint.

        Args:
            extra: Dictionary of states from dump_extra
            strict: Whether to throw error if keys not found
        """

    @abstractmethod
    def set_weights_uniform_random(self, bmin: float, bmax: float) -> None:
        """Set weights to uniform random numbers.

        Args:
            bmin: Minimum value
            bmax: Maximum value
        """

    @abstractmethod
    def get_meta_parameters(self) -> Any:
        """Get meta parameters.

        Returns:
            Meta parameters object
        """

    def decay_weights(self, alpha: float) -> None:
        """Apply weight decay: weights *= (1 - alpha).

        Default implementation uses get_weights/set_weights.
        Subclasses may override for efficiency.

        Args:
            alpha: Decay factor
        """
        w, _ = self.get_weights()
        self.set_weights(w * (1.0 - alpha))

    def drift_weights(self, t_inference: float) -> None:
        """Apply power-law weight drift.

        Default: no-op. Subclasses override for device-specific drift.

        Args:
            t_inference: Inference time
        """
        pass

    def diffuse_weights(self) -> None:
        """Apply weight diffusion (add noise).

        Default: no-op. Subclasses override for device-specific diffusion.
        """
        pass

    def reset_columns(
        self, start_col: int, num_cols: int, bmin: float, bmax: float
    ) -> None:
        """Reset specific columns of weights to uniform random values.

        Args:
            start_col: Starting column index
            num_cols: Number of columns to reset
            bmin: Minimum reset value
            bmax: Maximum reset value
        """
        w, _ = self.get_weights()
        if w.numel() > 0:
            w[:, start_col:start_col + num_cols].uniform_(bmin, bmax)
            self.set_weights(w)
