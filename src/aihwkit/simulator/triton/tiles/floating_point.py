# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Floating-point Triton tile: ideal MVM, no noise, no I/O management."""

import torch
from torch.nn import Module, Parameter
from typing import Optional, List, Dict, Any

from aihwkit.simulator.triton.base_tile import TritonBaseTile
from aihwkit.simulator.tiles.module import TileModule


class FloatingPointTritonTile(TritonBaseTile, TileModule):
    """Floating-point Triton tile: ideal MVM, no noise, no I/O management.

    This is the simplest tile implementation with:
    - Exact floating-point computation (no quantization)
    - No noise injection
    - No I/O management
    - Direct GEMM operations matching torch.mm
    """

    def __init__(self, out_size: int, in_size: int, rpu_config=None, bias: bool = False):
        """Initialize FloatingPointTritonTile.

        Args:
            out_size: Output dimension
            in_size: Input dimension
            rpu_config: RPU configuration (unused, for compatibility)
            bias: Whether to use bias (unused, for compatibility)
        """
        TileModule.__init__(self)
        self.out_size = out_size
        self.in_size = in_size
        # RPU convention: x_size = input size, d_size = output size
        self.x_size = in_size
        self.d_size = out_size
        self.rpu_config = rpu_config
        self._learning_rate = -0.01
        self.weight = Parameter(torch.zeros(out_size, in_size), requires_grad=False)

    def cuda(self, device=None):
        """Move tile to CUDA device (uses Module-level apply to avoid TileModule recursion)."""
        Module._apply(self, lambda t: t.cuda(device))
        return self

    def to(self, *args, **kwargs):
        """Move tile to device/dtype (uses Module-level apply to avoid TileModule complex logic)."""
        Module._apply(self, lambda t: t.to(*args, **kwargs))
        return self

    def extra_repr(self) -> str:
        """One-line tile description for print(model)."""
        return self.get_brief_info()

    def forward(
        self,
        x_input: torch.Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        is_test: bool = False,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        """Perform forward pass: y = x @ W.T (no noise, no bounds).

        Args:
            x_input: Input tensor [N, x_size] or [x_size, N] if in_trans
            bias: Whether to apply bias (unused)
            in_trans: Whether input is transposed
            out_trans: Whether output should be transposed
            is_test: Whether in test mode (unused)
            non_blocking: Whether to use non-blocking operations (unused)

        Returns:
            Output tensor [N, d_size] or [d_size, N] if out_trans
        """
        x = x_input.T if in_trans else x_input

        if x.dim() == 1:
            # Vector case: [x_size] @ [d_size, x_size].T -> [d_size]
            y = self.weight @ x
        else:
            # Matrix case: [N, x_size] @ [d_size, x_size].T -> [N, d_size]
            y = x @ self.weight.T

        return y.T if out_trans else y

    def backward(
        self,
        d_input: torch.Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        """Perform backward pass: dx = d @ W.

        Args:
            d_input: Gradient tensor [N, d_size] or [d_size, N] if out_trans
            bias: Whether to apply bias (unused)
            in_trans: Whether input is transposed
            out_trans: Whether output is transposed
            non_blocking: Whether to use non-blocking operations (unused)

        Returns:
            Gradient tensor [N, x_size] or [x_size, N] if in_trans
        """
        d = d_input.T if out_trans else d_input

        if d.dim() == 1:
            # Vector case: [d_size] @ [d_size, x_size] -> [x_size]
            dx = d @ self.weight
        else:
            # Matrix case: [N, d_size] @ [d_size, x_size] -> [N, x_size]
            dx = d @ self.weight

        return dx.T if in_trans else dx

    def update(
        self,
        x_input: torch.Tensor,
        d_input: torch.Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> None:
        """Perform weight update: w += lr * d.T @ x.

        Args:
            x_input: Input tensor [N, x_size] or [x_size, N] if in_trans
            d_input: Gradient tensor [N, d_size] or [d_size, N] if out_trans
            bias: Whether to apply bias (unused)
            in_trans: Whether input is transposed
            out_trans: Whether output is transposed
            non_blocking: Whether to use non-blocking operations (unused)
        """
        x = x_input.T if in_trans else x_input
        d = d_input.T if out_trans else d_input

        with torch.no_grad():
            if x.dim() == 1:
                # Vector case: outer product
                delta_w = torch.outer(d.flatten(), x.flatten())
            else:
                # Matrix case: [N, d_size].T @ [N, x_size] -> [d_size, x_size]
                delta_w = d.T @ x

            self.weight.data += self._learning_rate * delta_w

    def get_weights(self) -> tuple:
        """Get the analog weights.

        Returns:
            Tuple (weight_tensor, None) where weight is [d_size, x_size]
        """
        return self.weight.data.detach().cpu(), None

    def set_weights(self, weight: torch.Tensor, bias=None) -> None:
        """Set the analog weights.

        Args:
            weight: Weight tensor [d_size, x_size]
            bias: Optional bias tensor (ignored, kept for API compatibility)
        """
        self.weight.data = weight.to(self.weight.device).clone()

    def get_x_size(self) -> int:
        """Get input size of tile.

        Returns:
            Input dimension
        """
        return self.x_size

    def get_d_size(self) -> int:
        """Get output size of tile.

        Returns:
            Output dimension
        """
        return self.d_size

    def get_brief_info(self) -> str:
        """Get brief information about the tile.

        Returns:
            String description of tile configuration
        """
        return f"FloatingPointTritonTile({self.out_size},{self.in_size})"

    def get_hidden_parameters(self) -> torch.Tensor:
        """Get the hidden parameters of the tile.

        Returns:
            Empty tensor (no hidden parameters)
        """
        return torch.empty(0)

    def get_hidden_parameter_names(self) -> List[str]:
        """Get the names of hidden parameters.

        Returns:
            Empty list (no hidden parameters)
        """
        return []

    def set_hidden_parameters(self, params: torch.Tensor) -> None:
        """Set the hidden parameters of the tile.

        Args:
            params: Hidden parameter tensor (unused)
        """
        pass

    def get_learning_rate(self) -> Optional[float]:
        """Get the learning rate of the tile.

        Returns:
            Learning rate
        """
        return -self._learning_rate

    def set_learning_rate(self, learning_rate: Optional[float]) -> None:
        """Set the learning rate of the tile.

        Args:
            learning_rate: Learning rate to set
        """
        if learning_rate is not None:
            self._learning_rate = -float(learning_rate)

    def decay_weights(self, alpha: float) -> None:
        """Apply weight decay.

        Args:
            alpha: Decay factor
        """
        with torch.no_grad():
            self.weight.data *= 1.0 - alpha

    def drift_weights(self, t_inference: float) -> None:
        """Apply weight drift (no-op for floating-point).

        Args:
            t_inference: Inference time (unused)
        """
        pass

    def diffuse_weights(self) -> None:
        """Apply weight diffusion (no-op for floating-point)."""
        pass

    def reset_columns(
        self, start_col: int, num_cols: int, bmin: float, bmax: float
    ) -> None:
        """Reset specific columns of weights.

        Args:
            start_col: Starting column index
            num_cols: Number of columns to reset
            bmin: Minimum reset value
            bmax: Maximum reset value
        """
        with torch.no_grad():
            self.weight.data[:, start_col : start_col + num_cols].uniform_(bmin, bmax)

    def set_weights_uniform_random(self, bmin: float, bmax: float) -> None:
        """Set weights to uniform random numbers.

        Args:
            bmin: Minimum value
            bmax: Maximum value
        """
        with torch.no_grad():
            self.weight.data.uniform_(bmin, bmax)

    def get_meta_parameters(self) -> Any:
        """Get meta parameters.

        Returns:
            RPU configuration
        """
        return self.rpu_config

    def dump_extra(self) -> Optional[Dict]:
        """Dump any extra states for checkpointing.

        Returns:
            Dictionary with learning rate
        """
        return {"lr": self._learning_rate}

    def load_extra(self, extra: Dict, strict: bool = False) -> None:
        """Load extra states from checkpoint.

        Args:
            extra: Dictionary of states from dump_extra
            strict: Whether to throw error if keys not found (unused)
        """
        if extra:
            self._learning_rate = extra.get("lr", self._learning_rate)
