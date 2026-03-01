# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Triton-based analog tile: full forward/backward/update cycle."""

from typing import Optional, List, Dict, Any

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from aihwkit.simulator.tiles.module import TileModule
from aihwkit.simulator.triton.base_tile import TritonBaseTile


class _TritonAnalogFunction(torch.autograd.Function):
    """Bridges Triton tile forward/backward with PyTorch autograd.

    This allows ``loss.backward()`` to work and stores ``x_input`` /
    ``grad_output`` in the ``AnalogContext`` so that
    ``AnalogSGD.step()`` can call ``tile.update(x, d)``.
    """

    @staticmethod
    @torch.no_grad()
    def forward(ctx, analog_ctx, x_input, tile, is_test):  # type: ignore[override]
        ctx.analog_ctx = analog_ctx
        ctx.tile = tile
        ctx.save_for_backward(x_input)
        return tile._forward_impl(x_input, is_test=is_test)

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_output):  # type: ignore[override]
        (x_input,) = ctx.saved_tensors
        tile = ctx.tile
        analog_ctx = ctx.analog_ctx

        # Analog backward MVM
        dx = tile._backward_impl(grad_output)

        # Store for optimizer: AnalogSGD.step() will call tile.update(x, d).
        # Negate grad_output because the Triton pulsed/ideal update kernels add
        # +dw_min*sign(x)*sign(d); negating d gives gradient *descent* (W -= lr*dL/dW).
        analog_ctx.analog_input.append(x_input)
        analog_ctx.analog_grad_output.append(grad_output)

        # Gradients for: analog_ctx (None), x_input (dx), tile (None), is_test (None)
        return None, dx, None, None


class TritonAnalogTile(TritonBaseTile, TileModule):
    """Triton-accelerated analog tile.

    Implements a complete forward / backward / weight-update cycle using
    the fused Triton kernels (Wave 1 + 2).  Device model is pluggable; the
    default is ``TritonIdealDevice`` (gradient descent, no noise).

    Args:
        out_size: Output (row) dimension of the weight matrix.
        in_size: Input (column) dimension of the weight matrix.
        rpu_config: Optional RPU config object; reads ``forward`` / ``backward``
            sub-configs for I/O noise and bound parameters.
        bias: Unused flag kept for API compatibility.
    """

    supports_indexed: bool = False
    analog_bias: bool = False

    def __init__(
        self,
        out_size: int,
        in_size: int,
        rpu_config: Any = None,
        bias: bool = False,
    ) -> None:
        TileModule.__init__(self)
        self.out_size = out_size
        self.in_size = in_size
        # RPU convention: x_size = input size, d_size = output size
        self.x_size = in_size
        self.d_size = out_size
        self.rpu_config = rpu_config
        self._learning_rate: float = -0.01
        self._hidden_params: Dict[str, Any] = {}

        # Attributes required by AnalogSGD / AnalogContext
        self.in_trans: bool = False
        self.out_trans: bool = False

        # Device model (default: ideal) — stored as _tile_device to avoid
        # shadowing the Module.device property used by AnalogContext.
        from aihwkit.simulator.triton.devices.ideal import TritonIdealDevice

        self._tile_device = TritonIdealDevice()

        # Weight matrix; analog weights are managed manually, no autograd tracking
        self.weight = Parameter(torch.zeros(out_size, in_size), requires_grad=False)

        # Cache I/O sub-configs from rpu_config
        self._f_io = getattr(rpu_config, "forward", None) if rpu_config else None
        self._b_io = getattr(rpu_config, "backward", None) if rpu_config else None
        self._use_triton_gemm: bool = getattr(rpu_config, "use_triton_gemm", False)
        from aihwkit.simulator.triton.io_manager import TritonIOManager
        self._io_manager = TritonIOManager()

        # Register AnalogContext so AnalogSGD can find and drive this tile.
        # AnalogContext.__init__ calls self.analog_tile.analog_ctx = self which
        # triggers Module.__setattr__ and registers it as a Parameter here.
        from aihwkit.optim.context import AnalogContext

        AnalogContext(self)

    # ------------------------------------------------------------------
    # Device / dtype helpers (required by AnalogContext)
    # ------------------------------------------------------------------

    @property
    def device(self):  # type: ignore[override]
        """Return the torch device of the weight tensor."""
        return self.weight.device

    @property
    def is_cuda(self) -> bool:
        """True if the tile is on a CUDA device."""
        return self.weight.device.type == "cuda"

    @property
    def tile(self):
        """Return self — Triton tiles are their own underlying tile (no C++ wrapper)."""
        return self
    def get_dtype(self):
        """Return the dtype of the weight tensor."""
        return self.weight.dtype

    def get_runtime(self):
        """Return a minimal runtime config (offload disabled)."""
        from types import SimpleNamespace

        return SimpleNamespace(offload_input=False, offload_gradient=False)

    def post_update_step(self) -> None:
        """Post-update hook called by AnalogSGD after weight update (no-op)."""
        pass

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

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward_impl(self, x: Tensor, is_test: bool = False) -> Tensor:
        """Raw forward MVM without autograd wrapping."""
        from aihwkit.simulator.triton.kernels.forward import triton_forward_mvm

        # Handle 3D input from convolution unfold: [batch, patches, features]
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])  # [batch*patches, features]

        use_io_manager = self._should_use_io_manager(self._f_io)

        if use_io_manager and not is_test:
            # IO Manager path: pre-scale input, run GEMM, post-process output
            io_result = self._io_manager.manage_input(x, self._f_io)
            if self._use_triton_gemm:
                from aihwkit.simulator.triton.kernels.fused_gemm import triton_fused_gemm
                y = triton_fused_gemm(
                    io_result.x_scaled, self.weight,
                    noise_std=0.0, bound=float("inf"), seed=0
                )
            else:
                y = triton_forward_mvm(
                    self.weight, io_result.x_scaled,
                    noise_std=0.0, bound=float("inf"), is_test=False
                )
            y = self._io_manager.manage_output(y, self._f_io, io_result.scale)
        else:
            # Direct path (default): existing behavior
            noise_std = 0.0
            bound = float("inf")
            if self._f_io is not None:
                noise_std = float(getattr(self._f_io, "out_noise", 0.0))
                out_bound = float(getattr(self._f_io, "out_bound", 0.0))
                if out_bound > 0.0:
                    bound = out_bound
            if self._use_triton_gemm:
                from aihwkit.simulator.triton.kernels.fused_gemm import triton_fused_gemm
                y = triton_fused_gemm(
                    x, self.weight, noise_std=noise_std, bound=bound, seed=0
                )
            else:
                y = triton_forward_mvm(
                    self.weight, x, noise_std=noise_std, bound=bound, is_test=is_test
                )

        if len(orig_shape) == 3:
            y = y.reshape(orig_shape[0], orig_shape[1], -1)  # [batch, patches, out_size]

        return y

    def forward(
        self,
        x_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        is_test: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """Forward MVM: y = clamp(x @ W.T + noise, -bound, bound).

        In training mode the computation is routed through
        ``_TritonAnalogFunction`` so that PyTorch's autograd graph is
        built and ``AnalogSGD`` can trigger the weight update.

        Args:
            x_input: Input tensor ``[batch, in_size]`` (or transposed).
            bias: Unused; kept for API compatibility.
            in_trans: If True, ``x_input`` is ``[in_size, batch]``.
            out_trans: If True, return ``[out_size, batch]`` instead of ``[batch, out_size]``.
            is_test: If True, suppress noise (inference mode) and bypass autograd.
            non_blocking: Unused; kept for API compatibility.

        Returns:
            ``[batch, out_size]`` (or transposed when ``out_trans``).
        """
        x = x_input.T if in_trans else x_input

        if torch.is_grad_enabled() and not is_test:
            # Route through custom autograd function so that loss.backward()
            # works and x / grad are stored for AnalogSGD.step().
            y = _TritonAnalogFunction.apply(self.analog_ctx, x, self, is_test)
        else:
            y = self._forward_impl(x, is_test=is_test)

        return y.T if out_trans else y

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def _backward_impl(self, d: Tensor) -> Tensor:
        """Raw backward MVM without autograd wrapping."""
        from aihwkit.simulator.triton.kernels.backward import triton_backward_mvm

        # Handle 3D gradients from convolution unfold: [batch, patches, out_size]
        orig_shape = d.shape
        if d.dim() == 3:
            d = d.reshape(-1, d.shape[-1])  # [batch*patches, out_size]

        use_io_manager = self._should_use_io_manager(self._b_io)

        if use_io_manager:
            # IO Manager path: pre-scale gradient, run backward GEMM, post-process
            io_result = self._io_manager.manage_input(d, self._b_io)
            if self._use_triton_gemm:
                from aihwkit.simulator.triton.kernels.fused_gemm import triton_fused_gemm_backward
                dx = triton_fused_gemm_backward(
                    io_result.x_scaled, self.weight,
                    noise_std=0.0, bound=float("inf"), seed=1
                )
            else:
                dx = triton_backward_mvm(self.weight, io_result.x_scaled,
                                         noise_std=0.0, bound=float("inf"))
            dx = self._io_manager.manage_output(dx, self._b_io, io_result.scale)
        else:
            # Direct path (default): existing behavior
            noise_std = 0.0
            bound = float("inf")
            if self._b_io is not None:
                noise_std = float(getattr(self._b_io, "inp_noise", 0.0))
                out_bound = float(getattr(self._b_io, "out_bound", 0.0))
                if out_bound > 0.0:
                    bound = out_bound
            if self._use_triton_gemm:
                from aihwkit.simulator.triton.kernels.fused_gemm import triton_fused_gemm_backward
                dx = triton_fused_gemm_backward(
                    d, self.weight, noise_std=noise_std, bound=bound, seed=1
                )
            else:
                dx = triton_backward_mvm(self.weight, d, noise_std=noise_std, bound=bound)

        if len(orig_shape) == 3:
            dx = dx.reshape(orig_shape[0], orig_shape[1], -1)  # [batch, patches, in_size]

        return dx

    @staticmethod
    def _should_use_io_manager(io_pars) -> bool:
        """Return True if IOManager should be active for the given IO params."""
        if io_pars is None:
            return False
        noise_mgmt = getattr(io_pars, "noise_management", None)
        # Normalize to string
        if hasattr(noise_mgmt, "name"):
            nm_str = noise_mgmt.name.upper()
        else:
            nm_str = str(noise_mgmt).upper() if noise_mgmt else "NONE"
        if "." in nm_str:
            nm_str = nm_str.rsplit(".", 1)[-1]
        if nm_str not in ("NONE", ""):
            return True
        # Also activate for explicit input bounds or noise
        inp_bound = float(getattr(io_pars, "inp_bound", 0.0))
        inp_noise = float(getattr(io_pars, "inp_noise", 0.0))
        return inp_bound > 0.0 or inp_noise > 0.0

    def backward(
        self,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """Backward MVM: dx = clamp(d @ W + noise, -bound, bound).

        Args:
            d_input: Upstream gradient ``[batch, out_size]`` (or transposed).
            bias: Unused; kept for API compatibility.
            in_trans: If True, ``d_input`` is ``[out_size, batch]``.
            out_trans: If True, return ``[in_size, batch]``.
            non_blocking: Unused; kept for API compatibility.

        Returns:
            ``[batch, in_size]`` (or transposed when ``out_trans``).
        """
        d = d_input.T if in_trans else d_input
        dx = self._backward_impl(d)
        return dx.T if out_trans else dx

    # ------------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------------

    def update(
        self,
        x_input: Tensor,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """Weight update via outer-product accumulation.

        Computes delta_W = d.T @ x, then applies the device update rule.

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

        # Compute weight delta: dW = d.T @ x  →  [out_size, in_size]
        if x.dim() == 1:
            delta_w = torch.outer(d.flatten(), x.flatten())
        else:
            delta_w = d.T @ x  # [out_size, in_size]

        params = {"learning_rate": self._learning_rate}
        with torch.no_grad():
            self.weight.data = self._tile_device.apply_update(
                self.weight.data, delta_w, params
            )
        return self.weight.data

    # ------------------------------------------------------------------
    # Weight accessors
    # ------------------------------------------------------------------

    def get_weights(self) -> tuple:
        """Get weight matrix (CPU tensor, detached). Returns (weight, None)."""
        return self.weight.data.detach().cpu(), None

    def set_weights(self, weight: Tensor, bias=None) -> None:
        """Set weight matrix. bias arg is ignored (no bias in analog tiles).

        Args:
            weight: ``[out_size, in_size]`` tensor.
            bias: Optional bias tensor (ignored, kept for API compatibility).
        """
        self.weight.data = weight.to(self.weight.device).clone()

    def set_weights_uniform_random(self, bmin: float, bmax: float) -> None:
        """Initialize weights to uniform random U(bmin, bmax).

        Args:
            bmin: Minimum value.
            bmax: Maximum value.
        """
        with torch.no_grad():
            self.weight.data.uniform_(bmin, bmax)

    def reset_columns(
        self, start_col: int, num_cols: int, bmin: float, bmax: float
    ) -> None:
        """Reset a column slice to uniform random values.

        Args:
            start_col: First column to reset.
            num_cols: Number of columns to reset.
            bmin: Minimum reset value.
            bmax: Maximum reset value.
        """
        with torch.no_grad():
            self.weight.data[:, start_col : start_col + num_cols].uniform_(bmin, bmax)

    # ------------------------------------------------------------------
    # Size / info
    # ------------------------------------------------------------------

    def get_x_size(self) -> int:
        """Input (column) dimension."""
        return self.x_size

    def get_d_size(self) -> int:
        """Output (row) dimension."""
        return self.d_size

    def get_brief_info(self) -> str:
        """One-line tile description."""
        return f"TritonAnalogTile(out_size={self.out_size}, in_size={self.in_size})"

    # ------------------------------------------------------------------
    # Learning rate
    # ------------------------------------------------------------------

    def get_learning_rate(self) -> Optional[float]:
        """Get the current learning rate."""
        return -self._learning_rate

    def set_learning_rate(self, learning_rate: Optional[float]) -> None:
        """Set the learning rate.

        Args:
            learning_rate: New learning rate; ignored if None.
        """
        if learning_rate is not None:
            self._learning_rate = -float(learning_rate)

    # ------------------------------------------------------------------
    # Hidden parameters
    # ------------------------------------------------------------------

    def get_hidden_parameters(self) -> Tensor:
        """Get hidden parameters as a flat tensor.

        For the ideal device there are no hidden parameters; returns an empty
        1-D tensor.

        Returns:
            Empty tensor of shape ``[0]``.
        """
        return torch.empty(0)

    def get_hidden_parameter_names(self) -> List[str]:
        """Names of the hidden parameter slices (empty for ideal device)."""
        return []

    def set_hidden_parameters(self, params: Tensor) -> None:
        """Set hidden parameters.

        No-op for the ideal device.

        Args:
            params: Ignored.
        """
        pass  # No hidden params

    # ------------------------------------------------------------------
    # Weight dynamics (no-ops for ideal device)
    # ------------------------------------------------------------------

    def decay_weights(self, alpha: float) -> None:
        """Apply weight decay: w *= (1 - alpha).

        Args:
            alpha: Decay coefficient.
        """
        with torch.no_grad():
            self.weight.data *= 1.0 - alpha

    def drift_weights(self, t_inference: float) -> None:
        """Apply power-law drift (no-op for ideal device).

        Args:
            t_inference: Inference time (unused).
        """
        pass  # No drift for ideal device

    def diffuse_weights(self) -> None:
        """Apply weight diffusion (no-op for ideal device)."""
        pass  # No diffusion for ideal device

    # ------------------------------------------------------------------
    # Meta / checkpointing
    # ------------------------------------------------------------------

    def get_meta_parameters(self) -> Any:
        """Return the RPU config (meta parameters)."""
        return self.rpu_config

    def dump_extra(self) -> Optional[Dict]:
        """Serialize extra state for checkpointing.

        Returns:
            Dict with ``learning_rate`` and ``hidden_params``.
        """
        return {
            "learning_rate": self._learning_rate,
            "hidden_params": self._hidden_params,
        }

    def load_extra(self, extra: Dict, strict: bool = False) -> None:
        """Load extra state from checkpoint.

        Args:
            extra: Dict produced by :meth:`dump_extra`.
            strict: Unused; kept for API compatibility.
        """
        if extra:
            self._learning_rate = float(extra.get("learning_rate", self._learning_rate))
            self._hidden_params = extra.get("hidden_params", {})


__all__ = ["TritonAnalogTile"]
