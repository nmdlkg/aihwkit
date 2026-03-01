# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Tests for Triton-based analog tiles."""

import pytest
import torch

# Skip entire module if triton is not available
triton = pytest.importorskip("triton")

# Skip entire module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

X_SIZE = 8
D_SIZE = 4
BATCH = 16


def _make_floating_point_tile():
    from aihwkit.simulator.triton.tiles.floating_point import FloatingPointTritonTile

    tile = FloatingPointTritonTile(D_SIZE, X_SIZE)  # out_size=D_SIZE, in_size=X_SIZE
    tile.weight.data = tile.weight.data.cuda()
    tile.set_weights_uniform_random(-0.1, 0.1)
    return tile


def _make_analog_tile():
    from aihwkit.simulator.triton.tiles.analog import TritonAnalogTile
    from aihwkit.simulator.configs.configs import SingleRPUConfig
    from aihwkit.simulator.configs.devices import IdealDevice

    rpu_config = SingleRPUConfig(device=IdealDevice())
    tile = TritonAnalogTile(
        D_SIZE, X_SIZE, rpu_config=rpu_config
    )  # out_size=D_SIZE, in_size=X_SIZE
    tile.weight.data = tile.weight.data.cuda()
    tile.set_weights_uniform_random(-0.1, 0.1)
    return tile


def _make_constant_step_tile():
    from aihwkit.simulator.triton.tiles.constant_step import ConstantStepTritonTile
    from aihwkit.simulator.configs.configs import SingleRPUConfig
    from aihwkit.simulator.configs.devices import ConstantStepDevice

    rpu_config = SingleRPUConfig(device=ConstantStepDevice())
    tile = ConstantStepTritonTile(
        D_SIZE, X_SIZE, rpu_config=rpu_config
    )  # out_size=D_SIZE, in_size=X_SIZE
    tile.weight.data = tile.weight.data.cuda()
    tile.set_weights_uniform_random(-0.1, 0.1)
    return tile


def _make_exp_step_tile():
    from aihwkit.simulator.triton.tiles.exp_step import ExpStepTritonTile
    from aihwkit.simulator.configs.configs import SingleRPUConfig
    from aihwkit.simulator.configs.devices import ExpStepDevice

    rpu_config = SingleRPUConfig(device=ExpStepDevice())
    tile = ExpStepTritonTile(
        D_SIZE, X_SIZE, rpu_config=rpu_config
    )  # out_size=D_SIZE, in_size=X_SIZE
    tile.weight.data = tile.weight.data.cuda()
    tile.set_weights_uniform_random(-0.1, 0.1)
    return tile


# ===========================================================================
# FloatingPointTritonTile
# ===========================================================================


class TestFloatingPointTritonTile:
    """Tests for FloatingPointTritonTile."""

    def setup_method(self):
        self.tile = _make_floating_point_tile()

    def test_forward_shape(self):
        x = torch.randn(BATCH, X_SIZE, device="cuda")
        y = self.tile.forward(x)
        assert y.shape == (BATCH, D_SIZE)

    def test_backward_shape(self):
        d = torch.randn(BATCH, D_SIZE, device="cuda")
        dx = self.tile.backward(d)
        assert dx.shape == (BATCH, X_SIZE)

    def test_forward_correctness(self):
        W = torch.eye(D_SIZE, X_SIZE, device="cuda")
        self.tile.set_weights(W)
        x = torch.randn(BATCH, X_SIZE, device="cuda")
        y = self.tile.forward(x)
        expected = x @ W.T
        assert torch.allclose(y, expected, atol=1e-5)

    def test_update_changes_weights(self):
        w_before, _ = self.tile.get_weights()
        x = torch.randn(BATCH, X_SIZE, device="cuda")
        d = torch.randn(BATCH, D_SIZE, device="cuda")
        self.tile.update(x, d)
        w_after, _ = self.tile.get_weights()
        assert not torch.allclose(w_before, w_after)

    def test_get_set_weights(self):
        W = torch.randn(D_SIZE, X_SIZE)
        self.tile.set_weights(W)
        W_got, _ = self.tile.get_weights()
        assert torch.allclose(W, W_got, atol=1e-6)

    def test_learning_rate(self):
        self.tile.set_learning_rate(0.05)
        assert abs(self.tile.get_learning_rate() - 0.05) < 1e-6


# ===========================================================================
# TritonAnalogTile (IdealDevice)
# ===========================================================================


class TestTritonAnalogTile:
    """Tests for TritonAnalogTile with IdealDevice."""

    def setup_method(self):
        self.tile = _make_analog_tile()

    def test_forward_shape(self):
        x = torch.randn(BATCH, X_SIZE, device="cuda")
        y = self.tile.forward(x)
        assert y.shape == (BATCH, D_SIZE)

    def test_backward_shape(self):
        d = torch.randn(BATCH, D_SIZE, device="cuda")
        dx = self.tile.backward(d)
        assert dx.shape == (BATCH, X_SIZE)

    def test_forward_correctness(self):
        """Ideal analog tile forward (is_test=True) should match x @ W.T."""
        W = torch.eye(D_SIZE, X_SIZE, device="cuda")
        self.tile.set_weights(W)
        x = torch.randn(BATCH, X_SIZE, device="cuda")
        y = self.tile.forward(x, is_test=True)
        expected = x @ W.T
        assert torch.allclose(y, expected, atol=1e-5)

    def test_update_changes_weights(self):
        w_before, _ = self.tile.get_weights()
        x = torch.randn(BATCH, X_SIZE, device="cuda")
        d = torch.randn(BATCH, D_SIZE, device="cuda")
        self.tile.update(x, d)
        w_after, _ = self.tile.get_weights()
        assert not torch.allclose(w_before, w_after)

    def test_get_set_weights(self):
        W = torch.randn(D_SIZE, X_SIZE)
        self.tile.set_weights(W)
        W_got, _ = self.tile.get_weights()
        assert torch.allclose(W, W_got, atol=1e-6)

    def test_learning_rate(self):
        self.tile.set_learning_rate(0.05)
        assert abs(self.tile.get_learning_rate() - 0.05) < 1e-6


# ===========================================================================
# ConstantStepTritonTile
# ===========================================================================


class TestConstantStepTritonTile:
    """Tests for ConstantStepTritonTile."""

    def setup_method(self):
        self.tile = _make_constant_step_tile()

    def test_forward_shape(self):
        x = torch.randn(BATCH, X_SIZE, device="cuda")
        y = self.tile.forward(x)
        assert y.shape == (BATCH, D_SIZE)

    def test_backward_shape(self):
        d = torch.randn(BATCH, D_SIZE, device="cuda")
        dx = self.tile.backward(d)
        assert dx.shape == (BATCH, X_SIZE)

    def test_update_changes_weights(self):
        w_before, _ = self.tile.get_weights()
        x = torch.randn(BATCH, X_SIZE, device="cuda")
        d = torch.randn(BATCH, D_SIZE, device="cuda")
        self.tile.update(x, d)
        w_after, _ = self.tile.get_weights()
        assert not torch.allclose(w_before, w_after)

    def test_update_respects_bounds(self):
        """After update, weights should stay within [w_min, w_max]."""
        # Set weights near bounds
        W = torch.full((D_SIZE, X_SIZE), 0.55)
        self.tile.set_weights(W)
        # Large update to push past bounds
        x = torch.ones(BATCH, X_SIZE, device="cuda")
        d = torch.ones(BATCH, D_SIZE, device="cuda")
        for _ in range(50):
            self.tile.update(x, d)
        w_after, _ = self.tile.get_weights()
        assert w_after.max() <= self.tile._tile_device.w_max + 1e-6
        assert w_after.min() >= self.tile._tile_device.w_min - 1e-6

    def test_update_uses_bl_pulse_counts(self):
        self.tile._tile_device.dw_min_std = 0.0
        self.tile.set_learning_rate(0.01)
        self.tile.set_weights(torch.zeros(D_SIZE, X_SIZE))

        x = torch.ones(BATCH, X_SIZE, device="cuda")
        d = torch.ones(BATCH, D_SIZE, device="cuda")
        self.tile.update(x, d)

        w_after, _ = self.tile.get_weights()
        lr = abs(self.tile.get_learning_rate())
        dw_min = self.tile._tile_device.dw_min
        k_val = lr / dw_min
        bl = min(
            max(1, int(torch.ceil(torch.tensor(k_val)).item())), self.tile.bl_count
        )
        expected = -dw_min * BATCH * bl

        assert torch.allclose(w_after, torch.full_like(w_after, expected), atol=1e-6)

    def test_get_set_weights(self):
        W = torch.randn(D_SIZE, X_SIZE) * 0.1
        self.tile.set_weights(W)
        W_got, _ = self.tile.get_weights()
        assert torch.allclose(W, W_got, atol=1e-6)

    def test_learning_rate(self):
        self.tile.set_learning_rate(0.05)
        assert abs(self.tile.get_learning_rate() - 0.05) < 1e-6


# ===========================================================================
# ExpStepTritonTile
# ===========================================================================


class TestExpStepTritonTile:
    """Tests for ExpStepTritonTile."""

    def setup_method(self):
        self.tile = _make_exp_step_tile()

    def test_forward_shape(self):
        x = torch.randn(BATCH, X_SIZE, device="cuda")
        y = self.tile.forward(x)
        assert y.shape == (BATCH, D_SIZE)

    def test_backward_shape(self):
        d = torch.randn(BATCH, D_SIZE, device="cuda")
        dx = self.tile.backward(d)
        assert dx.shape == (BATCH, X_SIZE)

    def test_update_changes_weights(self):
        w_before, _ = self.tile.get_weights()
        x = torch.randn(BATCH, X_SIZE, device="cuda")
        d = torch.randn(BATCH, D_SIZE, device="cuda")
        self.tile.update(x, d)
        w_after, _ = self.tile.get_weights()
        assert not torch.allclose(w_before, w_after)

    def test_update_respects_bounds(self):
        """After update, weights should stay within [w_min, w_max]."""
        W = torch.full((D_SIZE, X_SIZE), 0.55)
        self.tile.set_weights(W)
        x = torch.ones(BATCH, X_SIZE, device="cuda")
        d = torch.ones(BATCH, D_SIZE, device="cuda")
        for _ in range(50):
            self.tile.update(x, d)
        w_after, _ = self.tile.get_weights()
        assert w_after.max() <= self.tile._tile_device.w_max + 1e-6
        assert w_after.min() >= self.tile._tile_device.w_min - 1e-6

    def test_get_set_weights(self):
        W = torch.randn(D_SIZE, X_SIZE) * 0.1
        self.tile.set_weights(W)
        W_got, _ = self.tile.get_weights()
        assert torch.allclose(W, W_got, atol=1e-6)

    def test_learning_rate(self):
        self.tile.set_learning_rate(0.05)
        assert abs(self.tile.get_learning_rate() - 0.05) < 1e-6
