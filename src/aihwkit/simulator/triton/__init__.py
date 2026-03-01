# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Triton-based backend for aihwkit simulator."""

_triton_available = False
try:
    import triton  # noqa: F401

    _triton_available = True
except ImportError:
    pass


def is_triton_available() -> bool:
    """Return True if triton is installed and available."""
    return _triton_available


if _triton_available:
    from aihwkit.simulator.triton.base_tile import TritonBaseTile
    from aihwkit.simulator.triton.devices import BaseTritonDevice
else:
    TritonBaseTile = None
    BaseTritonDevice = None

__all__ = [
    "TritonBaseTile",
    "BaseTritonDevice",
    "is_triton_available",
]
