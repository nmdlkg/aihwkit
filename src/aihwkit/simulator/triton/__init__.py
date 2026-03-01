# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Triton-based backend for aihwkit simulator."""

try:
    import triton  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "Triton is not installed. Install it with: pip install triton"
    ) from exc

from aihwkit.simulator.triton.base_tile import TritonBaseTile
from aihwkit.simulator.triton.devices import BaseTritonDevice

__all__ = [
    "TritonBaseTile",
    "BaseTritonDevice",
]
