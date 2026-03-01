# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Backend selection mechanism for Triton-based tile implementations."""

import os
from typing import Type, Optional, Any


class TritonBackend:
    """Backend selection and registry for Triton tile implementations.

    This class manages:
    - Availability checking (is Triton installed?)
    - Preference detection (AIHWKIT_USE_TRITON env var)
    - Tile class registration and retrieval
    """

    REGISTRY: dict = {}
    """Registry mapping tile names to Triton tile classes."""

    @classmethod
    def is_available(cls) -> bool:
        """Check if Triton is available (installed and importable).

        Returns:
            True if triton can be imported, False otherwise.
        """
        try:
            import triton  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def is_preferred(cls) -> bool:
        """Check if Triton backend is preferred via environment variable.

        Reads AIHWKIT_USE_TRITON environment variable.
        Returns True if set to '1', 'true', or 'True'.

        Returns:
            True if AIHWKIT_USE_TRITON is set to a truthy value, False otherwise.
        """
        env_value = os.environ.get("AIHWKIT_USE_TRITON", "0")
        return env_value in ("1", "true", "True")

    @classmethod
    def get_tile_class(cls, config: Any) -> Optional[Type]:
        """Get the appropriate Triton tile class for a given config.

        Selects tile class based on the device type in the config:
        - FloatingPointDevice → FloatingPointTritonTile
        - IdealDevice → TritonAnalogTile
        - ConstantStepDevice → ConstantStepTritonTile
        - ExpStepDevice → ExpStepTritonTile
        - Others → ConstantStepTritonTile (default analog)

        Args:
            config: An RPUConfig instance (SingleRPUConfig, InferenceRPUConfig, etc.)

        Returns:
            The Triton tile class if available, None otherwise.
        """
        try:
            # Check for InferenceRPUConfig first (by class name to avoid import cycles)
            config_type = type(config).__name__
            if config_type in ('InferenceRPUConfig', 'TorchInferenceRPUConfig'):
                from aihwkit.simulator.triton.tiles.inference import InferenceTritonTile
                return InferenceTritonTile

            device = getattr(config, 'device', None)
            device_type = type(device).__name__ if device is not None else ''

            if device_type == 'FloatingPointDevice':
                from aihwkit.simulator.triton.tiles.floating_point import FloatingPointTritonTile
                return FloatingPointTritonTile
            elif device_type == 'IdealDevice':
                from aihwkit.simulator.triton.tiles.analog import TritonAnalogTile
                return TritonAnalogTile
            elif device_type == 'ConstantStepDevice':
                from aihwkit.simulator.triton.tiles.constant_step import ConstantStepTritonTile
                return ConstantStepTritonTile
            elif device_type == 'ExpStepDevice':
                from aihwkit.simulator.triton.tiles.exp_step import ExpStepTritonTile
                return ExpStepTritonTile
            else:
                # Default: use ConstantStep tile for any pulsed device
                from aihwkit.simulator.triton.tiles.constant_step import ConstantStepTritonTile
                return ConstantStepTritonTile
        except ImportError:
            return None

    @classmethod
    def register_tile(cls, name: str, tile_class: Type) -> None:
        """Register a Triton tile class in the registry.

        Args:
            name: Name/key for the tile class (e.g., 'AnalogTile', 'InferenceTile')
            tile_class: The Triton tile class to register
        """
        cls.REGISTRY[name] = tile_class

    @classmethod
    def get_registered_tile(cls, name: str) -> Optional[Type]:
        """Retrieve a registered Triton tile class by name.

        Args:
            name: Name/key of the tile class

        Returns:
            The registered tile class, or None if not found.
        """
        return cls.REGISTRY.get(name)
