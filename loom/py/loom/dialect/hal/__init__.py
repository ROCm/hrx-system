# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""HAL dialect: hardware abstraction layer types.

Provides opaque type declarations for HAL resource handles that
appear in loom IR. These are dialect-specific types (DialectType
in ir.py), not built-in types — they are registered by code that
needs them (pool ops, host-side scheduling, etc.).

hal.buffer — an opaque device buffer handle. Produced by pool.buffer,
consumed by HAL queue operations (copies, fills, etc.).
"""

from loom.dsl import TypeDef

# ============================================================================
# hal.buffer — opaque device buffer handle
# ============================================================================

hal_buffer_type = TypeDef(
    name="hal.buffer",
    doc="Opaque device buffer handle for HAL queue operations.",
)

# ============================================================================
# Registry
# ============================================================================

ALL_HAL_TYPES: tuple[TypeDef, ...] = (hal_buffer_type,)

__all__ = [
    "hal_buffer_type",
    "ALL_HAL_TYPES",
]
