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

hal.buffer_view — a managed buffer range with shape, element type and encoding.

hal.fence — a retained set of device semaphore timepoints.
"""

from loom.dsl import ReferenceTypeKey, TypeDef

# ============================================================================
# hal.buffer — opaque device buffer handle
# ============================================================================

hal_buffer_type = TypeDef(
    name="hal.buffer",
    doc="Opaque device buffer handle for HAL queue operations.",
    reference=ReferenceTypeKey("hal", "buffer"),
)

hal_buffer_view_type = TypeDef(
    name="hal.buffer_view",
    doc="Managed HAL buffer range with shape, element type and encoding.",
    reference=ReferenceTypeKey("hal", "buffer_view"),
)

hal_fence_type = TypeDef(
    name="hal.fence",
    doc="Retained completion set of HAL semaphore timepoints.",
    reference=ReferenceTypeKey("hal", "fence"),
)

# ============================================================================
# Registry
# ============================================================================

ALL_HAL_TYPES: tuple[TypeDef, ...] = (
    hal_buffer_type,
    hal_buffer_view_type,
    hal_fence_type,
)

__all__ = [
    "hal_buffer_type",
    "hal_buffer_view_type",
    "hal_fence_type",
    "ALL_HAL_TYPES",
]
