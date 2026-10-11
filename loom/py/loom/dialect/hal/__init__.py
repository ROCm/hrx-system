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

hal_device_group_type = TypeDef(
    name="hal.device_group",
    doc="Managed device group with immutable device topology.",
    reference=ReferenceTypeKey("hal", "device_group"),
)

hal_queue_type = TypeDef(
    name="hal.queue",
    doc="Managed HAL queue retaining its device group.",
    reference=ReferenceTypeKey("hal", "queue"),
)

hal_command_buffer_type = TypeDef(
    name="hal.command_buffer",
    doc="Managed command buffer retaining its queue family's device group.",
    reference=ReferenceTypeKey("hal", "command_buffer"),
)

hal_semaphore_type = TypeDef(
    name="hal.semaphore",
    doc="Managed timeline semaphore retaining its device group.",
    reference=ReferenceTypeKey("hal", "semaphore"),
)

hal_pool_type = TypeDef(
    name="hal.pool",
    doc="Managed memory pool retaining its device group, with explicit native work lifetime.",
    reference=ReferenceTypeKey("hal", "pool"),
)

# ============================================================================
# Registry
# ============================================================================

ALL_HAL_TYPES: tuple[TypeDef, ...] = (
    hal_buffer_type,
    hal_buffer_view_type,
    hal_device_group_type,
    hal_queue_type,
    hal_command_buffer_type,
    hal_semaphore_type,
    hal_pool_type,
)

__all__ = [
    "hal_buffer_type",
    "hal_buffer_view_type",
    "hal_device_group_type",
    "hal_queue_type",
    "hal_command_buffer_type",
    "hal_semaphore_type",
    "hal_pool_type",
    "ALL_HAL_TYPES",
]
