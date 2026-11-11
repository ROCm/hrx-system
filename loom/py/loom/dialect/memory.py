# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared target-independent memory vocabulary."""

from loom.dsl import EnumCase, EnumDef

MemorySpace = EnumDef(
    "MemorySpace",
    [
        EnumCase("unknown", 0, doc="No target-independent memory space is known."),
        EnumCase("global", 1, doc="Device-visible global storage."),
        EnumCase("workgroup", 2, doc="Workgroup/shared storage."),
        EnumCase("private", 3, doc="Invocation-private storage."),
        EnumCase("constant", 4, doc="Read-only constant storage."),
        EnumCase("host", 5, doc="Host-visible storage."),
        EnumCase("descriptor", 6, doc="Descriptor-backed storage identity."),
        EnumCase(
            "generic",
            7,
            doc=("Target-generic device storage. Targets may lower this to a generic address space such as AMDGPU flat memory."),
        ),
    ],
    doc="Target-independent memory space for storage roots, views, and fences.",
    c_type="loom_value_fact_memory_space_t",
    c_const_prefix="LOOM_VALUE_FACT_MEMORY_SPACE",
    c_include="loom/ir/memory_space.h",
)

MemoryAccessFlags = EnumDef(
    "MemoryAccessFlags",
    [
        EnumCase(
            "volatile",
            1,
            doc=(
                "Require every dynamic access to execute as an independently "
                "observable operation and preserve its order relative to other "
                "volatile accesses. A target may decompose one logical access "
                "into multiple physical accesses. This does not provide "
                "atomicity, synchronization, or a cache-coherence guarantee."
            ),
        ),
        EnumCase(
            "root_in_bounds",
            8,
            doc=(
                "Promise that the complete physical access lies within its "
                "allocation root. Logical view indices must still be proven "
                "in bounds. This carries a source-language access contract "
                "through opaque pointer transport and has no target execution "
                "semantics."
            ),
        ),
    ],
    doc="Semantic modifiers shared by scalar and vector memory accesses.",
    c_type="loom_memory_access_flags_t",
    c_const_prefix="LOOM_MEMORY_ACCESS_FLAG",
    c_include="loom/ops/op_defs.h",
)

LowMemoryAccessFlags = EnumDef(
    "LowMemoryAccessFlags",
    [
        MemoryAccessFlags.case("volatile"),
        EnumCase(
            "atomic",
            4,
            doc=(
                "Mark one physical load/store as an indivisible atomic observation. "
                "Every dynamic access is observable and same-location issue order "
                "is preserved. The instruction and address alignment supply atomicity; "
                "explicit fences and cache controls supply synchronization and visibility. "
                "Independent addresses do not require ordered completion."
            ),
        ),
    ],
    doc="Physical memory observation semantics for descriptor-backed instructions.",
    c_type="loom_memory_access_flags_t",
    c_const_prefix="LOOM_MEMORY_ACCESS_FLAG",
    c_include="loom/ops/op_defs.h",
)

__all__ = ["LowMemoryAccessFlags", "MemoryAccessFlags", "MemorySpace"]
