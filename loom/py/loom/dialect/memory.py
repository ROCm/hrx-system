# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared target-independent memory-space vocabulary."""

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
    c_include="loom/ir/facts.h",
)

__all__ = ["MemorySpace"]
