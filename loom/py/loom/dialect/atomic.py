# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared atomic operation vocabulary."""

from dataclasses import replace

from loom.dialect.memory import MemoryAccessFlags
from loom.dsl import EnumCase, EnumDef

AtomicKind = EnumDef(
    "AtomicKind",
    [
        EnumCase("xchgi", 0, doc="Integer exchange."),
        EnumCase("xchgf", 1, doc="Floating-point exchange."),
        EnumCase("addi", 2, doc="Integer addition."),
        EnumCase("addf", 3, doc="Floating-point addition with native subnormal handling unless preservation is explicitly requested."),
        EnumCase("subi", 4, doc="Integer subtraction."),
        EnumCase("andi", 5, doc="Bitwise AND."),
        EnumCase("ori", 6, doc="Bitwise OR."),
        EnumCase("xori", 7, doc="Bitwise XOR."),
        EnumCase("minsi", 8, doc="Signed integer minimum."),
        EnumCase("maxsi", 9, doc="Signed integer maximum."),
        EnumCase("minui", 10, doc="Unsigned integer minimum."),
        EnumCase("maxui", 11, doc="Unsigned integer maximum."),
        EnumCase("minimumf", 12, doc="IEEE 754 floating-point minimum."),
        EnumCase("maximumf", 13, doc="IEEE 754 floating-point maximum."),
        EnumCase("minnumf", 14, doc="C99 fmin-style floating-point minimum."),
        EnumCase("maxnumf", 15, doc="C99 fmax-style floating-point maximum."),
    ],
    doc="Read-modify-write operation kind supported by view and vector atomics.",
    c_type="loom_atomic_kind_t",
    c_const_prefix="LOOM_ATOMIC_KIND",
    c_include="loom/ops/atomic.h",
)

AtomicMemoryFlags = EnumDef(
    "AtomicMemoryFlags",
    [
        MemoryAccessFlags.case("root_in_bounds"),
        EnumCase(
            "noftz",
            2,
            doc=(
                "Preserve subnormal inputs and results of floating-point atomic "
                "addition. Without this flag the target may flush subnormals. "
                "The selected target must support the requested element width; "
                "preservation may require a compare-exchange loop. Returned "
                "old values retain their bits with or without this flag."
            ),
        ),
    ],
    doc="Semantic requirements for atomic memory updates.",
    c_type="loom_memory_access_flags_t",
    c_const_prefix="LOOM_MEMORY_ACCESS_FLAG",
    c_include="loom/ops/op_defs.h",
)

AtomicAccessFlags = replace(
    AtomicMemoryFlags,
    name="AtomicAccessFlags",
    cases=(AtomicMemoryFlags.case("root_in_bounds"),),
    doc="Source address promises shared by all atomic memory accesses.",
)

AtomicOrdering = EnumDef(
    "AtomicOrdering",
    [
        EnumCase("relaxed", 0, doc="Atomicity without inter-address synchronization."),
        EnumCase("acquire", 1, doc="Acquire ordering."),
        EnumCase("release", 2, doc="Release ordering."),
        EnumCase("acq_rel", 3, doc="Acquire and release ordering."),
        EnumCase("seq_cst", 4, doc="Sequentially consistent ordering."),
    ],
    doc="Atomic memory ordering between memory accesses.",
    c_type="loom_atomic_ordering_t",
    c_const_prefix="LOOM_ATOMIC_ORDERING",
    c_include="loom/ops/atomic.h",
)

AtomicLoadOrdering = replace(
    AtomicOrdering,
    name="AtomicLoadOrdering",
    cases=tuple(AtomicOrdering.case(case) for case in ("relaxed", "acquire", "seq_cst")),
    doc="Memory ordering permitted on an atomic load.",
)

AtomicStoreOrdering = replace(
    AtomicOrdering,
    name="AtomicStoreOrdering",
    cases=tuple(AtomicOrdering.case(case) for case in ("relaxed", "release", "seq_cst")),
    doc="Memory ordering permitted on an atomic store.",
)

AtomicFenceOrdering = replace(
    AtomicOrdering,
    name="AtomicFenceOrdering",
    cases=tuple(AtomicOrdering.case(case) for case in ("acquire", "release", "acq_rel", "seq_cst")),
    doc="Memory ordering permitted on a standalone fence.",
)

AtomicScope = EnumDef(
    "AtomicScope",
    [
        EnumCase("thread", 0, doc="Current invocation or thread."),
        EnumCase("subgroup", 1, doc="Current SIMD subgroup or wave."),
        EnumCase("workgroup", 2, doc="Current workgroup or block."),
        EnumCase("device", 3, doc="Current device."),
        EnumCase("system", 4, doc="Whole system."),
    ],
    doc="Synchronization scope for atomic memory effects.",
    c_type="loom_atomic_scope_t",
    c_const_prefix="LOOM_ATOMIC_SCOPE",
    c_include="loom/ops/atomic.h",
)
