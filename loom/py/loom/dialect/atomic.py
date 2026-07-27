# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared atomic operation vocabulary."""

from loom.dsl import EnumCase, EnumDef

AtomicKind = EnumDef(
    "AtomicKind",
    [
        EnumCase("xchgi", 0, doc="Integer exchange."),
        EnumCase("xchgf", 1, doc="Floating-point exchange."),
        EnumCase("addi", 2, doc="Integer addition."),
        EnumCase("addf", 3, doc="Floating-point addition."),
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

AtomicOrdering = EnumDef(
    "AtomicOrdering",
    [
        EnumCase("relaxed", 0, doc="Atomicity without inter-address synchronization."),
        EnumCase("acquire", 1, doc="Acquire ordering."),
        EnumCase("release", 2, doc="Release ordering."),
        EnumCase("acq_rel", 3, doc="Acquire and release ordering."),
        EnumCase("seq_cst", 4, doc="Sequentially consistent ordering."),
    ],
    doc="Atomic memory ordering. The relaxed case lowers to LLVM monotonic RMW ordering.",
    c_type="loom_atomic_ordering_t",
    c_const_prefix="LOOM_ATOMIC_ORDERING",
    c_include="loom/ops/atomic.h",
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
