# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Sanitizer dialect op definitions."""

from loom.assembly import (
    ARROW,
    COLON,
    AttrDict,
    IndexList,
    PredicateList,
    Ref,
    Refs,
    ResultType,
    TemplateParam,
    TypeOf,
    TypesOf,
)
from loom.dialect.atomic import AtomicOrdering, AtomicScope
from loom.dialect.memory import MemorySpace
from loom.dsl import (
    ANY,
    ATTR_TYPE_BOOL,
    ATTR_TYPE_ENUM,
    ATTR_TYPE_I64,
    ATTR_TYPE_I64_ARRAY,
    CONVERGENT,
    FACT_IDENTITY,
    INDEX,
    REFINABLE_RESULT_TYPE_REFS,
    UNKNOWN_EFFECTS,
    VIEW,
    AttrDef,
    ContractFamily,
    Dialect,
    EnumCase,
    EnumDef,
    Op,
    Operand,
    RanksMatch,
    Result,
    SameElementType,
    VariadicValuesMatch,
)

__all__ = [
    "ALL_SANITIZER_OPS",
    "SanitizerAccessKind",
    "SanitizerAccessesKind",
    "SanitizerRaceAccessKind",
    "sanitizer_assert_access",
    "sanitizer_assert_accesses",
    "sanitizer_assert_layout",
    "sanitizer_assert_op",
    "sanitizer_assert_value",
    "sanitizer_race_access",
    "sanitizer_race_sync",
    "sanitizer_ops",
]


sanitizer_ops = Dialect(
    "sanitizer",
    dialect_id=0x1D,
    doc=(
        "Executable diagnostic assertions. Assertions refine SSA facts on the "
        "continuing path while preserving the dynamic trap/report boundary "
        "until canonicalization proves that boundary unreachable."
    ),
)


SanitizerAccessKind = EnumDef(
    "SanitizerAccessKind",
    [
        EnumCase("read", 0, doc="Assertion covers a logical read access."),
        EnumCase("write", 1, doc="Assertion covers a logical write access."),
        EnumCase(
            "read_write",
            2,
            doc="Assertion covers a logical read-modify-write access.",
        ),
    ],
    doc="Logical memory access kind covered by a sanitizer access assertion.",
)


SanitizerAccessesKind = EnumDef(
    "SanitizerAccessesKind",
    [
        EnumCase("read", 0, doc="Assertion covers logical read accesses."),
        EnumCase("write", 1, doc="Assertion covers logical write accesses."),
        EnumCase(
            "read_write",
            2,
            doc="Assertion covers logical read-modify-write accesses.",
        ),
    ],
    doc="Logical memory access kind covered by a repeated sanitizer access assertion.",
)


SanitizerRaceAccessKind = EnumDef(
    "SanitizerRaceAccessKind",
    [
        EnumCase("read", 0, doc="Race observation covers a logical read access."),
        EnumCase("write", 1, doc="Race observation covers a logical write access."),
        EnumCase(
            "read_write",
            2,
            doc="Race observation covers a logical read-modify-write access.",
        ),
    ],
    doc="Logical memory access kind covered by a sanitizer race observation.",
)


# ============================================================================
# sanitizer.assert.access
# ============================================================================

sanitizer_assert_access = Op(
    "sanitizer.assert.access",
    group=sanitizer_ops,
    doc=(
        "Assert that a logical indexed view access is valid. The assertion "
        "has the same index-list shape as ordinary view memory operations so "
        "source-level memory contracts remain typed until target lowering "
        "materializes address checks."
    ),
    operands=[
        Operand("view", VIEW, doc="Typed view being accessed."),
        Operand(
            "indices",
            INDEX,
            variadic=True,
            doc="Dynamic logical element indices.",
        ),
    ],
    attrs=[
        AttrDef(
            "kind",
            ATTR_TYPE_ENUM,
            enum_def=SanitizerAccessKind,
            doc="Logical access kind being asserted.",
        ),
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical element indices with INT64_MIN sentinels for dynamics.",
        ),
        AttrDef(
            "static_extents",
            ATTR_TYPE_I64_ARRAY,
            optional=True,
            doc=("Optional full-rank static logical footprint extents. Absent means a scalar element access."),
        ),
    ],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_sanitizer_assert_access_verify",
    format=[
        TemplateParam("kind"),
        Ref("view"),
        IndexList("indices", "static_indices"),
        AttrDict(),
        COLON,
        TypeOf("view"),
    ],
    examples=[
        "sanitizer.assert.access<read> %view[%row, %col] : view<[%M]x[%N]xf32, %layout>",
    ],
)


# ============================================================================
# sanitizer.assert.accesses
# ============================================================================

sanitizer_assert_accesses = Op(
    "sanitizer.assert.accesses",
    group=sanitizer_ops,
    doc=(
        "Assert that a static sequence of regularly-strided logical indexed "
        "view accesses is valid. Each sub-access has the same static extent "
        "tuple, and each successive sub-access origin is advanced by the "
        "static stride tuple. This keeps structured fragment and vector "
        "footprints as one executable assertion site instead of exploding one "
        "source memory operation into many independent assertions."
    ),
    operands=[
        Operand("view", VIEW, doc="Typed view being accessed."),
        Operand(
            "indices",
            INDEX,
            variadic=True,
            doc="Dynamic logical element indices for the first sub-access origin.",
        ),
    ],
    attrs=[
        AttrDef(
            "kind",
            ATTR_TYPE_ENUM,
            enum_def=SanitizerAccessesKind,
            doc="Logical access kind being asserted.",
        ),
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical element indices for the first sub-access origin with INT64_MIN sentinels for dynamics.",
        ),
        AttrDef(
            "static_extents",
            ATTR_TYPE_I64_ARRAY,
            doc="Full-rank static logical footprint extents for each sub-access.",
        ),
        AttrDef(
            "static_strides",
            ATTR_TYPE_I64_ARRAY,
            doc="Full-rank static logical origin stride between consecutive sub-accesses.",
        ),
        AttrDef(
            "static_count",
            ATTR_TYPE_I64,
            doc="Number of sub-accesses in the static sequence.",
        ),
    ],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_sanitizer_assert_accesses_verify",
    format=[
        TemplateParam("kind"),
        Ref("view"),
        IndexList("indices", "static_indices"),
        AttrDict(),
        COLON,
        TypeOf("view"),
    ],
    examples=[
        "sanitizer.assert.accesses<write> %view[%row, %col] {static_extents = [1, 16], static_strides = [1, 0], static_count = 16} : view<128x128xf32, #dense>",
    ],
)


# ============================================================================
# sanitizer.assert.value
# ============================================================================

sanitizer_assert_value = Op(
    "sanitizer.assert.value",
    group=sanitizer_ops,
    doc=(
        "Assert predicate constraints over SSA values and return checked "
        "identity aliases. Unlike assume ops, this op is executable: failing "
        "the assertion reports the site and aborts execution. Passing the "
        "assertion refines facts for the returned aliases."
    ),
    operands=[
        Operand(
            "values",
            ANY,
            variadic=True,
            doc="Values observed by the assertion predicates.",
        ),
    ],
    results=[
        Result(
            "results",
            ANY,
            variadic=True,
            doc="Same values on the path where every predicate held.",
        ),
    ],
    attrs=[
        AttrDef(
            "predicates",
            "predicate_list",
            doc="Predicate constraints that must hold at runtime.",
        ),
    ],
    constraints=[VariadicValuesMatch("values", "results")],
    traits=[UNKNOWN_EFFECTS, FACT_IDENTITY],
    canonicalize="loom_sanitizer_assert_value_canonicalize",
    facts="loom_sanitizer_assert_value_facts",
    verify="loom_sanitizer_assert_value_verify",
    format=[
        Refs("values"),
        PredicateList("predicates"),
        COLON,
        TypesOf("results"),
    ],
    examples=[
        "%n_checked = sanitizer.assert.value %n [range(%n, 0, 4096), mul(%n, 16)] : index",
        "%x_checked, %y_checked = sanitizer.assert.value %x, %y [lt(%x, %y)] : i32, i32",
    ],
)


# ============================================================================
# sanitizer.assert.op
# ============================================================================

sanitizer_assert_op = Op(
    "sanitizer.assert.op",
    group=sanitizer_ops,
    doc=(
        "Assert operation-level predicate constraints without producing "
        "checked aliases. This is the executable form for contracts such as "
        "valid divide operands, shift counts, overflow preconditions, and "
        "other facts where the checked operation itself remains the semantic "
        "anchor."
    ),
    operands=[
        Operand(
            "values",
            ANY,
            variadic=True,
            doc="Values observed by the assertion predicates.",
        ),
    ],
    attrs=[
        AttrDef(
            "predicates",
            "predicate_list",
            doc="Predicate constraints that must hold at runtime.",
        ),
    ],
    traits=[UNKNOWN_EFFECTS],
    canonicalize="loom_sanitizer_assert_op_canonicalize",
    verify="loom_sanitizer_assert_op_verify",
    format=[
        Refs("values"),
        PredicateList("predicates"),
        COLON,
        TypesOf("values"),
    ],
    examples=[
        "sanitizer.assert.op %lhs, %rhs [ne(%rhs, 0)] : i32, i32",
    ],
)


# ============================================================================
# sanitizer.assert.layout
# ============================================================================

sanitizer_assert_layout = Op(
    "sanitizer.assert.layout",
    group=sanitizer_ops,
    doc=(
        "Assert that a view satisfies a refined layout, shape, or encoding "
        "contract and return the same view on the continuing path. This is "
        "the executable counterpart to view.refine for diagnostics that must "
        "abort instead of trusting unchecked layout facts."
    ),
    operands=[Operand("view", VIEW, doc="Source view to assert and refine.")],
    results=[
        Result(
            "result",
            VIEW,
            doc="Same view on the path where the layout assertion held.",
        )
    ],
    constraints=[
        SameElementType("view", "result"),
        RanksMatch("view", "result"),
    ],
    traits=[UNKNOWN_EFFECTS, REFINABLE_RESULT_TYPE_REFS, FACT_IDENTITY],
    facts="loom_sanitizer_assert_layout_facts",
    type_transfer="loom_sanitizer_assert_layout_type_transfer",
    verify="loom_sanitizer_assert_layout_verify",
    format=[
        Ref("view"),
        COLON,
        TypeOf("view"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%checked = sanitizer.assert.layout %view : view<[%M]x[%N]xf32, %layout> -> view<16x[%N]xf32, %layout>",
    ],
)


# ============================================================================
# sanitizer.race.access
# ============================================================================

sanitizer_race_access = Op(
    "sanitizer.race.access",
    group=sanitizer_ops,
    doc=(
        "Observe a logical indexed view access for race detection. Unlike "
        "sanitizer.assert.access, this op does not assert that the access is "
        "individually valid and does not refine the continuing path. It records "
        "a memory event that target materialization can compare against prior "
        "unordered events in the selected race detector."
    ),
    operands=[
        Operand("view", VIEW, doc="Typed view being accessed."),
        Operand(
            "indices",
            INDEX,
            variadic=True,
            doc="Dynamic logical element indices.",
        ),
    ],
    attrs=[
        AttrDef(
            "kind",
            ATTR_TYPE_ENUM,
            enum_def=SanitizerRaceAccessKind,
            doc="Logical access kind being observed.",
        ),
        AttrDef(
            "atomic",
            ATTR_TYPE_BOOL,
            default=False,
            elide_default=True,
            doc="Whether the observed access is an atomic memory operation.",
        ),
        AttrDef(
            "ordering",
            ATTR_TYPE_ENUM,
            enum_def=AtomicOrdering,
            optional=True,
            doc="Atomic memory ordering when atomic is true.",
        ),
        AttrDef(
            "scope",
            ATTR_TYPE_ENUM,
            enum_def=AtomicScope,
            optional=True,
            doc="Atomic synchronization scope when atomic is true.",
        ),
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical element indices with INT64_MIN sentinels for dynamics.",
        ),
    ],
    traits=[UNKNOWN_EFFECTS],
    contracts=[ContractFamily.SANITIZER_RACE],
    verify="loom_sanitizer_race_access_verify",
    format=[
        TemplateParam("kind"),
        Ref("view"),
        IndexList("indices", "static_indices"),
        AttrDict(),
        COLON,
        TypeOf("view"),
    ],
    examples=[
        "sanitizer.race.access<read> %view[%lane] : view<64xi32, #dense>",
        "sanitizer.race.access<read_write> %view[%lane] {atomic = true, ordering = acq_rel, scope = workgroup} : view<64xi32, #dense>",
    ],
)


# ============================================================================
# sanitizer.race.sync
# ============================================================================

sanitizer_race_sync = Op(
    "sanitizer.race.sync",
    group=sanitizer_ops,
    doc=(
        "Observe a synchronization boundary for race detection. The original "
        "synchronization operation remains the semantic barrier or fence; this "
        "op records the boundary needed by race-detector materialization."
    ),
    attrs=[
        AttrDef(
            "memory_space",
            ATTR_TYPE_ENUM,
            enum_def=MemorySpace,
            doc="Memory space whose accesses are ordered by this boundary.",
        ),
        AttrDef(
            "ordering",
            ATTR_TYPE_ENUM,
            enum_def=AtomicOrdering,
            doc="Memory ordering applied to the synchronized accesses.",
        ),
        AttrDef(
            "scope",
            ATTR_TYPE_ENUM,
            enum_def=AtomicScope,
            doc="Synchronization scope.",
        ),
    ],
    traits=[UNKNOWN_EFFECTS, CONVERGENT],
    contracts=[ContractFamily.SANITIZER_RACE],
    verify="loom_sanitizer_race_sync_verify",
    format=[
        TemplateParam("memory_space"),
        AttrDict(),
    ],
    examples=[
        "sanitizer.race.sync<workgroup> {ordering = acq_rel, scope = workgroup}",
    ],
)


ALL_SANITIZER_OPS: tuple[Op, ...] = (
    sanitizer_assert_access,
    sanitizer_assert_value,
    sanitizer_assert_op,
    sanitizer_assert_layout,
    sanitizer_race_access,
    sanitizer_race_sync,
    sanitizer_assert_accesses,
)
