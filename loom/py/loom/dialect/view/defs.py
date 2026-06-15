# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""View dialect op definitions."""

from loom.assembly import (
    ARROW,
    COLON,
    COMMA,
    AttrDict,
    IndexList,
    Ref,
    ResultType,
    TemplateParam,
    TypeOf,
)
from loom.dialect.atomic import AtomicKind, AtomicOrdering, AtomicScope
from loom.dialect.cache import CacheScope, CacheTemporal
from loom.dsl import (
    ATTR_TYPE_ENUM,
    ATTR_TYPE_I64_ARRAY,
    FACT_IDENTITY,
    HINT,
    INDEX,
    PURE,
    REFINABLE_RESULT_TYPE_REFS,
    SCALAR,
    VIEW,
    AttrDef,
    ContractFamily,
    Dialect,
    EnumCase,
    EnumDef,
    HasIndexOrNonI1IntegerScalar,
    MemoryAccessInterface,
    Op,
    Operand,
    OpPhase,
    RanksMatch,
    Reads,
    ReadWrites,
    Result,
    SameElementType,
    SameEncoding,
    SameType,
    Writes,
)

# ============================================================================
# Group
# ============================================================================

view_ops = Dialect(
    "view",
    dialect_id=0x0D,
    doc="Logical view operations.",
    default_phase=OpPhase.EXECUTABLE,
)

PrefetchIntent = EnumDef(
    "PrefetchIntent",
    [
        EnumCase("read", 0, doc="Prefetch for an expected future read."),
        EnumCase("write", 1, doc="Prefetch for an expected future write."),
    ],
    doc="Intended future access kind for a prefetch hint.",
)

PrefetchLocality = EnumDef(
    "PrefetchLocality",
    [
        EnumCase("none", 0, doc="No temporal locality expected."),
        EnumCase("l1", 1, doc="Prefer L1 or the nearest target cache."),
        EnumCase("l2", 2, doc="Prefer L2 or the nearest mid-level target cache."),
        EnumCase("l3", 3, doc="Prefer L3 or the nearest outer target cache."),
    ],
    doc="Target-independent prefetch locality hint.",
)


def _cache_policy_attrs() -> list[AttrDef]:
    return [
        AttrDef(
            "cache_scope",
            ATTR_TYPE_ENUM,
            optional=True,
            enum_def=CacheScope,
            doc="Optional cache/coherency scope required by target lowering.",
        ),
        AttrDef(
            "cache_temporal",
            ATTR_TYPE_ENUM,
            optional=True,
            enum_def=CacheTemporal,
            doc="Optional temporal cache policy required by target lowering.",
        ),
    ]


def _indexed_memory_attrs() -> list[AttrDef]:
    return [
        *_cache_policy_attrs(),
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical element indices with INT64_MIN sentinels for dynamics.",
        ),
    ]


def _memory_access_interface(
    *,
    value: str | None = None,
    expected: str | None = None,
    replacement: str | None = None,
    cache: bool = True,
    atomic_kind: str | None = None,
    atomic_ordering: str | None = None,
    atomic_success_ordering: str | None = None,
    atomic_failure_ordering: str | None = None,
    atomic_scope: str | None = None,
) -> MemoryAccessInterface:
    cache_fields = {} if cache else {"cache_scope": None, "cache_temporal": None}
    return MemoryAccessInterface(
        value=value,
        expected=expected,
        replacement=replacement,
        atomic_kind=atomic_kind,
        atomic_ordering=atomic_ordering,
        atomic_success_ordering=atomic_success_ordering,
        atomic_failure_ordering=atomic_failure_ordering,
        atomic_scope=atomic_scope,
        **cache_fields,
    )


def _atomic_memory_access_interface(*, value: str) -> MemoryAccessInterface:
    return _memory_access_interface(
        value=value,
        atomic_kind="kind",
        atomic_ordering="ordering",
        atomic_scope="scope",
    )


# ============================================================================
# view.subview — logical subview with explicit offsets
# ============================================================================

view_subview = Op(
    name="view.subview",
    group=view_ops,
    doc=("Form a logical subview from an existing view. Offsets select the logical origin; result type dimensions provide the subview extents."),
    operands=[
        Operand("source", VIEW, doc="Source view."),
        Operand("offsets", INDEX, doc="Dynamic logical offsets.", variadic=True),
    ],
    results=[Result("result", VIEW, doc="Subview over the same storage root.")],
    attrs=[
        AttrDef(
            "static_offsets",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical offsets with INT64_MIN sentinels for dynamics.",
        ),
    ],
    constraints=[
        SameElementType("source", "result"),
        SameEncoding("source", "result"),
        RanksMatch("source", "result"),
    ],
    traits=[PURE, REFINABLE_RESULT_TYPE_REFS],
    verify="loom_view_subview_verify",
    facts="loom_view_subview_facts",
    format=[
        Ref("source"),
        IndexList("offsets", "static_offsets"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%sub = view.subview %source[%row, 0] : view<[%M]x[%N]xf32, %layout> -> view<16x[%N]xf32, %layout>",
    ],
)

# ============================================================================
# view.refine — type/fact refinement for an existing view
# ============================================================================

view_refine = Op(
    name="view.refine",
    group=view_ops,
    doc=(
        "Refine the static type information attached to an existing view while "
        "preserving the same storage root and byte base. This is an explicit "
        "SSA assertion point for layout, shape, and encoding facts discovered "
        "or required by earlier analysis."
    ),
    operands=[Operand("source", VIEW, doc="Source view to refine.")],
    results=[Result("result", VIEW, doc="Same view with refined type information.")],
    constraints=[
        SameElementType("source", "result"),
        RanksMatch("source", "result"),
    ],
    traits=[PURE, REFINABLE_RESULT_TYPE_REFS, FACT_IDENTITY],
    verify="loom_view_refine_verify",
    facts="loom_view_refine_facts",
    type_transfer="loom_view_refine_type_transfer",
    format=[
        Ref("source"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%refined = view.refine %view : view<[%M]xf32, %layout> -> view<16xf32, #dense>",
    ],
)

# ============================================================================
# view.load/view.store — scalar logical memory access
# ============================================================================

view_load = Op(
    name="view.load",
    group=view_ops,
    doc=("Load one scalar element from a typed view at a full-rank logical index. The index list is expressed in view coordinates and must name one position per view axis."),
    operands=[
        Operand("view", VIEW, doc="Typed source view."),
        Operand("indices", INDEX, doc="Dynamic logical element indices.", variadic=True),
    ],
    results=[Result("result", SCALAR, doc="Loaded scalar element.")],
    attrs=_indexed_memory_attrs(),
    constraints=[SameElementType("view", "result")],
    effects=[Reads("view")],
    interfaces=[_memory_access_interface()],
    verify="loom_view_load_verify",
    format=[
        Ref("view"),
        IndexList("indices", "static_indices"),
        AttrDict(),
        COLON,
        TypeOf("view"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%x = view.load %view[%row, %col] : view<[%M]x[%N]xf32, %layout> -> f32",
    ],
)

view_store = Op(
    name="view.store",
    group=view_ops,
    doc=("Store one scalar element into a typed view at a full-rank logical index. The index list is expressed in view coordinates and must name one position per view axis."),
    operands=[
        Operand("value", SCALAR, doc="Scalar element to store."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("indices", INDEX, doc="Dynamic logical element indices.", variadic=True),
    ],
    attrs=_indexed_memory_attrs(),
    constraints=[SameElementType("value", "view")],
    effects=[Writes("view")],
    interfaces=[_memory_access_interface(value="value")],
    verify="loom_view_store_verify",
    format=[
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        AttrDict(),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("view"),
    ],
    examples=[
        "view.store %x, %view[%row, %col] : f32, view<[%M]x[%N]xf32, %layout>",
    ],
)

# ============================================================================
# view.atomic.* — scalar logical atomic memory access
# ============================================================================


def _atomic_memory_attrs() -> list[AttrDef]:
    return [
        AttrDef("kind", ATTR_TYPE_ENUM, enum_def=AtomicKind),
        AttrDef(
            "ordering",
            ATTR_TYPE_ENUM,
            enum_def=AtomicOrdering,
            doc="Required atomic memory ordering.",
        ),
        AttrDef(
            "scope",
            ATTR_TYPE_ENUM,
            enum_def=AtomicScope,
            doc="Required atomic synchronization scope.",
        ),
        *_cache_policy_attrs(),
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical element indices with INT64_MIN sentinels for dynamics.",
        ),
    ]


def _atomic_cmpxchg_memory_attrs() -> list[AttrDef]:
    return [
        AttrDef(
            "success_ordering",
            ATTR_TYPE_ENUM,
            enum_def=AtomicOrdering,
            doc="Memory ordering used when the compare-exchange succeeds.",
        ),
        AttrDef(
            "failure_ordering",
            ATTR_TYPE_ENUM,
            enum_def=AtomicOrdering,
            doc="Memory ordering used when the compare-exchange fails.",
        ),
        AttrDef(
            "scope",
            ATTR_TYPE_ENUM,
            enum_def=AtomicScope,
            doc="Required atomic synchronization scope.",
        ),
        *_cache_policy_attrs(),
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical element indices with INT64_MIN sentinels for dynamics.",
        ),
    ]


view_atomic_reduce = Op(
    name="view.atomic.reduce",
    group=view_ops,
    doc=(
        "Atomically combine one scalar value into a typed view element at a "
        "full-rank logical index. Atomic ordering and scope are required so "
        "the synchronization contract remains explicit after vector-to-scalar "
        "lowering."
    ),
    operands=[
        Operand("value", SCALAR, doc="Scalar contribution."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("indices", INDEX, doc="Dynamic logical element indices.", variadic=True),
    ],
    attrs=_atomic_memory_attrs(),
    constraints=[SameElementType("value", "view")],
    effects=[ReadWrites("view")],
    contracts=[ContractFamily.MEMORY_ATOMIC],
    interfaces=[_atomic_memory_access_interface(value="value")],
    verify="loom_view_atomic_reduce_verify",
    format=[
        TemplateParam("kind"),
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        AttrDict(),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("view"),
    ],
    examples=[
        "view.atomic.reduce<addi> %value, %view[%row, %col] {ordering = relaxed, scope = workgroup} : i32, view<[%M]x[%N]xi32, %layout>",
    ],
)

view_atomic_rmw = Op(
    name="view.atomic.rmw",
    group=view_ops,
    doc=("Atomically read one scalar view element, combine it with a scalar update value, write the combined value back, and return the old value observed by that atomic operation."),
    operands=[
        Operand("value", SCALAR, doc="Scalar update value."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("indices", INDEX, doc="Dynamic logical element indices.", variadic=True),
    ],
    results=[Result("result", SCALAR, doc="Old memory value read by the atomic operation.")],
    attrs=_atomic_memory_attrs(),
    constraints=[
        SameElementType("value", "view", "result"),
        SameType("value", "result"),
    ],
    effects=[ReadWrites("view")],
    contracts=[ContractFamily.MEMORY_ATOMIC],
    interfaces=[_atomic_memory_access_interface(value="value")],
    verify="loom_view_atomic_rmw_verify",
    format=[
        TemplateParam("kind"),
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        AttrDict(),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("view"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%old = view.atomic.rmw<addi> %value, %view[%row, %col] {ordering = relaxed, scope = workgroup} : i32, view<[%M]x[%N]xi32, %layout> -> i32",
    ],
)

view_atomic_cmpxchg = Op(
    name="view.atomic.cmpxchg",
    group=view_ops,
    doc=(
        "Atomically compare one scalar view element with an expected value, write a replacement value when they match, and return the old observed value. Success is derived by comparing old == expected."
    ),
    operands=[
        Operand("expected", SCALAR, doc="Scalar value compared against memory."),
        Operand("replacement", SCALAR, doc="Scalar value written on success."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("indices", INDEX, doc="Dynamic logical element indices.", variadic=True),
    ],
    results=[Result("old", SCALAR, doc="Old memory value observed by the atomic operation.")],
    attrs=_atomic_cmpxchg_memory_attrs(),
    constraints=[
        HasIndexOrNonI1IntegerScalar("expected"),
        SameElementType("expected", "replacement", "view", "old"),
        SameType("expected", "replacement", "old"),
    ],
    effects=[ReadWrites("view")],
    contracts=[ContractFamily.MEMORY_ATOMIC],
    interfaces=[
        _memory_access_interface(
            expected="expected",
            replacement="replacement",
            atomic_success_ordering="success_ordering",
            atomic_failure_ordering="failure_ordering",
            atomic_scope="scope",
        )
    ],
    verify="loom_view_atomic_cmpxchg_verify",
    format=[
        Ref("expected"),
        COMMA,
        Ref("replacement"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        AttrDict(),
        COLON,
        TypeOf("expected"),
        COMMA,
        TypeOf("view"),
        ARROW,
        ResultType("old"),
    ],
    examples=[
        "%old = view.atomic.cmpxchg %expected, %replacement, %view[%row, %col] {success_ordering = acq_rel, failure_ordering = acquire, scope = workgroup} : i32, view<[%M]x[%N]xi32, %layout> -> i32",
    ],
)

# ============================================================================
# view.prefetch — discardable compiler hint for a future view access
# ============================================================================

view_prefetch = Op(
    name="view.prefetch",
    group=view_ops,
    doc=(
        "Compiler hint for a future access to a logical view origin. Prefetch "
        "has no semantic memory effects and may not fault semantically, but it "
        "is intentionally preserved by ordinary canonicalization/DCE until an "
        "explicit hint-stripping pass removes it."
    ),
    operands=[
        Operand("view", VIEW, doc="Typed view whose address should be prefetched."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    attrs=[
        AttrDef("intent", ATTR_TYPE_ENUM, enum_def=PrefetchIntent, doc="Required expected future access kind."),
        AttrDef("locality", ATTR_TYPE_ENUM, enum_def=PrefetchLocality, doc="Required target-independent locality hint."),
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical origin indices with INT64_MIN sentinels for dynamics.",
        ),
    ],
    traits=[HINT],
    interfaces=[_memory_access_interface(cache=False)],
    verify="loom_view_prefetch_verify",
    format=[
        Ref("view"),
        IndexList("indices", "static_indices"),
        AttrDict(),
        COLON,
        TypeOf("view"),
    ],
    examples=[
        "view.prefetch %view[%row, %col] {intent = read, locality = l2} : view<[%M]x[%N]xf32, %layout>",
    ],
)

# ============================================================================
# Registry
# ============================================================================

ALL_VIEW_OPS: tuple[Op, ...] = (
    view_subview,
    view_refine,
    view_load,
    view_store,
    view_atomic_reduce,
    view_atomic_rmw,
    view_atomic_cmpxchg,
    view_prefetch,
)
