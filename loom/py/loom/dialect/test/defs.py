# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test dialect op definitions.

These ops exist solely to validate the DSL, format elements, printer,
parser, and builder infrastructure. They are not part of the loom
compiler — they are the acceptance tests for the toolchain.

Every format element type appears in at least one test op. Every
DSL feature (variadic, optional, tied results, regions, enums,
constraints, traits) is exercised. If the test ops round-trip
through printer and parser, real dialect ops will too.
"""

from loom.assembly import (
    ARROW,
    BINDING_ELEMENT,
    COLON,
    COMMA,
    EQUALS,
    GLUE,
    LBRACKET,
    LPAREN,
    RBRACKET,
    RPAREN,
    Attr,
    AttrDict,
    AttrParams,
    AttrTable,
    BindingList,
    BlockArgs,
    BlockRef,
    Clause,
    FuncArgs,
    IndexList,
    OperandDict,
    OptionalGroup,
    Param,
    PredicateList,
    Ref,
    Refs,
    Region,
    RegionTable,
    ResultType,
    ResultTypeList,
    Scope,
    SymbolRef,
    TemplateParam,
    TemplateParamFlags,
    TypedRefs,
    TypeOf,
    TypesOf,
    kw,
)
from loom.dialect.target import target_record_attrs
from loom.dsl import (
    ANY,
    ANY_ENCODING,
    ATTR_TYPE_ENUM,
    ATTR_TYPE_ENUM_ARRAY,
    ATTR_TYPE_FLAGS,
    ATTR_TYPE_I64_ARRAY,
    ATTR_TYPE_PARAMETERIZED,
    ATTR_TYPE_PARAMETERIZED_ARRAY,
    ATTR_TYPE_SYMBOL,
    BY_REFERENCE,
    CONSTANT_LIKE,
    CONVERGENT,
    ELEMENTWISE,
    FLOAT,
    I1,
    INDEX,
    INTEGER,
    INVOLUTION,
    ISOLATED_FROM_ABOVE,
    POISON_BOUNDARY,
    POOL,
    PURE,
    STORAGE,
    SYMBOL_DEFINE,
    TENSOR,
    TERMINATOR,
    TILE,
    UNKNOWN_EFFECTS,
    AliasResult,
    AllShapesMatch,
    AttrDef,
    BlockArgCount,
    BlockArgsMatchElementTypes,
    BlockArgsMatchTypes,
    Borrow,
    BorrowedResult,
    CallLikeInterface,
    CallLikeKind,
    ConditionRefinement,
    ConditionRefinementTruth,
    Consume,
    Dialect,
    DimIndexInBounds,
    Discard,
    EnumCase,
    EnumDef,
    Escape,
    FreshResult,
    FuncLikeInterface,
    ImplicitTerminator,
    IterArgsMatchResults,
    LiteralMatchesElementType,
    LoopLikeInterface,
    OffsetCountMatchesRank,
    Op,
    Operand,
    OperandRole,
    ParameterizedAttrDef,
    Reads,
    ReadWrites,
    RegionBranchInterface,
    RegionDef,
    Release,
    Result,
    Retain,
    RetainedResult,
    SameElementType,
    SameType,
    Successor,
    SymbolDefinition,
    SymbolReference,
    TargetLikeInterface,
    TiedResult,
    TypeDef,
    Writes,
    YieldCountMatchesResults,
    YieldElementTypesMatchResults,
    YieldTypesMatchResults,
    binary_op,
    cast_op,
    comparison_op,
    unary_op,
)

# ============================================================================
# Group and enums
# ============================================================================

test_ops = Dialect(
    "test",
    dialect_id=0x01,
    doc="Test ops for infrastructure validation.",
    register_by_default=False,
)

# Enums for test.func (same values as the func dialect, defined
# locally to avoid cross-dialect import).
_Visibility = EnumDef(
    "Visibility",
    [EnumCase("public", 1, doc="Visible outside the module.")],
    doc="Function visibility. Absent (0) means private.",
)
_CallingConv = EnumDef(
    "CallingConv",
    [
        EnumCase("host", 1, doc="Host calling convention."),
        EnumCase("device", 2, doc="Device calling convention."),
        EnumCase("initializer", 3, doc="Module initialization."),
        EnumCase("deinitializer", 4, doc="Module deinitialization."),
    ],
    doc="Function calling convention. Absent (0) means host.",
)

_RecordKind = EnumDef(
    "RecordKind",
    [
        EnumCase("target", 1, doc="Target-like record."),
        EnumCase("artifact", 2, doc="Artifact-like record."),
    ],
    doc="Synthetic record kind. Open for bytecode future-ordinal tests.",
)

_TargetKind = EnumDef(
    "TargetKind",
    [
        EnumCase("low_core", 1, doc="Generic target-low test core."),
        EnumCase("quirky", 2, doc="Synthetic edge-case target."),
    ],
    doc="Synthetic target kind for target-like interface tests.",
)

_TemplateFlags = EnumDef(
    "TemplateFlags",
    [
        EnumCase("debug", 1, doc="Synthetic debug flag."),
        EnumCase("trace", 2, doc="Synthetic trace flag."),
    ],
    doc="Synthetic flags for TemplateParamFlags parser/printer coverage.",
)

_ArrayElement = EnumDef(
    "ArrayElement",
    [
        EnumCase("low", 1, doc="Sparse low value."),
        EnumCase("middle", 7, doc="Sparse middle value."),
        EnumCase("high", 255, doc="Maximum stable byte value."),
    ],
    doc="Synthetic sparse enum for enum-array lifecycle coverage.",
)

_ParameterizedMode = EnumDef(
    "ParameterizedMode",
    [
        EnumCase("fast", 1, doc="Synthetic fast mode."),
        EnumCase("precise", 2, doc="Synthetic precise mode."),
    ],
    doc="Synthetic mode for parameterized attribute coverage.",
)

_ParameterizedScope = EnumDef(
    "ParameterizedScope",
    [
        EnumCase("workgroup", 1, doc="Synthetic workgroup scope."),
        EnumCase("subgroup", 2, doc="Synthetic subgroup scope."),
    ],
    doc="Synthetic scope for parameterized value coverage.",
)

test_tile_attr = ParameterizedAttrDef(
    "test.tile",
    group=test_ops,
    parameters=[
        AttrDef("width", "i64", doc="Tile width in elements."),
    ],
    doc="Minimal parameterized attribute family.",
)

test_options_attr = ParameterizedAttrDef(
    "test.options",
    group=test_ops,
    parameters=[
        AttrDef(
            "mode",
            ATTR_TYPE_ENUM,
            enum_def=_ParameterizedMode,
            doc="Required execution mode.",
        ),
        AttrDef(
            "scopes",
            ATTR_TYPE_ENUM_ARRAY,
            enum_def=_ParameterizedScope,
            optional=True,
            open_enum=True,
            doc="Optional ordered scopes.",
        ),
        AttrDef("element_type", "type", optional=True),
        AttrDef(
            "tile",
            ATTR_TYPE_PARAMETERIZED,
            optional=True,
            parameterized_attr=test_tile_attr,
        ),
        AttrDef(
            "target",
            ATTR_TYPE_SYMBOL,
            optional=True,
            symbol_ref=SymbolReference("record", ["record"]),
            doc="Optional record symbol dependency.",
        ),
        AttrDef(
            "tiles",
            ATTR_TYPE_PARAMETERIZED_ARRAY,
            optional=True,
            parameterized_attr=test_tile_attr,
            doc="Optional ordered tile alternatives.",
        ),
    ],
    doc="Structured parameterized attribute lifecycle witness.",
)

test_compact_attr = ParameterizedAttrDef(
    "test.compact",
    group=test_ops,
    parameters=[
        AttrDef("label", "string", optional=True, doc="Optional display label."),
        AttrDef("value", "i64", doc="Primary compact value."),
    ],
    primary_parameter="value",
    doc="Compact primary parameter lifecycle witness.",
)

test_node_attr = ParameterizedAttrDef(
    "test.node",
    group=test_ops,
    parameters=[
        AttrDef("value", "i64", doc="Synthetic node value."),
        AttrDef(
            "children",
            ATTR_TYPE_PARAMETERIZED_ARRAY,
            optional=True,
            doc="Ordered child nodes or other registered attributes.",
        ),
    ],
    primary_parameter="value",
    doc="Recursive open-family parameterized-array witness.",
)

ALL_TEST_PARAMETERIZED_ATTRS = (
    test_tile_attr,
    test_options_attr,
    test_compact_attr,
    test_node_attr,
)

test_scope_type = TypeDef(
    "test.scope",
    params=[
        AttrDef("scope", ATTR_TYPE_ENUM, enum_def=_ParameterizedScope),
    ],
    format=[Param("scope")],
    doc="Positional descriptor-backed type parameter witness.",
)

test_matrix_type = TypeDef(
    "test.matrix",
    params=[
        AttrDef("element_type", "type"),
        AttrDef("scope", ATTR_TYPE_ENUM, enum_def=_ParameterizedScope),
        AttrDef("rows", "i64"),
        AttrDef(
            "target",
            ATTR_TYPE_SYMBOL,
            optional=True,
            symbol_ref=SymbolReference("record", ["record"]),
        ),
    ],
    format=[
        Param("element_type"),
        COMMA,
        kw("scope"),
        EQUALS,
        Param("scope"),
        COMMA,
        kw("rows"),
        EQUALS,
        Param("rows"),
        OptionalGroup(
            [COMMA, kw("target"), EQUALS, Param("target")],
            anchor="target",
        ),
    ],
    doc="Mixed positional and keyed descriptor-backed type witness.",
)

test_compact_matrix_type = TypeDef(
    "test.compact_matrix",
    params=[
        AttrDef("rows", "i64"),
        AttrDef("columns", "i64"),
        AttrDef("element_type", "type"),
        AttrDef("scope", ATTR_TYPE_ENUM, enum_def=_ParameterizedScope),
    ],
    format=[
        Param("rows"),
        GLUE,
        kw("x"),
        GLUE,
        Param("columns"),
        GLUE,
        kw("x"),
        GLUE,
        Param("element_type"),
        COMMA,
        Param("scope"),
    ],
    doc="Compact shape syntax witness for descriptor-backed type parsing.",
)

test_array_type = TypeDef(
    "test.array",
    params=[
        AttrDef("element_type", "type"),
        AttrDef("alignment", "i64", optional=True),
        AttrDef("metadata", "dict", optional=True),
    ],
    format=[
        Param("element_type"),
        OptionalGroup(
            [COMMA, kw("alignment"), EQUALS, Param("alignment")],
            anchor="alignment",
        ),
        OptionalGroup(
            [COMMA, kw("metadata"), EQUALS, Param("metadata")],
            anchor="metadata",
        ),
    ],
    doc="Optional and nested descriptor-backed type parameter witness.",
)

test_variant_set_type = TypeDef(
    "test.variant_set",
    params=[
        AttrDef(
            "values",
            ATTR_TYPE_PARAMETERIZED_ARRAY,
            doc="Ordered parameterized values from any registered family.",
        ),
        AttrDef(
            "alternatives",
            ATTR_TYPE_PARAMETERIZED_ARRAY,
            optional=True,
            doc="Optional ordered values from any registered family.",
        ),
    ],
    format=[
        Param("values"),
        OptionalGroup(
            [COMMA, kw("alternatives"), EQUALS, Param("alternatives")],
            anchor="alternatives",
        ),
    ],
    doc="Parameterized-array type-parameter lifecycle witness.",
)

ALL_TEST_TYPES = (
    test_scope_type,
    test_matrix_type,
    test_compact_matrix_type,
    test_array_type,
    test_variant_set_type,
)

cmp_predicates = EnumDef(
    "TestCmpPredicate",
    [
        EnumCase("eq", 0, doc="Equal."),
        EnumCase("ne", 1, doc="Not equal."),
        EnumCase("lt", 2, doc="Less than."),
        EnumCase("le", 3, doc="Less or equal."),
        EnumCase("gt", 4, doc="Greater than."),
        EnumCase("ge", 5, doc="Greater or equal."),
    ],
)

# ============================================================================
# test.addi — binary integer op
# ============================================================================

test_addi = binary_op(
    "test.addi",
    group=test_ops,
    type_constraint=INTEGER,
    doc="Test binary integer op.",
    commutative=True,
    facts="loom_test_addi_facts",
    canonicalize="loom_test_addi_canonicalize",
    examples=["%result = test.addi %lhs, %rhs : i32"],
)

# ============================================================================
# test.neg — unary float op
# ============================================================================

test_neg = unary_op(
    "test.neg",
    group=test_ops,
    type_constraint=FLOAT,
    doc="Test unary float op.",
    traits=[PURE, INVOLUTION],
    examples=["%result = test.neg %input : f32"],
)

# ============================================================================
# test.cast — type-casting op
# ============================================================================

test_cast = cast_op(
    "test.cast",
    group=test_ops,
    from_constraint=INTEGER,
    to_constraint=FLOAT,
    doc="Test cast op.",
    examples=["%result = test.cast %input : i32 to f32"],
)

# ============================================================================
# test.constant — constant materialization
# ============================================================================

test_constant = Op(
    "test.constant",
    group=test_ops,
    doc="Test constant materialization.",
    results=[Result("result", ANY)],
    attrs=[AttrDef("value", "any", doc="The constant value.")],
    traits=[PURE, CONSTANT_LIKE],
    facts="loom_test_constant_facts",
    verify="loom_test_constant_verify",
    format=[Attr("value"), COLON, TypeOf("result")],
    examples=[
        "%c42 = test.constant 42 : i32",
        "%pi = test.constant 3.14 : f32",
    ],
)

# ============================================================================
# test.effectful_constant — exact facts behind an observable boundary
# ============================================================================

test_effectful_constant = Op(
    "test.effectful_constant",
    group=test_ops,
    doc=("Unknown-effect test op with exact result facts. This guards the canonicalizer contract that fact folding may use exact facts from an effectful op without erasing the op itself."),
    results=[Result("result", INTEGER)],
    attrs=[AttrDef("value", "any", doc="The exact integer value fact.")],
    traits=[UNKNOWN_EFFECTS],
    facts="loom_test_effectful_constant_facts",
    verify="loom_test_constant_verify",
    format=[Attr("value"), COLON, TypeOf("result")],
    examples=[
        "%value = test.effectful_constant 42 : i64",
    ],
)

# ============================================================================
# test.clause_* — named clause format helpers
# ============================================================================

test_clause_constant = Op(
    "test.clause_constant",
    group=test_ops,
    doc="Test constant materialization using a named value clause.",
    results=[Result("result", INTEGER)],
    attrs=[AttrDef("value", "any", doc="The constant payload.")],
    constraints=[LiteralMatchesElementType("value", "result")],
    traits=[PURE, CONSTANT_LIKE],
    format=[Clause("value", Attr("value")), COLON, TypeOf("result")],
    examples=["%c42 = test.clause_constant value(42) : i32"],
)

test_clause_copy = Op(
    "test.clause_copy",
    group=test_ops,
    doc="Test dynamic operand clauses that model source/target-style syntax.",
    operands=[
        Operand("source", ANY),
        Operand("target", ANY),
    ],
    constraints=[SameType("source", "target")],
    traits=[UNKNOWN_EFFECTS],
    format=[
        Clause("source", Ref("source")),
        Clause("target", Ref("target")),
        COLON,
        TypeOf("source"),
    ],
    examples=["test.clause_copy source(%src) target(%dst) : i32"],
)

# ============================================================================
# test.use — side-effecting value sink for testing
# ============================================================================

test_use = Op(
    "test.use",
    group=test_ops,
    doc="Side-effecting sink that observes values without producing results. Not DCE-able. Use in tests to keep values alive for inspection.",
    operands=[Operand("values", ANY, variadic=True)],
    traits=[UNKNOWN_EFFECTS],
    format=[Refs("values"), COLON, TypesOf("values")],
    examples=[
        "test.use %a : i32",
        "test.use %a, %b : i32, f32",
    ],
)

test_convergent = Op(
    "test.convergent",
    group=test_ops,
    doc="Pure value transform whose dynamic participant set is semantically observable.",
    operands=[Operand("input", ANY)],
    results=[Result("result", ANY)],
    constraints=[SameType("input", "result")],
    traits=[PURE, CONVERGENT],
    format=[Ref("input"), COLON, TypeOf("input")],
    examples=["%result = test.convergent %input : i32"],
)

test_typed_use = Op(
    "test.typed_use",
    group=test_ops,
    doc="Side-effecting sink with adjacent SSA type annotations in its format.",
    operands=[Operand("values", ANY, variadic=True)],
    traits=[UNKNOWN_EFFECTS],
    format=[TypedRefs("values")],
    examples=[
        "test.typed_use %a: i32",
        "test.typed_use %a: i32, %b: f32",
    ],
)

test_segmented = Op(
    "test.segmented",
    group=test_ops,
    doc="Pure test op with independent operand spans sharing one flat operand array.",
    operands=[
        Operand("root", ANY),
        Operand("guard", ANY, optional=True),
        Operand("lhs", ANY, variadic=True),
        Operand("rhs", ANY, variadic=True),
    ],
    results=[Result("result", ANY)],
    constraints=[
        SameType("root", "result"),
        SameType("root", "lhs", "rhs"),
    ],
    traits=[PURE],
    format=[
        Ref("root"),
        OptionalGroup([kw("base"), Ref("guard")], anchor="guard"),
        kw("values"),
        Refs("lhs"),
        kw("expected"),
        Refs("rhs"),
        COLON,
        TypeOf("root"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%result = test.segmented %root base %guard values %lhs0, %lhs1 expected %rhs : i32 -> i32",
        "%result = test.segmented %root values expected %rhs0, %rhs1 : i32 -> i32",
    ],
)

# ============================================================================
# test.fact_* — value facts inspection ops for testing
# ============================================================================
#
# Each op reads one property from the analysis facts of its input and
# exposes it as a result. During canonicalization with facts enabled,
# the fact inference function returns exact facts, which the rewriter materializes
# as scalar.constant ops. This makes internal analysis state observable
# in .loom-test files.

test_fact_range_lo = Op(
    "test.fact_range_lo",
    group=test_ops,
    doc="Exposes the analysis range lower bound as an i64 constant.",
    operands=[Operand("value", ANY)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_range_lo_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%lo = test.fact_range_lo %x : index -> i64"],
)

test_fact_range_hi = Op(
    "test.fact_range_hi",
    group=test_ops,
    doc="Exposes the analysis range upper bound as an i64 constant.",
    operands=[Operand("value", ANY)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_range_hi_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%hi = test.fact_range_hi %x : index -> i64"],
)

test_fact_all_equal_range_lo = Op(
    "test.fact_all_equal_range_lo",
    group=test_ops,
    doc="Exposes the all-equal element range lower bound as an i64 constant.",
    operands=[Operand("value", ANY)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_all_equal_range_lo_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%lo = test.fact_all_equal_range_lo %x : reg<test.i32 x4> -> i64"],
)

test_fact_all_equal_range_hi = Op(
    "test.fact_all_equal_range_hi",
    group=test_ops,
    doc="Exposes the all-equal element range upper bound as an i64 constant.",
    operands=[Operand("value", ANY)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_all_equal_range_hi_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%hi = test.fact_all_equal_range_hi %x : reg<test.i32 x4> -> i64"],
)

test_fact_divisor = Op(
    "test.fact_divisor",
    group=test_ops,
    doc="Exposes the analysis known divisor as an i64 constant.",
    operands=[Operand("value", ANY)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_divisor_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%div = test.fact_divisor %x : index -> i64"],
)

test_fact_non_negative = Op(
    "test.fact_non_negative",
    group=test_ops,
    doc="Returns 1 if the input is provably non-negative, 0 otherwise.",
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_non_negative_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%nn = test.fact_non_negative %x : index -> i1"],
)

test_fact_non_zero = Op(
    "test.fact_non_zero",
    group=test_ops,
    doc="Returns 1 if the input is provably non-zero, 0 otherwise.",
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_non_zero_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%nz = test.fact_non_zero %x : index -> i1"],
)

test_fact_positive = Op(
    "test.fact_positive",
    group=test_ops,
    doc="Returns 1 if the input is provably positive (> 0), 0 otherwise.",
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_positive_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%pos = test.fact_positive %x : index -> i1"],
)

test_fact_power_of_two = Op(
    "test.fact_power_of_two",
    group=test_ops,
    doc="Returns 1 if the input is provably a power of two, 0 otherwise.",
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_power_of_two_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%p2 = test.fact_power_of_two %x : index -> i1"],
)

test_fact_finite = Op(
    "test.fact_finite",
    group=test_ops,
    doc="Returns 1 if the input is provably finite, 0 otherwise.",
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_finite_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%finite = test.fact_finite %x : f32 -> i1"],
)

test_fact_not_subnormal = Op(
    "test.fact_not_subnormal",
    group=test_ops,
    doc="Returns 1 if the input is provably not subnormal, 0 otherwise.",
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_not_subnormal_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%not_subnormal = test.fact_not_subnormal %x : f32 -> i1"],
)

test_fact_subgroup_uniform = Op(
    "test.fact_subgroup_uniform",
    group=test_ops,
    doc="Returns 1 if the input is provably uniform across a subgroup, 0 otherwise.",
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_subgroup_uniform_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%uniform = test.fact_subgroup_uniform %x : index -> i1"],
)

test_fact_workgroup_uniform = Op(
    "test.fact_workgroup_uniform",
    group=test_ops,
    doc="Returns 1 if the input is provably uniform across a workgroup, 0 otherwise.",
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_workgroup_uniform_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%uniform = test.fact_workgroup_uniform %x : index -> i1"],
)

test_fact_cluster_uniform = Op(
    "test.fact_cluster_uniform",
    group=test_ops,
    doc=("Returns 1 if the input is provably uniform across every workgroup in a workgroup cluster, 0 otherwise."),
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_cluster_uniform_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%uniform = test.fact_cluster_uniform %x : index -> i1"],
)

test_fact_lane_varying = Op(
    "test.fact_lane_varying",
    group=test_ops,
    doc="Returns 1 if the input may vary across active lanes, 0 otherwise.",
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_lane_varying_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%varying = test.fact_lane_varying %x : index -> i1"],
)

test_fact_lane_predicate = Op(
    "test.fact_lane_predicate",
    group=test_ops,
    doc="Returns 1 if the input is an i1 lane predicate, 0 otherwise.",
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_lane_predicate_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%predicate = test.fact_lane_predicate %x : i1 -> i1"],
)

test_fact_subgroup_lane_mask = Op(
    "test.fact_subgroup_lane_mask",
    group=test_ops,
    doc="Returns 1 if the input is an integer subgroup lane mask, 0 otherwise.",
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_subgroup_lane_mask_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%mask = test.fact_subgroup_lane_mask %x : i64 -> i1"],
)

test_fact_is_vector_iota = Op(
    "test.fact_is_vector_iota",
    group=test_ops,
    doc="Returns 1 if the input has a vector.iota analysis summary, 0 otherwise.",
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_is_vector_iota_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%is = test.fact_is_vector_iota %x : vector<[%n]xindex> -> i1"],
)

test_fact_is_vector_prefix_mask = Op(
    "test.fact_is_vector_prefix_mask",
    group=test_ops,
    doc="Returns 1 if the input has a vector.mask.range analysis summary, 0 otherwise.",
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_is_vector_prefix_mask_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%is = test.fact_is_vector_prefix_mask %x : vector<[%n]xi1> -> i1"],
)

test_fact_encoding_layout_kind = Op(
    "test.fact_encoding_layout_kind",
    group=test_ops,
    doc="Exposes an encoding-summary address-layout kind as an i64 constant.",
    operands=[Operand("value", ANY_ENCODING)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_encoding_layout_kind_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%kind = test.fact_encoding_layout_kind %layout : encoding<layout> -> i64"],
)

test_fact_encoding_layout_stride_hi = Op(
    "test.fact_encoding_layout_stride_hi",
    group=test_ops,
    doc="Exposes an encoding-summary strided-layout stride upper bound as an i64 constant.",
    operands=[Operand("value", ANY_ENCODING)],
    attrs=[AttrDef("axis", "i64", doc="Stride axis to inspect.")],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_encoding_layout_stride_hi_facts",
    format=[
        Ref("value"),
        LBRACKET,
        Attr("axis"),
        RBRACKET,
        COLON,
        TypeOf("value"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%hi = test.fact_encoding_layout_stride_hi %layout[0] : encoding<layout> -> i64",
    ],
)

test_fact_encoding_operand_field = Op(
    "test.fact_encoding_operand_field",
    group=test_ops,
    doc=(
        "Exposes an encoded-operand storage-schema summary field as an i64 "
        "constant. Supported fields are element_format, payload_packing, "
        "scale_topology, scale_format, secondary_scale_format, affine, "
        "rounding, codebook, sparsity, sparsity_group_elements, "
        "sparsity_group_nonzero_elements, payload_registers, "
        "payload_elements, scale_group_elements, scale_group_rank, "
        "scale_group_dim0 through scale_group_dim3, scale_operands, "
        "zero_scale_fallback, and static_spec."
    ),
    operands=[Operand("value", ANY_ENCODING)],
    attrs=[AttrDef("field", "string", doc="Encoded operand field to inspect.")],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_encoding_operand_field_facts",
    format=[
        Ref("value"),
        LBRACKET,
        Attr("field"),
        RBRACKET,
        COLON,
        TypeOf("value"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        '%format = test.fact_encoding_operand_field %schema["element_format"] : encoding<schema> -> i64',
    ],
)

test_fact_is_buffer_reference = Op(
    "test.fact_is_buffer_reference",
    group=test_ops,
    doc="Returns 1 if the input has a buffer-reference analysis summary, 0 otherwise.",
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_is_buffer_reference_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%is = test.fact_is_buffer_reference %buffer : buffer -> i1"],
)

test_fact_is_view_reference = Op(
    "test.fact_is_view_reference",
    group=test_ops,
    doc="Returns 1 if the input has a view-reference analysis summary, 0 otherwise.",
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_is_view_reference_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%is = test.fact_is_view_reference %view : view<4xf32, %layout> -> i1"],
)

test_fact_buffer_memory_space = Op(
    "test.fact_buffer_memory_space",
    group=test_ops,
    doc="Exposes a buffer-reference memory-space enum value as an i64 constant, or -1 when unknown.",
    operands=[Operand("value", ANY)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_buffer_memory_space_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%space = test.fact_buffer_memory_space %buffer : buffer -> i64"],
)

test_fact_view_memory_space = Op(
    "test.fact_view_memory_space",
    group=test_ops,
    doc="Exposes a view-reference memory-space enum value as an i64 constant, or -1 when unknown.",
    operands=[Operand("value", ANY)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_view_memory_space_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%space = test.fact_view_memory_space %view : view<4xf32, %layout> -> i64"],
)

test_fact_view_root_matches = Op(
    "test.fact_view_root_matches",
    group=test_ops,
    doc="Returns 1 if a view reference and another reference share the same root identity.",
    operands=[Operand("view", ANY), Operand("root", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_view_root_matches_facts",
    format=[
        Ref("view"),
        COMMA,
        Ref("root"),
        COLON,
        TypeOf("view"),
        COMMA,
        TypeOf("root"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%same = test.fact_view_root_matches %view, %buffer : view<4xf32, %layout>, buffer -> i1",
    ],
)

test_fact_alias_scope_known = Op(
    "test.fact_alias_scope_known",
    group=test_ops,
    doc="Returns 1 if the input has a comparable storage alias scope, 0 otherwise.",
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_alias_scope_known_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%known = test.fact_alias_scope_known %buffer : buffer -> i1"],
)

test_fact_alias_scope_matches = Op(
    "test.fact_alias_scope_matches",
    group=test_ops,
    doc="Returns 1 if both inputs have the same comparable storage alias scope, 0 otherwise.",
    operands=[
        Operand("lhs", ANY, doc="First buffer or view value."),
        Operand("rhs", ANY, doc="Second buffer or view value."),
    ],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_alias_scope_matches_facts",
    format=[
        Ref("lhs"),
        COMMA,
        Ref("rhs"),
        COLON,
        TypeOf("lhs"),
        COMMA,
        TypeOf("rhs"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%same = test.fact_alias_scope_matches %lhs, %rhs : buffer, buffer -> i1",
    ],
)

test_fact_view_byte_offset_lo = Op(
    "test.fact_view_byte_offset_lo",
    group=test_ops,
    doc="Exposes a view-reference byte-offset lower bound as an i64 constant.",
    operands=[Operand("value", ANY)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_view_byte_offset_lo_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%lo = test.fact_view_byte_offset_lo %view : view<4xf32, %layout> -> i64"],
)

test_fact_view_byte_offset_hi = Op(
    "test.fact_view_byte_offset_hi",
    group=test_ops,
    doc="Exposes a view-reference byte-offset upper bound as an i64 constant.",
    operands=[Operand("value", ANY)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_view_byte_offset_hi_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%hi = test.fact_view_byte_offset_hi %view : view<4xf32, %layout> -> i64"],
)

test_fact_view_byte_length_lo = Op(
    "test.fact_view_byte_length_lo",
    group=test_ops,
    doc="Exposes a view-reference footprint byte-length lower bound as an i64 constant.",
    operands=[Operand("value", ANY)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_view_byte_length_lo_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%lo = test.fact_view_byte_length_lo %view : view<4xf32, %layout> -> i64"],
)

test_fact_view_byte_length_hi = Op(
    "test.fact_view_byte_length_hi",
    group=test_ops,
    doc="Exposes a view-reference footprint byte-length upper bound as an i64 constant.",
    operands=[Operand("value", ANY)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_view_byte_length_hi_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%hi = test.fact_view_byte_length_hi %view : view<4xf32, %layout> -> i64"],
)

test_fact_view_min_alignment = Op(
    "test.fact_view_min_alignment",
    group=test_ops,
    doc="Exposes the minimum provable view byte-offset alignment as an i64 constant.",
    operands=[Operand("value", ANY)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_view_min_alignment_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%align = test.fact_view_min_alignment %view : view<4xf32, %layout> -> i64"],
)

test_fact_buffer_min_alignment = Op(
    "test.fact_buffer_min_alignment",
    group=test_ops,
    doc="Exposes the minimum provable buffer root byte alignment as an i64 constant.",
    operands=[Operand("value", ANY)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_buffer_min_alignment_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%align = test.fact_buffer_min_alignment %buffer : buffer -> i64"],
)

test_fact_view_root_min_alignment = Op(
    "test.fact_view_root_min_alignment",
    group=test_ops,
    doc="Exposes the minimum provable view root byte alignment as an i64 constant.",
    operands=[Operand("value", ANY)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_view_root_min_alignment_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%align = test.fact_view_root_min_alignment %view : view<4xf32, %layout> -> i64"],
)

test_fact_view_element_bytes = Op(
    "test.fact_view_element_bytes",
    group=test_ops,
    doc="Exposes the static addressed element byte count, or -1 when unknown.",
    operands=[Operand("value", ANY)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_view_element_bytes_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%bytes = test.fact_view_element_bytes %view : view<4xf32, %layout> -> i64"],
)

test_fact_is_storage_reference = Op(
    "test.fact_is_storage_reference",
    group=test_ops,
    doc="Returns 1 if the input has a storage reference analysis summary, 0 otherwise.",
    operands=[Operand("value", STORAGE)],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_is_storage_reference_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%is = test.fact_is_storage_reference %storage : low.storage<workgroup> -> i1"],
)

test_fact_storage_same_backing = Op(
    "test.fact_storage_same_backing",
    group=test_ops,
    doc="Returns 1 if two storage references share the same backing reservation.",
    operands=[
        Operand("lhs", STORAGE),
        Operand("rhs", STORAGE),
    ],
    results=[Result("result", I1)],
    traits=[PURE],
    facts="loom_test_fact_storage_same_backing_facts",
    format=[
        Ref("lhs"),
        COMMA,
        Ref("rhs"),
        COLON,
        TypeOf("lhs"),
        COMMA,
        TypeOf("rhs"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%same = test.fact_storage_same_backing %lhs, %rhs : low.storage<workgroup>, low.storage<workgroup> -> i1",
    ],
)

test_fact_storage_byte_offset_lo = Op(
    "test.fact_storage_byte_offset_lo",
    group=test_ops,
    doc="Exposes a storage reference byte-offset lower bound as an i64 constant.",
    operands=[Operand("value", STORAGE)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_storage_byte_offset_lo_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%lo = test.fact_storage_byte_offset_lo %storage : low.storage<workgroup> -> i64"],
)

test_fact_storage_byte_offset_hi = Op(
    "test.fact_storage_byte_offset_hi",
    group=test_ops,
    doc="Exposes a storage reference byte-offset upper bound as an i64 constant.",
    operands=[Operand("value", STORAGE)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_storage_byte_offset_hi_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%hi = test.fact_storage_byte_offset_hi %storage : low.storage<workgroup> -> i64"],
)

test_fact_storage_byte_offset_divisor = Op(
    "test.fact_storage_byte_offset_divisor",
    group=test_ops,
    doc="Exposes a storage reference byte-offset divisor as an i64 constant.",
    operands=[Operand("value", STORAGE)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_storage_byte_offset_divisor_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%div = test.fact_storage_byte_offset_divisor %storage : low.storage<workgroup> -> i64"],
)

test_fact_storage_byte_length_lo = Op(
    "test.fact_storage_byte_length_lo",
    group=test_ops,
    doc="Exposes a storage reference byte-length lower bound as an i64 constant.",
    operands=[Operand("value", STORAGE)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_storage_byte_length_lo_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%lo = test.fact_storage_byte_length_lo %storage : low.storage<workgroup> -> i64"],
)

test_fact_storage_byte_length_hi = Op(
    "test.fact_storage_byte_length_hi",
    group=test_ops,
    doc="Exposes a storage reference byte-length upper bound as an i64 constant.",
    operands=[Operand("value", STORAGE)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_storage_byte_length_hi_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%hi = test.fact_storage_byte_length_hi %storage : low.storage<workgroup> -> i64"],
)

test_fact_storage_min_alignment = Op(
    "test.fact_storage_min_alignment",
    group=test_ops,
    doc="Exposes a storage reference minimum byte alignment as an i64 constant.",
    operands=[Operand("value", STORAGE)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_storage_min_alignment_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%align = test.fact_storage_min_alignment %storage : low.storage<workgroup> -> i64"],
)

test_fact_storage_space = Op(
    "test.fact_storage_space",
    group=test_ops,
    doc="Exposes a storage reference storage-space enum value as an i64 constant, or -1 when unknown.",
    operands=[Operand("value", STORAGE)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    facts="loom_test_fact_storage_space_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%space = test.fact_storage_space %storage : low.storage<workgroup> -> i64"],
)

# ============================================================================
# test.cmp — comparison with enum predicate
# ============================================================================

test_cmp = comparison_op(
    "test.cmp",
    group=test_ops,
    type_constraint=INTEGER,
    predicates=cmp_predicates,
    doc="Test comparison op.",
    examples=["%result = test.cmp lt, %lhs, %rhs : i32"],
)

# ============================================================================
# test.map — region capture (elementwise-like)
# ============================================================================

test_map = Op(
    "test.map",
    group=test_ops,
    doc="Test region-capture elementwise op.",
    operands=[Operand("inputs", TILE, variadic=True)],
    results=[Result("result", TILE)],
    regions=[
        RegionDef(
            "body",
            doc="Element-wise body.",
            single_block=True,
            terminator="test.yield",
        )
    ],
    constraints=[
        AllShapesMatch("inputs"),
        BlockArgCount("body", "inputs"),
        BlockArgsMatchElementTypes("body", "inputs"),
        YieldCountMatchesResults("body", "result"),
        YieldElementTypesMatchResults("body", "result"),
    ],
    traits=[PURE, ELEMENTWISE, ImplicitTerminator("test.implicit_yield")],
    format=[
        BindingList("inputs", kind=BINDING_ELEMENT),
        Region("body"),
        ARROW,
        ResultTypeList("result"),
    ],
    examples=[
        "%result = test.map(%element = %input : tile<4xf32>) {\n  %negated = test.neg %element : f32\n  test.yield %negated : f32\n} -> (tile<4xf32>)",
    ],
)

# ============================================================================
# test.update — tied result with index list
# ============================================================================

test_update = Op(
    "test.update",
    group=test_ops,
    doc="Test tied result with index list.",
    operands=[
        Operand("source", TILE),
        Operand("target", TENSOR),
        Operand("offsets", INDEX, variadic=True),
    ],
    results=[TiedResult("result", "target", TENSOR)],
    attrs=[
        AttrDef(
            "static_offsets",
            "i64_array",
            doc="Static offset values (sentinel for dynamic).",
        ),
    ],
    constraints=[SameElementType("source", "target", "result")],
    traits=[PURE],
    format=[
        Ref("source"),
        COMMA,
        Ref("target"),
        IndexList("offsets", "static_offsets"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultTypeList("result"),
    ],
    examples=[
        "%result = test.update %tile, %tensor[%offset] : tile<4xf32> -> (%tensor as tensor<[%M]xf32>)",
    ],
)

# ============================================================================
# test.invoke — variadic call-like with tied results
# ============================================================================

test_invoke = Op(
    "test.invoke",
    group=test_ops,
    doc="Test variadic call-like op with tied results. The verifier checks that the invoke signature matches the referenced function declaration or definition.",
    operands=[
        Operand("operands", ANY, variadic=True),
    ],
    attrs=[
        AttrDef(
            "callee",
            "symbol",
            symbol_ref=SymbolReference("function", ["callable"]),
        ),
    ],
    results=[Result("results", ANY, variadic=True)],
    traits=[UNKNOWN_EFFECTS],
    interfaces=[
        CallLikeInterface(
            callee="callee",
            operands="operands",
            results="results",
            kind=CallLikeKind.SEMANTIC,
        ),
    ],
    verify="loom_test_call_like_verify",
    format=[
        SymbolRef("callee"),
        GLUE,
        LPAREN,
        Refs("operands"),
        RPAREN,
        COLON,
        LPAREN,
        TypesOf("operands"),
        RPAREN,
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
    ],
    examples=[
        "%output, %count = test.invoke @callee(%weights, %input) : (tile<4xf32>, index) -> (%weights as tile<4xf32>, index)",
    ],
)

# ============================================================================
# test.low_call / test.low_invoke — non-semantic call-like kind fixtures
# ============================================================================

test_low_call = Op(
    "test.low_call",
    group=test_ops,
    doc="Test call-like op classified like a target-low internal call.",
    operands=[
        Operand("operands", ANY, variadic=True),
    ],
    attrs=[
        AttrDef(
            "callee",
            "symbol",
            symbol_ref=SymbolReference("function", ["callable"]),
        ),
    ],
    results=[Result("results", ANY, variadic=True)],
    traits=[UNKNOWN_EFFECTS],
    interfaces=[
        CallLikeInterface(
            callee="callee",
            operands="operands",
            results="results",
            kind=CallLikeKind.LOW_INTERNAL,
        ),
    ],
    verify="loom_test_call_like_verify",
    format=[
        SymbolRef("callee"),
        GLUE,
        LPAREN,
        Refs("operands"),
        RPAREN,
        COLON,
        LPAREN,
        TypesOf("operands"),
        RPAREN,
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
    ],
    examples=[
        "test.low_call @callee() : ()",
    ],
)

test_low_invoke = Op(
    "test.low_invoke",
    group=test_ops,
    doc="Test call-like op classified like an explicit target-low invocation.",
    operands=[
        Operand("operands", ANY, variadic=True),
    ],
    attrs=[
        AttrDef(
            "callee",
            "symbol",
            symbol_ref=SymbolReference("function", ["callable"]),
        ),
    ],
    results=[Result("results", ANY, variadic=True)],
    traits=[UNKNOWN_EFFECTS],
    interfaces=[
        CallLikeInterface(
            callee="callee",
            operands="operands",
            results="results",
            kind=CallLikeKind.LOW_INVOKE,
        ),
    ],
    verify="loom_test_call_like_verify",
    format=[
        SymbolRef("callee"),
        GLUE,
        LPAREN,
        Refs("operands"),
        RPAREN,
        COLON,
        LPAREN,
        TypesOf("operands"),
        RPAREN,
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
    ],
    examples=[
        "test.low_invoke @callee() : ()",
    ],
)

# ============================================================================
# test.partitioned_call — call-like operand partitions
# ============================================================================

test_partitioned_call = Op(
    "test.partitioned_call",
    group=test_ops,
    doc="Test call-like op spanning trailing operand partitions after an unrelated prefix.",
    operands=[
        Operand("prefix", ANY, variadic=True),
        Operand("specializations", ANY, variadic=True),
        Operand("bindings", ANY, variadic=True),
    ],
    attrs=[
        AttrDef(
            "callee",
            "symbol",
            symbol_ref=SymbolReference("function", ["callable"]),
        ),
    ],
    traits=[UNKNOWN_EFFECTS],
    interfaces=[
        CallLikeInterface(
            callee="callee",
            operands="specializations",
            results=None,
            kind=CallLikeKind.COMMAND_PROGRAM,
        ),
    ],
    format=[
        SymbolRef("callee"),
        GLUE,
        LBRACKET,
        Refs("prefix"),
        RBRACKET,
        GLUE,
        LBRACKET,
        Refs("specializations"),
        RBRACKET,
        GLUE,
        LPAREN,
        Refs("bindings"),
        RPAREN,
        COLON,
        LBRACKET,
        TypesOf("prefix"),
        RBRACKET,
        GLUE,
        LBRACKET,
        TypesOf("specializations"),
        RBRACKET,
        GLUE,
        LPAREN,
        TypesOf("bindings"),
        RPAREN,
    ],
    examples=[
        "test.partitioned_call @callee[%prefix][%specialization](%binding) : [index][index](i32)",
    ],
)

# ============================================================================
# test.slice — index list with mixed static/dynamic offsets
# ============================================================================

test_slice = Op(
    "test.slice",
    group=test_ops,
    doc="Test index list with mixed static/dynamic offsets.",
    operands=[
        Operand("source", TILE),
        Operand("offsets", INDEX, variadic=True),
    ],
    results=[Result("result", TILE)],
    attrs=[
        AttrDef("static_offsets", "i64_array", doc="Static offset values."),
    ],
    constraints=[SameElementType("source", "result"), OffsetCountMatchesRank("source", "offsets")],
    traits=[PURE],
    format=[
        Ref("source"),
        IndexList("offsets", "static_offsets"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultTypeList("result"),
    ],
    examples=[
        "%subtile = test.slice %source[0, %offset] : tile<64x64xf16> -> (tile<16x16xf16>)",
    ],
)

# ============================================================================
# test.shape — unglued named index list
# ============================================================================

test_shape = Op(
    "test.shape",
    group=test_ops,
    doc="Test named index-list clause that keeps a space before '['.",
    operands=[
        Operand("value", TILE),
        Operand("dims", INDEX, variadic=True),
    ],
    attrs=[
        AttrDef("static_dims", "i64_array", doc="Static shape values."),
    ],
    constraints=[OffsetCountMatchesRank("value", "dims")],
    traits=[PURE],
    format=[
        Ref("value"),
        kw("shape"),
        IndexList("dims", "static_dims", glue=False),
        COLON,
        TypeOf("value"),
    ],
    examples=[
        "test.shape %value shape [%m, 4] : tile<[%m]x4xf32>",
    ],
)

# ============================================================================
# test.loop — for-loop with iter_args and tied results
# ============================================================================

test_loop = Op(
    "test.loop",
    group=test_ops,
    doc="Test for-loop with iter_args and tied results.",
    operands=[
        Operand("lower_bound", INDEX),
        Operand("upper_bound", INDEX),
        Operand("step", INDEX),
        Operand("iter_args", ANY, variadic=True),
    ],
    results=[Result("results", ANY, variadic=True)],
    regions=[
        RegionDef(
            "body",
            doc="Loop body.",
            single_block=True,
            terminator="test.yield",
            implicit_args=(("iv", "index"),),
            arg_source="iter_args",
        )
    ],
    interfaces=[
        LoopLikeInterface(
            body="body",
            iter_args="iter_args",
            iv="iv",
            lower_bound="lower_bound",
            upper_bound="upper_bound",
            step="step",
        ),
    ],
    constraints=[
        IterArgsMatchResults("iter_args", "results"),
        YieldCountMatchesResults("body", "results"),
        YieldTypesMatchResults("body", "results"),
    ],
    traits=[ImplicitTerminator("test.implicit_yield")],
    format=[
        Ref("iv"),
        EQUALS,
        Ref("lower_bound"),
        kw("to"),
        Ref("upper_bound"),
        kw("step"),
        Ref("step"),
        OptionalGroup(
            [kw("iter_args"), BindingList("iter_args")],
            anchor="iter_args",
        ),
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
        Region("body"),
    ],
    examples=[
        "%result = test.loop %i = %c0 to %count step %c1 iter_args(%accumulator = %init : f32) -> (%init as f32) {\n  %next = test.neg %accumulator : f32\n  test.yield %next : f32\n}",
    ],
)

# ============================================================================
# test.block_args — explicit region entry args
# ============================================================================

test_block_args = Op(
    "test.block_args",
    group=test_ops,
    doc="Test op with explicit BlockArgs syntax for a region entry block.",
    operands=[Operand("inputs", ANY, variadic=True)],
    regions=[
        RegionDef(
            "body",
            doc="Body with explicitly named entry block args.",
            single_block=True,
            terminator="test.yield",
            arg_source="inputs",
        )
    ],
    constraints=[
        BlockArgCount("body", "inputs"),
        BlockArgsMatchTypes("body", "inputs"),
    ],
    traits=[ImplicitTerminator("test.implicit_yield")],
    format=[
        Refs("inputs"),
        COLON,
        TypesOf("inputs"),
        kw("do"),
        BlockArgs("body"),
        Region("body"),
    ],
    examples=[
        "test.block_args %value : i32 do(%arg: i32) {\n  test.yield\n}",
    ],
)

# ============================================================================
# test.branch — if/else with both regions present
# ============================================================================

test_branch = Op(
    "test.branch",
    group=test_ops,
    doc="Test if/else with both regions always present.",
    operands=[
        Operand("condition", INTEGER, role=OperandRole.CONTROL_CONDITION),
    ],
    results=[Result("results", ANY, variadic=True)],
    regions=[
        RegionDef(
            "then_region",
            doc="Then branch.",
            single_block=True,
            terminator="test.yield",
        ),
        RegionDef(
            "else_region",
            doc="Else branch.",
            single_block=True,
            terminator="test.yield",
        ),
    ],
    interfaces=[RegionBranchInterface(selector="condition")],
    traits=[ImplicitTerminator("test.implicit_yield")],
    format=[
        Ref("condition"),
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
        Region("then_region"),
        kw("else"),
        Region("else_region"),
    ],
    examples=[
        "%result = test.branch %condition -> (f32) {\n  test.yield %true_value : f32\n} else {\n  test.yield %false_value : f32\n}",
    ],
)

# ============================================================================
# test.optional_region — trailing optional region
# ============================================================================

test_optional_region = Op(
    "test.optional_region",
    group=test_ops,
    doc="Test op with a required body and a trailing optional region.",
    operands=[Operand("condition", INTEGER)],
    regions=[
        RegionDef(
            "body",
            doc="Required body.",
            single_block=True,
            terminator="test.yield",
        ),
        RegionDef(
            "else_region",
            doc="Optional else body.",
            single_block=True,
            optional=True,
            terminator="test.yield",
        ),
    ],
    traits=[ImplicitTerminator("test.implicit_yield")],
    format=[
        Ref("condition"),
        Region("body"),
        OptionalGroup(
            [kw("else"), Region("else_region")],
            anchor="else_region",
        ),
    ],
    examples=[
        "test.optional_region %condition {\n  test.yield\n}",
        "test.optional_region %condition {\n  test.yield\n} else {\n  test.yield\n}",
    ],
)

# ============================================================================
# test.implicit_yield — dedicated implicit region terminator
# ============================================================================

test_implicit_yield = Op(
    "test.implicit_yield",
    group=test_ops,
    doc="Dedicated zero-field implicit terminator synthesized for elidable test regions.",
    traits=[TERMINATOR],
    examples=["test.implicit_yield"],
)

# ============================================================================
# test.yield — variadic yield terminator
# ============================================================================

test_yield = Op(
    "test.yield",
    group=test_ops,
    doc="Test yield terminator.",
    operands=[Operand("values", ANY, variadic=True)],
    traits=[TERMINATOR, POISON_BOUNDARY],
    format=[
        OptionalGroup(
            [Refs("values"), COLON, TypesOf("values")],
            anchor="values",
        ),
    ],
    examples=["test.yield", "test.yield %first, %second : f32, i32"],
)

# ============================================================================
# test.br — CFG-style branch terminator
# ============================================================================

test_br = Op(
    "test.br",
    group=test_ops,
    doc="Test CFG branch terminator with a semantic successor edge.",
    successors=[Successor("dest", doc="Destination block.")],
    traits=[TERMINATOR],
    format=[BlockRef("dest")],
    examples=["test.br ^dest"],
)

# ============================================================================
# test.func / test.decl — function definitions and declarations
# ============================================================================

test_func = Op(
    "test.func",
    group=test_ops,
    doc="Test function definition with body always present.",
    traits=[SYMBOL_DEFINE, ISOLATED_FROM_ABOVE],
    interfaces=[
        FuncLikeInterface(
            callee="callee",
            visibility="visibility",
            cc="cc",
            predicates="predicates",
            body="body",
        )
    ],
    symbol_def=SymbolDefinition(
        field="callee",
        name="function",
        interfaces=["func_like", "callable"],
        bytecode_kind="LOOM_SYMBOL_FUNC_DEF",
    ),
    attrs=[
        AttrDef("callee", "symbol"),
        AttrDef("visibility", "enum", enum_def=_Visibility, optional=True),
        AttrDef("cc", "enum", enum_def=_CallingConv, optional=True),
        AttrDef("predicates", "predicate_list", optional=True),
    ],
    results=[Result("results", ANY, variadic=True)],
    regions=[RegionDef("body", doc="Function body.", terminator="test.yield")],
    format=[
        OptionalGroup([Attr("visibility")], anchor="visibility"),
        OptionalGroup([Attr("cc")], anchor="cc"),
        SymbolRef("callee"),
        Scope(
            [
                FuncArgs("args"),
                OptionalGroup(
                    [ARROW, ResultTypeList("results")],
                    anchor="results",
                ),
                OptionalGroup(
                    [kw("where"), PredicateList("predicates")],
                    anchor="predicates",
                ),
            ]
        ),
        Region("body"),
    ],
    examples=[
        "test.func @identity(%input: f32) -> (f32) {\n  test.yield %input : f32\n}",
    ],
)

# ============================================================================
# test.split_func — multi-region function definition
# ============================================================================

test_split_func = Op(
    "test.split_func",
    group=test_ops,
    doc="Test function definition with projected signature args in a side region.",
    traits=[SYMBOL_DEFINE, ISOLATED_FROM_ABOVE],
    interfaces=[
        FuncLikeInterface(
            callee="callee",
            visibility="visibility",
            cc="cc",
            body="body",
        )
    ],
    symbol_def=SymbolDefinition(
        field="callee",
        name="function",
        interfaces=["func_like", "callable"],
        bytecode_kind="LOOM_SYMBOL_FUNC_DEF",
    ),
    attrs=[
        AttrDef("callee", "symbol"),
        AttrDef("visibility", "enum", enum_def=_Visibility, optional=True),
        AttrDef("cc", "enum", enum_def=_CallingConv, optional=True),
    ],
    regions=[
        RegionDef(
            "config",
            doc="Projected configuration region.",
            single_block=True,
            terminator="test.yield",
            arg_source="args",
        ),
        RegionDef("body", doc="Function body.", terminator="test.yield"),
    ],
    constraints=[
        BlockArgCount("config", "body"),
        BlockArgsMatchTypes("config", "body"),
    ],
    format=[
        OptionalGroup([Attr("visibility")], anchor="visibility"),
        OptionalGroup([Attr("cc")], anchor="cc"),
        SymbolRef("callee"),
        Scope([FuncArgs("args")]),
        Region("config"),
        kw("launch"),
        Region("body"),
    ],
    examples=[
        "test.split_func @projected(%arg: i32) {\n  test.yield\n} launch {\n  test.yield\n}",
    ],
)

test_decl = Op(
    "test.decl",
    group=test_ops,
    doc="Test function declaration with no body and signature arguments stored as op operands.",
    traits=[SYMBOL_DEFINE],
    operands=[Operand("args", ANY, variadic=True)],
    interfaces=[
        FuncLikeInterface(
            callee="callee",
            visibility="visibility",
            cc="cc",
            args="args",
        )
    ],
    symbol_def=SymbolDefinition(
        field="callee",
        name="function",
        interfaces=["func_like", "callable"],
        bytecode_kind="LOOM_SYMBOL_FUNC_DEF",
    ),
    attrs=[
        AttrDef("callee", "symbol"),
        AttrDef("visibility", "enum", enum_def=_Visibility, optional=True),
        AttrDef("cc", "enum", enum_def=_CallingConv, optional=True),
    ],
    results=[Result("results", ANY, variadic=True)],
    format=[
        OptionalGroup([Attr("visibility")], anchor="visibility"),
        OptionalGroup([Attr("cc")], anchor="cc"),
        SymbolRef("callee"),
        Scope(
            [
                FuncArgs("args"),
                OptionalGroup(
                    [ARROW, ResultTypeList("results")],
                    anchor="results",
                ),
            ]
        ),
    ],
    examples=[
        "test.decl @identity(%input: f32) -> (%input as f32)",
    ],
)

# ============================================================================
# test.record — attr-only module record symbol
# ============================================================================

test_record = Op(
    "test.record",
    group=test_ops,
    doc="Test named module record with generic symbol payload metadata.",
    traits=[SYMBOL_DEFINE],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="record",
        interfaces=["record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
        fact_domain="loom_test_record_symbol_fact_domain",
    ),
    attrs=[
        AttrDef("symbol", "symbol"),
        AttrDef("kind", "enum", enum_def=_RecordKind, optional=True, open_enum=True),
        AttrDef("dict", "dict", optional=True),
    ],
    format=[
        OptionalGroup([Attr("kind")], anchor="kind"),
        SymbolRef("symbol"),
        AttrDict("dict"),
    ],
    examples=[
        'test.record target @target {arch = "gfx1100", lanes = 64}',
    ],
)

# ============================================================================
# test.template_param_symbol — symbol-valued TemplateParam coverage
# ============================================================================

test_template_param_symbol = Op(
    "test.template_param_symbol",
    group=test_ops,
    doc="Test op with a real symbol reference spelled as an angle parameter.",
    attrs=[
        AttrDef(
            "target",
            "symbol",
            symbol_ref=SymbolReference("record", ["record"]),
        ),
    ],
    format=[
        TemplateParam("target"),
    ],
    examples=[
        "test.template_param_symbol<@target>",
    ],
)

test_template_param_symbol_flags = Op(
    "test.template_param_symbol_flags",
    group=test_ops,
    doc="Test op with a symbol angle parameter followed by instance flags.",
    attrs=[
        AttrDef(
            "target",
            "symbol",
            symbol_ref=SymbolReference("record", ["record"]),
        ),
        AttrDef("flags", ATTR_TYPE_FLAGS, optional=True, enum_def=_TemplateFlags),
    ],
    format=[
        TemplateParamFlags("target", "flags"),
    ],
    examples=[
        "test.template_param_symbol_flags<@target, debug|trace>",
    ],
)

# ============================================================================
# test.attrs — op with attribute dictionary
# ============================================================================

test_attrs = Op(
    "test.attrs",
    group=test_ops,
    doc="Test op with attribute dictionary.",
    operands=[Operand("input", ANY)],
    results=[Result("result", ANY)],
    attrs=[AttrDef("dict", "dict", optional=True)],
    constraints=[SameType("input", "result")],
    traits=[PURE],
    format=[
        Ref("input"),
        AttrDict("dict"),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        '%result = test.attrs %input {axis = 0, label = "foo"} : f32',
    ],
)

# ============================================================================
# test.parameterized_attr — descriptor-backed parameterized attribute
# ============================================================================

test_parameterized_attr = Op(
    "test.parameterized_attr",
    group=test_ops,
    doc="Test op carrying an exact descriptor-backed attribute family.",
    attrs=[
        AttrDef(
            "options",
            ATTR_TYPE_PARAMETERIZED,
            parameterized_attr=test_options_attr,
        ),
    ],
    format=[Attr("options")],
    examples=[
        "test.parameterized_attr #test.options<mode = fast>",
    ],
)

# ============================================================================
# test.attr_params — exact-family parameter payload syntax
# ============================================================================

test_attr_params = Op(
    "test.attr_params",
    group=test_ops,
    doc="Test op carrying a known-family parameter payload in angle brackets.",
    attrs=[
        AttrDef(
            "options",
            ATTR_TYPE_PARAMETERIZED,
            parameterized_attr=test_options_attr,
        ),
    ],
    format=[AttrParams("options")],
    examples=[
        "test.attr_params<mode = fast, scopes = [workgroup]>",
    ],
)

# ============================================================================
# test.parameterized_attr_array — ordered parameterized attributes
# ============================================================================

test_parameterized_attr_array = Op(
    "test.parameterized_attr_array",
    group=test_ops,
    doc="Test op carrying open- and exact-family parameterized arrays.",
    attrs=[
        AttrDef(
            "values",
            ATTR_TYPE_PARAMETERIZED_ARRAY,
            doc="Ordered parameterized attributes from any registered family.",
        ),
        AttrDef(
            "tiles",
            ATTR_TYPE_PARAMETERIZED_ARRAY,
            optional=True,
            parameterized_attr=test_tile_attr,
            doc="Optional ordered attributes from the tile family.",
        ),
    ],
    format=[
        Attr("values"),
        OptionalGroup([kw("using"), Attr("tiles")], anchor="tiles"),
    ],
    examples=[
        "test.parameterized_attr_array [#test.tile<width = 8>, #test.options<mode = fast>, #test.tile<width = 16>] using [#test.tile<width = 4>]",
    ],
)

# ============================================================================
# test.compact_parameterized_attr — compact parameterized attribute
# ============================================================================

test_compact_parameterized_attr = Op(
    "test.compact_parameterized_attr",
    group=test_ops,
    doc="Test op carrying a compact descriptor-backed attribute family.",
    attrs=[
        AttrDef(
            "value",
            ATTR_TYPE_PARAMETERIZED,
            parameterized_attr=test_compact_attr,
        ),
    ],
    format=[Attr("value")],
    examples=[
        'test.compact_parameterized_attr #test.compact<64, label = "wave">',
    ],
)

# ============================================================================
# test.enum_array_attrs — descriptor-backed enum-array attributes
# ============================================================================

test_enum_array_attrs = Op(
    "test.enum_array_attrs",
    group=test_ops,
    doc="Test op with closed and open descriptor-backed enum arrays.",
    attrs=[
        AttrDef(
            "required_values",
            ATTR_TYPE_ENUM_ARRAY,
            enum_def=_ArrayElement,
            doc="Required ordered enum values.",
        ),
        AttrDef(
            "optional_values",
            ATTR_TYPE_ENUM_ARRAY,
            enum_def=_ArrayElement,
            optional=True,
            open_enum=True,
            doc="Optional ordered enum values admitting unknown bytes.",
        ),
        AttrDef("dict", "dict", optional=True),
    ],
    format=[
        Attr("required_values"),
        OptionalGroup(
            [kw("using"), Attr("optional_values")],
            anchor="optional_values",
        ),
        AttrDict("dict"),
    ],
    examples=[
        "test.enum_array_attrs [low, high] using [middle, <42>]",
    ],
)

# ============================================================================
# test.operand_dict — op with keyed SSA operand dictionary
# ============================================================================

test_operand_dict = Op(
    "test.operand_dict",
    group=test_ops,
    doc="Test op with a keyed SSA operand dictionary.",
    operands=[
        Operand("input", ANY),
        Operand("params", ANY, variadic=True),
    ],
    results=[Result("result", ANY)],
    attrs=[
        AttrDef(
            "param_names",
            "dict",
            optional=True,
            doc="Sorted operand dictionary keys mapped to operand ordinals.",
        ),
    ],
    constraints=[SameType("input", "result")],
    traits=[PURE],
    format=[
        Ref("input"),
        OperandDict("params", "param_names"),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%result = test.operand_dict %input {alpha = %a : i32, beta = %b : f32} : f32",
    ],
)

# ============================================================================
# test.attr_table — op with static-attribute-keyed SSA value table
# ============================================================================

test_attr_table = Op(
    "test.attr_table",
    group=test_ops,
    doc="Test op with a static-attribute-keyed SSA value table.",
    operands=[
        Operand("selector", INDEX),
        Operand("values", ANY, variadic=True),
    ],
    results=[Result("results", ANY, variadic=True)],
    attrs=[
        AttrDef(
            "case_keys",
            ATTR_TYPE_I64_ARRAY,
            doc="Sorted selector values for explicit table rows.",
        ),
    ],
    traits=[PURE],
    format=[
        Ref("selector"),
        AttrTable("case_keys", "values"),
        COLON,
        ResultTypeList("results", parens=False),
    ],
    examples=[
        "%a, %b = test.attr_table %selector {0 = (%a0, %b0), 1 = (%a1, %b1)} default(%ad, %bd) : i32, f32",
    ],
)

# ============================================================================
# test.region_table — op with static-attribute-keyed region table
# ============================================================================

test_region_table = Op(
    "test.region_table",
    group=test_ops,
    doc="Test op with a static-attribute-keyed region table.",
    operands=[Operand("selector", INDEX)],
    attrs=[
        AttrDef(
            "case_keys",
            ATTR_TYPE_I64_ARRAY,
            doc="Sorted selector values for explicit region rows.",
        ),
    ],
    regions=[
        RegionDef(
            "default_region",
            doc="Fallback region.",
            single_block=True,
            terminator="test.yield",
        ),
        RegionDef(
            "case_regions",
            doc="Case regions in the same order as case_keys.",
            single_block=True,
            variadic=True,
            terminator="test.yield",
        ),
    ],
    interfaces=[RegionBranchInterface(selector="selector")],
    format=[
        Ref("selector"),
        RegionTable("case_keys", "case_regions", "default_region"),
    ],
    examples=[
        "test.region_table %selector {\n  case 0 {\n    test.yield\n  }\n  default {\n    test.yield\n  }\n}",
    ],
)

# ============================================================================
# test.deflate — result dim references
# ============================================================================

test_deflate = Op(
    "test.deflate",
    group=test_ops,
    doc="Test op with result type referencing a co-result dim.",
    operands=[Operand("input", TENSOR)],
    results=[Result("results", ANY, variadic=True)],
    traits=[PURE],
    format=[
        Ref("input"),
        COLON,
        TypeOf("input"),
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
    ],
    examples=[
        "%output, %length = test.deflate %input : tensor<[%M]xf32> -> (tensor<[%length]xf32>, index)",
    ],
)

# ============================================================================
# test.assume — predicate-constrained identity
# ============================================================================

test_assume = Op(
    "test.assume",
    group=test_ops,
    doc="Test predicate-constrained identity (SSA assume).",
    operands=[Operand("values", ANY, variadic=True)],
    results=[Result("results", ANY, variadic=True)],
    attrs=[AttrDef("predicates", "predicate_list")],
    traits=[PURE],
    format=[
        Refs("values"),
        PredicateList("predicates"),
        COLON,
        TypesOf("results"),
    ],
    examples=[
        "%M2 = test.assume %M [mul(%M, 16)] : index",
        "%M2, %K2 = test.assume %M, %K [mul(%M, 16), lt(%K, 1024)] : index, index",
    ],
)

# ============================================================================
# test.condition_refines_positive — generic condition-refinement metadata
# ============================================================================

test_condition_refines_positive = Op(
    "test.condition_refines_positive",
    group=test_ops,
    doc="Test a boolean query that refines its source on the true edge.",
    operands=[Operand("value", INDEX)],
    results=[Result("result", I1)],
    traits=[PURE],
    condition_refinement=ConditionRefinement(
        source="value",
        truth=ConditionRefinementTruth.TRUE,
        materialize="loom_test_condition_refines_positive_materialize",
    ),
    format=[Ref("value"), COLON, TypeOf("value")],
    examples=[
        "%positive = test.condition_refines_positive %value : index",
    ],
)

# ============================================================================
# test.convert — single bare result type (ResultType, no parens)
# ============================================================================

test_convert = Op(
    "test.convert",
    group=test_ops,
    doc="Test op with bare result type (no parentheses).",
    operands=[Operand("input", ANY)],
    results=[Result("result", ANY)],
    traits=[PURE],
    format=[
        Ref("input"),
        COLON,
        TypeOf("input"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%result = test.convert %input : i32 -> f32",
        "%tile = test.convert %x : tile<4xi8> -> tile<4xf32>",
    ],
)

# ============================================================================
# test.reduce — variadic SameType constraint
# ============================================================================

test_reduce = Op(
    "test.reduce",
    group=test_ops,
    doc="Test variadic operands with SameType constraint across variadic and result.",
    operands=[Operand("inputs", ANY, variadic=True)],
    results=[Result("result", ANY)],
    constraints=[SameType("inputs", "result")],
    traits=[PURE],
    format=[
        Refs("inputs"),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%sum = test.reduce %a, %b, %c : i32",
    ],
)

# ============================================================================
# test.read_resource — read-only resource op (for effect testing)
# ============================================================================

test_read_resource = Op(
    "test.read_resource",
    group=test_ops,
    doc="Test op that reads from a resource operand.",
    operands=[Operand("source", POOL, doc="Resource to read from.")],
    results=[Result("result", ANY, doc="Data read from the resource.")],
    effects=[Reads("source")],
    format=[
        Ref("source"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%tile = test.read_resource %pool : pool<[%BS]> -> tile<4xf32>",
    ],
)

# ============================================================================
# test.write_resource — write-only resource op (for effect testing)
# ============================================================================

test_write_resource = Op(
    "test.write_resource",
    group=test_ops,
    doc="Test op that writes to a resource operand.",
    operands=[
        Operand("target", POOL, doc="Resource to write to."),
        Operand("data", ANY, doc="Data to write."),
    ],
    effects=[Writes("target")],
    format=[
        Ref("target"),
        COMMA,
        Ref("data"),
        COLON,
        TypeOf("target"),
        COMMA,
        TypeOf("data"),
    ],
    examples=[
        "test.write_resource %pool, %tile : pool<[%BS]>, tile<4xf32>",
    ],
)

# ============================================================================
# test.mutate_resource — atomic read-modify-write on a resource (for effect testing)
# ============================================================================

test_mutate_resource = Op(
    "test.mutate_resource",
    group=test_ops,
    doc="Test op that atomically reads and writes a resource operand.",
    operands=[
        Operand("target", POOL, doc="Resource to read-modify-write."),
        Operand("value", ANY, doc="Value to combine with the resource."),
    ],
    results=[Result("old_value", ANY, doc="Previous value from the resource.")],
    effects=[ReadWrites("target")],
    format=[
        Ref("target"),
        COMMA,
        Ref("value"),
        COLON,
        TypeOf("target"),
        COMMA,
        TypeOf("value"),
        ARROW,
        ResultType("old_value"),
    ],
    examples=[
        "%old = test.mutate_resource %pool, %delta : pool<[%BS]>, i32 -> i32",
    ],
)

# ============================================================================
# test.alloc — allocation op with unique identity (for CSE/DCE testing)
# ============================================================================

test_alloc = Op(
    "test.alloc",
    group=test_ops,
    doc="Test allocation op. Each execution produces a distinct identity even with identical operands. Prevents CSE but allows DCE when unused.",
    operands=[Operand("size", INDEX, doc="Allocation size.")],
    results=[Result("result", POOL, doc="Allocated pool.", allocates=True)],
    format=[
        Ref("size"),
        COLON,
        TypeOf("size"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%pool = test.alloc %sz : index -> pool<[%BS]>",
    ],
)

# ============================================================================
# test.resource.* — owned resource operations
# ============================================================================

test_resource_alloc = Op(
    "test.resource.alloc",
    group=test_ops,
    doc="Test owned-resource allocation.",
    operands=[Operand("size", INDEX, doc="Allocation size.")],
    results=[Result("result", POOL, doc="Fresh owned resource.", allocates=True)],
    ownership_effects=[FreshResult("result")],
    format=[
        Ref("size"),
        COLON,
        TypeOf("size"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%resource = test.resource.alloc %sz : index -> pool<[%BS]>",
    ],
)

test_resource_borrow = Op(
    "test.resource.borrow",
    group=test_ops,
    doc="Test borrowed use of an owned resource.",
    operands=[Operand("resource", POOL, doc="Resource to borrow.")],
    ownership_effects=[Borrow("resource")],
    format=[Ref("resource"), COLON, TypeOf("resource")],
    examples=[
        "test.resource.borrow %resource : pool<[%BS]>",
    ],
)

test_resource_borrow_ref = Op(
    "test.resource.borrow_ref",
    group=test_ops,
    doc="Test borrowed by-reference use of an owned resource carrier.",
    operands=[Operand("resource", POOL, doc="Resource carrier to borrow.")],
    ownership_effects=[Borrow("resource", BY_REFERENCE)],
    format=[Ref("resource"), COLON, TypeOf("resource")],
    examples=[
        "test.resource.borrow_ref %resource : pool<[%BS]>",
    ],
)

test_resource_consume = Op(
    "test.resource.consume",
    group=test_ops,
    doc="Test consuming use of an owned resource.",
    operands=[Operand("resource", POOL, doc="Resource to consume.")],
    ownership_effects=[Consume("resource")],
    format=[Ref("resource"), COLON, TypeOf("resource")],
    examples=[
        "test.resource.consume %resource : pool<[%BS]>",
    ],
)

test_resource_retain = Op(
    "test.resource.retain",
    group=test_ops,
    doc="Test retaining an additional owned reference to a resource.",
    operands=[Operand("resource", POOL, doc="Resource to retain.")],
    results=[Result("result", POOL, doc="Retained owned resource.")],
    constraints=[SameType("resource", "result")],
    ownership_effects=[Retain("resource"), RetainedResult("result")],
    format=[
        Ref("resource"),
        COLON,
        TypeOf("resource"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%retained = test.resource.retain %resource : pool<[%BS]> -> pool<[%BS]>",
    ],
)

test_resource_release = Op(
    "test.resource.release",
    group=test_ops,
    doc="Test releasing an owned resource.",
    operands=[Operand("resource", POOL, doc="Resource to release.")],
    ownership_effects=[Release("resource")],
    format=[Ref("resource"), COLON, TypeOf("resource")],
    examples=[
        "test.resource.release %resource : pool<[%BS]>",
    ],
)

test_resource_discard = Op(
    "test.resource.discard",
    group=test_ops,
    doc="Test discarding compiler ownership without releasing the resource.",
    operands=[Operand("resource", POOL, doc="Resource to discard.")],
    ownership_effects=[Discard("resource")],
    format=[Ref("resource"), COLON, TypeOf("resource")],
    examples=[
        "test.resource.discard %resource : pool<[%BS]>",
    ],
)

test_resource_escape = Op(
    "test.resource.escape",
    group=test_ops,
    doc="Test transferring a resource to an untracked owner.",
    operands=[Operand("resource", POOL, doc="Resource to escape.")],
    ownership_effects=[Escape("resource")],
    format=[Ref("resource"), COLON, TypeOf("resource")],
    examples=[
        "test.resource.escape %resource : pool<[%BS]>",
    ],
)

test_resource_alias = Op(
    "test.resource.alias",
    group=test_ops,
    doc="Test producing a borrowed alias of a resource.",
    operands=[Operand("resource", POOL, doc="Resource to alias.")],
    results=[Result("result", POOL, doc="Borrowed alias result.")],
    constraints=[SameType("resource", "result")],
    ownership_effects=[
        Borrow("resource"),
        AliasResult("result", "resource"),
    ],
    format=[
        Ref("resource"),
        COLON,
        TypeOf("resource"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%alias = test.resource.alias %resource : pool<[%BS]> -> pool<[%BS]>",
    ],
)

test_resource_borrowed = Op(
    "test.resource.borrowed",
    group=test_ops,
    doc="Test materializing a borrowed resource result.",
    operands=[Operand("resource", POOL, doc="Resource to borrow.")],
    results=[Result("result", POOL, doc="Borrowed resource result.")],
    constraints=[SameType("resource", "result")],
    ownership_effects=[
        Borrow("resource"),
        BorrowedResult("result"),
    ],
    format=[
        Ref("resource"),
        COLON,
        TypeOf("resource"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%borrowed = test.resource.borrowed %resource : pool<[%BS]> -> pool<[%BS]>",
    ],
)

# ============================================================================
# test.isolated_region — isolated single-block region (for CSE testing)
# ============================================================================

test_isolated_region = Op(
    "test.isolated_region",
    group=test_ops,
    doc="Test op with an isolated single-block region. Values from the enclosing scope are not visible inside the body.",
    results=[Result("results", ANY, variadic=True)],
    regions=[
        RegionDef(
            "body",
            doc="Isolated region body.",
            single_block=True,
            terminator="test.yield",
        ),
    ],
    traits=[ISOLATED_FROM_ABOVE],
    format=[
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
        Region("body"),
    ],
    examples=[
        "%r = test.isolated_region -> (i32) {\n  %c = test.constant 42 : i32\n  test.yield %c : i32\n}",
    ],
)

# ============================================================================
# test.counter — canonicalize test harness (multi-step, error, fixed point)
# ============================================================================

test_counter = Op(
    "test.counter",
    group=test_ops,
    doc="Test op for canonicalize multi-step and error path testing. Canonicalize: value < 0 returns error, value > 0 decrements, value == 0 is fixed point.",
    results=[Result("result", ANY)],
    attrs=[AttrDef("value", "i64", doc="Counter value.")],
    traits=[PURE],
    canonicalize="loom_test_counter_canonicalize",
    format=[Attr("value"), COLON, TypeOf("result")],
    examples=[
        "%c = test.counter 3 : i32",
        "%c = test.counter 0 : i32",
    ],
)

# ============================================================================
# test.dim — dimension query with ATTR_IN_RANGE_RANK constraint
# ============================================================================

test_dim = Op(
    "test.dim",
    group=test_ops,
    doc="Test dimension query to exercise ATTR_IN_RANGE_RANK constraint.",
    operands=[Operand("source", TILE)],
    results=[Result("result", INDEX)],
    attrs=[
        AttrDef("dim_index", "i64", doc="Dimension index to query."),
    ],
    constraints=[DimIndexInBounds("source", "dim_index")],
    traits=[PURE],
    format=[
        Ref("source"),
        LBRACKET,
        Attr("dim_index"),
        RBRACKET,
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%d = test.dim %t[0] : tile<4xf32> -> index",
    ],
)

# ============================================================================
# test.region_syntax — alternate region surface syntax
# ============================================================================

test_region_syntax = Op(
    "test.region_syntax",
    group=test_ops,
    doc="Test op whose body uses an alternate declarative region syntax while preserving ordinary region storage.",
    regions=[
        RegionDef(
            "body",
            doc="Body parsed through a named region syntax.",
            single_block=True,
            terminator="test.yield",
        )
    ],
    traits=[ImplicitTerminator("test.implicit_yield")],
    format=[Region("body", syntax="test.do")],
    examples=[
        "test.region_syntax do {\n  test.yield\n}",
    ],
)

# ============================================================================
# test.target — target-like symbol record
# ============================================================================

test_target = Op(
    "test.target",
    group=test_ops,
    doc="Test target-like module record with structural interface metadata.",
    traits=[SYMBOL_DEFINE],
    interfaces=[
        TargetLikeInterface(
            symbol="symbol",
            selector="kind",
            bundle_table="loom_test_target_bundles",
            fact_type="loom_test_target_fact_type",
        )
    ],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="target",
        interfaces=["target", "record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
        fact_domain="loom_target_symbol_fact_domain",
    ),
    attrs=target_record_attrs(_TargetKind),
    verify="loom_target_record_verify",
    format=[
        TemplateParam("kind"),
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        "test.target<low_core> @target {subgroup_size = 64}",
    ],
)

# ============================================================================
# Registry: all test ops in declaration order
# ============================================================================

ALL_TEST_OPS: tuple[Op, ...] = (
    test_addi,
    test_neg,
    test_cast,
    test_constant,
    test_effectful_constant,
    test_use,
    test_convergent,
    test_cmp,
    test_map,
    test_update,
    test_invoke,
    test_low_call,
    test_low_invoke,
    test_slice,
    test_loop,
    test_block_args,
    test_branch,
    test_optional_region,
    test_implicit_yield,
    test_yield,
    test_br,
    test_func,
    test_split_func,
    test_decl,
    test_record,
    test_attrs,
    test_operand_dict,
    test_attr_table,
    test_region_table,
    test_deflate,
    test_assume,
    test_convert,
    test_reduce,
    test_read_resource,
    test_write_resource,
    test_mutate_resource,
    test_alloc,
    test_isolated_region,
    test_counter,
    test_dim,
    test_fact_range_lo,
    test_fact_range_hi,
    test_fact_all_equal_range_lo,
    test_fact_all_equal_range_hi,
    test_fact_divisor,
    test_fact_non_negative,
    test_fact_non_zero,
    test_fact_positive,
    test_fact_power_of_two,
    test_fact_subgroup_uniform,
    test_fact_workgroup_uniform,
    test_fact_lane_varying,
    test_fact_lane_predicate,
    test_fact_subgroup_lane_mask,
    test_fact_is_vector_iota,
    test_fact_is_vector_prefix_mask,
    test_fact_encoding_layout_kind,
    test_fact_encoding_layout_stride_hi,
    test_fact_encoding_operand_field,
    test_fact_is_buffer_reference,
    test_fact_is_view_reference,
    test_fact_buffer_memory_space,
    test_fact_view_memory_space,
    test_fact_view_root_matches,
    test_fact_alias_scope_known,
    test_fact_alias_scope_matches,
    test_fact_view_byte_offset_lo,
    test_fact_view_byte_offset_hi,
    test_fact_view_byte_length_lo,
    test_fact_view_byte_length_hi,
    test_fact_view_min_alignment,
    test_fact_buffer_min_alignment,
    test_fact_view_root_min_alignment,
    test_fact_view_element_bytes,
    test_fact_is_storage_reference,
    test_fact_storage_same_backing,
    test_fact_storage_byte_offset_lo,
    test_fact_storage_byte_offset_hi,
    test_fact_storage_byte_offset_divisor,
    test_fact_storage_byte_length_lo,
    test_fact_storage_byte_length_hi,
    test_fact_storage_min_alignment,
    test_fact_storage_space,
    test_region_syntax,
    test_clause_constant,
    test_clause_copy,
    test_typed_use,
    test_shape,
    test_target,
    test_resource_alloc,
    test_resource_borrow,
    test_resource_borrow_ref,
    test_resource_consume,
    test_resource_retain,
    test_resource_release,
    test_resource_discard,
    test_resource_escape,
    test_resource_alias,
    test_resource_borrowed,
    test_segmented,
    test_template_param_symbol,
    test_template_param_symbol_flags,
    test_fact_finite,
    test_fact_not_subnormal,
    test_fact_cluster_uniform,
    test_enum_array_attrs,
    test_parameterized_attr,
    test_compact_parameterized_attr,
    test_parameterized_attr_array,
    test_attr_params,
    test_condition_refines_positive,
    test_partitioned_call,
)
