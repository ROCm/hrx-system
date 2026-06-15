# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Vector dialect op definitions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from loom.assembly import (
    ARROW,
    COLON,
    COMMA,
    GLUE,
    LBRACKET,
    RBRACKET,
    Attr,
    AttrDict,
    Flags,
    FormatElement,
    IndexList,
    OperandDict,
    OptionalGroup,
    PredicateList,
    Ref,
    Refs,
    ResultType,
    ResultTypeList,
    TemplateParam,
    TemplateParamFlags,
    TypeOf,
    TypesOf,
    kw,
)
from loom.dialect.atomic import AtomicKind, AtomicOrdering, AtomicScope
from loom.dialect.cache import CacheScope, CacheTemporal
from loom.dialect.combining import CombiningKind
from loom.dialect.scalar import ClampFMode, FastMathFlags, GeluVariant, IntOverflowFlags
from loom.dialect.scalar.comparison import CmpFPredicate, CmpIPredicate
from loom.dsl import (
    ANY,
    ATTR_TYPE_ANY,
    ATTR_TYPE_DICT,
    ATTR_TYPE_ENUM,
    ATTR_TYPE_F64,
    ATTR_TYPE_FLAGS,
    ATTR_TYPE_I64,
    ATTR_TYPE_I64_ARRAY,
    ATTR_TYPE_PREDICATE_LIST,
    COMMUTATIVE,
    CONSTANT_LIKE,
    ELEMENTWISE,
    ENCODING_SCHEMA,
    ENCODING_TRANSFORM,
    FLOAT_ELEMENT,
    I1_ELEMENT,
    IDEMPOTENT,
    INDEX,
    INTEGER_ELEMENT,
    INVOLUTION,
    PURE,
    REFINABLE_RESULT_TYPE_REFS,
    SCALAR,
    VALUE_ALIAS,
    VECTOR,
    VIEW,
    AttrDef,
    AttrMatchesElementType,
    BitRangeWithinElementWidth,
    Constraint,
    ContractFamily,
    Dialect,
    DimIndexInBounds,
    ElementWidthAtLeastAttr,
    ElementWidthGreaterThan,
    ElementWidthLessThan,
    EnumCase,
    EnumDef,
    HasAllStaticRankOneVector,
    HasAllStaticVector,
    HasF16OrBf16Element,
    HasF32Element,
    HasFloatElement,
    HasI1Element,
    HasI8Element,
    HasI32Element,
    HasIndexOrNonI1IntegerElement,
    HasIndexOrNonI1IntegerScalar,
    HasIntegerElement,
    HasRankOneVector,
    LastAxisGroupedBy,
    MemoryAccessInterface,
    Op,
    OpCategory,
    Operand,
    OpPhase,
    PackedPayloadBitCountMatchesStorage,
    PositiveBitWidthAttr,
    Reads,
    ReadWrites,
    Result,
    SameElementType,
    SameKind,
    SameShape,
    SameType,
    TotalBitCountEqual,
    Trait,
    TypeConstraint,
    UnpackedPayloadBitCountMatchesStorage,
    ValueCountMatchesStaticElementCount,
    Writes,
)

# ============================================================================
# Group and shared attrs
# ============================================================================

VECTOR_CONSTRUCTION_CATEGORY = OpCategory(
    "construction",
    doc="Vector constants, splats, shape producers, and aggregate construction.",
)
VECTOR_AGGREGATE_CATEGORY = OpCategory(
    "aggregate",
    doc="Vector aggregate lane extraction, insertion, and rearrangement.",
)
VECTOR_TABLE_CATEGORY = OpCategory(
    "table",
    doc="Register-table lookup, quantization, and transform descriptors.",
)
VECTOR_MEMORY_CATEGORY = OpCategory(
    "memory",
    doc="Vector memory transfer operations.",
)
VECTOR_ATOMIC_CATEGORY = OpCategory(
    "atomic",
    doc="Vector atomic memory operations.",
)
VECTOR_COMPARE_CATEGORY = OpCategory(
    "compare",
    doc="Vector selection and comparison operations.",
)
VECTOR_FLOAT_ARITHMETIC_CATEGORY = OpCategory(
    "float_arithmetic",
    doc="Lanewise floating-point arithmetic operations.",
)
VECTOR_INTEGER_ARITHMETIC_CATEGORY = OpCategory(
    "integer_arithmetic",
    doc="Lanewise integer arithmetic and bitwise operations.",
)
VECTOR_MATH_CATEGORY = OpCategory(
    "math",
    doc="Lanewise transcendental, rounding, and classification operations.",
)
VECTOR_CAST_CATEGORY = OpCategory(
    "cast",
    doc="Vector element-type conversion operations.",
)
VECTOR_BITPACK_CATEGORY = OpCategory(
    "bitpack",
    doc="Vector bitfield and packed-payload operations.",
)
VECTOR_CONTRACTION_CATEGORY = OpCategory(
    "contraction",
    doc="Vector dot and contraction operations.",
)
VECTOR_REDUCTION_CATEGORY = OpCategory(
    "reduction",
    doc="Vector horizontal reduction operations.",
)
VECTOR_ENCODING_CATEGORY = OpCategory(
    "encoding",
    doc="Vector encoded numeric interpretation boundaries.",
)

VECTOR_OP_CATEGORIES = (
    VECTOR_CONSTRUCTION_CATEGORY,
    VECTOR_AGGREGATE_CATEGORY,
    VECTOR_TABLE_CATEGORY,
    VECTOR_MEMORY_CATEGORY,
    VECTOR_ATOMIC_CATEGORY,
    VECTOR_COMPARE_CATEGORY,
    VECTOR_FLOAT_ARITHMETIC_CATEGORY,
    VECTOR_INTEGER_ARITHMETIC_CATEGORY,
    VECTOR_MATH_CATEGORY,
    VECTOR_CAST_CATEGORY,
    VECTOR_BITPACK_CATEGORY,
    VECTOR_CONTRACTION_CATEGORY,
    VECTOR_REDUCTION_CATEGORY,
    VECTOR_ENCODING_CATEGORY,
)

vector_ops = Dialect(
    "vector",
    dialect_id=0x0E,
    doc=(
        "Vector register-lane construction, lanewise computation, memory, and "
        "reduction ops. Static zero-lane vector values are empty aggregates, "
        "not poison; canonicalization erases their pure computation and "
        "zero-lane memory effects. Typed vector poison represents invalid "
        "observations that must be removed or diagnosed before target lowering."
    ),
    categories=VECTOR_OP_CATEGORIES,
)

VectorFragmentRole = EnumDef(
    "VectorFragmentRole",
    [
        EnumCase("lhs", 0, doc="Left-hand matrix operand fragment."),
        EnumCase("rhs", 1, doc="Right-hand matrix operand fragment."),
        EnumCase("init", 2, doc="Initial accumulator matrix fragment."),
        EnumCase("result", 3, doc="Result accumulator matrix fragment."),
    ],
)

IntegerDot4Kind = EnumDef(
    "IntegerDot4Kind",
    [
        EnumCase("s8s8", 0, doc="Signed i8 lhs times signed i8 rhs."),
        EnumCase("u8s8", 1, doc="Unsigned i8 lhs times signed i8 rhs."),
        EnumCase("s8u8", 2, doc="Signed i8 lhs times unsigned i8 rhs."),
        EnumCase("u8u8", 3, doc="Unsigned i8 lhs times unsigned i8 rhs."),
    ],
    doc="Signedness variants for four-lane i8 dot products accumulated into i32 lanes.",
)

IntegerDot8I4Kind = EnumDef(
    "IntegerDot8I4Kind",
    [
        EnumCase("s4s4", 0, doc="Signed packed i4 lhs times signed packed i4 rhs."),
        EnumCase("u4s4", 1, doc="Unsigned packed u4 lhs times signed packed i4 rhs."),
        EnumCase("s4u4", 2, doc="Signed packed i4 lhs times unsigned packed u4 rhs."),
        EnumCase("u4u4", 3, doc="Unsigned packed u4 lhs times unsigned packed u4 rhs."),
    ],
    doc="Signedness variants for packed eight-lane i4 dot products accumulated into i32 lanes.",
)

FloatDot4F8Kind = EnumDef(
    "FloatDot4F8Kind",
    [
        EnumCase("fp8bf8", 0, doc="Packed fp8/E4M3 lhs times packed bf8/E5M2 rhs."),
        EnumCase("bf8fp8", 1, doc="Packed bf8/E5M2 lhs times packed fp8/E4M3 rhs."),
        EnumCase("fp8fp8", 2, doc="Packed fp8/E4M3 lhs times packed fp8/E4M3 rhs."),
        EnumCase("bf8bf8", 3, doc="Packed bf8/E5M2 lhs times packed bf8/E5M2 rhs."),
    ],
    doc="Format variants for packed four-lane fp8/bf8 dot products accumulated into f32 lanes.",
)

QuantizeNaN = EnumDef(
    "QuantizeNaN",
    [
        EnumCase("zero", 0, doc="Map NaN lanes to code 0."),
        EnumCase("max", 1, doc="Map NaN lanes to the last code."),
    ],
    doc="NaN handling policy for table-based scalar quantization.",
)

QuantizeTie = EnumDef(
    "QuantizeTie",
    [
        EnumCase("lower", 0, doc="Equal-to-threshold lanes remain in the lower bin."),
        EnumCase("upper", 1, doc="Equal-to-threshold lanes advance to the upper bin."),
    ],
    doc="Threshold equality policy for table-based scalar quantization.",
)


def _lanewise_binary(
    name: str,
    *,
    result_constraint: TypeConstraint,
    doc: str,
    commutative: bool = False,
    flags: tuple[str, EnumDef] | None = None,
    facts: str = "",
    canonicalize: str = "",
    **kwargs: Any,
) -> Op:
    result_element_constraint = _element_constraint_for(result_constraint)
    attrs: list[AttrDef] = []
    fmt: list[FormatElement] = []
    if flags:
        attr_name, enum_def = flags
        attrs.append(AttrDef(attr_name, ATTR_TYPE_FLAGS, optional=True, enum_def=enum_def))
        fmt.append(Flags(attr_name))
    fmt.extend([Ref("lhs"), COMMA, Ref("rhs"), COLON, TypeOf("result")])

    traits: list[Trait] = [PURE, ELEMENTWISE]
    if commutative:
        traits.append(COMMUTATIVE)
    return Op(
        name,
        group=vector_ops,
        doc=doc,
        operands=[
            Operand("lhs", VECTOR),
            Operand("rhs", VECTOR),
        ],
        results=[Result("result", VECTOR)],
        attrs=attrs,
        constraints=[
            result_element_constraint("result"),
            SameType("lhs", "rhs", "result"),
        ],
        traits=traits,
        facts=facts,
        canonicalize=canonicalize,
        format=fmt,
        **kwargs,
    )


def _lanewise_unary(
    name: str,
    *,
    result_constraint: TypeConstraint,
    doc: str,
    traits: list[Trait] | None = None,
    flags: tuple[str, EnumDef] | None = None,
    facts: str = "",
    canonicalize: str = "",
    **kwargs: Any,
) -> Op:
    result_element_constraint = _element_constraint_for(result_constraint)
    attrs: list[AttrDef] = []
    fmt: list[FormatElement] = []
    if flags:
        attr_name, enum_def = flags
        attrs.append(AttrDef(attr_name, ATTR_TYPE_FLAGS, optional=True, enum_def=enum_def))
        fmt.append(Flags(attr_name))
    fmt.extend([Ref("input"), COLON, TypeOf("result")])

    op_traits: list[Trait] = [PURE, ELEMENTWISE]
    if traits:
        op_traits.extend(traits)
    return Op(
        name,
        group=vector_ops,
        doc=doc,
        operands=[Operand("input", VECTOR)],
        results=[Result("result", VECTOR)],
        attrs=attrs,
        constraints=[
            result_element_constraint("result"),
            SameType("input", "result"),
        ],
        traits=op_traits,
        facts=facts,
        canonicalize=canonicalize,
        format=fmt,
        **kwargs,
    )


def _lanewise_unary_shape_change(
    name: str,
    *,
    source_constraint: Callable[[str], Constraint],
    result_constraint: TypeConstraint,
    doc: str,
    facts: str = "",
    canonicalize: str = "",
    **kwargs: Any,
) -> Op:
    result_element_constraint = _element_constraint_for(result_constraint)
    return Op(
        name,
        group=vector_ops,
        doc=doc,
        operands=[Operand("input", VECTOR)],
        results=[Result("result", VECTOR)],
        constraints=[
            source_constraint("input"),
            result_element_constraint("result"),
            SameKind("input", "result"),
            SameShape("input", "result"),
        ],
        traits=[PURE, ELEMENTWISE],
        facts=facts,
        canonicalize=canonicalize,
        format=[
            Ref("input"),
            COLON,
            TypeOf("input"),
            ARROW,
            ResultType("result"),
        ],
        **kwargs,
    )


def _vector_cast(
    name: str,
    *,
    result_constraint: TypeConstraint,
    source_constraint: Callable[[str], Constraint],
    doc: str,
    constraints: Sequence[Constraint] = (),
    facts: str = "",
    canonicalize: str = "",
    **kwargs: Any,
) -> Op:
    result_element_constraint = _element_constraint_for(result_constraint)
    return Op(
        name,
        group=vector_ops,
        doc=doc,
        operands=[Operand("input", VECTOR)],
        results=[Result("result", VECTOR)],
        constraints=[
            source_constraint("input"),
            result_element_constraint("result"),
            SameKind("input", "result"),
            SameShape("input", "result"),
            *constraints,
        ],
        facts=facts,
        canonicalize=canonicalize,
        traits=[PURE, ELEMENTWISE],
        format=[
            Ref("input"),
            COLON,
            TypeOf("input"),
            kw("to"),
            TypeOf("result"),
        ],
        **kwargs,
    )


def _element_constraint_for(
    type_constraint: TypeConstraint,
) -> Callable[[str], Constraint]:
    if type_constraint == FLOAT_ELEMENT:
        return HasFloatElement
    if type_constraint == INTEGER_ELEMENT:
        return HasIntegerElement
    if type_constraint == I1_ELEMENT:
        return HasI1Element
    raise ValueError(f"unsupported vector element constraint: {type_constraint}")


# ============================================================================
# Construction
# ============================================================================

vector_constant = Op(
    "vector.constant",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Materialize a compile-time vector value whose every lane has the "
        "same scalar attribute payload. The result type supplies both the "
        "vector shape and the element type used to interpret the payload."
    ),
    results=[Result("result", VECTOR)],
    attrs=[AttrDef("value", ATTR_TYPE_ANY, doc="The constant payload.")],
    constraints=[AttrMatchesElementType("value", "result")],
    facts="loom_vector_constant_facts",
    traits=[PURE, CONSTANT_LIKE, REFINABLE_RESULT_TYPE_REFS],
    format=[Attr("value"), COLON, ResultType("result")],
    examples=["%v = vector.constant 0.0 : vector<4xf32>"],
)

vector_poison = Op(
    "vector.poison",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Materialize a typed Loom poison vector. Poison represents an invalid "
        "vector value and propagates through pure vector ops until dead-code "
        "elimination removes it or a boundary diagnoses it. A zero-lane vector "
        "such as vector<0xf32> is not poison: it is an empty aggregate whose "
        "pure lane-wise computation and zero-lane memory effects should "
        "canonicalize away. Poison is introduced when IR observes something "
        "that cannot exist, such as a lane extracted from a vector proven to "
        "have zero lanes."
    ),
    results=[Result("result", VECTOR)],
    verify="loom_vector_poison_verify",
    traits=[PURE, REFINABLE_RESULT_TYPE_REFS],
    format=[COLON, ResultType("result")],
    examples=[
        "%p = vector.poison : vector<4xf32>",
    ],
)

vector_empty = Op(
    "vector.empty",
    group=vector_ops,
    doc=("Materialize the unique empty aggregate value for a static zero-lane vector type. Empty vectors are ordinary values, not poison, and pure zero-lane computation canonicalizes to this op."),
    results=[Result("result", VECTOR)],
    verify="loom_vector_empty_verify",
    traits=[PURE, CONSTANT_LIKE, REFINABLE_RESULT_TYPE_REFS],
    format=[COLON, ResultType("result")],
    examples=["%v = vector.empty : vector<0xf32>"],
)

vector_splat = Op(
    "vector.splat",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Replicate one scalar value to every lane of a vector result. The "
        "annotation after ':' is the result vector type; the scalar operand "
        "must already have the same element type, so conversions must be "
        "spelled with scalar/vector cast ops before or after the splat."
    ),
    operands=[Operand("scalar", SCALAR)],
    results=[Result("result", VECTOR)],
    constraints=[SameElementType("scalar", "result")],
    facts="loom_vector_splat_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
    traits=[PURE, REFINABLE_RESULT_TYPE_REFS],
    format=[Ref("scalar"), COLON, ResultType("result")],
    examples=[
        "%vec = vector.splat %scalar : vector<16xf32>",
        "%vec = vector.splat %scalar : vector<[%n]xi32>",
    ],
)

vector_iota = Op(
    "vector.iota",
    group=vector_ops,
    contracts=[ContractFamily.VECTOR_COORDINATE],
    doc=(
        "Construct a vector of lane-coordinate values. Lane order is the "
        "logical row-major order of the result shape; result lane ordinal i "
        "contains base + i * step. The result element type must be index or a "
        "non-i1 integer payload, and base/step must be scalar values with the "
        "same element type. Dynamic result extents are allowed: the result "
        "type supplies the lane count symbolically and later specialization "
        "fixes the concrete number of produced coordinates."
    ),
    operands=[
        Operand("base", SCALAR, doc="First coordinate value."),
        Operand("step", SCALAR, doc="Coordinate delta between adjacent logical lanes."),
    ],
    results=[Result("result", VECTOR)],
    constraints=[
        HasIndexOrNonI1IntegerScalar("base"),
        SameType("base", "step"),
        SameElementType("base", "step", "result"),
    ],
    facts="loom_vector_iota_facts",
    canonicalize="loom_vector_iota_canonicalize",
    traits=[PURE, REFINABLE_RESULT_TYPE_REFS],
    format=[
        Ref("base"),
        kw("step"),
        Ref("step"),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%lanes = vector.iota %c0 step %c1 : vector<16xindex>",
        "%offsets = vector.iota %base step %stride : vector<4xi32>",
        "%dyn = vector.iota %c0 step %c1 : vector<[%n]xindex>",
    ],
)

vector_mask_range = Op(
    "vector.mask.range",
    group=vector_ops,
    doc=(
        "Construct an i1 tail mask from an explicit scalar coordinate range. "
        "For logical lane ordinal i, the lane is true when "
        "lower_bound + i * step is strictly less than upper_bound using the "
        "coordinate domain's signed ordering. The bracketed syntax mirrors "
        "scf.for ranges because the same inclusive-lower, exclusive-upper "
        "semantics are being tested; the result vector type supplies the "
        "number and shape of lanes to test."
    ),
    operands=[
        Operand("lower_bound", SCALAR, doc="First coordinate tested by lane 0."),
        Operand("upper_bound", SCALAR, doc="Exclusive coordinate bound."),
        Operand("step", SCALAR, doc="Coordinate delta between adjacent logical lanes."),
    ],
    results=[Result("result", VECTOR)],
    constraints=[
        HasI1Element("result"),
        HasIndexOrNonI1IntegerScalar("lower_bound"),
        SameType("lower_bound", "upper_bound", "step"),
    ],
    facts="loom_vector_mask_range_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
    traits=[PURE, REFINABLE_RESULT_TYPE_REFS],
    format=[
        LBRACKET,
        Ref("lower_bound"),
        kw("to"),
        Ref("upper_bound"),
        kw("step"),
        Ref("step"),
        RBRACKET,
        COLON,
        TypeOf("lower_bound"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%mask = vector.mask.range [%iv to %n step %c1] : index -> vector<16xi1>",
        "%mask = vector.mask.range [%base to %limit step %stride] : i32 -> vector<8xi1>",
    ],
)

vector_broadcast = Op(
    "vector.broadcast",
    group=vector_ops,
    doc=(
        "Broadcast a vector value to a larger-rank or same-rank vector "
        "result. Source axes align with the trailing result axes, and each "
        "static source extent must either be 1 or match the corresponding "
        "result extent."
    ),
    operands=[Operand("source", VECTOR)],
    results=[Result("result", VECTOR)],
    constraints=[SameElementType("source", "result")],
    verify="loom_vector_broadcast_verify",
    facts="loom_vector_broadcast_facts",
    traits=[PURE],
    format=[Ref("source"), COLON, TypeOf("source"), ARROW, ResultType("result")],
    examples=["%wide = vector.broadcast %v : vector<4xf32> -> vector<16x4xf32>"],
)

vector_from_elements = Op(
    "vector.from_elements",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Build an all-static vector from scalar element operands in logical "
        "lane order. The result vector type defines both the lane count and "
        "element type: the number of operands must equal the static element "
        "count, and every operand must have the vector element type."
    ),
    operands=[Operand("elements", SCALAR, variadic=True)],
    results=[Result("result", VECTOR)],
    constraints=[
        HasAllStaticVector("result"),
        SameElementType("elements", "result"),
        ValueCountMatchesStaticElementCount("result", "elements"),
    ],
    facts="loom_vector_from_elements_facts",
    canonicalize="loom_vector_from_elements_canonicalize",
    traits=[PURE],
    format=[
        Refs("elements"),
        COLON,
        ResultType("result"),
    ],
    examples=["%v = vector.from_elements %a, %b, %c, %d : vector<4xf32>"],
)


# ============================================================================
# Access
# ============================================================================

vector_extract = Op(
    "vector.extract",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Extract a scalar or tail subvector from a vector at explicit leading "
        "indices. Supplying one index consumes the first source axis, two "
        "indices consume the first two axes, and consuming all axes produces "
        "a scalar element."
    ),
    operands=[
        Operand("source", VECTOR),
        Operand("indices", INDEX, variadic=True),
    ],
    results=[Result("result", ANY)],
    attrs=[
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static indices with INT64_MIN sentinels for dynamics.",
        ),
    ],
    constraints=[SameElementType("source", "result")],
    verify="loom_vector_extract_verify",
    facts="loom_vector_extract_facts",
    canonicalize="loom_vector_extract_canonicalize",
    traits=[PURE],
    format=[
        Ref("source"),
        IndexList("indices", "static_indices"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=["%x = vector.extract %v[%i] : vector<[%n]xf32> -> f32"],
)

vector_insert = Op(
    "vector.insert",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Insert a scalar or tail subvector into a vector at explicit leading "
        "indices. The inserted value must match the destination tail shape "
        "remaining after the supplied indices, and the result type is the "
        "same as the destination type."
    ),
    operands=[
        Operand("value", ANY),
        Operand("dest", VECTOR),
        Operand("indices", INDEX, variadic=True),
    ],
    results=[Result("result", VECTOR)],
    attrs=[
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static indices with INT64_MIN sentinels for dynamics.",
        ),
    ],
    constraints=[
        SameElementType("value", "dest", "result"),
        SameType("dest", "result"),
    ],
    verify="loom_vector_insert_verify",
    facts="loom_vector_insert_facts",
    traits=[PURE],
    format=[
        Ref("value"),
        kw("into"),
        Ref("dest"),
        IndexList("indices", "static_indices"),
        COLON,
        TypeOf("value"),
        COMMA,
        ResultType("result"),
    ],
    examples=["%r = vector.insert %x into %v[%i] : f32, vector<[%n]xf32>"],
)


# ============================================================================
# Register layout
# ============================================================================

vector_slice = Op(
    "vector.slice",
    group=vector_ops,
    contracts=[ContractFamily.REGISTER_PERMUTATION],
    doc=(
        "Extract a rank-preserving contiguous register subvector at explicit "
        "offsets. The offset list has one entry per source axis; each result "
        "axis extent describes how many lanes are kept from that source axis."
    ),
    operands=[
        Operand("source", VECTOR),
        Operand("offsets", INDEX, variadic=True),
    ],
    results=[Result("result", VECTOR)],
    attrs=[
        AttrDef(
            "static_offsets",
            ATTR_TYPE_I64_ARRAY,
            doc="Static offsets with INT64_MIN sentinels for dynamics.",
        ),
    ],
    constraints=[SameElementType("source", "result")],
    verify="loom_vector_slice_verify",
    facts="loom_vector_slice_facts",
    traits=[PURE],
    format=[
        Ref("source"),
        IndexList("offsets", "static_offsets"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%tail = vector.slice %v[%i] : vector<[%n]xf32> -> vector<4xf32>",
        "%tile = vector.slice %v[0, 4] : vector<8x16xf32> -> vector<4x8xf32>",
    ],
)

vector_concat = Op(
    "vector.concat",
    group=vector_ops,
    doc=(
        "Concatenate one or more same-rank vectors along the template axis. "
        "All non-concatenated axes must match the result shape, and when "
        "static the result axis extent must equal the sum of input extents."
    ),
    operands=[Operand("inputs", VECTOR, variadic=True)],
    results=[Result("result", VECTOR)],
    attrs=[
        AttrDef(
            "axis",
            ATTR_TYPE_I64,
            doc="Axis along which input extents concatenate.",
        ),
    ],
    constraints=[
        SameElementType("inputs", "result"),
        DimIndexInBounds("result", "axis"),
    ],
    verify="loom_vector_concat_verify",
    facts="loom_vector_concat_facts",
    traits=[PURE],
    format=[
        TemplateParam("axis"),
        Refs("inputs"),
        COLON,
        TypesOf("inputs"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%wide = vector.concat<0> %a, %b : vector<4xf32>, vector<4xf32> -> vector<8xf32>",
        "%cols = vector.concat<1> %a, %b : vector<4x8xf32>, vector<4x8xf32> -> vector<4x16xf32>",
    ],
)

vector_transpose = Op(
    "vector.transpose",
    group=vector_ops,
    doc=(
        "Permute vector register axes. The template list maps each result "
        "axis to a source axis: permutation[i] is the source axis used for "
        "result axis i, so <[1, 0]> maps vector<MxN> to vector<NxM>. This "
        "does not touch memory layout; it only reorders lanes in the register "
        "value."
    ),
    operands=[Operand("source", VECTOR)],
    results=[Result("result", VECTOR)],
    attrs=[
        AttrDef(
            "permutation",
            ATTR_TYPE_I64_ARRAY,
            doc=("Result-axis to source-axis permutation. The list length must equal the source rank and must mention each source axis exactly once."),
        ),
    ],
    constraints=[SameElementType("source", "result")],
    verify="loom_vector_transpose_verify",
    type_transfer="loom_vector_transpose_type_transfer",
    facts="loom_vector_transpose_facts",
    traits=[PURE],
    format=[
        TemplateParam("permutation"),
        Ref("source"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%t = vector.transpose<[1, 0]> %v : vector<4x8xf32> -> vector<8x4xf32>",
    ],
)

vector_shuffle = Op(
    "vector.shuffle",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Reorder a static rank-1 vector with a static lane map. Entry i of "
        "source_lanes selects the source lane for result lane i; duplicate "
        "source lanes are allowed, but the result type is the same as the "
        "source type."
    ),
    operands=[Operand("source", VECTOR)],
    results=[Result("result", VECTOR)],
    attrs=[
        AttrDef(
            "source_lanes",
            ATTR_TYPE_I64_ARRAY,
            doc="Source lane index for each result lane.",
        ),
    ],
    constraints=[
        HasAllStaticRankOneVector("source"),
        SameType("source", "result"),
    ],
    verify="loom_vector_shuffle_verify",
    facts="loom_vector_shuffle_facts",
    canonicalize="loom_vector_shuffle_canonicalize",
    traits=[PURE],
    format=[
        TemplateParam("source_lanes"),
        Ref("source"),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%rev = vector.shuffle<[3, 2, 1, 0]> %v : vector<4xf32>",
        "%dup = vector.shuffle<[0, 0, 2, 2]> %v : vector<4xf32>",
    ],
)


vector_interleave = Op(
    "vector.interleave",
    group=vector_ops,
    doc=(
        "Interleave two same-typed vectors along the template axis. Result "
        "positions with even coordinates along that axis come from the first "
        "operand, odd coordinates come from the second operand, and the "
        "result extent on that axis is doubled."
    ),
    operands=[
        Operand("even", VECTOR, doc="Vector providing result lanes at even positions along the interleaved axis."),
        Operand("odd", VECTOR, doc="Vector providing result lanes at odd positions along the interleaved axis."),
    ],
    results=[Result("result", VECTOR)],
    attrs=[
        AttrDef(
            "axis",
            ATTR_TYPE_I64,
            doc="Axis whose extent is doubled in the result.",
        ),
    ],
    constraints=[
        SameType("even", "odd"),
        SameElementType("even", "result"),
        DimIndexInBounds("even", "axis"),
    ],
    verify="loom_vector_interleave_verify",
    facts="loom_vector_interleave_facts",
    traits=[PURE],
    format=[
        TemplateParam("axis"),
        Ref("even"),
        COMMA,
        Ref("odd"),
        COLON,
        TypeOf("even"),
        COMMA,
        TypeOf("odd"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%r = vector.interleave<0> %lo, %hi : vector<16xi8>, vector<16xi8> -> vector<32xi8>",
        "%r = vector.interleave<1> %a, %b : vector<4x8xi8>, vector<4x8xi8> -> vector<4x16xi8>",
    ],
)

vector_deinterleave = Op(
    "vector.deinterleave",
    group=vector_ops,
    doc=(
        "Split one vector along the template axis into two same-typed "
        "results. The first result receives even coordinates along that axis, "
        "the second receives odd coordinates, and each result extent on that "
        "axis is half of the source extent."
    ),
    operands=[Operand("source", VECTOR)],
    results=[Result("results", VECTOR, variadic=True, doc="Even-position result followed by odd-position result.")],
    attrs=[
        AttrDef(
            "axis",
            ATTR_TYPE_I64,
            doc="Axis whose extent is halved in each result.",
        ),
    ],
    constraints=[
        SameType("results"),
        SameElementType("source", "results"),
        DimIndexInBounds("source", "axis"),
    ],
    verify="loom_vector_deinterleave_verify",
    facts="loom_vector_deinterleave_facts",
    traits=[PURE],
    format=[
        TemplateParam("axis"),
        Ref("source"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultTypeList("results", parens=False),
    ],
    examples=[
        "%lo, %hi = vector.deinterleave<0> %r : vector<32xi8> -> vector<16xi8>, vector<16xi8>",
        "%a, %b = vector.deinterleave<1> %r : vector<4x16xi8> -> vector<4x8xi8>, vector<4x8xi8>",
    ],
)


vector_table_lookup = Op(
    "vector.table.lookup",
    group=vector_ops,
    contracts=[ContractFamily.VECTOR_TABLE_LOOKUP],
    doc=(
        "Select values from a rank-1 register table using integer index "
        "lanes. Each result lane reads table[indices lane]; the result shape "
        "matches the index vector shape and the result element type matches "
        "the table element type."
    ),
    operands=[
        Operand("table", VECTOR, doc="Rank-1 register table containing selectable lane values."),
        Operand("indices", VECTOR, doc="Index vector selecting one table lane for each result lane."),
    ],
    results=[Result("result", VECTOR)],
    constraints=[
        HasRankOneVector("table"),
        HasIndexOrNonI1IntegerElement("indices"),
        SameElementType("table", "result"),
        SameShape("indices", "result"),
    ],
    verify="loom_vector_table_lookup_verify",
    traits=[PURE],
    facts="loom_vector_table_lookup_facts",
    format=[
        Ref("table"),
        GLUE,
        LBRACKET,
        Ref("indices"),
        RBRACKET,
        COLON,
        TypeOf("table"),
        COMMA,
        TypeOf("indices"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%values = vector.table.lookup %grid[%codes] : vector<16xf16>, vector<32xi8> -> vector<32xf16>",
        "%values = vector.table.lookup %grid[%codes] : vector<16xf32>, vector<4x8xi8> -> vector<4x8xf32>",
    ],
)

vector_table_quantize = Op(
    "vector.table.quantize",
    group=vector_ops,
    doc=(
        "Map floating-point lanes to integer ordinal code lanes using an "
        "ordered rank-1 threshold table. For each input lane, the result code "
        "is the selected quantization bin; nan and tie attributes make NaN "
        "and threshold equality behavior explicit."
    ),
    operands=[
        Operand("input", VECTOR, doc="Floating-point lanes to quantize."),
        Operand("thresholds", VECTOR, doc="Rank-1 ordered floating-point threshold table."),
    ],
    results=[Result("result", VECTOR, doc="Unsigned ordinal code lanes stored in integer elements.")],
    attrs=[
        AttrDef("nan", ATTR_TYPE_ENUM, enum_def=QuantizeNaN, doc="Required NaN handling policy."),
        AttrDef("tie", ATTR_TYPE_ENUM, enum_def=QuantizeTie, doc="Required threshold equality policy."),
    ],
    constraints=[
        HasFloatElement("input"),
        HasFloatElement("thresholds"),
        HasRankOneVector("thresholds"),
        HasIntegerElement("result"),
        SameElementType("input", "thresholds"),
        SameShape("input", "result"),
    ],
    verify="loom_vector_table_quantize_verify",
    traits=[PURE],
    facts="loom_vector_table_quantize_facts",
    format=[
        Ref("input"),
        COMMA,
        Ref("thresholds"),
        AttrDict(),
        COLON,
        TypeOf("input"),
        COMMA,
        TypeOf("thresholds"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%codes = vector.table.quantize %values, %thresholds {nan = zero, tie = lower} : vector<32xf32>, vector<15xf32> -> vector<32xi8>",
    ],
)

vector_transform = Op(
    "vector.transform",
    group=vector_ops,
    doc=(
        "Apply an explicit numeric transform descriptor to vector register "
        "lanes. The transform operand is an encoding<transform> value that "
        "names the numeric mapping, such as scale/zero-point decode, "
        "whitening, or projection; verifier rules keep supported transform "
        "families and shape-changing parameters explicit. Hadamard-like "
        "families act along the last axis. `hadamard_sign` applies either an "
        "explicit per-lane sign table or deterministic seed-derived signs "
        "from the low bit of SplitMix64(seed + input lane) before the "
        "Hadamard. `sign_permute_hadamard` applies explicit signs to source "
        "lanes, gathers lanes through the explicit permutation vector, then "
        "applies the Hadamard."
    ),
    operands=[
        Operand("source", VECTOR, doc="Vector lanes to transform."),
        Operand("transform", ENCODING_TRANSFORM, doc="Numeric transform descriptor."),
    ],
    results=[Result("result", VECTOR, doc="Transformed vector lanes.")],
    constraints=[
        HasFloatElement("source"),
        HasFloatElement("result"),
        SameElementType("source", "result"),
    ],
    verify="loom_vector_transform_verify",
    facts="loom_vector_transform_facts",
    traits=[PURE],
    format=[
        Ref("source"),
        COMMA,
        Ref("transform"),
        COLON,
        TypeOf("source"),
        COMMA,
        TypeOf("transform"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%r = vector.transform %v, %xf : vector<128xf32>, encoding<transform> -> vector<128xf32>",
        "%sketch = vector.transform %v, %jl : vector<128xf32>, encoding<transform> -> vector<64xf32>",
    ],
)

vector_decode = Op(
    "vector.decode",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Decode physical encoded vector payload lanes into logical numeric "
        "lanes using an explicit encoding<schema> witness. The schema value "
        "carries compact representation facts such as element format, block "
        "extent, packing order, rounding, and sparsity kind. Bulk or "
        "runtime-varying interpretation data such as scales, zero-points, "
        "codebook rows, sparse metadata, residual streams, signs, and online "
        "amax values stay visible as auxiliary SSA operands instead of being "
        "hidden inside the encoding value."
    ),
    operands=[
        Operand("payload", VECTOR, doc="Physical encoded payload lanes."),
        Operand("schema", ENCODING_SCHEMA, doc="Schema witness for interpreting the payload."),
        Operand("auxiliary", VECTOR, variadic=True, doc="Explicit scale, table, metadata, or online state operands."),
    ],
    results=[Result("result", VECTOR, doc="Decoded logical numeric lanes.")],
    attrs=[
        AttrDef(
            "auxiliary_names",
            ATTR_TYPE_DICT,
            optional=True,
            doc="Sorted auxiliary operand keys mapped to auxiliary operand ordinals.",
        ),
    ],
    verify="loom_vector_decode_verify",
    canonicalize="loom_vector_decode_canonicalize",
    traits=[PURE, REFINABLE_RESULT_TYPE_REFS],
    format=[
        Ref("payload"),
        kw("using"),
        Ref("schema"),
        OperandDict("auxiliary", "auxiliary_names"),
        COLON,
        TypeOf("payload"),
        COMMA,
        TypeOf("schema"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%values = vector.decode %payload using %schema {scale = %scale : vector<1xf16>} : vector<4xi32>, encoding<schema> -> vector<32xf32>",
        "%values = vector.decode %payload using %schema : vector<4xi32>, encoding<schema> -> vector<32xf32>",
    ],
)

vector_encode = Op(
    "vector.encode",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Encode logical numeric vector lanes into a physical encoded payload "
        "using an explicit encoding<schema> witness. This is the inverse "
        "boundary to vector.decode for runtime-created encoded data such as "
        "KV-cache pages, online quantization records, and target prepack "
        "buffers. Rounding, saturation, affine terms, table lookup policy, "
        "and sparse/codebook structure are described by schema facts; the "
        "actual scale/table/metadata/state values are ordinary auxiliary SSA "
        "operands."
    ),
    operands=[
        Operand("source", VECTOR, doc="Logical numeric lanes to encode."),
        Operand("schema", ENCODING_SCHEMA, doc="Schema witness for producing the payload."),
        Operand("auxiliary", VECTOR, variadic=True, doc="Explicit scale, table, metadata, or online state operands."),
    ],
    results=[Result("result", VECTOR, doc="Physical encoded payload lanes.")],
    attrs=[
        AttrDef(
            "auxiliary_names",
            ATTR_TYPE_DICT,
            optional=True,
            doc="Sorted auxiliary operand keys mapped to auxiliary operand ordinals.",
        ),
    ],
    verify="loom_vector_encode_verify",
    traits=[PURE, REFINABLE_RESULT_TYPE_REFS],
    format=[
        Ref("source"),
        kw("using"),
        Ref("schema"),
        OperandDict("auxiliary", "auxiliary_names"),
        COLON,
        TypeOf("source"),
        COMMA,
        TypeOf("schema"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%payload = vector.encode %values using %schema {amax = %amax : vector<1xf32>, scale = %scale : vector<1xf16>} : vector<32xf32>, encoding<schema> -> vector<4xi32>",
        "%payload = vector.encode %values using %schema : vector<32xf32>, encoding<schema> -> vector<4xi32>",
    ],
)

vector_fragment = Op(
    "vector.fragment",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Attach a matrix-fragment interpretation to a physical vector value "
        "without changing the physical vector type. The role selects how the "
        "two shape operands are interpreted: lhs is [m, k], rhs is [k, n], "
        "and init/result are [m, n]. Dense/default fragments need only the "
        "data value and shape SSA values. Encoded fragments carry schema and "
        "scale/table/sparse metadata values in the keyed using dictionary so "
        "bulk runtime data remains ordinary SSA while lowering can consume a "
        "compact resolved fragment fact."
    ),
    operands=[
        Operand("data", VECTOR, doc="Physical vector lanes or packed registers."),
        Operand("rows", INDEX, doc="Logical matrix row count for this fragment role."),
        Operand("columns", INDEX, doc="Logical matrix column count for this fragment role."),
        Operand(
            "params",
            ANY,
            variadic=True,
            doc="Optional schema, scale, table, metadata, or online state operands.",
        ),
    ],
    results=[Result("result", VECTOR, doc="The same physical vector value with fragment facts.")],
    attrs=[
        AttrDef("role", ATTR_TYPE_ENUM, enum_def=VectorFragmentRole),
        AttrDef(
            "param_names",
            ATTR_TYPE_DICT,
            optional=True,
            doc="Sorted fragment parameter names mapped to parameter operand ordinals.",
        ),
        AttrDef(
            "predicates",
            ATTR_TYPE_PREDICATE_LIST,
            optional=True,
            doc="Optional local facts constraining fragment shape or parameter values.",
        ),
    ],
    constraints=[SameType("data", "result")],
    verify="loom_vector_fragment_verify",
    facts="loom_vector_fragment_facts",
    traits=[PURE, REFINABLE_RESULT_TYPE_REFS, VALUE_ALIAS],
    format=[
        TemplateParam("role"),
        Ref("data"),
        kw("shape"),
        LBRACKET,
        Ref("rows"),
        COMMA,
        Ref("columns"),
        RBRACKET,
        OptionalGroup(
            [kw("using"), OperandDict("params", "param_names")],
            anchor="params",
        ),
        OptionalGroup(
            [kw("where"), PredicateList("predicates")],
            anchor="predicates",
        ),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%fragment = vector.fragment<lhs> %payload shape [%m, %k] : vector<4xi32>",
        "%fragment = vector.fragment<rhs> %payload shape [%k, %n] using {scale = %scale : vector<1xf16>, schema = %schema : encoding<schema>} : vector<4xi32>",
    ],
)


# ============================================================================
# Memory
# ============================================================================


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
            doc="Static logical origin indices with INT64_MIN sentinels for dynamics.",
        ),
    ]


def _memory_access_interface(
    *,
    value: str | None = None,
    expected: str | None = None,
    replacement: str | None = None,
    mask: str | None = None,
    passthrough: str | None = None,
    offsets: str | None = None,
    atomic_kind: str | None = None,
    atomic_ordering: str | None = None,
    atomic_success_ordering: str | None = None,
    atomic_failure_ordering: str | None = None,
    atomic_scope: str | None = None,
) -> MemoryAccessInterface:
    return MemoryAccessInterface(
        value=value,
        expected=expected,
        replacement=replacement,
        mask=mask,
        passthrough=passthrough,
        offsets=offsets,
        cache_scope="cache_scope",
        cache_temporal="cache_temporal",
        atomic_kind=atomic_kind,
        atomic_ordering=atomic_ordering,
        atomic_success_ordering=atomic_success_ordering,
        atomic_failure_ordering=atomic_failure_ordering,
        atomic_scope=atomic_scope,
    )


def _atomic_memory_access_interface(
    *,
    value: str | None = None,
    mask: str | None = None,
    passthrough: str | None = None,
) -> MemoryAccessInterface:
    return _memory_access_interface(
        value=value,
        mask=mask,
        passthrough=passthrough,
        offsets="offsets",
        atomic_kind="kind",
        atomic_ordering="ordering",
        atomic_scope="scope",
    )


vector_fragment_load = Op(
    "vector.fragment.load",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Load a target-shaped matrix fragment payload from a typed view at a "
        "full-rank logical origin. Unlike vector.load, the result vector shape "
        "is the physical fragment payload selected by role, logical matrix "
        "shape, view layout, and target legality; it is not an ordinary "
        "trailing-axis footprint of the view. The result carries fragment "
        "facts directly so vector.mma can consume it without a separate "
        "vector.fragment wrapper."
    ),
    operands=[
        Operand("view", VIEW, doc="Typed source view holding logical matrix data."),
        Operand("rows", INDEX, doc="Logical matrix row count for this fragment role."),
        Operand("columns", INDEX, doc="Logical matrix column count for this fragment role."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    results=[Result("result", VECTOR, doc="Loaded physical matrix fragment payload.")],
    attrs=[
        AttrDef("role", ATTR_TYPE_ENUM, enum_def=VectorFragmentRole),
        *_indexed_memory_attrs(),
    ],
    constraints=[SameElementType("view", "result")],
    traits=[REFINABLE_RESULT_TYPE_REFS],
    effects=[Reads("view")],
    interfaces=[_memory_access_interface()],
    verify="loom_vector_fragment_load_verify",
    facts="loom_vector_fragment_load_facts",
    format=[
        TemplateParam("role"),
        Ref("view"),
        IndexList("indices", "static_indices"),
        kw("shape"),
        LBRACKET,
        Ref("rows"),
        COMMA,
        Ref("columns"),
        RBRACKET,
        AttrDict(),
        COLON,
        TypeOf("view"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%lhs = vector.fragment.load<lhs> %a[%row, %k0] shape [%m, %k] : view<[%M]x[%K]xf16, %layout> -> vector<16xf16>",
    ],
)

vector_fragment_store = Op(
    "vector.fragment.store",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Store a target-shaped matrix fragment payload into a typed view at a "
        "full-rank logical origin. The value is interpreted as the physical "
        "payload for the given fragment role and logical matrix shape; the "
        "store is therefore a matrix-fragment movement boundary, not an "
        "ordinary vector.store footprint."
    ),
    operands=[
        Operand("value", VECTOR, doc="Physical matrix fragment payload to store."),
        Operand("view", VIEW, doc="Typed destination view for logical matrix data."),
        Operand("rows", INDEX, doc="Logical matrix row count for this fragment role."),
        Operand("columns", INDEX, doc="Logical matrix column count for this fragment role."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    attrs=[
        AttrDef("role", ATTR_TYPE_ENUM, enum_def=VectorFragmentRole),
        *_indexed_memory_attrs(),
    ],
    constraints=[SameElementType("value", "view")],
    effects=[Writes("view")],
    interfaces=[_memory_access_interface(value="value")],
    verify="loom_vector_fragment_store_verify",
    format=[
        TemplateParam("role"),
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        kw("shape"),
        LBRACKET,
        Ref("rows"),
        COMMA,
        Ref("columns"),
        RBRACKET,
        AttrDict(),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("view"),
    ],
    examples=[
        "vector.fragment.store<result> %acc, %c[%row, %col] shape [%m, %n] : vector<8xf32>, view<[%M]x[%N]xf32, %layout>",
    ],
)


vector_load = Op(
    "vector.load",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Load a vector footprint from a typed view at a full-rank logical "
        "origin. The index list addresses the origin in view coordinates; "
        "vector axes map onto the trailing view axes, so leading view axes "
        "select a slice and trailing axes describe the loaded footprint."
    ),
    operands=[
        Operand("view", VIEW, doc="Typed source view."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    results=[Result("result", VECTOR, doc="Loaded vector value.")],
    attrs=_indexed_memory_attrs(),
    constraints=[SameElementType("view", "result")],
    traits=[REFINABLE_RESULT_TYPE_REFS],
    effects=[Reads("view")],
    interfaces=[_memory_access_interface()],
    verify="loom_vector_load_verify",
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
        "%v = vector.load %view[%row, %col] : view<[%m]x[%n]xf32, %layout> -> vector<4x8xf32>",
    ],
)

vector_store = Op(
    "vector.store",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Store a vector footprint into a typed view at a full-rank logical "
        "origin. The index list addresses the origin in view coordinates; "
        "vector axes map onto the trailing view axes, matching vector.load."
    ),
    operands=[
        Operand("value", VECTOR, doc="Vector value to store."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    attrs=_indexed_memory_attrs(),
    constraints=[SameElementType("value", "view")],
    effects=[Writes("view")],
    interfaces=[_memory_access_interface(value="value")],
    verify="loom_vector_store_verify",
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
        "vector.store %v, %view[%row, %col] : vector<4x8xf32>, view<[%m]x[%n]xf32, %layout>",
    ],
)

vector_load_mask = Op(
    "vector.load.mask",
    group=vector_ops,
    doc=(
        "Masked vector load from a typed view. Mask lanes with true values "
        "perform the same access as vector.load, while false lanes do not "
        "access memory and instead take the corresponding passthrough lane."
    ),
    operands=[
        Operand("view", VIEW, doc="Typed source view."),
        Operand("mask", VECTOR, doc="i1 vector mask selecting loaded lanes."),
        Operand("passthrough", VECTOR, doc="Value used for masked-off lanes."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    results=[Result("result", VECTOR, doc="Loaded vector value.")],
    attrs=_indexed_memory_attrs(),
    constraints=[
        HasI1Element("mask"),
        SameElementType("view", "passthrough", "result"),
        SameShape("mask", "passthrough", "result"),
        SameType("passthrough", "result"),
    ],
    effects=[Reads("view")],
    interfaces=[_memory_access_interface(mask="mask", passthrough="passthrough")],
    verify="loom_vector_load_mask_verify",
    canonicalize="loom_vector_masked_memory_canonicalize",
    format=[
        Ref("view"),
        IndexList("indices", "static_indices"),
        COMMA,
        Ref("mask"),
        COMMA,
        Ref("passthrough"),
        AttrDict(),
        COLON,
        TypeOf("view"),
        COMMA,
        TypeOf("mask"),
        COMMA,
        ResultType("result"),
    ],
    examples=[
        "%v = vector.load.mask %view[%row, %col], %mask, %old : view<[%m]x[%n]xf32, %layout>, vector<4x8xi1>, vector<4x8xf32>",
    ],
)

vector_store_mask = Op(
    "vector.store.mask",
    group=vector_ops,
    doc=("Masked vector store into a typed view. True mask lanes store the corresponding value lane, and false mask lanes do not access memory and leave the destination unchanged."),
    operands=[
        Operand("value", VECTOR, doc="Vector value to store."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("mask", VECTOR, doc="i1 vector mask selecting stored lanes."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    attrs=_indexed_memory_attrs(),
    constraints=[
        HasI1Element("mask"),
        SameElementType("value", "view"),
        SameShape("mask", "value"),
    ],
    effects=[Writes("view")],
    interfaces=[_memory_access_interface(value="value", mask="mask")],
    verify="loom_vector_store_mask_verify",
    canonicalize="loom_vector_masked_memory_canonicalize",
    format=[
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        COMMA,
        Ref("mask"),
        AttrDict(),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("view"),
        COMMA,
        TypeOf("mask"),
    ],
    examples=[
        "vector.store.mask %v, %view[%row, %col], %mask : vector<4x8xf32>, view<[%m]x[%n]xf32, %layout>, vector<4x8xi1>",
    ],
)

vector_load_expand = Op(
    "vector.load.expand",
    group=vector_ops,
    doc=(
        "Rank-1 masked expand load from consecutive view elements. Active "
        "lanes consume memory densely in increasing lane order; inactive "
        "lanes do not consume memory and take the corresponding passthrough "
        "lane."
    ),
    operands=[
        Operand("view", VIEW, doc="Typed source view."),
        Operand("mask", VECTOR, doc="i1 rank-1 vector mask selecting loaded lanes."),
        Operand("passthrough", VECTOR, doc="Value used for inactive lanes."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    results=[Result("result", VECTOR, doc="Expanded loaded vector value.")],
    attrs=_indexed_memory_attrs(),
    constraints=[
        HasRankOneVector("result"),
        HasI1Element("mask"),
        SameElementType("view", "passthrough", "result"),
        SameShape("mask", "passthrough", "result"),
        SameType("passthrough", "result"),
    ],
    effects=[Reads("view")],
    interfaces=[_memory_access_interface(mask="mask", passthrough="passthrough")],
    verify="loom_vector_load_expand_verify",
    canonicalize="loom_vector_masked_memory_canonicalize",
    format=[
        Ref("view"),
        IndexList("indices", "static_indices"),
        COMMA,
        Ref("mask"),
        COMMA,
        Ref("passthrough"),
        AttrDict(),
        COLON,
        TypeOf("view"),
        COMMA,
        TypeOf("mask"),
        COMMA,
        ResultType("result"),
    ],
    examples=[
        "%v = vector.load.expand %view[%row, %col], %mask, %old : view<[%m]x[%n]xf32, %layout>, vector<4xi1>, vector<4xf32>",
    ],
)

vector_store_compress = Op(
    "vector.store.compress",
    group=vector_ops,
    doc=("Rank-1 masked compress store to consecutive view elements. Active lanes write densely in increasing lane order; inactive lanes do not produce memory elements."),
    operands=[
        Operand("value", VECTOR, doc="Rank-1 vector value to store."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("mask", VECTOR, doc="i1 rank-1 vector mask selecting stored lanes."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    attrs=_indexed_memory_attrs(),
    constraints=[
        HasRankOneVector("value"),
        HasI1Element("mask"),
        SameElementType("value", "view"),
        SameShape("mask", "value"),
    ],
    effects=[Writes("view")],
    interfaces=[_memory_access_interface(value="value", mask="mask")],
    verify="loom_vector_store_compress_verify",
    canonicalize="loom_vector_masked_memory_canonicalize",
    format=[
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        COMMA,
        Ref("mask"),
        AttrDict(),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("view"),
        COMMA,
        TypeOf("mask"),
    ],
    examples=[
        "vector.store.compress %v, %view[%row, %col], %mask : vector<4xf32>, view<[%m]x[%n]xf32, %layout>, vector<4xi1>",
    ],
)

vector_gather = Op(
    "vector.gather",
    group=vector_ops,
    doc=(
        "Gather a vector from per-lane signed logical offsets added to the "
        "last view axis of a full-rank view origin. Each result lane reads "
        "origin with the final coordinate adjusted by offsets[lane]; the "
        "offset vector shape matches the result shape."
    ),
    operands=[
        Operand("view", VIEW, doc="Typed source view."),
        Operand("offsets", VECTOR, doc="Per-lane signed element offsets from the logical origin."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    results=[Result("result", VECTOR, doc="Gathered vector value.")],
    attrs=_indexed_memory_attrs(),
    constraints=[
        HasIndexOrNonI1IntegerElement("offsets"),
        SameElementType("view", "result"),
        SameShape("offsets", "result"),
    ],
    effects=[Reads("view")],
    interfaces=[_memory_access_interface(offsets="offsets")],
    verify="loom_vector_gather_verify",
    canonicalize="loom_vector_gather_scatter_canonicalize",
    format=[
        Ref("view"),
        IndexList("indices", "static_indices"),
        GLUE,
        LBRACKET,
        Ref("offsets"),
        RBRACKET,
        AttrDict(),
        COLON,
        TypeOf("view"),
        COMMA,
        TypeOf("offsets"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%v = vector.gather %view[%row, %col][%offsets] : view<[%m]x[%n]xf32, %layout>, vector<4xindex> -> vector<4xf32>",
    ],
)

vector_scatter = Op(
    "vector.scatter",
    group=vector_ops,
    doc=(
        "Non-atomic scatter of a vector to per-lane signed logical offsets "
        "added to the last view axis of a full-rank view origin. Each lane "
        "writes origin with the final coordinate adjusted by offsets[lane], "
        "and active lane addresses must be distinct because no atomic "
        "conflict resolution is implied."
    ),
    operands=[
        Operand("value", VECTOR, doc="Vector value to store."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("offsets", VECTOR, doc="Per-lane signed element offsets from the logical origin."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    attrs=_indexed_memory_attrs(),
    constraints=[
        HasIndexOrNonI1IntegerElement("offsets"),
        SameElementType("value", "view"),
        SameShape("offsets", "value"),
    ],
    effects=[Writes("view")],
    interfaces=[_memory_access_interface(value="value", offsets="offsets")],
    verify="loom_vector_scatter_verify",
    canonicalize="loom_vector_gather_scatter_canonicalize",
    format=[
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        GLUE,
        LBRACKET,
        Ref("offsets"),
        RBRACKET,
        AttrDict(),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("view"),
        COMMA,
        TypeOf("offsets"),
    ],
    examples=[
        "vector.scatter %v, %view[%row, %col][%offsets] : vector<4xf32>, view<[%m]x[%n]xf32, %layout>, vector<4xindex>",
    ],
)

vector_gather_mask = Op(
    "vector.gather.mask",
    group=vector_ops,
    doc=(
        "Masked vector gather from per-lane signed logical offsets added to "
        "the last view axis. True mask lanes read the adjusted coordinate, "
        "while false mask lanes do not access memory and take the "
        "corresponding passthrough lane."
    ),
    operands=[
        Operand("view", VIEW, doc="Typed source view."),
        Operand("offsets", VECTOR, doc="Per-lane signed element offsets from the logical origin."),
        Operand("mask", VECTOR, doc="i1 vector mask selecting gathered lanes."),
        Operand("passthrough", VECTOR, doc="Value used for masked-off lanes."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    results=[Result("result", VECTOR, doc="Gathered vector value.")],
    attrs=_indexed_memory_attrs(),
    constraints=[
        HasI1Element("mask"),
        HasIndexOrNonI1IntegerElement("offsets"),
        SameElementType("view", "passthrough", "result"),
        SameShape("offsets", "mask", "passthrough", "result"),
        SameType("passthrough", "result"),
    ],
    effects=[Reads("view")],
    interfaces=[_memory_access_interface(offsets="offsets", mask="mask", passthrough="passthrough")],
    verify="loom_vector_gather_mask_verify",
    canonicalize="loom_vector_masked_memory_canonicalize",
    format=[
        Ref("view"),
        IndexList("indices", "static_indices"),
        GLUE,
        LBRACKET,
        Ref("offsets"),
        RBRACKET,
        COMMA,
        Ref("mask"),
        COMMA,
        Ref("passthrough"),
        AttrDict(),
        COLON,
        TypeOf("view"),
        COMMA,
        TypeOf("offsets"),
        COMMA,
        TypeOf("mask"),
        COMMA,
        ResultType("result"),
    ],
    examples=[
        "%v = vector.gather.mask %view[%row, %col][%offsets], %mask, %old : view<[%m]x[%n]xf32, %layout>, vector<4xindex>, vector<4xi1>, vector<4xf32>",
    ],
)

vector_scatter_mask = Op(
    "vector.scatter.mask",
    group=vector_ops,
    doc=(
        "Masked non-atomic scatter. True mask lanes write the full-rank origin with the last coordinate adjusted by offsets[lane], false mask lanes do not access memory, and active lane addresses must be distinct."
    ),
    operands=[
        Operand("value", VECTOR, doc="Vector value to store."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("offsets", VECTOR, doc="Per-lane signed element offsets from the logical origin."),
        Operand("mask", VECTOR, doc="i1 vector mask selecting stored lanes."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    attrs=_indexed_memory_attrs(),
    constraints=[
        HasI1Element("mask"),
        HasIndexOrNonI1IntegerElement("offsets"),
        SameElementType("value", "view"),
        SameShape("offsets", "mask", "value"),
    ],
    effects=[Writes("view")],
    interfaces=[_memory_access_interface(value="value", offsets="offsets", mask="mask")],
    verify="loom_vector_scatter_mask_verify",
    canonicalize="loom_vector_masked_memory_canonicalize",
    format=[
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        GLUE,
        LBRACKET,
        Ref("offsets"),
        RBRACKET,
        COMMA,
        Ref("mask"),
        AttrDict(),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("view"),
        COMMA,
        TypeOf("offsets"),
        COMMA,
        TypeOf("mask"),
    ],
    examples=[
        "vector.scatter.mask %v, %view[%row, %col][%offsets], %mask : vector<4xf32>, view<[%m]x[%n]xf32, %layout>, vector<4xindex>, vector<4xi1>",
    ],
)


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
            doc="Static logical origin indices with INT64_MIN sentinels for dynamics.",
        ),
    ]


def _atomic_cmpxchg_memory_attrs() -> list[AttrDef]:
    return [
        AttrDef(
            "success_ordering",
            ATTR_TYPE_ENUM,
            enum_def=AtomicOrdering,
            doc="Memory ordering used when a lane compare-exchange succeeds.",
        ),
        AttrDef(
            "failure_ordering",
            ATTR_TYPE_ENUM,
            enum_def=AtomicOrdering,
            doc="Memory ordering used when a lane compare-exchange fails.",
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
            doc="Static logical origin indices with INT64_MIN sentinels for dynamics.",
        ),
    ]


vector_atomic_reduce = Op(
    "vector.atomic.reduce",
    group=vector_ops,
    doc=(
        "Atomic no-result scatter reduction/update into per-lane signed "
        "element offsets. Each lane atomically combines its value into origin "
        "+ offsets[lane]; duplicate active addresses are valid and are "
        "serialized by the required ordering and scope attributes."
    ),
    operands=[
        Operand("value", VECTOR, doc="Vector contribution for each lane."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("offsets", VECTOR, doc="Per-lane signed element offsets from the logical origin."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    attrs=_atomic_memory_attrs(),
    constraints=[
        HasIndexOrNonI1IntegerElement("offsets"),
        SameElementType("value", "view"),
        SameShape("offsets", "value"),
    ],
    effects=[ReadWrites("view")],
    interfaces=[_atomic_memory_access_interface(value="value")],
    verify="loom_vector_atomic_reduce_verify",
    format=[
        TemplateParam("kind"),
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        GLUE,
        LBRACKET,
        Ref("offsets"),
        RBRACKET,
        AttrDict(),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("view"),
        COMMA,
        TypeOf("offsets"),
    ],
    examples=[
        "vector.atomic.reduce<addi> %v, %view[%row, %col][%offsets] {ordering = relaxed, scope = workgroup} : vector<4xi32>, view<[%m]x[%n]xi32, %layout>, vector<4xindex>",
    ],
)

vector_atomic_reduce_mask = Op(
    "vector.atomic.reduce.mask",
    group=vector_ops,
    doc=("Masked atomic no-result scatter reduction/update. True mask lanes perform vector.atomic.reduce, while false mask lanes do not access memory."),
    operands=[
        Operand("value", VECTOR, doc="Vector contribution for each lane."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("offsets", VECTOR, doc="Per-lane signed element offsets from the logical origin."),
        Operand("mask", VECTOR, doc="i1 vector mask selecting active atomic lanes."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    attrs=_atomic_memory_attrs(),
    constraints=[
        HasI1Element("mask"),
        HasIndexOrNonI1IntegerElement("offsets"),
        SameElementType("value", "view"),
        SameShape("offsets", "mask", "value"),
    ],
    effects=[ReadWrites("view")],
    interfaces=[_atomic_memory_access_interface(value="value", mask="mask")],
    verify="loom_vector_atomic_reduce_mask_verify",
    canonicalize="loom_vector_masked_memory_canonicalize",
    format=[
        TemplateParam("kind"),
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        GLUE,
        LBRACKET,
        Ref("offsets"),
        RBRACKET,
        COMMA,
        Ref("mask"),
        AttrDict(),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("view"),
        COMMA,
        TypeOf("offsets"),
        COMMA,
        TypeOf("mask"),
    ],
    examples=[
        "vector.atomic.reduce.mask<addf> %v, %view[%row, %col][%offsets], %mask {ordering = relaxed, scope = device} : vector<4xf32>, view<[%m]x[%n]xf32, %layout>, vector<4xindex>, vector<4xi1>",
    ],
)

vector_atomic_rmw = Op(
    "vector.atomic.rmw",
    group=vector_ops,
    doc=(
        "Atomic read-modify-write at per-lane signed element offsets. Each "
        "lane atomically combines its value with origin + offsets[lane] and "
        "the result lane is the old memory value observed by that atomic "
        "operation."
    ),
    operands=[
        Operand("value", VECTOR, doc="Vector update value for each lane."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("offsets", VECTOR, doc="Per-lane signed element offsets from the logical origin."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    results=[Result("result", VECTOR, doc="Old memory values read by the atomic operations.")],
    attrs=_atomic_memory_attrs(),
    constraints=[
        HasIndexOrNonI1IntegerElement("offsets"),
        SameElementType("value", "view", "result"),
        SameShape("offsets", "value", "result"),
        SameType("value", "result"),
    ],
    effects=[ReadWrites("view")],
    interfaces=[_atomic_memory_access_interface(value="value")],
    verify="loom_vector_atomic_rmw_verify",
    format=[
        TemplateParam("kind"),
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        GLUE,
        LBRACKET,
        Ref("offsets"),
        RBRACKET,
        AttrDict(),
        COLON,
        ResultType("result"),
        COMMA,
        TypeOf("view"),
        COMMA,
        TypeOf("offsets"),
    ],
    examples=[
        "%old = vector.atomic.rmw<addi> %v, %view[%row, %col][%offsets] {ordering = relaxed, scope = workgroup} : vector<4xi32>, view<[%m]x[%n]xi32, %layout>, vector<4xindex>",
    ],
)

vector_atomic_rmw_mask = Op(
    "vector.atomic.rmw.mask",
    group=vector_ops,
    doc=("Masked atomic read-modify-write. True mask lanes perform vector.atomic.rmw, while false mask lanes do not access memory and take the corresponding passthrough lane in the result."),
    operands=[
        Operand("value", VECTOR, doc="Vector update value for each lane."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("offsets", VECTOR, doc="Per-lane signed element offsets from the logical origin."),
        Operand("mask", VECTOR, doc="i1 vector mask selecting active atomic lanes."),
        Operand("passthrough", VECTOR, doc="Value used for masked-off result lanes."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    results=[Result("result", VECTOR, doc="Old memory values for active lanes and passthrough for inactive lanes.")],
    attrs=_atomic_memory_attrs(),
    constraints=[
        HasI1Element("mask"),
        HasIndexOrNonI1IntegerElement("offsets"),
        SameElementType("value", "view", "passthrough", "result"),
        SameShape("offsets", "mask", "value"),
        SameType("value", "passthrough", "result"),
    ],
    effects=[ReadWrites("view")],
    interfaces=[_atomic_memory_access_interface(value="value", mask="mask", passthrough="passthrough")],
    verify="loom_vector_atomic_rmw_mask_verify",
    canonicalize="loom_vector_masked_memory_canonicalize",
    format=[
        TemplateParam("kind"),
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        GLUE,
        LBRACKET,
        Ref("offsets"),
        RBRACKET,
        COMMA,
        Ref("mask"),
        COMMA,
        Ref("passthrough"),
        AttrDict(),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("view"),
        COMMA,
        TypeOf("offsets"),
        COMMA,
        TypeOf("mask"),
        COMMA,
        ResultType("result"),
    ],
    examples=[
        "%old = vector.atomic.rmw.mask<addf> %v, %view[%row, %col][%offsets], %mask, %passthrough {ordering = relaxed, scope = device} : vector<4xf32>, view<[%m]x[%n]xf32, %layout>, vector<4xindex>, vector<4xi1>, vector<4xf32>",
    ],
)

vector_atomic_cmpxchg = Op(
    "vector.atomic.cmpxchg",
    group=vector_ops,
    doc=(
        "Atomic compare-exchange at per-lane signed element offsets. Each "
        "lane compares origin + offsets[lane] with expected[lane], writes "
        "replacement[lane] on success, and returns the old memory value. "
        "Success lanes are derived by comparing old == expected."
    ),
    operands=[
        Operand("expected", VECTOR, doc="Expected memory value for each lane."),
        Operand("replacement", VECTOR, doc="Replacement value written for each successful lane."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("offsets", VECTOR, doc="Per-lane signed element offsets from the logical origin."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    results=[Result("old", VECTOR, doc="Old memory values read by the atomic operations.")],
    attrs=_atomic_cmpxchg_memory_attrs(),
    constraints=[
        HasIndexOrNonI1IntegerElement("expected"),
        HasIndexOrNonI1IntegerElement("offsets"),
        SameElementType("expected", "replacement", "view", "old"),
        SameShape("offsets", "expected", "replacement", "old"),
        SameType("expected", "replacement", "old"),
    ],
    effects=[ReadWrites("view")],
    interfaces=[
        _memory_access_interface(
            expected="expected",
            replacement="replacement",
            offsets="offsets",
            atomic_success_ordering="success_ordering",
            atomic_failure_ordering="failure_ordering",
            atomic_scope="scope",
        )
    ],
    verify="loom_vector_atomic_cmpxchg_verify",
    format=[
        Ref("expected"),
        COMMA,
        Ref("replacement"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        GLUE,
        LBRACKET,
        Ref("offsets"),
        RBRACKET,
        AttrDict(),
        COLON,
        TypeOf("expected"),
        COMMA,
        TypeOf("view"),
        COMMA,
        TypeOf("offsets"),
        ARROW,
        ResultType("old"),
    ],
    examples=[
        "%old = vector.atomic.cmpxchg %expected, %replacement, %view[%row, %col][%offsets] {success_ordering = acq_rel, failure_ordering = acquire, scope = workgroup} : vector<4xi32>, view<[%m]x[%n]xi32, %layout>, vector<4xindex> -> vector<4xi32>",
    ],
)


# ============================================================================
# Selection and comparison
# ============================================================================

vector_select = Op(
    "vector.select",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Lanewise select from two same-typed vector values using an i1 mask vector. True condition lanes choose true_value; false lanes choose false_value."),
    operands=[
        Operand("condition", VECTOR),
        Operand("true_value", VECTOR),
        Operand("false_value", VECTOR),
    ],
    results=[Result("result", VECTOR)],
    constraints=[
        HasI1Element("condition"),
        SameKind("condition", "result"),
        SameShape("condition", "true_value", "false_value", "result"),
        SameType("true_value", "false_value", "result"),
    ],
    facts="loom_vector_select_facts",
    canonicalize="loom_vector_select_canonicalize",
    traits=[PURE, ELEMENTWISE],
    format=[
        Ref("condition"),
        COMMA,
        Ref("true_value"),
        COMMA,
        Ref("false_value"),
        COLON,
        TypeOf("result"),
    ],
    examples=["%r = vector.select %mask, %a, %b : vector<16xf32>"],
)

vector_cmpi = Op(
    "vector.cmpi",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Lanewise integer comparison producing an i1 mask vector. The predicate attribute uses the scalar.cmpi predicate names and applies independently to each lane."),
    operands=[
        Operand("lhs", VECTOR),
        Operand("rhs", VECTOR),
    ],
    results=[Result("result", VECTOR)],
    attrs=[AttrDef("predicate", ATTR_TYPE_ENUM, enum_def=CmpIPredicate)],
    constraints=[
        HasIntegerElement("lhs"),
        HasI1Element("result"),
        SameKind("lhs", "result"),
        SameShape("lhs", "rhs", "result"),
        SameElementType("lhs", "rhs"),
    ],
    traits=[PURE, ELEMENTWISE],
    facts="loom_vector_cmpi_facts",
    canonicalize="loom_vector_comparison_canonicalize",
    format=[
        Attr("predicate"),
        COMMA,
        Ref("lhs"),
        COMMA,
        Ref("rhs"),
        COLON,
        TypeOf("lhs"),
        ARROW,
        ResultType("result"),
    ],
    examples=["%m = vector.cmpi slt, %lhs, %rhs : vector<16xi32> -> vector<16xi1>"],
)

vector_cmpf = Op(
    "vector.cmpf",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Lanewise floating-point comparison producing an i1 mask vector. The predicate attribute uses the scalar.cmpf ordered/unordered predicate names and applies independently to each lane."),
    operands=[
        Operand("lhs", VECTOR),
        Operand("rhs", VECTOR),
    ],
    results=[Result("result", VECTOR)],
    attrs=[AttrDef("predicate", ATTR_TYPE_ENUM, enum_def=CmpFPredicate)],
    constraints=[
        HasFloatElement("lhs"),
        HasI1Element("result"),
        SameKind("lhs", "result"),
        SameShape("lhs", "rhs", "result"),
        SameElementType("lhs", "rhs"),
    ],
    traits=[PURE, ELEMENTWISE],
    facts="loom_vector_cmpf_facts",
    canonicalize="loom_vector_comparison_canonicalize",
    format=[
        Attr("predicate"),
        COMMA,
        Ref("lhs"),
        COMMA,
        Ref("rhs"),
        COLON,
        TypeOf("lhs"),
        ARROW,
        ResultType("result"),
    ],
    examples=["%m = vector.cmpf olt, %lhs, %rhs : vector<16xf32> -> vector<16xi1>"],
)


# ============================================================================
# Lanewise arithmetic and math
# ============================================================================

_VF = ("fastmath", FastMathFlags)

vector_addf = _lanewise_binary(
    "vector.addf",
    phase=OpPhase.EXECUTABLE,
    result_constraint=FLOAT_ELEMENT,
    doc=(
        "Lanewise floating-point addition of same-typed vector operands. Optional fastmath flags carry the same per-lane floating-point permissions as scalar.addf; reduction reassociation belongs on vector.reduce instead."
    ),
    commutative=True,
    flags=_VF,
    facts="loom_vector_addf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_subf = _lanewise_binary(
    "vector.subf",
    phase=OpPhase.EXECUTABLE,
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise floating-point subtraction of same-typed vector operands. Optional fastmath flags carry the same per-lane floating-point permissions as scalar.subf."),
    flags=_VF,
    facts="loom_vector_subf_facts",
    canonicalize="loom_vector_subf_canonicalize",
)

vector_mulf = _lanewise_binary(
    "vector.mulf",
    phase=OpPhase.EXECUTABLE,
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise floating-point multiplication of same-typed vector operands. Optional fastmath flags carry the same per-lane floating-point permissions as scalar.mulf."),
    commutative=True,
    flags=_VF,
    facts="loom_vector_mulf_facts",
    canonicalize="loom_vector_mulf_canonicalize",
)

vector_divf = _lanewise_binary(
    "vector.divf",
    phase=OpPhase.EXECUTABLE,
    result_constraint=FLOAT_ELEMENT,
    doc=(
        "Lanewise floating-point division of same-typed vector operands. Optional fastmath flags carry the same per-lane floating-point permissions as scalar.divf, including arcp for reciprocal formation."
    ),
    flags=_VF,
    facts="loom_vector_divf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_remf = _lanewise_binary(
    "vector.remf",
    phase=OpPhase.EXECUTABLE,
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise floating-point remainder with C fmod semantics over same-typed vector operands."),
    flags=_VF,
    facts="loom_vector_remf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_negf = _lanewise_unary(
    "vector.negf",
    phase=OpPhase.EXECUTABLE,
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise floating-point negation of a same-typed vector operand."),
    traits=[INVOLUTION],
    flags=_VF,
    facts="loom_vector_negf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_absf = _lanewise_unary(
    "vector.absf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise floating-point absolute value of a same-typed vector operand."),
    traits=[IDEMPOTENT],
    flags=_VF,
    facts="loom_vector_absf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_minimumf = _lanewise_binary(
    "vector.minimumf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise IEEE 754 floating-point minimum of same-typed vector operands; NaN lanes propagate."),
    commutative=True,
    flags=_VF,
    facts="loom_vector_minimumf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_maximumf = _lanewise_binary(
    "vector.maximumf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise IEEE 754 floating-point maximum of same-typed vector operands; NaN lanes propagate."),
    commutative=True,
    flags=_VF,
    facts="loom_vector_maximumf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_minnumf = _lanewise_binary(
    "vector.minnumf",
    phase=OpPhase.EXECUTABLE,
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise C99 fmin-style floating-point minimum of same-typed vector operands; NaN lanes select the non-NaN operand."),
    commutative=True,
    flags=_VF,
    facts="loom_vector_minnumf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_maxnumf = _lanewise_binary(
    "vector.maxnumf",
    phase=OpPhase.EXECUTABLE,
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise C99 fmax-style floating-point maximum of same-typed vector operands; NaN lanes select the non-NaN operand."),
    commutative=True,
    flags=_VF,
    facts="loom_vector_maxnumf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_clampf = Op(
    "vector.clampf",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Lanewise floating-point clamp with explicit NaN/comparison policy. "
        "The ordered mode preserves strict compare/select semantics, number "
        "mode uses minnum/maxnum semantics, and ieee mode propagates NaNs."
    ),
    operands=[
        Operand("value", VECTOR),
        Operand("lower", VECTOR),
        Operand("upper", VECTOR),
    ],
    results=[Result("result", VECTOR)],
    attrs=[
        AttrDef("mode", ATTR_TYPE_ENUM, enum_def=ClampFMode),
        AttrDef("fastmath", ATTR_TYPE_FLAGS, optional=True, enum_def=FastMathFlags),
    ],
    constraints=[
        HasFloatElement("result"),
        SameType("value", "lower", "upper", "result"),
    ],
    traits=[PURE, ELEMENTWISE],
    facts="loom_vector_clampf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
    format=[
        TemplateParamFlags("mode", "fastmath"),
        Ref("value"),
        COMMA,
        Ref("lower"),
        COMMA,
        Ref("upper"),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%result = vector.clampf<ordered> %value, %lower, %upper : vector<16xf32>",
        "%result = vector.clampf<number, nnan|nsz> %value, %lower, %upper : vector<16xf32>",
        "%result = vector.clampf<ieee> %value, %lower, %upper : vector<16xf32>",
    ],
)

vector_copysignf = _lanewise_binary(
    "vector.copysignf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise copy sign of rhs lanes onto lhs lane magnitudes."),
    flags=_VF,
    facts="loom_vector_copysignf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_fmaf = Op(
    "vector.fmaf",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Lanewise fused multiply-add of same-typed floating-point vectors. "
        "Each result lane computes a*b + c with one final rounding; use "
        "separate vector.mulf/vector.addf when unfused rounding is required."
    ),
    operands=[
        Operand("a", VECTOR),
        Operand("b", VECTOR),
        Operand("c", VECTOR),
    ],
    results=[Result("result", VECTOR)],
    attrs=[
        AttrDef(
            "fastmath",
            ATTR_TYPE_FLAGS,
            optional=True,
            enum_def=FastMathFlags,
        )
    ],
    constraints=[
        HasFloatElement("result"),
        SameType("a", "b", "c", "result"),
    ],
    facts="loom_vector_fmaf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
    traits=[PURE, ELEMENTWISE],
    format=[
        Flags("fastmath"),
        Ref("a"),
        COMMA,
        Ref("b"),
        COMMA,
        Ref("c"),
        COLON,
        TypeOf("result"),
    ],
    examples=["%r = vector.fmaf %a, %b, %c : vector<16xf32>"],
)

vector_addi = _lanewise_binary(
    "vector.addi",
    phase=OpPhase.EXECUTABLE,
    result_constraint=INTEGER_ELEMENT,
    doc=("Lanewise integer addition of same-typed vector operands. Optional overflow flags state required no-wrap facts for every lane."),
    commutative=True,
    flags=("overflow", IntOverflowFlags),
    facts="loom_vector_addi_facts",
    canonicalize="loom_vector_binary_identity_canonicalize",
)

vector_subi = _lanewise_binary(
    "vector.subi",
    phase=OpPhase.EXECUTABLE,
    result_constraint=INTEGER_ELEMENT,
    doc=("Lanewise integer subtraction of same-typed vector operands. Optional overflow flags state required no-wrap facts for every lane."),
    flags=("overflow", IntOverflowFlags),
    facts="loom_vector_subi_facts",
    canonicalize="loom_vector_binary_identity_canonicalize",
)

vector_muli = _lanewise_binary(
    "vector.muli",
    phase=OpPhase.EXECUTABLE,
    result_constraint=INTEGER_ELEMENT,
    doc=("Lanewise integer multiplication of same-typed vector operands. Optional overflow flags state required no-wrap facts for every lane."),
    commutative=True,
    flags=("overflow", IntOverflowFlags),
    facts="loom_vector_muli_facts",
    canonicalize="loom_vector_binary_identity_canonicalize",
)

vector_divsi = _lanewise_binary(
    "vector.divsi",
    phase=OpPhase.EXECUTABLE,
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise signed integer division of same-typed vector operands; each lane rounds toward zero.",
    facts="loom_vector_divsi_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_divui = _lanewise_binary(
    "vector.divui",
    phase=OpPhase.EXECUTABLE,
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise unsigned integer division of same-typed vector operands.",
    facts="loom_vector_divui_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_remsi = _lanewise_binary(
    "vector.remsi",
    phase=OpPhase.EXECUTABLE,
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise signed integer remainder of same-typed vector operands.",
    facts="loom_vector_remsi_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_remui = _lanewise_binary(
    "vector.remui",
    phase=OpPhase.EXECUTABLE,
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise unsigned integer remainder of same-typed vector operands.",
    facts="loom_vector_remui_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_ceildivsi = _lanewise_binary(
    "vector.ceildivsi",
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise signed integer division rounding toward positive infinity.",
)

vector_ceildivui = _lanewise_binary(
    "vector.ceildivui",
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise unsigned integer division rounding toward positive infinity.",
)

vector_floordivsi = _lanewise_binary(
    "vector.floordivsi",
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise signed integer division rounding toward negative infinity.",
)

vector_negi = _lanewise_unary(
    "vector.negi",
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise integer negation of a same-typed vector operand.",
    traits=[INVOLUTION],
    facts="loom_vector_negi_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_absi = _lanewise_unary(
    "vector.absi",
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise integer absolute value of a same-typed vector operand.",
    traits=[IDEMPOTENT],
    facts="loom_vector_absi_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_minsi = _lanewise_binary(
    "vector.minsi",
    phase=OpPhase.EXECUTABLE,
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise signed integer minimum of same-typed vector operands.",
    commutative=True,
    facts="loom_vector_minsi_facts",
    canonicalize="loom_vector_binary_identity_canonicalize",
)

vector_maxsi = _lanewise_binary(
    "vector.maxsi",
    phase=OpPhase.EXECUTABLE,
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise signed integer maximum of same-typed vector operands.",
    commutative=True,
    facts="loom_vector_maxsi_facts",
    canonicalize="loom_vector_binary_identity_canonicalize",
)

vector_minui = _lanewise_binary(
    "vector.minui",
    phase=OpPhase.EXECUTABLE,
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise unsigned integer minimum of same-typed vector operands.",
    commutative=True,
    facts="loom_vector_minui_facts",
    canonicalize="loom_vector_binary_identity_canonicalize",
)

vector_maxui = _lanewise_binary(
    "vector.maxui",
    phase=OpPhase.EXECUTABLE,
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise unsigned integer maximum of same-typed vector operands.",
    commutative=True,
    facts="loom_vector_maxui_facts",
    canonicalize="loom_vector_binary_identity_canonicalize",
)

vector_fmai = Op(
    "vector.fmai",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Lanewise fused integer multiply-add a*b + c over same-typed vector operands. Optional overflow flags state required no-wrap facts for every lane."),
    operands=[
        Operand("a", VECTOR),
        Operand("b", VECTOR),
        Operand("c", VECTOR),
    ],
    results=[Result("result", VECTOR)],
    attrs=[AttrDef("overflow", ATTR_TYPE_FLAGS, optional=True, enum_def=IntOverflowFlags)],
    constraints=[
        HasIntegerElement("result"),
        SameType("a", "b", "c", "result"),
    ],
    traits=[PURE, ELEMENTWISE],
    facts="loom_vector_fmai_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
    format=[
        Flags("overflow"),
        Ref("a"),
        COMMA,
        Ref("b"),
        COMMA,
        Ref("c"),
        COLON,
        TypeOf("result"),
    ],
    examples=["%r = vector.fmai %a, %b, %c : vector<16xi32>"],
)

vector_andi = _lanewise_binary(
    "vector.andi",
    phase=OpPhase.EXECUTABLE,
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise bitwise AND of same-typed integer vector operands.",
    commutative=True,
    facts="loom_vector_andi_facts",
    canonicalize="loom_vector_binary_identity_canonicalize",
)

vector_ori = _lanewise_binary(
    "vector.ori",
    phase=OpPhase.EXECUTABLE,
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise bitwise OR of same-typed integer vector operands.",
    commutative=True,
    facts="loom_vector_ori_facts",
    canonicalize="loom_vector_binary_identity_canonicalize",
)

vector_xori = _lanewise_binary(
    "vector.xori",
    phase=OpPhase.EXECUTABLE,
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise bitwise XOR of same-typed integer vector operands.",
    commutative=True,
    facts="loom_vector_xori_facts",
    canonicalize="loom_vector_binary_identity_canonicalize",
)

vector_shli = _lanewise_binary(
    "vector.shli",
    phase=OpPhase.EXECUTABLE,
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise left shift of same-typed integer vector operands.",
    flags=("overflow", IntOverflowFlags),
    facts="loom_vector_shli_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_shrsi = _lanewise_binary(
    "vector.shrsi",
    phase=OpPhase.EXECUTABLE,
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise arithmetic right shift of same-typed integer vector operands.",
    facts="loom_vector_shrsi_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_shrui = _lanewise_binary(
    "vector.shrui",
    phase=OpPhase.EXECUTABLE,
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise logical right shift of same-typed integer vector operands.",
    facts="loom_vector_shrui_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_rotli = _lanewise_binary(
    "vector.rotli",
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise left rotate of same-typed integer vector operands.",
)

vector_rotri = _lanewise_binary(
    "vector.rotri",
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise right rotate of same-typed integer vector operands.",
)

vector_ctlzi = _lanewise_unary(
    "vector.ctlzi",
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise count leading zeros over integer lanes.",
    facts="loom_vector_ctlzi_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_cttzi = _lanewise_unary(
    "vector.cttzi",
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise count trailing zeros over integer lanes.",
    facts="loom_vector_cttzi_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_ctpopi = _lanewise_unary(
    "vector.ctpopi",
    result_constraint=INTEGER_ELEMENT,
    doc=("Lanewise population count over integer lanes. Each result lane is the number of set bits in the corresponding input lane and has the same integer element type as the input."),
    facts="loom_vector_ctpopi_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_expf = _lanewise_unary(
    "vector.expf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise natural exponential e^x."),
    flags=_VF,
    facts="loom_vector_expf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_exp2f = _lanewise_unary(
    "vector.exp2f",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise base-2 exponential 2^x."),
    flags=_VF,
    facts="loom_vector_exp2f_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_expm1f = _lanewise_unary(
    "vector.expm1f",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise exp(x)-1, preserving the scalar operation's near-zero numerical semantics."),
    flags=_VF,
    facts="loom_vector_expm1f_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_logf = _lanewise_unary(
    "vector.logf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise natural logarithm ln(x)."),
    flags=_VF,
    facts="loom_vector_logf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_log2f = _lanewise_unary(
    "vector.log2f",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise base-2 logarithm."),
    flags=_VF,
    facts="loom_vector_log2f_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_log10f = _lanewise_unary(
    "vector.log10f",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise base-10 logarithm."),
    flags=_VF,
    facts="loom_vector_log10f_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_log1pf = _lanewise_unary(
    "vector.log1pf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise log(1+x), preserving the scalar operation's near-zero numerical semantics."),
    flags=_VF,
    facts="loom_vector_log1pf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_powf = _lanewise_binary(
    "vector.powf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise floating-point power lhs^rhs over same-typed vector operands."),
    flags=_VF,
    facts="loom_vector_powf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_sqrtf = _lanewise_unary(
    "vector.sqrtf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise floating-point square root. Optional fastmath flags carry the same per-lane floating-point permissions as scalar.sqrtf."),
    flags=_VF,
    facts="loom_vector_sqrtf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_rsqrtf = _lanewise_unary(
    "vector.rsqrtf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise reciprocal square root 1/sqrt(x)."),
    flags=_VF,
    facts="loom_vector_rsqrtf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_cbrtf = _lanewise_unary(
    "vector.cbrtf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise cube root."),
    flags=_VF,
    facts="loom_vector_cbrtf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_sinf = _lanewise_unary(
    "vector.sinf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise sine."),
    flags=_VF,
    facts="loom_vector_sinf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_cosf = _lanewise_unary(
    "vector.cosf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise cosine."),
    flags=_VF,
    facts="loom_vector_cosf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_sinturnsf = _lanewise_unary(
    "vector.sinturnsf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise sine over turns: sin(2*pi*x), where 1.0 is one full revolution."),
    flags=_VF,
    facts="loom_vector_sinturnsf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_costurnsf = _lanewise_unary(
    "vector.costurnsf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise cosine over turns: cos(2*pi*x), where 1.0 is one full revolution."),
    flags=_VF,
    facts="loom_vector_costurnsf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_tanf = _lanewise_unary(
    "vector.tanf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise tangent."),
    flags=_VF,
    facts="loom_vector_tanf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_asinf = _lanewise_unary(
    "vector.asinf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise arcsine."),
    flags=_VF,
    facts="loom_vector_asinf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_acosf = _lanewise_unary(
    "vector.acosf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise arccosine."),
    flags=_VF,
    facts="loom_vector_acosf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_atanf = _lanewise_unary(
    "vector.atanf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise arctangent."),
    flags=_VF,
    facts="loom_vector_atanf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_atan2f = _lanewise_binary(
    "vector.atan2f",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise two-argument arctangent atan2(lhs, rhs) over same-typed vector operands."),
    flags=_VF,
    facts="loom_vector_atan2f_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_sinhf = _lanewise_unary(
    "vector.sinhf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise hyperbolic sine."),
    flags=_VF,
    facts="loom_vector_sinhf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_coshf = _lanewise_unary(
    "vector.coshf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise hyperbolic cosine."),
    flags=_VF,
    facts="loom_vector_coshf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_tanhf = _lanewise_unary(
    "vector.tanhf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise hyperbolic tangent."),
    flags=_VF,
    facts="loom_vector_tanhf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_asinhf = _lanewise_unary(
    "vector.asinhf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise inverse hyperbolic sine."),
    flags=_VF,
    facts="loom_vector_asinhf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_acoshf = _lanewise_unary(
    "vector.acoshf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise inverse hyperbolic cosine."),
    flags=_VF,
    facts="loom_vector_acoshf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_atanhf = _lanewise_unary(
    "vector.atanhf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise inverse hyperbolic tangent."),
    flags=_VF,
    facts="loom_vector_atanhf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_erff = _lanewise_unary(
    "vector.erff",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise error function, used by GeLU-style activations."),
    flags=_VF,
    facts="loom_vector_erff_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_erfcf = _lanewise_unary(
    "vector.erfcf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise complementary error function 1-erf(x)."),
    flags=_VF,
    facts="loom_vector_erfcf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_logisticf = _lanewise_unary(
    "vector.logisticf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise logistic sigmoid 1 / (1 + exp(-x))."),
    flags=_VF,
    facts="loom_vector_logisticf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_siluf = _lanewise_unary(
    "vector.siluf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise SiLU activation x * logistic(x)."),
    flags=_VF,
    facts="loom_vector_siluf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_softplusf = _lanewise_unary(
    "vector.softplusf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise softplus activation log(1 + exp(x))."),
    flags=_VF,
    facts="loom_vector_softplusf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_geluf = Op(
    "vector.geluf",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Lanewise GELU activation preserving the chosen formula family. The "
        "logistic variant carries its scale as an explicit attribute so "
        "importers do not encode approximation identity through arithmetic "
        "constants."
    ),
    operands=[Operand("input", VECTOR)],
    results=[Result("result", VECTOR)],
    attrs=[
        AttrDef("variant", ATTR_TYPE_ENUM, enum_def=GeluVariant),
        AttrDef("fastmath", ATTR_TYPE_FLAGS, optional=True, enum_def=FastMathFlags),
        AttrDef("scale", ATTR_TYPE_F64, optional=True),
    ],
    constraints=[
        HasFloatElement("result"),
        SameType("input", "result"),
    ],
    verify="loom_vector_geluf_verify",
    facts="loom_vector_geluf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
    traits=[PURE, ELEMENTWISE],
    format=[
        TemplateParamFlags("variant", "fastmath"),
        Ref("input"),
        AttrDict(),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%result = vector.geluf<erf> %input : vector<16xf32>",
        "%result = vector.geluf<tanh, afn> %input : vector<16xf32>",
        "%result = vector.geluf<logistic> %input {scale = 1.702} : vector<16xf32>",
    ],
)

vector_ceilf = _lanewise_unary(
    "vector.ceilf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise round toward positive infinity."),
    traits=[IDEMPOTENT],
    flags=_VF,
    facts="loom_vector_ceilf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_floorf = _lanewise_unary(
    "vector.floorf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise round toward negative infinity."),
    traits=[IDEMPOTENT],
    flags=_VF,
    facts="loom_vector_floorf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_roundf = _lanewise_unary(
    "vector.roundf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise round to nearest, ties away from zero."),
    traits=[IDEMPOTENT],
    flags=_VF,
    facts="loom_vector_roundf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_roundevenf = _lanewise_unary(
    "vector.roundevenf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise round to nearest, ties to even."),
    traits=[IDEMPOTENT],
    flags=_VF,
    facts="loom_vector_roundevenf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_truncf = _lanewise_unary(
    "vector.truncf",
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise round toward zero."),
    traits=[IDEMPOTENT],
    flags=_VF,
    facts="loom_vector_truncf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_isnanf = _lanewise_unary_shape_change(
    "vector.isnanf",
    source_constraint=HasFloatElement,
    result_constraint=I1_ELEMENT,
    doc="Lanewise floating-point NaN test producing an i1 mask vector.",
    facts="loom_vector_isnanf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_isinff = _lanewise_unary_shape_change(
    "vector.isinff",
    source_constraint=HasFloatElement,
    result_constraint=I1_ELEMENT,
    doc="Lanewise floating-point infinity test producing an i1 mask vector.",
    facts="loom_vector_isinff_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_isfinitef = _lanewise_unary_shape_change(
    "vector.isfinitef",
    source_constraint=HasFloatElement,
    result_constraint=I1_ELEMENT,
    doc="Lanewise floating-point finite test producing an i1 mask vector.",
    facts="loom_vector_isfinitef_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_signf = _lanewise_unary(
    "vector.signf",
    result_constraint=FLOAT_ELEMENT,
    doc="Lanewise floating-point sign, returning -1.0, 0.0, or 1.0 per lane.",
    facts="loom_vector_signf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_signi = _lanewise_unary(
    "vector.signi",
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise integer sign, returning -1, 0, or 1 per lane.",
    facts="loom_vector_signi_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)


# ============================================================================
# Conversions
# ============================================================================

vector_extf = _vector_cast(
    "vector.extf",
    phase=OpPhase.EXECUTABLE,
    source_constraint=HasFloatElement,
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise floating-point precision extension. Source and result shapes match exactly; only the floating-point element type widens."),
    constraints=[ElementWidthGreaterThan("result", "input")],
    facts="loom_vector_extf_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_fptrunc = _vector_cast(
    "vector.fptrunc",
    phase=OpPhase.EXECUTABLE,
    source_constraint=HasFloatElement,
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise floating-point precision truncation. Source and result shapes match exactly; only the floating-point element type narrows."),
    constraints=[ElementWidthLessThan("result", "input")],
)

vector_extsi = _vector_cast(
    "vector.extsi",
    phase=OpPhase.EXECUTABLE,
    source_constraint=HasIntegerElement,
    result_constraint=INTEGER_ELEMENT,
    doc=("Lanewise signed integer extension. Source and result shapes match exactly, and each source lane is sign-extended to the result element width."),
    constraints=[ElementWidthGreaterThan("result", "input")],
    facts="loom_vector_extsi_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_extui = _vector_cast(
    "vector.extui",
    phase=OpPhase.EXECUTABLE,
    source_constraint=HasIntegerElement,
    result_constraint=INTEGER_ELEMENT,
    doc=("Lanewise unsigned integer extension. Source and result shapes match exactly, and each source lane is zero-extended to the result element width."),
    constraints=[ElementWidthGreaterThan("result", "input")],
)

vector_trunci = _vector_cast(
    "vector.trunci",
    phase=OpPhase.EXECUTABLE,
    source_constraint=HasIntegerElement,
    result_constraint=INTEGER_ELEMENT,
    doc=("Lanewise integer truncation. Source and result shapes match exactly, and each lane keeps the low bits required by the result element width."),
    constraints=[ElementWidthLessThan("result", "input")],
)

vector_sitofp = _vector_cast(
    "vector.sitofp",
    phase=OpPhase.EXECUTABLE,
    source_constraint=HasIntegerElement,
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise signed integer to floating-point conversion with unchanged shape."),
    facts="loom_vector_sitofp_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
)

vector_uitofp = _vector_cast(
    "vector.uitofp",
    phase=OpPhase.EXECUTABLE,
    source_constraint=HasIntegerElement,
    result_constraint=FLOAT_ELEMENT,
    doc=("Lanewise unsigned integer to floating-point conversion with unchanged shape."),
)

vector_fptosi = _vector_cast(
    "vector.fptosi",
    phase=OpPhase.EXECUTABLE,
    source_constraint=HasFloatElement,
    result_constraint=INTEGER_ELEMENT,
    doc=("Lanewise floating-point to signed integer conversion with unchanged shape."),
)

vector_fptoui = _vector_cast(
    "vector.fptoui",
    phase=OpPhase.EXECUTABLE,
    source_constraint=HasFloatElement,
    result_constraint=INTEGER_ELEMENT,
    doc=("Lanewise floating-point to unsigned integer conversion with unchanged shape."),
)

vector_bitcast = Op(
    "vector.bitcast",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Bitwise reinterpretation between vector register types with the same total bit count. No numeric conversion is performed; only the lane shape and element interpretation change."),
    operands=[Operand("input", VECTOR)],
    results=[Result("result", VECTOR)],
    constraints=[TotalBitCountEqual("input", "result")],
    facts="loom_vector_bitcast_facts",
    traits=[PURE],
    format=[
        Ref("input"),
        COLON,
        TypeOf("input"),
        kw("to"),
        TypeOf("result"),
    ],
    examples=[
        "%r = vector.bitcast %input : vector<16xf32> to vector<16xi32>",
        "%s = vector.bitcast %bytes : vector<2xi8> to vector<1xf16>",
    ],
)


# ============================================================================
# Bitfield packing
# ============================================================================


def _bitfield_attrs() -> list[AttrDef]:
    return [
        AttrDef(
            "offset",
            ATTR_TYPE_I64,
            doc="Least-significant bit position of the field within each source lane.",
        ),
        AttrDef(
            "width",
            ATTR_TYPE_I64,
            doc="Number of bits in the field extracted or inserted in each lane.",
        ),
    ]


vector_bitfield_extractu = Op(
    "vector.bitfield.extractu",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Extract one fixed bitfield from each integer source lane and zero-extend it into the corresponding result lane. The bitfield is identified by least-significant-bit offset and width."),
    operands=[Operand("source", VECTOR)],
    results=[Result("result", VECTOR)],
    attrs=_bitfield_attrs(),
    constraints=[
        HasIntegerElement("source"),
        HasIntegerElement("result"),
        SameKind("source", "result"),
        SameShape("source", "result"),
        BitRangeWithinElementWidth("source", "offset", "width"),
        ElementWidthAtLeastAttr("result", "width"),
    ],
    facts="loom_vector_bitfield_extractu_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
    traits=[PURE, ELEMENTWISE],
    format=[
        Ref("source"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%lo = vector.bitfield.extractu %bytes {offset = 0, width = 4} : vector<16xi8> -> vector<16xi32>",
    ],
)

vector_bitfield_extracts = Op(
    "vector.bitfield.extracts",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Extract one fixed bitfield from each integer source lane and sign-extend it into the corresponding result lane. The bitfield is identified by least-significant-bit offset and width."),
    operands=[Operand("source", VECTOR)],
    results=[Result("result", VECTOR)],
    attrs=_bitfield_attrs(),
    constraints=[
        HasIntegerElement("source"),
        HasIntegerElement("result"),
        SameKind("source", "result"),
        SameShape("source", "result"),
        BitRangeWithinElementWidth("source", "offset", "width"),
        ElementWidthAtLeastAttr("result", "width"),
    ],
    facts="loom_vector_bitfield_extracts_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
    traits=[PURE, ELEMENTWISE],
    format=[
        Ref("source"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%signed = vector.bitfield.extracts %bytes {offset = 4, width = 4} : vector<16xi8> -> vector<16xi32>",
    ],
)

vector_bitfield_insert = Op(
    "vector.bitfield.insert",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Insert the low bits of each integer field lane into a fixed bitfield of the corresponding integer base lane. Bits outside the target field are preserved from the base lane."),
    operands=[
        Operand("field", VECTOR, doc="Integer field values. Only the low `width` bits are inserted."),
        Operand("base", VECTOR, doc="Integer base lanes whose target bitfield is replaced."),
    ],
    results=[Result("result", VECTOR)],
    attrs=_bitfield_attrs(),
    constraints=[
        HasIntegerElement("field"),
        HasIntegerElement("base"),
        HasIntegerElement("result"),
        SameShape("field", "base", "result"),
        SameType("base", "result"),
        BitRangeWithinElementWidth("base", "offset", "width"),
        ElementWidthAtLeastAttr("field", "width"),
    ],
    facts="loom_vector_bitfield_insert_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
    traits=[PURE, ELEMENTWISE],
    format=[
        Ref("field"),
        kw("into"),
        Ref("base"),
        AttrDict(),
        COLON,
        TypeOf("field"),
        COMMA,
        ResultType("result"),
    ],
    examples=[
        "%packed = vector.bitfield.insert %lo into %zero {offset = 0, width = 4} : vector<16xi32>, vector<16xi8>",
    ],
)


vector_bitpack = Op(
    "vector.bitpack",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Pack the low bits of each integer source lane into a contiguous "
        "little-endian bitstream stored in integer result lanes. Source lanes "
        "are consumed in logical lane order and width gives the number of "
        "bits taken from each source lane."
    ),
    operands=[Operand("source", VECTOR, doc="Integer lanes whose low `width` bits are packed.")],
    results=[Result("result", VECTOR, doc="Integer storage lanes containing the packed bitstream.")],
    attrs=[
        AttrDef(
            "width",
            ATTR_TYPE_I64,
            doc="Number of low source bits consumed from each lane.",
        ),
    ],
    constraints=[
        HasIntegerElement("source"),
        HasIntegerElement("result"),
        PositiveBitWidthAttr("width"),
        ElementWidthAtLeastAttr("source", "width"),
        PackedPayloadBitCountMatchesStorage("source", "width", "result", "result"),
    ],
    facts="loom_vector_bitpack_facts",
    traits=[PURE],
    format=[
        TemplateParam("width"),
        Ref("source"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%packed = vector.bitpack<4> %codes : vector<32xi8> -> vector<16xi8>",
        "%packed = vector.bitpack<1> %mask : vector<128xi1> -> vector<16xi8>",
    ],
)

vector_bitunpacku = Op(
    "vector.bitunpacku",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Unpack unsigned fixed-width fields from a contiguous little-endian integer bitstream into zero-extended integer result lanes. Result lanes are produced in logical lane order."),
    operands=[Operand("source", VECTOR, doc="Integer storage lanes containing the packed bitstream.")],
    results=[Result("result", VECTOR, doc="Integer lanes receiving zero-extended unpacked fields.")],
    attrs=[
        AttrDef(
            "width",
            ATTR_TYPE_I64,
            doc="Number of packed bits read for each result lane.",
        ),
    ],
    constraints=[
        HasIntegerElement("source"),
        HasIntegerElement("result"),
        PositiveBitWidthAttr("width"),
        ElementWidthAtLeastAttr("result", "width"),
        UnpackedPayloadBitCountMatchesStorage("result", "width", "source", "result"),
    ],
    facts="loom_vector_bitunpacku_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
    traits=[PURE],
    format=[
        TemplateParam("width"),
        Ref("source"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%codes = vector.bitunpacku<4> %packed : vector<16xi8> -> vector<32xi8>",
        "%mask = vector.bitunpacku<1> %packed : vector<16xi8> -> vector<128xi1>",
    ],
)

vector_bitunpacks = Op(
    "vector.bitunpacks",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Unpack signed fixed-width fields from a contiguous little-endian integer bitstream into sign-extended integer result lanes. Result lanes are produced in logical lane order."),
    operands=[Operand("source", VECTOR, doc="Integer storage lanes containing the packed bitstream.")],
    results=[Result("result", VECTOR, doc="Integer lanes receiving sign-extended unpacked fields.")],
    attrs=[
        AttrDef(
            "width",
            ATTR_TYPE_I64,
            doc="Number of packed bits read for each result lane.",
        ),
    ],
    constraints=[
        HasIntegerElement("source"),
        HasIntegerElement("result"),
        PositiveBitWidthAttr("width"),
        ElementWidthAtLeastAttr("result", "width"),
        UnpackedPayloadBitCountMatchesStorage("result", "width", "source", "result"),
    ],
    facts="loom_vector_bitunpacks_facts",
    canonicalize="loom_vector_uniform_result_canonicalize",
    traits=[PURE],
    format=[
        TemplateParam("width"),
        Ref("source"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%deltas = vector.bitunpacks<3> %packed : vector<12xi8> -> vector<32xi8>",
    ],
)


# ============================================================================
# Dot products
# ============================================================================

vector_dotf = Op(
    "vector.dotf",
    group=vector_ops,
    contracts=[ContractFamily.VECTOR_CONTRACTION],
    doc=(
        "Compute a same-element floating-point dot product with an explicit "
        "scalar accumulator. Semantics are equivalent to accumulating "
        "scalar.fmaf(lhs_lane, rhs_lane, acc) over lanes in logical lane order; "
        "use vector.mulf followed by vector.reduce<addf> when separately "
        "rounded products and additions are required. The source vectors must "
        "have the same shape and element type, and the init/result scalar type "
        "matches that element type. Zero-lane inputs return init. Optional "
        "fastmath flags carry the same floating-point permissions as scalar "
        "arithmetic, including reassociation of the fused dot terms."
    ),
    operands=[
        Operand("lhs", VECTOR, doc="Floating-point source lanes."),
        Operand("rhs", VECTOR, doc="Floating-point source lanes."),
        Operand("init", SCALAR, doc="Scalar accumulator seed."),
    ],
    results=[Result("result", SCALAR, doc="Scalar dot-product accumulator result.")],
    attrs=[
        AttrDef(
            "fastmath",
            ATTR_TYPE_FLAGS,
            optional=True,
            enum_def=FastMathFlags,
        ),
    ],
    constraints=[
        HasFloatElement("lhs"),
        SameShape("lhs", "rhs"),
        SameType("init", "result"),
        SameElementType("lhs", "rhs", "init"),
    ],
    facts="loom_vector_dotf_facts",
    traits=[PURE],
    format=[
        Flags("fastmath"),
        Ref("lhs"),
        COMMA,
        Ref("rhs"),
        COMMA,
        Ref("init"),
        COLON,
        TypeOf("lhs"),
        COMMA,
        TypeOf("rhs"),
        COMMA,
        TypeOf("result"),
    ],
    examples=[
        "%r = vector.dotf %lhs, %rhs, %acc : vector<16xf32>, vector<16xf32>, f32",
        "%r = vector.dotf<reassoc|contract> %lhs, %rhs, %acc : vector<16xf32>, vector<16xf32>, f32",
    ],
)

vector_dot2f = Op(
    "vector.dot2f",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Group adjacent two-lane f16 or bf16 products along the last axis and "
        "add each two-product fused sum into an f32 accumulator lane. "
        "Semantics are equivalent to extending each source lane to f32, then "
        "accumulating scalar.fmaf(lhs0_f32, rhs0_f32, acc) followed by "
        "scalar.fmaf(lhs1_f32, rhs1_f32, partial) for each result lane. This "
        "models AMDGPU fdot2-style widened register dots without making f16 "
        "dot accumulation implicit in vector.dotf."
    ),
    operands=[
        Operand("lhs", VECTOR, doc="f16 or bf16 source lanes grouped in pairs along the last axis."),
        Operand("rhs", VECTOR, doc="f16 or bf16 source lanes grouped in pairs along the last axis."),
        Operand("acc", VECTOR, doc="f32 accumulator lanes updated by each two-lane dot product."),
    ],
    results=[Result("result", VECTOR, doc="Updated f32 accumulator lanes.")],
    constraints=[
        HasF16OrBf16Element("lhs"),
        HasF16OrBf16Element("rhs"),
        HasF32Element("acc"),
        SameShape("lhs", "rhs"),
        SameElementType("lhs", "rhs"),
        SameType("acc", "result"),
        LastAxisGroupedBy("lhs", "result", 2),
    ],
    facts="loom_vector_dot2f_facts",
    traits=[PURE],
    format=[
        Ref("lhs"),
        COMMA,
        Ref("rhs"),
        COMMA,
        Ref("acc"),
        COLON,
        TypeOf("lhs"),
        COMMA,
        TypeOf("rhs"),
        COMMA,
        TypeOf("result"),
    ],
    examples=[
        "%r = vector.dot2f %lhs, %rhs, %acc : vector<16xf16>, vector<16xf16>, vector<8xf32>",
        "%r = vector.dot2f %lhs, %rhs, %acc : vector<2x16xbf16>, vector<2x16xbf16>, vector<2x8xf32>",
    ],
)

vector_dot4i = Op(
    "vector.dot4i",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Group adjacent four-lane i8 products along the last axis and add "
        "each four-product sum into an i32 accumulator lane. The signedness "
        "template chooses how lhs and rhs i8 lanes are interpreted, matching "
        "dp4a/VNNI-style hardware operations."
    ),
    operands=[
        Operand("lhs", VECTOR, doc="Integer source lanes grouped in fours along the last axis."),
        Operand("rhs", VECTOR, doc="Integer source lanes grouped in fours along the last axis."),
        Operand("acc", VECTOR, doc="Integer accumulator lanes updated by each four-lane dot product."),
    ],
    results=[Result("result", VECTOR, doc="Updated i32 accumulator lanes.")],
    attrs=[AttrDef("kind", ATTR_TYPE_ENUM, enum_def=IntegerDot4Kind)],
    constraints=[
        HasI8Element("lhs"),
        HasI8Element("rhs"),
        HasI32Element("acc"),
        SameShape("lhs", "rhs"),
        SameType("acc", "result"),
        LastAxisGroupedBy("lhs", "result", 4),
    ],
    facts="loom_vector_dot4i_facts",
    traits=[PURE],
    format=[
        TemplateParam("kind"),
        Ref("lhs"),
        COMMA,
        Ref("rhs"),
        COMMA,
        Ref("acc"),
        COLON,
        TypeOf("lhs"),
        COMMA,
        TypeOf("rhs"),
        COMMA,
        TypeOf("result"),
    ],
    examples=[
        "%r = vector.dot4i<s8s8> %lhs, %rhs, %acc : vector<16xi8>, vector<16xi8>, vector<4xi32>",
        "%r = vector.dot4i<u8s8> %lhs, %rhs, %acc : vector<2x16xi8>, vector<2x16xi8>, vector<2x4xi32>",
    ],
)

vector_dot8i4 = Op(
    "vector.dot8i4",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Treat each i32 source lane as a little-endian pack of eight 4-bit "
        "integer fields, multiply corresponding packed fields using the "
        "signedness template, and add the eight-product sum into the matching "
        "i32 accumulator lane. This is a packed-storage register dot: use "
        "vector.bitpack<4> when starting from unpacked byte lanes. The "
        "semantics match AMDGPU sdot8/udot8/sudot8 with clamp disabled."
    ),
    operands=[
        Operand("lhs", VECTOR, doc="i32 lanes holding packed lhs 4-bit fields."),
        Operand("rhs", VECTOR, doc="i32 lanes holding packed rhs 4-bit fields."),
        Operand("acc", VECTOR, doc="i32 accumulator lanes updated by each packed eight-lane dot product."),
    ],
    results=[Result("result", VECTOR, doc="Updated i32 accumulator lanes.")],
    attrs=[AttrDef("kind", ATTR_TYPE_ENUM, enum_def=IntegerDot8I4Kind)],
    constraints=[
        HasI32Element("lhs"),
        SameType("lhs", "rhs", "acc", "result"),
    ],
    facts="loom_vector_dot8i4_facts",
    traits=[PURE],
    format=[
        TemplateParam("kind"),
        Ref("lhs"),
        COMMA,
        Ref("rhs"),
        COMMA,
        Ref("acc"),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%r = vector.dot8i4<s4s4> %lhs, %rhs, %acc : vector<4xi32>",
        "%r = vector.dot8i4<u4s4> %lhs, %rhs, %acc : vector<[%N]xi32>",
    ],
)

vector_dot4f8 = Op(
    "vector.dot4f8",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Treat each i32 source lane as a little-endian pack of four 8-bit "
        "floating-point fields, decode fields according to the fp8/bf8 "
        "template, and add the four-product fused sum into the matching f32 "
        "accumulator lane. The fp8 spelling names the E4M3 primitive float "
        "format and bf8 names the E5M2 primitive float format. This is a "
        "packed-storage register dot matching AMDGPU dot4.f32.fp8/bf8 "
        "families without requiring unpacked f8 vector source lanes."
    ),
    operands=[
        Operand("lhs", VECTOR, doc="i32 lanes holding packed lhs 8-bit float fields."),
        Operand("rhs", VECTOR, doc="i32 lanes holding packed rhs 8-bit float fields."),
        Operand("acc", VECTOR, doc="f32 accumulator lanes updated by each packed four-lane dot product."),
    ],
    results=[Result("result", VECTOR, doc="Updated f32 accumulator lanes.")],
    attrs=[AttrDef("kind", ATTR_TYPE_ENUM, enum_def=FloatDot4F8Kind)],
    constraints=[
        HasI32Element("lhs"),
        HasF32Element("acc"),
        SameType("lhs", "rhs"),
        SameType("acc", "result"),
        SameShape("lhs", "acc"),
    ],
    facts="loom_vector_dot4f8_facts",
    traits=[PURE],
    format=[
        TemplateParam("kind"),
        Ref("lhs"),
        COMMA,
        Ref("rhs"),
        COMMA,
        Ref("acc"),
        COLON,
        TypeOf("lhs"),
        COMMA,
        TypeOf("result"),
    ],
    examples=[
        "%r = vector.dot4f8<fp8bf8> %lhs, %rhs, %acc : vector<4xi32>, vector<4xf32>",
        "%r = vector.dot4f8<bf8fp8> %lhs, %rhs, %acc : vector<[%N]xi32>, vector<[%N]xf32>",
    ],
)

vector_mma = Op(
    "vector.mma",
    group=vector_ops,
    contracts=[ContractFamily.VECTOR_CONTRACTION],
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Compute a matrix multiply-accumulate over target-shaped vector "
        "fragments. The op consumes only the physical lhs, rhs, and init "
        "vectors; logical M/N/K shape, fragment role, packed storage schema, "
        "scales, codebooks, sparse metadata, and other interpretation data are "
        "carried by vector.fragment facts on those operands. Lowering queries "
        "those facts to select native matrix instructions or a reference "
        "decomposition without baking target-specific witnesses into the MMA "
        "syntax."
    ),
    operands=[
        Operand("lhs", VECTOR, doc="Physical lhs fragment vector."),
        Operand("rhs", VECTOR, doc="Physical rhs fragment vector."),
        Operand("init", VECTOR, doc="Physical accumulator/addend fragment vector."),
    ],
    results=[Result("result", VECTOR, doc="Updated accumulator/result fragment vector.")],
    constraints=[SameType("init", "result")],
    facts="loom_vector_mma_facts",
    traits=[PURE],
    format=[
        Ref("lhs"),
        COMMA,
        Ref("rhs"),
        COMMA,
        Ref("init"),
        COLON,
        TypeOf("lhs"),
        COMMA,
        TypeOf("rhs"),
        COMMA,
        ResultType("result"),
    ],
    examples=[
        "%r = vector.mma %lhs, %rhs, %init : vector<8xf16>, vector<8xf16>, vector<8xf32>",
        "%r = vector.mma %lhs, %rhs, %init : vector<6xi32>, vector<6xi32>, vector<8xf32>",
    ],
)


# ============================================================================
# Reductions
# ============================================================================

vector_reduce = Op(
    "vector.reduce",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Reduce all lanes of a vector into a scalar accumulator/result using "
        "the template combining kind. The init operand and result have the "
        "same scalar type, and the combining kind must be valid for the input "
        "element type. Optional fastmath flags carry the same floating-point "
        "permissions as scalar arithmetic; contraction may fuse producer "
        "products into FMA accumulation when the producer permits it too."
    ),
    operands=[
        Operand("input", VECTOR),
        Operand("init", SCALAR),
    ],
    results=[Result("result", SCALAR)],
    attrs=[
        AttrDef("kind", ATTR_TYPE_ENUM, enum_def=CombiningKind),
        AttrDef(
            "fastmath",
            ATTR_TYPE_FLAGS,
            optional=True,
            enum_def=FastMathFlags,
        ),
    ],
    constraints=[
        SameType("init", "result"),
        SameElementType("input", "init", "result"),
    ],
    verify="loom_vector_reduce_verify",
    facts="loom_vector_reduce_facts",
    canonicalize="loom_vector_reduce_canonicalize",
    traits=[PURE],
    format=[
        TemplateParamFlags("kind", "fastmath"),
        Ref("input"),
        COMMA,
        Ref("init"),
        COLON,
        TypeOf("input"),
        COMMA,
        TypeOf("result"),
    ],
    examples=["%sum = vector.reduce<addf> %v, %zero : vector<16xf32>, f32"],
)

vector_reduce_axes = Op(
    "vector.reduce.axes",
    group=vector_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Reduce the explicit source axes of a vector while preserving the "
        "remaining axes in their original order. The init operand and result "
        "have the same type: scalar when every source axis is reduced, or a "
        "vector whose shape is the source shape with the reduced axes removed. "
        "Optional fastmath flags carry the same floating-point permissions as "
        "scalar arithmetic."
    ),
    operands=[
        Operand("input", VECTOR),
        Operand("init", ANY),
    ],
    results=[Result("result", ANY)],
    attrs=[
        AttrDef("kind", ATTR_TYPE_ENUM, enum_def=CombiningKind),
        AttrDef(
            "fastmath",
            ATTR_TYPE_FLAGS,
            optional=True,
            enum_def=FastMathFlags,
        ),
        AttrDef("axes", ATTR_TYPE_I64_ARRAY),
    ],
    constraints=[
        SameType("init", "result"),
        SameElementType("input", "init", "result"),
    ],
    verify="loom_vector_reduce_axes_verify",
    facts="loom_vector_reduce_axes_facts",
    canonicalize="loom_vector_reduce_axes_canonicalize",
    builder_name="reduce_axes",
    traits=[PURE],
    format=[
        TemplateParamFlags("kind", "fastmath"),
        Ref("input"),
        COMMA,
        Ref("init"),
        kw("axes"),
        Attr("axes"),
        COLON,
        TypeOf("input"),
        COMMA,
        ResultType("result"),
    ],
    examples=[
        "%cols = vector.reduce.axes<addf> %src, %init axes [0] : vector<4x8xf32>, vector<8xf32>",
        "%sum = vector.reduce.axes<addf> %src, %zero axes [0, 1] : vector<4x8xf32>, f32",
    ],
)


# ============================================================================
# Registry
# ============================================================================

VECTOR_CONSTRUCTION_OPS: tuple[Op, ...] = (
    vector_constant,
    vector_poison,
    vector_empty,
    vector_splat,
    vector_iota,
    vector_mask_range,
    vector_broadcast,
    vector_from_elements,
)

VECTOR_AGGREGATE_OPS: tuple[Op, ...] = (
    vector_extract,
    vector_insert,
    vector_slice,
    vector_concat,
    vector_transpose,
    vector_shuffle,
    vector_interleave,
    vector_deinterleave,
)

VECTOR_TABLE_OPS: tuple[Op, ...] = (
    vector_table_lookup,
    vector_table_quantize,
    vector_transform,
)

VECTOR_MEMORY_OPS: tuple[Op, ...] = (
    vector_fragment_load,
    vector_fragment_store,
    vector_load,
    vector_store,
    vector_load_mask,
    vector_store_mask,
    vector_load_expand,
    vector_store_compress,
    vector_gather,
    vector_scatter,
    vector_gather_mask,
    vector_scatter_mask,
)

VECTOR_ATOMIC_OPS: tuple[Op, ...] = (
    vector_atomic_reduce,
    vector_atomic_reduce_mask,
    vector_atomic_rmw,
    vector_atomic_rmw_mask,
    vector_atomic_cmpxchg,
)

VECTOR_COMPARE_OPS: tuple[Op, ...] = (
    vector_select,
    vector_cmpi,
    vector_cmpf,
)

VECTOR_FLOAT_ARITHMETIC_OPS: tuple[Op, ...] = (
    vector_addf,
    vector_subf,
    vector_mulf,
    vector_divf,
    vector_remf,
    vector_negf,
    vector_absf,
    vector_minimumf,
    vector_maximumf,
    vector_minnumf,
    vector_maxnumf,
    vector_clampf,
    vector_copysignf,
    vector_fmaf,
)

VECTOR_INTEGER_ARITHMETIC_OPS: tuple[Op, ...] = (
    vector_addi,
    vector_subi,
    vector_muli,
    vector_divsi,
    vector_divui,
    vector_remsi,
    vector_remui,
    vector_ceildivsi,
    vector_ceildivui,
    vector_floordivsi,
    vector_negi,
    vector_absi,
    vector_minsi,
    vector_maxsi,
    vector_minui,
    vector_maxui,
    vector_fmai,
    vector_andi,
    vector_ori,
    vector_xori,
    vector_shli,
    vector_shrsi,
    vector_shrui,
    vector_rotli,
    vector_rotri,
    vector_ctlzi,
    vector_cttzi,
    vector_ctpopi,
)

VECTOR_MATH_OPS: tuple[Op, ...] = (
    vector_expf,
    vector_exp2f,
    vector_expm1f,
    vector_logf,
    vector_log2f,
    vector_log10f,
    vector_log1pf,
    vector_powf,
    vector_sqrtf,
    vector_rsqrtf,
    vector_cbrtf,
    vector_sinf,
    vector_cosf,
    vector_sinturnsf,
    vector_costurnsf,
    vector_tanf,
    vector_asinf,
    vector_acosf,
    vector_atanf,
    vector_atan2f,
    vector_sinhf,
    vector_coshf,
    vector_tanhf,
    vector_asinhf,
    vector_acoshf,
    vector_atanhf,
    vector_erff,
    vector_erfcf,
    vector_logisticf,
    vector_siluf,
    vector_softplusf,
    vector_geluf,
    vector_ceilf,
    vector_floorf,
    vector_roundf,
    vector_roundevenf,
    vector_truncf,
    vector_isnanf,
    vector_isinff,
    vector_isfinitef,
    vector_signf,
    vector_signi,
)

VECTOR_CAST_OPS: tuple[Op, ...] = (
    vector_extf,
    vector_fptrunc,
    vector_extsi,
    vector_extui,
    vector_trunci,
    vector_sitofp,
    vector_uitofp,
    vector_fptosi,
    vector_fptoui,
    vector_bitcast,
)

VECTOR_BITPACK_OPS: tuple[Op, ...] = (
    vector_bitfield_extractu,
    vector_bitfield_extracts,
    vector_bitfield_insert,
    vector_bitpack,
    vector_bitunpacku,
    vector_bitunpacks,
)

VECTOR_CONTRACTION_OPS: tuple[Op, ...] = (
    vector_dotf,
    vector_dot2f,
    vector_dot4i,
    vector_dot8i4,
    vector_dot4f8,
    vector_mma,
)

VECTOR_REDUCTION_OPS: tuple[Op, ...] = (vector_reduce, vector_reduce_axes)

VECTOR_ENCODING_OPS: tuple[Op, ...] = (
    vector_decode,
    vector_encode,
    vector_fragment,
)

VECTOR_OP_CATEGORY_GROUPS: tuple[tuple[OpCategory, tuple[Op, ...]], ...] = (
    (VECTOR_CONSTRUCTION_CATEGORY, VECTOR_CONSTRUCTION_OPS),
    (VECTOR_AGGREGATE_CATEGORY, VECTOR_AGGREGATE_OPS),
    (VECTOR_TABLE_CATEGORY, VECTOR_TABLE_OPS),
    (VECTOR_MEMORY_CATEGORY, VECTOR_MEMORY_OPS),
    (VECTOR_ATOMIC_CATEGORY, VECTOR_ATOMIC_OPS),
    (VECTOR_COMPARE_CATEGORY, VECTOR_COMPARE_OPS),
    (VECTOR_FLOAT_ARITHMETIC_CATEGORY, VECTOR_FLOAT_ARITHMETIC_OPS),
    (VECTOR_INTEGER_ARITHMETIC_CATEGORY, VECTOR_INTEGER_ARITHMETIC_OPS),
    (VECTOR_MATH_CATEGORY, VECTOR_MATH_OPS),
    (VECTOR_CAST_CATEGORY, VECTOR_CAST_OPS),
    (VECTOR_BITPACK_CATEGORY, VECTOR_BITPACK_OPS),
    (VECTOR_CONTRACTION_CATEGORY, VECTOR_CONTRACTION_OPS),
    (VECTOR_REDUCTION_CATEGORY, VECTOR_REDUCTION_OPS),
    (VECTOR_ENCODING_CATEGORY, VECTOR_ENCODING_OPS),
)

ALL_VECTOR_OPS: tuple[Op, ...] = tuple(op for _, category_ops in VECTOR_OP_CATEGORY_GROUPS for op in category_ops)
