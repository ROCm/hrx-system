# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Index dialect definitions.

The index dialect owns scalar address-domain values: logical coordinates
(`index`), physical byte offsets (`offset`), and explicit conversion to or from
fixed-width integer payloads. It intentionally keeps a small op surface so that
agents can author address math without falling back to scalar integer syntax.
"""

from loom.assembly import (
    ARROW,
    COLON,
    COMMA,
    Attr,
    PredicateList,
    Ref,
    Refs,
    ResultType,
    TypeOf,
    TypesOf,
)
from loom.dsl import (
    ADDRESS,
    CONSTANT_LIKE,
    DISTRIBUTION_TRANSFER,
    FACT_IDENTITY,
    INDEX,
    OFFSET,
    PURE,
    SCALAR,
    AttrDef,
    AttrMatchesElementType,
    Dialect,
    EnumCase,
    EnumDef,
    Op,
    Operand,
    OpPhase,
    Result,
    SameType,
    binary_op,
    cast_op,
    comparison_op,
    unary_op,
)

__all__ = [
    "index_ops",
    "IndexPredicate",
    "ALL_INDEX_OPS",
]

index_ops = Dialect(
    "index",
    dialect_id=0x0F,
    doc=("Scalar address-domain operations over index, offset, and explicit integer boundary casts."),
)

IndexPredicate = EnumDef(
    "IndexPredicate",
    [
        EnumCase("eq", 0, doc="Equal."),
        EnumCase("ne", 1, doc="Not equal."),
        EnumCase("slt", 2, doc="Signed less than."),
        EnumCase("sle", 3, doc="Signed less or equal."),
        EnumCase("sgt", 4, doc="Signed greater than."),
        EnumCase("sge", 5, doc="Signed greater or equal."),
        EnumCase("ult", 6, doc="Unsigned less than."),
        EnumCase("ule", 7, doc="Unsigned less or equal."),
        EnumCase("ugt", 8, doc="Unsigned greater than."),
        EnumCase("uge", 9, doc="Unsigned greater or equal."),
    ],
    doc="Address-domain comparison predicates.",
)

# ============================================================================
# Construction and conversion
# ============================================================================

index_constant = Op(
    "index.constant",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Materialize a compile-time address-domain constant. The result type "
        "must be index for logical coordinates or offset for physical byte "
        "counts; fixed-width integer payload constants remain scalar.constant."
    ),
    results=[Result("result", ADDRESS)],
    attrs=[AttrDef("value", "any", doc="The constant integer value.")],
    constraints=[AttrMatchesElementType("value", "result")],
    traits=[PURE, CONSTANT_LIKE],
    facts="loom_index_constant_facts",
    format=[Attr("value"), COLON, TypeOf("result")],
    examples=[
        "%c0 = index.constant 0 : index",
        "%base = index.constant 0 : offset",
    ],
)

index_cast = cast_op(
    "index.cast",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    from_constraint=SCALAR,
    to_constraint=SCALAR,
    doc=("Explicit conversion at an address boundary. At least one side must be index or offset; pure integer width changes use scalar.extsi, scalar.extui, or scalar.trunci."),
    canonicalize="loom_index_cast_canonicalize",
    facts="loom_index_cast_facts",
    traits=[DISTRIBUTION_TRANSFER],
    verify="loom_index_cast_verify",
    examples=[
        "%i = index.cast %n : i64 to index",
        "%bytes = index.cast %raw : i64 to offset",
        "%n = index.cast %i : index to i64",
    ],
)

# ============================================================================
# Assumptions
# ============================================================================

index_assume = Op(
    "index.assume",
    group=index_ops,
    doc="Identity with predicate constraints on index or offset results.",
    operands=[Operand("values", ADDRESS, variadic=True)],
    results=[Result("results", ADDRESS, variadic=True)],
    attrs=[AttrDef("predicates", "predicate_list")],
    traits=[PURE, FACT_IDENTITY],
    facts="loom_index_assume_facts",
    verify="loom_index_assume_verify",
    format=[
        Refs("values"),
        PredicateList("predicates"),
        COLON,
        TypesOf("results"),
    ],
    examples=[
        "%n2 = index.assume %n [mul(%n, 16)] : index",
        "%end2 = index.assume %end [range(%end, 0, 4096)] : offset",
        "%n2, %off2 = index.assume %n, %off [mul(%n, 16), mul(%off, 64)] : index, offset",
    ],
)

# ============================================================================
# Address-domain arithmetic
# ============================================================================

index_add = binary_op(
    "index.add",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=ADDRESS,
    doc="Address-domain addition. Operands and result must all be index or all offset.",
    commutative=True,
    canonicalize="loom_index_add_canonicalize",
    facts="loom_index_add_facts",
    traits=[DISTRIBUTION_TRANSFER],
    examples=[
        "%r = index.add %lhs, %rhs : index",
        "%bytes = index.add %base, %delta : offset",
    ],
)

index_sub = binary_op(
    "index.sub",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=ADDRESS,
    doc="Address-domain subtraction. Operands and result must all be index or all offset.",
    canonicalize="loom_index_sub_canonicalize",
    facts="loom_index_sub_facts",
    traits=[DISTRIBUTION_TRANSFER],
    examples=[
        "%r = index.sub %lhs, %rhs : index",
        "%delta = index.sub %end, %base : offset",
    ],
)

index_mul = binary_op(
    "index.mul",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INDEX,
    doc="Logical coordinate multiplication. Offsets are physical byte counts and cannot be multiplied with this op.",
    commutative=True,
    canonicalize="loom_index_mul_canonicalize",
    facts="loom_index_mul_facts",
    traits=[DISTRIBUTION_TRANSFER],
    examples=["%r = index.mul %lhs, %rhs : index"],
)

index_scale = Op(
    "index.scale",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Scale a logical coordinate by a physical byte stride to produce a "
        "physical byte offset. This is the address-domain boundary for packed "
        "payload layouts: logical indices stay index values, byte strides stay "
        "offset values, and the result feeds byte-addressed buffer/view bases."
    ),
    operands=[
        Operand("index", INDEX, doc="Logical coordinate to scale."),
        Operand(
            "stride",
            OFFSET,
            doc="Physical byte stride for one coordinate step.",
        ),
    ],
    results=[Result("result", OFFSET)],
    traits=[PURE, DISTRIBUTION_TRANSFER],
    canonicalize="loom_index_scale_canonicalize",
    facts="loom_index_scale_facts",
    format=[
        Ref("index"),
        COMMA,
        Ref("stride"),
        COLON,
        TypeOf("index"),
        COMMA,
        TypeOf("stride"),
        ARROW,
        ResultType("result"),
    ],
    examples=["%bytes = index.scale %lane, %byte_stride : index, offset -> offset"],
)

index_div = binary_op(
    "index.div",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INDEX,
    doc=(
        "Logical coordinate quotient for non-negative index values with a "
        "positive divisor. Offsets are physical byte counts and cannot be "
        "divided with this op; use an explicit layout or storage mapping "
        "before deriving physical address pieces."
    ),
    canonicalize="loom_index_div_canonicalize",
    facts="loom_index_div_facts",
    traits=[DISTRIBUTION_TRANSFER],
    examples=["%q = index.div %lane, %group_size : index"],
)

index_rem = binary_op(
    "index.rem",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INDEX,
    doc=(
        "Logical coordinate remainder for non-negative index values with a "
        "positive divisor. Offsets are physical byte counts and cannot use "
        "remainder with this op; use an explicit layout or storage mapping "
        "before deriving physical address pieces."
    ),
    canonicalize="loom_index_rem_canonicalize",
    facts="loom_index_rem_facts",
    traits=[DISTRIBUTION_TRANSFER],
    examples=["%r = index.rem %lane, %group_size : index"],
)

index_min = binary_op(
    "index.min",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INDEX,
    doc="Signed minimum of two logical coordinate values.",
    commutative=True,
    canonicalize="loom_index_min_canonicalize",
    facts="loom_index_min_facts",
    traits=[DISTRIBUTION_TRANSFER],
    examples=["%r = index.min %lhs, %rhs : index"],
)

index_max = binary_op(
    "index.max",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INDEX,
    doc="Signed maximum of two logical coordinate values.",
    commutative=True,
    canonicalize="loom_index_max_canonicalize",
    facts="loom_index_max_facts",
    traits=[DISTRIBUTION_TRANSFER],
    examples=["%r = index.max %lhs, %rhs : index"],
)

index_madd = Op(
    "index.madd",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Logical coordinate multiply-add: a*b + c. Offsets are physical byte counts and cannot be multiplied with this op.",
    operands=[
        Operand("a", INDEX),
        Operand("b", INDEX),
        Operand("c", INDEX),
    ],
    results=[Result("result", INDEX)],
    constraints=[SameType("a", "b", "c", "result")],
    traits=[PURE, DISTRIBUTION_TRANSFER],
    canonicalize="loom_index_madd_canonicalize",
    format=[
        Ref("a"),
        COMMA,
        Ref("b"),
        COMMA,
        Ref("c"),
        COLON,
        TypeOf("result"),
    ],
    facts="loom_index_madd_facts",
    examples=["%r = index.madd %a, %b, %c : index"],
)

# ============================================================================
# Logical coordinate bitwise operations
# ============================================================================

index_andi = binary_op(
    "index.andi",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INDEX,
    doc="Bitwise AND over logical coordinate values. Offsets are physical byte counts and cannot use this op.",
    commutative=True,
    canonicalize="loom_index_andi_canonicalize",
    facts="loom_index_andi_facts",
    traits=[DISTRIBUTION_TRANSFER],
    examples=["%r = index.andi %lhs, %rhs : index"],
)

index_ori = binary_op(
    "index.ori",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INDEX,
    doc="Bitwise OR over logical coordinate values. Offsets are physical byte counts and cannot use this op.",
    commutative=True,
    canonicalize="loom_index_ori_canonicalize",
    facts="loom_index_ori_facts",
    traits=[DISTRIBUTION_TRANSFER],
    examples=["%r = index.ori %lhs, %rhs : index"],
)

index_xori = binary_op(
    "index.xori",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INDEX,
    doc="Bitwise XOR over logical coordinate values. Offsets are physical byte counts and cannot use this op.",
    commutative=True,
    canonicalize="loom_index_xori_canonicalize",
    facts="loom_index_xori_facts",
    traits=[DISTRIBUTION_TRANSFER],
    examples=["%r = index.xori %lhs, %rhs : index"],
)

index_shli = binary_op(
    "index.shli",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INDEX,
    doc="Left shift over logical coordinate values. Offsets are physical byte counts and cannot use this op.",
    canonicalize="loom_index_shli_canonicalize",
    facts="loom_index_shli_facts",
    traits=[DISTRIBUTION_TRANSFER],
    examples=["%r = index.shli %lhs, %rhs : index"],
)

index_shrsi = binary_op(
    "index.shrsi",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INDEX,
    doc="Arithmetic right shift over logical coordinate values. Offsets are physical byte counts and cannot use this op.",
    canonicalize="loom_index_shrsi_canonicalize",
    facts="loom_index_shrsi_facts",
    traits=[DISTRIBUTION_TRANSFER],
    examples=["%r = index.shrsi %lhs, %rhs : index"],
)

index_shrui = binary_op(
    "index.shrui",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INDEX,
    doc="Logical right shift over logical coordinate values. Offsets are physical byte counts and cannot use this op.",
    canonicalize="loom_index_shrui_canonicalize",
    facts="loom_index_shrui_facts",
    traits=[DISTRIBUTION_TRANSFER],
    examples=["%r = index.shrui %lhs, %rhs : index"],
)

index_rotli = binary_op(
    "index.rotli",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INDEX,
    doc="Left rotate over logical coordinate values. Offsets are physical byte counts and cannot use this op.",
    canonicalize="loom_index_rotli_canonicalize",
    facts="loom_index_rotli_facts",
    examples=["%r = index.rotli %lhs, %rhs : index"],
)

index_rotri = binary_op(
    "index.rotri",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INDEX,
    doc="Right rotate over logical coordinate values. Offsets are physical byte counts and cannot use this op.",
    canonicalize="loom_index_rotri_canonicalize",
    facts="loom_index_rotri_facts",
    examples=["%r = index.rotri %lhs, %rhs : index"],
)

index_ctlzi = unary_op(
    "index.ctlzi",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INDEX,
    doc="Count leading zeros in a logical coordinate value.",
    facts="loom_index_ctlzi_facts",
    examples=["%r = index.ctlzi %input : index"],
)

index_cttzi = unary_op(
    "index.cttzi",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INDEX,
    doc="Count trailing zeros in a logical coordinate value.",
    facts="loom_index_cttzi_facts",
    examples=["%r = index.cttzi %input : index"],
)

index_ctpopi = unary_op(
    "index.ctpopi",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=INDEX,
    doc="Count set bits in a logical coordinate value.",
    facts="loom_index_ctpopi_facts",
    examples=["%r = index.ctpopi %input : index"],
)

# ============================================================================
# Predicates
# ============================================================================

index_cmp = comparison_op(
    "index.cmp",
    group=index_ops,
    phase=OpPhase.EXECUTABLE,
    type_constraint=ADDRESS,
    predicates=IndexPredicate,
    doc="Address-domain comparison. Operands must both be index or both be offset.",
    canonicalize="loom_index_cmp_canonicalize",
    facts="loom_index_cmp_facts",
    examples=[
        "%p = index.cmp slt, %i, %n : index",
        "%inside = index.cmp ule, %byte_offset, %limit : offset",
    ],
)

# ============================================================================
# Registry
# ============================================================================

ALL_INDEX_OPS: tuple[Op, ...] = (
    index_constant,
    index_cast,
    index_assume,
    index_add,
    index_sub,
    index_mul,
    index_scale,
    index_div,
    index_rem,
    index_min,
    index_max,
    index_madd,
    index_andi,
    index_ori,
    index_xori,
    index_shli,
    index_shrsi,
    index_shrui,
    index_rotli,
    index_rotri,
    index_ctlzi,
    index_cttzi,
    index_ctpopi,
    index_cmp,
)
