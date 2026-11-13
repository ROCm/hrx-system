# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact binary32 comparison contracts for AMD XDNA AIE2P."""

from __future__ import annotations

from loom.dialect.scalar import comparison as scalar_comparison
from loom.target.arch.amd.xdna.aie2p.contracts.scalar_program import ScalarProgram
from loom.target.contracts import DescriptorRule, Guard, Scalar, ValueRef

_I1 = Scalar("i1")
_F32 = Scalar("f32")
_FLOAT_PREDICATES = (
    "oeq",
    "ogt",
    "oge",
    "olt",
    "ole",
    "one",
    "ord",
    "ueq",
    "ugt",
    "uge",
    "ult",
    "ule",
    "une",
    "uno",
)


def _comparison_result(
    program: ScalarProgram,
    predicate: str,
    relation_value: ValueRef,
    unordered: ValueRef | None,
) -> None:
    if unordered is None:
        program.binary(None, "or.i32", relation_value, relation_value)
    elif predicate.startswith("o"):
        one = program.constant("ordered_one", 1)
        ordered = program.binary("ordered", "xor.i32", unordered, one)
        program.binary(None, "and.i32", relation_value, ordered)
    else:
        program.binary(None, "or.i32", relation_value, unordered)


def _comparison_program(predicate: str, *, no_nan: bool) -> ScalarProgram:
    """Builds only the encoding relations consumed by this predicate."""

    program = ScalarProgram()
    if no_nan and predicate in ("ord", "uno"):
        program.constant(None, int(predicate == "ord"))
        return program

    lhs = ValueRef.operand("lhs")
    rhs = ValueRef.operand("rhs")
    absolute_mask = program.constant("absolute_mask", 0x7FFFFFFF)
    lhs_absolute = program.binary("lhs_absolute", "and.i32", lhs, absolute_mask)
    rhs_absolute = program.binary("rhs_absolute", "and.i32", rhs, absolute_mask)
    unordered = None
    if not no_nan:
        infinity = program.constant("infinity", 0x7F800000)
        lhs_nan = program.binary("lhs_nan", "cmp.ult.i32", infinity, lhs_absolute)
        rhs_nan = program.binary("rhs_nan", "cmp.ult.i32", infinity, rhs_absolute)
        unordered = program.binary("unordered", "or.i32", lhs_nan, rhs_nan)

    if predicate == "ord":
        one = program.constant("one", 1)
        program.binary(None, "xor.i32", unordered, one)
        return program
    if predicate == "uno":
        program.binary(None, "or.i32", unordered, unordered)
        return program

    absolute_union = program.binary(
        "absolute_union", "or.i32", lhs_absolute, rhs_absolute
    )
    both_zero = program.unary("both_zero", "cmp.eqz.i32", absolute_union)
    relation = predicate[1:]
    if relation in ("eq", "ne", "ge", "le"):
        same_bits = program.binary("same_bits", "cmp.eq.i32", lhs, rhs)
        equal = program.binary("equal", "or.i32", same_bits, both_zero)
    if relation in ("eq", "ne"):
        relation_value = equal
        if relation == "ne":
            one = program.constant("one", 1)
            relation_value = program.binary("not_equal", "xor.i32", equal, one)
    else:
        sign_mask = program.constant("sign_mask", -(2**31))
        all_bits = program.constant("all_bits", -1)
        keys = []
        for operand_name, operand in (("lhs", lhs), ("rhs", rhs)):
            sign = program.binary(f"{operand_name}_sign", "and.i32", operand, sign_mask)
            negative_key = program.binary(
                f"{operand_name}_negative_key", "xor.i32", operand, all_bits
            )
            positive_key = program.binary(
                f"{operand_name}_positive_key", "xor.i32", operand, sign_mask
            )
            keys.append(
                program.select(f"{operand_name}_key", negative_key, positive_key, sign)
            )
        lhs_key, rhs_key = keys
        if relation in ("gt", "ge"):
            lhs_key, rhs_key = rhs_key, lhs_key
        raw_relation = program.binary("raw_relation", "cmp.ult.i32", lhs_key, rhs_key)
        if relation in ("ge", "le"):
            relation_value = program.binary("inclusive", "or.i32", raw_relation, equal)
        else:
            one = program.constant("one", 1)
            nonzero_pair = program.binary("nonzero_pair", "xor.i32", both_zero, one)
            relation_value = program.binary(
                "strict", "and.i32", raw_relation, nonzero_pair
            )
    _comparison_result(program, predicate, relation_value, unordered)
    return program


def _zero_comparison_program(
    predicate: str, *, no_nan: bool, zero_operand: str
) -> ScalarProgram:
    program = ScalarProgram()
    if no_nan and predicate in ("ord", "uno"):
        program.constant(None, int(predicate == "ord"))
        return program
    relation = predicate[1:]
    if zero_operand == "lhs":
        relation = {"gt": "lt", "ge": "le", "lt": "gt", "le": "ge"}.get(
            relation, relation
        )
    operand = ValueRef.operand("rhs" if zero_operand == "lhs" else "lhs")
    absolute_mask = program.constant("absolute_mask", 0x7FFFFFFF)
    magnitude = program.binary("magnitude", "and.i32", operand, absolute_mask)
    if predicate in ("ord", "uno"):
        infinity = program.constant("infinity", 0x7F800000)
        program.binary(
            None,
            "cmp.uge.i32" if predicate == "ord" else "cmp.ult.i32",
            infinity,
            magnitude,
        )
        return program
    if relation in ("eq", "ne"):
        natural_nan = predicate in ("oeq", "une")
        result = program.unary(
            None if no_nan or natural_nan else "relation",
            "cmp.eqz.i32" if relation == "eq" else "cmp.nez.i32",
            magnitude,
        )
        if no_nan or natural_nan:
            return program
    else:
        zero = program.constant("zero", 0)
        left, right = (zero, operand) if relation in ("gt", "le") else (operand, zero)
        result = program.binary(
            "relation",
            "cmp.sge.i32" if relation in ("ge", "le") else "cmp.slt.i32",
            left,
            right,
        )
        if relation in ("ge", "lt"):
            zero_test = program.unary(
                "zero_test",
                "cmp.eqz.i32" if relation == "ge" else "cmp.nez.i32",
                magnitude,
            )
            result = program.binary(
                "numeric_relation",
                "or.i32" if relation == "ge" else "and.i32",
                result,
                zero_test,
            )
    unordered = None
    if not no_nan:
        infinity = program.constant("infinity", 0x7F800000)
        unordered = program.binary("unordered", "cmp.ult.i32", infinity, magnitude)
    _comparison_result(program, predicate, result, unordered)
    return program


def _comparison_rule(
    predicate: str,
    *,
    no_nan: bool,
    zero_operand: str | None = None,
    zero_value: float = 0.0,
) -> DescriptorRule:
    program = (
        _comparison_program(predicate, no_nan=no_nan)
        if zero_operand is None
        else _zero_comparison_program(
            predicate, no_nan=no_nan, zero_operand=zero_operand
        )
    )
    report_key = "exact_binary32_compare"
    if zero_operand is not None:
        report_key += "_zero"
    if no_nan:
        report_key += "_nnan"
    return DescriptorRule(
        source_op=scalar_comparison.scalar_cmpf,
        descriptor=program.emits[-1].descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            Guard.value_type("lhs", _F32),
            Guard.value_type("rhs", _F32),
            Guard.value_type("result", _I1),
            *((Guard.instance_flags_has_all("fastmath", "nnan"),) if no_nan else ()),
            *(
                (Guard.value_float_equals(zero_operand, zero_value),)
                if zero_operand
                else ()
            ),
        ),
        emit=program.emits,
        report_key=report_key,
    )


AIE2P_F32_COMPARE_RULES = tuple(
    _comparison_rule(
        predicate, no_nan=no_nan, zero_operand=zero_operand, zero_value=zero_value
    )
    for zero_operand in ("rhs", "lhs", None)
    for zero_value in ((0.0, -0.0) if zero_operand is not None else (0.0,))
    for no_nan in (True, False)
    for predicate in _FLOAT_PREDICATES
)
