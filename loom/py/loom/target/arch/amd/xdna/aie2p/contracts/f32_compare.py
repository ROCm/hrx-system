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


def _comparison_program(predicate: str) -> ScalarProgram:
    """Builds one IEEE-754 comparison over binary32 bit patterns."""

    program = ScalarProgram()
    lhs = ValueRef.operand("lhs")
    rhs = ValueRef.operand("rhs")
    absolute_mask = program.constant("absolute_mask", 0x7FFFFFFF)
    sign_mask = program.constant("sign_mask", -(2**31))
    all_bits = program.constant("all_bits", -1)
    one = program.constant("one", 1)
    infinity = program.constant("infinity", 0x7F800000)

    lhs_absolute = program.binary("lhs_absolute", "and.i32", lhs, absolute_mask)
    rhs_absolute = program.binary("rhs_absolute", "and.i32", rhs, absolute_mask)
    lhs_nan = program.binary("lhs_nan", "cmp.ult.i32", infinity, lhs_absolute)
    rhs_nan = program.binary("rhs_nan", "cmp.ult.i32", infinity, rhs_absolute)
    unordered = program.binary("unordered", "or.i32", lhs_nan, rhs_nan)

    absolute_union = program.binary(
        "absolute_union", "or.i32", lhs_absolute, rhs_absolute
    )
    both_zero = program.unary("both_zero", "cmp.eqz.i32", absolute_union)
    same_bits = program.binary("same_bits", "cmp.eq.i32", lhs, rhs)
    equal = program.binary("equal", "or.i32", same_bits, both_zero)

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
            program.select(
                f"{operand_name}_key",
                negative_key,
                positive_key,
                sign,
            )
        )
    lhs_key, rhs_key = keys
    nonzero_pair = program.binary("nonzero_pair", "xor.i32", both_zero, one)
    raw_less = program.binary("raw_less", "cmp.ult.i32", lhs_key, rhs_key)
    less = program.binary("less", "and.i32", raw_less, nonzero_pair)
    raw_greater = program.binary("raw_greater", "cmp.ult.i32", rhs_key, lhs_key)
    greater = program.binary("greater", "and.i32", raw_greater, nonzero_pair)

    ordered_predicate = predicate.startswith("o")
    relation = predicate[1:]
    if relation == "eq":
        relation_value = equal
    elif relation == "gt":
        relation_value = greater
    elif relation == "ge":
        relation_value = program.binary("greater_equal", "or.i32", greater, equal)
    elif relation == "lt":
        relation_value = less
    elif relation == "le":
        relation_value = program.binary("less_equal", "or.i32", less, equal)
    elif relation == "ne":
        relation_value = program.binary("not_equal", "xor.i32", equal, one)
    elif relation == "rd":
        program.binary(None, "xor.i32", unordered, one)
        return program
    elif relation == "no":
        program.binary(None, "or.i32", unordered, unordered)
        return program
    else:
        raise ValueError(f"unsupported binary32 comparison predicate {predicate}")

    if ordered_predicate:
        ordered = program.binary("ordered", "xor.i32", unordered, one)
        program.binary(None, "and.i32", relation_value, ordered)
    else:
        program.binary(None, "or.i32", relation_value, unordered)
    return program


def _comparison_rule(predicate: str) -> DescriptorRule:
    program = _comparison_program(predicate)
    return DescriptorRule(
        source_op=scalar_comparison.scalar_cmpf,
        descriptor=program.emits[-1].descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            Guard.value_type("lhs", _F32),
            Guard.value_type("rhs", _F32),
            Guard.value_type("result", _I1),
        ),
        emit=program.emits,
        report_key="exact_binary32_compare",
    )


AIE2P_F32_COMPARE_RULES = tuple(
    _comparison_rule(predicate) for predicate in _FLOAT_PREDICATES
)
