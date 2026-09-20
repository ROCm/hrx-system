# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exactness tests for AMD XDNA AIE2P scalar-pair i64 recipes."""

import random

from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.target.arch.amd.xdna.aie2p.contracts.i64 import AIE2P_I64_RULES
from loom.target.contracts import (
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    GuardKind,
    ValueRef,
)

_U32_MASK = 2**32 - 1
_U64_MASK = 2**64 - 1


def _u32(value: int) -> int:
    return value & _U32_MASK


def _s32(value: int) -> int:
    value = _u32(value)
    return value if value < 2**31 else value - 2**32


def _split_i64(value: int) -> tuple[int, int]:
    value &= _U64_MASK
    return value & _U32_MASK, value >> 32


def _join_i64(value: tuple[int, int]) -> int:
    return value[0] | (value[1] << 32)


def _logical_shift(value: int, amount: int) -> int:
    amount = _s32(amount)
    assert -32 <= amount < 32
    if amount < 0:
        return value >> (-amount & 31)
    return _u32(value << amount)


def _evaluate_rule(rule: DescriptorRule, lhs: int, rhs: int) -> int:
    values: dict[ValueRef, int | tuple[int, int]] = {
        ValueRef.operand("lhs"): _split_i64(lhs),
        ValueRef.operand("rhs"): _split_i64(rhs),
    }
    for emit in rule.emit:
        if isinstance(emit, EmitRegisterSlice):
            source = values[emit.source]
            assert isinstance(source, tuple)
            assert emit.unit_count == 1
            values[emit.result] = source[emit.unit_offset]
            continue
        if isinstance(emit, EmitRegisterConcat):
            assert len(emit.sources) == 2
            low = values[emit.sources[0]]
            high = values[emit.sources[1]]
            assert isinstance(low, int)
            assert isinstance(high, int)
            values[emit.result] = (_u32(low), _u32(high))
            continue
        assert isinstance(emit, EmitDescriptorOp)
        operands = {name: values[ref] for name, ref in emit.operands.items()}
        assert all(isinstance(value, int) for value in operands.values())
        semantic_tag = emit.descriptor.semantic_tag
        result = 0
        if semantic_tag == "integer.const.i32":
            result = emit.immediates["i"]
            assert isinstance(result, int)
        elif semantic_tag == "integer.and.i32":
            result = operands["s0"] & operands["s1"]
        elif semantic_tag == "integer.or.i32":
            result = operands["s0"] | operands["s1"]
        elif semantic_tag == "integer.xor.i32":
            result = operands["s0"] ^ operands["s1"]
        elif semantic_tag == "integer.mul.i32":
            result = operands["s0"] * operands["s1"]
        elif semantic_tag == "integer.madd.i32":
            result = operands["a0"] + operands["s0"] * operands["s1"]
        elif semantic_tag in (
            "integer.add.i32",
            "integer.add.carry_out.i32",
            "integer.add.carry_in_out.i32",
        ):
            rhs_value = (
                emit.immediates["imm"] if "imm" in emit.immediates else operands["s1"]
            )
            assert isinstance(rhs_value, int)
            result = operands["s0"] + rhs_value + operands.get("carry_in", 0)
            if "carry_out" in emit.results:
                values[emit.results["carry_out"]] = int(result > _U32_MASK)
        elif semantic_tag in (
            "integer.sub.i32",
            "integer.sub.borrow_out.i32",
            "integer.sub.borrow_in_out.i32",
        ):
            result = operands["s0"] - operands["s1"] - operands.get("carry_in", 0)
            if "carry_out" in emit.results:
                values[emit.results["carry_out"]] = int(result < 0)
        elif semantic_tag == "integer.lshl.i32":
            result = _logical_shift(operands["s0"], operands["s1"])
        elif semantic_tag == "integer.ashl.i32":
            result = _logical_shift(_s32(operands["s0"]), operands["s1"])
        elif semantic_tag == "integer.cmp.eq.i32":
            result = int(operands["s0"] == operands["s1"])
        elif semantic_tag == "integer.cmp.ne.i32":
            result = int(operands["s0"] != operands["s1"])
        elif semantic_tag == "integer.cmp.slt.i32":
            result = int(_s32(operands["s0"]) < _s32(operands["s1"]))
        elif semantic_tag == "integer.cmp.ult.i32":
            result = int(operands["s0"] < operands["s1"])
        elif semantic_tag == "integer.select.nonzero.i32":
            result = operands["s0"] if operands["s2"] else operands["s1"]
        else:
            raise AssertionError(f"unsupported recipe operation {semantic_tag}")
        data_results = [
            ref for name, ref in emit.results.items() if name != "carry_out"
        ]
        assert len(data_results) == 1
        values[data_results[0]] = _u32(result)

    result = values[ValueRef.result("result")]
    if isinstance(result, tuple):
        return _join_i64(result)
    return result


def _rule(source_op, *, predicate: str | None = None) -> DescriptorRule:
    candidates = [
        rule
        for rule in AIE2P_I64_RULES
        if isinstance(rule, DescriptorRule) and rule.source_op is source_op
    ]
    if predicate is not None:
        candidates = [
            rule
            for rule in candidates
            if any(guard.enum_keyword == predicate for guard in rule.guards)
        ]
    assert len(candidates) == 1
    return candidates[0]


def _i64_samples() -> list[tuple[int, int]]:
    edges = (
        0,
        1,
        2,
        2**31 - 1,
        2**31,
        2**32 - 1,
        2**32,
        2**63 - 1,
        2**63,
        2**64 - 1,
    )
    samples = [(lhs, rhs) for lhs in edges for rhs in edges]
    generator = random.Random(0xA1E2_0064)
    samples.extend(
        (generator.getrandbits(64), generator.getrandbits(64)) for _ in range(4096)
    )
    return samples


def test_i64_binary_recipes_are_exact() -> None:
    binary_rules = (
        (_rule(scalar_bitwise.scalar_andi), lambda lhs, rhs: lhs & rhs),
        (_rule(scalar_bitwise.scalar_ori), lambda lhs, rhs: lhs | rhs),
        (_rule(scalar_bitwise.scalar_xori), lambda lhs, rhs: lhs ^ rhs),
        (_rule(scalar_arithmetic.scalar_addi), lambda lhs, rhs: lhs + rhs),
        (_rule(scalar_arithmetic.scalar_subi), lambda lhs, rhs: lhs - rhs),
        (_rule(scalar_arithmetic.scalar_muli), lambda lhs, rhs: lhs * rhs),
    )
    for lhs, rhs in _i64_samples():
        for rule, reference in binary_rules:
            assert _evaluate_rule(rule, lhs, rhs) == reference(lhs, rhs) & _U64_MASK


def test_i64_shift_recipes_are_exact() -> None:
    references = {
        scalar_bitwise.scalar_shli: lambda value, amount: value << amount,
        scalar_bitwise.scalar_shrui: lambda value, amount: value >> amount,
        scalar_bitwise.scalar_shrsi: lambda value, amount: (
            (value if value < 2**63 else value - 2**64) >> amount
        ),
    }
    rules = [
        rule
        for rule in AIE2P_I64_RULES
        if isinstance(rule, DescriptorRule) and rule.source_op in references
    ]
    values = [lhs for lhs, _ in _i64_samples()[:1024]]
    for rule in rules:
        reference = references[rule.source_op]
        count_range = next(
            (guard for guard in rule.guards if guard.kind == GuardKind.VALUE_I64_RANGE),
            None,
        )
        amounts = range(
            count_range.minimum if count_range else 0,
            count_range.maximum + 1 if count_range else 64,
        )
        for value in values:
            for amount in amounts:
                assert (
                    _evaluate_rule(rule, value, amount)
                    == reference(value, amount) & _U64_MASK
                )


def test_i64_comparison_recipes_are_exact() -> None:
    def signed(value: int) -> int:
        return value if value < 2**63 else value - 2**64

    references = {
        "eq": lambda lhs, rhs: lhs == rhs,
        "ne": lambda lhs, rhs: lhs != rhs,
        "slt": lambda lhs, rhs: signed(lhs) < signed(rhs),
        "sle": lambda lhs, rhs: signed(lhs) <= signed(rhs),
        "sgt": lambda lhs, rhs: signed(lhs) > signed(rhs),
        "sge": lambda lhs, rhs: signed(lhs) >= signed(rhs),
        "ult": lambda lhs, rhs: lhs < rhs,
        "ule": lambda lhs, rhs: lhs <= rhs,
        "ugt": lambda lhs, rhs: lhs > rhs,
        "uge": lambda lhs, rhs: lhs >= rhs,
    }
    rules = {
        predicate: _rule(scalar_comparison.scalar_cmpi, predicate=predicate)
        for predicate in references
    }
    for lhs, rhs in _i64_samples():
        for predicate, reference in references.items():
            assert _evaluate_rule(rules[predicate], lhs, rhs) == int(
                reference(lhs, rhs)
            )
