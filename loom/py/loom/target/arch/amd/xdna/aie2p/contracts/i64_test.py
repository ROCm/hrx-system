# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exactness tests for AMD XDNA AIE2P scalar-pair i64 recipes."""

import random

import pytest

from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.vector import defs as vector
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
_I64_UNSIGNED_DIVISION_DIVISORS = (
    1,
    2,
    3,
    2**24 - 1,
    2**31 - 1,
    2**31,
    2**31 + 1,
    2**32 - 1,
)
_I64_UNSIGNED_DIVISION_OPS = (
    scalar_arithmetic.scalar_divui,
    scalar_arithmetic.scalar_remui,
    scalar_arithmetic.scalar_divremui,
)


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


def _evaluate_rule(
    rule: DescriptorRule, lhs: int, rhs: int, result_name: str = "result"
) -> int:
    values: dict[ValueRef, int | tuple[int, int]] = {
        ValueRef.operand("lhs"): _split_i64(lhs),
        ValueRef.operand("rhs"): _split_i64(rhs),
    }
    carry_state = 0
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
            carry_state = int(result > _U32_MASK)
            if "carry_out" in emit.results:
                values[emit.results["carry_out"]] = carry_state
        elif semantic_tag in (
            "integer.sub.i32",
            "integer.sub.borrow_out.i32",
            "integer.sub.borrow_in_out.i32",
        ):
            result = operands["s0"] - operands["s1"] - operands.get("carry_in", 0)
            carry_state = int(result < 0)
            if "carry_out" in emit.results:
                values[emit.results["carry_out"]] = carry_state
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
        elif semantic_tag in (
            "register.move.to.division-state",
            "register.move.from.division-state",
        ):
            result = operands["src"]
        elif semantic_tag == "register.move.carry-to-scalar":
            result = carry_state
        elif semantic_tag == "integer.divide.step.unsigned.i32":
            state = operands["sd"]
            remainder = _u32((operands["s0"] << 1) | (state >> 31))
            state = _u32(state << 1)
            if remainder >= operands["s1"]:
                remainder -= operands["s1"]
                state |= 1
            values[emit.results["d0"]] = _u32(remainder)
            values[emit.results["sd_out"]] = state
            continue
        else:
            raise AssertionError(f"unsupported recipe operation {semantic_tag}")
        data_results = [
            ref for name, ref in emit.results.items() if name != "carry_out"
        ]
        assert len(data_results) == 1
        values[data_results[0]] = _u32(result)

    result = values[ValueRef.result(result_name)]
    if isinstance(result, tuple):
        return _join_i64(result)
    return result


def _evaluate_vector_select_rule(
    rule: DescriptorRule,
    condition: int,
    true_words: tuple[int, ...],
    false_words: tuple[int, ...],
) -> tuple[int, ...]:
    values: dict[ValueRef, int | tuple[int, ...]] = {
        ValueRef.operand("condition"): condition,
        ValueRef.operand("true_value"): true_words,
        ValueRef.operand("false_value"): false_words,
    }
    for emit in rule.emit:
        assert isinstance(emit, EmitDescriptorOp)
        operands = {name: values[ref] for name, ref in emit.operands.items()}
        semantic_tag = emit.descriptor.semantic_tag
        if semantic_tag == "integer.const.i32":
            result: int | tuple[int, ...] = emit.immediates["i"]
        elif semantic_tag in (
            "integer.and.i32",
            "integer.predicate.mask.low32",
        ):
            assert isinstance(operands["s0"], int)
            assert isinstance(operands["s1"], int)
            result = operands["s0"] & operands["s1"]
        elif semantic_tag == "integer.or.i32":
            assert isinstance(operands["s0"], int)
            assert isinstance(operands["s1"], int)
            result = operands["s0"] | operands["s1"]
        elif semantic_tag == "integer.lshl.i32":
            assert isinstance(operands["s0"], int)
            assert isinstance(operands["s1"], int)
            result = _logical_shift(operands["s0"], operands["s1"])
        elif semantic_tag == "integer.select.i32x16":
            selector = operands["sel"]
            true_value = operands["s2"]
            false_value = operands["s1"]
            assert isinstance(selector, int)
            assert isinstance(true_value, tuple)
            assert isinstance(false_value, tuple)
            result = tuple(
                true_value[word] if selector & (1 << word) else false_value[word]
                for word in range(16)
            )
        else:
            raise AssertionError(f"unsupported vector-select operation {semantic_tag}")
        assert len(emit.results) == 1
        values[next(iter(emit.results.values()))] = (
            _u32(result) if isinstance(result, int) else result
        )

    result = values[ValueRef.result("result")]
    assert isinstance(result, tuple)
    return result


def _evaluate_pair_vector_rule(
    rule: DescriptorRule,
    lhs: tuple[int, ...],
    rhs: tuple[int, ...],
) -> int | tuple[int, ...]:
    values: dict[ValueRef, int | tuple[int, ...]] = {
        ValueRef.operand("lhs"): lhs,
        ValueRef.operand("rhs"): rhs,
    }
    for emit in rule.emit:
        assert isinstance(emit, EmitDescriptorOp)
        operands = {name: values[ref] for name, ref in emit.operands.items()}
        semantic_tag = emit.descriptor.semantic_tag
        result: int | tuple[int, ...]
        if semantic_tag == "integer.const.i32":
            result = emit.immediates["i"]
            assert isinstance(result, int)
        elif semantic_tag in ("integer.and.bits512", "integer.or.bits512"):
            lhs_words = operands["s1"]
            rhs_words = operands["s2"]
            assert isinstance(lhs_words, tuple)
            assert isinstance(rhs_words, tuple)
            operation = (
                int.__and__ if semantic_tag == "integer.and.bits512" else int.__or__
            )
            result = tuple(
                operation(a, b) for a, b in zip(lhs_words, rhs_words, strict=True)
            )
        elif semantic_tag == "integer.sub.i32x16":
            lhs_words = operands["s1"]
            rhs_words = operands["s2"]
            assert isinstance(lhs_words, tuple)
            assert isinstance(rhs_words, tuple)
            result = tuple(
                _u32(a - b) for a, b in zip(lhs_words, rhs_words, strict=True)
            )
        elif semantic_tag == "register.shuffle.x.configured":
            source = operands["s1"]
            control = operands["mod"]
            assert isinstance(source, tuple)
            assert isinstance(control, int)
            assert control in (4, 5)
            selected = source[(control - 4) :: 2]
            result = (*selected, *selected)
        elif semantic_tag == "integer.cmp.eq.i32x16.low32":
            words = operands["s2"]
            assert isinstance(words, tuple)
            result = sum((word == 0) << index for index, word in enumerate(words))
        elif semantic_tag in (
            "integer.cmp.lt.signed.i32x16.low32",
            "integer.cmp.lt.unsigned.i32x16.low32",
            "integer.cmp.ge.unsigned.i32x16.low32",
        ):
            lhs_words = operands["s1"]
            rhs_words = operands["s2"]
            assert isinstance(lhs_words, tuple)
            assert isinstance(rhs_words, tuple)
            signed = ".signed." in semantic_tag
            relation = int.__ge__ if ".ge." in semantic_tag else int.__lt__
            result = sum(
                (relation(_s32(a), _s32(b)) if signed else relation(a, b)) << index
                for index, (a, b) in enumerate(zip(lhs_words, rhs_words, strict=True))
            )
        elif semantic_tag in (
            "integer.predicate.and.low32",
            "integer.predicate.or.low32",
        ):
            lhs_value = operands["s0"]
            rhs_value = operands["s1"]
            assert isinstance(lhs_value, int)
            assert isinstance(rhs_value, int)
            operation = (
                int.__and__
                if semantic_tag == "integer.predicate.and.low32"
                else int.__or__
            )
            result = operation(lhs_value, rhs_value)
        elif semantic_tag == "integer.predicate.complete.zero.high32":
            result = operands["storage"]
            assert isinstance(result, int)
        else:
            raise AssertionError(f"unsupported pair-vector operation {semantic_tag}")
        assert len(emit.results) == 1
        values[next(iter(emit.results.values()))] = (
            _u32(result) if isinstance(result, int) else result
        )

    return values[ValueRef.result("result")]


def _rule(source_op, *, predicate: str | None = None) -> DescriptorRule:
    candidates = [
        rule
        for rule in AIE2P_I64_RULES
        if isinstance(rule, DescriptorRule)
        and rule.source_op is source_op
        and rule.priority == 0
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


def _i64_shift_basis() -> tuple[int, ...]:
    single_bits = tuple(1 << bit for bit in range(64))
    return (
        0,
        _U64_MASK,
        *single_bits,
        *(_U64_MASK ^ bit for bit in single_bits),
    )


def _i64_unsigned_division_rules() -> tuple[tuple[DescriptorRule, int], ...]:
    rules: list[tuple[DescriptorRule, int]] = []
    for rule in AIE2P_I64_RULES:
        if not isinstance(rule, DescriptorRule):
            continue
        if rule.source_op not in _I64_UNSIGNED_DIVISION_OPS:
            continue
        divisor_range = next(
            guard for guard in rule.guards if guard.kind is GuardKind.VALUE_I64_RANGE
        )
        assert divisor_range.minimum == 1
        assert isinstance(divisor_range.maximum, int)
        rules.append((rule, divisor_range.maximum))
    assert {(rule.source_op, divisor_maximum) for rule, divisor_maximum in rules} == {
        (operation, divisor_maximum)
        for operation in _I64_UNSIGNED_DIVISION_OPS
        for divisor_maximum in (2**31, 2**32 - 1)
    }
    return tuple(rules)


def _i64_unsigned_division_transition_values(divisor: int) -> tuple[int, ...]:
    quotient_transitions = (
        1,
        2,
        (2**32 - 1) // divisor,
        2**32 // divisor,
        2**32 - 1,
        2**32,
        _U64_MASK // divisor,
    )
    values = {0, _U64_MASK}
    for quotient in quotient_transitions:
        if quotient == 0:
            continue
        threshold = quotient * divisor
        for value in (threshold - 1, threshold, threshold + 1):
            if 0 <= value <= _U64_MASK:
                values.add(value)
    return tuple(sorted(values))


def _i64_unsigned_division_basis() -> list[tuple[int, int]]:
    samples: list[tuple[int, int]] = []
    for divisor in _I64_UNSIGNED_DIVISION_DIVISORS:
        samples.extend((1 << bit, divisor) for bit in range(64))
        samples.extend(
            (value, divisor)
            for value in _i64_unsigned_division_transition_values(divisor)
        )
    return list(dict.fromkeys(samples))


def _check_i64_unsigned_division_samples(
    samples: list[tuple[int, int]],
) -> None:
    for rule, divisor_maximum in _i64_unsigned_division_rules():
        for value, divisor in samples:
            if divisor > divisor_maximum:
                continue
            quotient, remainder = divmod(value, divisor)
            if rule.source_op is scalar_arithmetic.scalar_divremui:
                assert _evaluate_rule(rule, value, divisor, "quotient") == quotient
                assert _evaluate_rule(rule, value, divisor, "remainder") == remainder
            else:
                expected = (
                    quotient
                    if rule.source_op is scalar_arithmetic.scalar_divui
                    else remainder
                )
                assert _evaluate_rule(rule, value, divisor) == expected


def _pair_vector_words(values: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(word for value in values for word in _split_i64(value))


def _pair_vector_samples() -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    edges = (
        0,
        1,
        2**31 - 1,
        2**31,
        2**32 - 1,
        2**32,
        2**63 - 1,
        2**63,
        2**64 - 1,
    )
    samples = [
        (
            tuple(edges[(offset + lane) % len(edges)] for lane in range(8)),
            tuple(edges[(offset - lane - 1) % len(edges)] for lane in range(8)),
        )
        for offset in range(len(edges))
    ]
    generator = random.Random(0xA1E2_5164)
    samples.extend(
        (
            tuple(generator.getrandbits(64) for _ in range(8)),
            tuple(generator.getrandbits(64) for _ in range(8)),
        )
        for _ in range(256)
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


@pytest.mark.parametrize(
    "source_op", [scalar_bitwise.scalar_ori, scalar_bitwise.scalar_xori]
)
@pytest.mark.parametrize("low_field", ["lhs", "rhs"])
def test_i64_word_join_recipes_are_exact(source_op, low_field: str) -> None:
    rule = next(
        rule
        for rule in AIE2P_I64_RULES
        if isinstance(rule, DescriptorRule)
        and rule.source_op is source_op
        and any(
            guard.kind is GuardKind.VALUE_I64_RANGE and guard.field == low_field
            for guard in rule.guards
        )
    )
    samples = _i64_samples()
    samples.extend((value, value) for value in _i64_shift_basis())
    for low, high in samples:
        low &= _U32_MASK
        high &= _U64_MASK ^ _U32_MASK
        lhs, rhs = (low, high) if low_field == "lhs" else (high, low)
        reference = lhs | rhs if source_op is scalar_bitwise.scalar_ori else lhs ^ rhs
        assert _evaluate_rule(rule, lhs, rhs) == reference


def test_i64_carry_and_borrow_recipes_are_exact() -> None:
    carry_and_borrow_rules = (
        (
            _rule(scalar_arithmetic.scalar_addcarryui),
            "sum",
            "carry",
            lambda lhs, rhs: lhs + rhs,
        ),
        (
            _rule(scalar_arithmetic.scalar_subborrowui),
            "difference",
            "borrow",
            lambda lhs, rhs: lhs - rhs,
        ),
    )
    for lhs, rhs in _i64_samples():
        for rule, value_name, flag_name, reference in carry_and_borrow_rules:
            mathematical_result = reference(lhs, rhs)
            assert _evaluate_rule(rule, lhs, rhs, value_name) == (
                mathematical_result & _U64_MASK
            )
            assert _evaluate_rule(rule, lhs, rhs, flag_name) == int(
                mathematical_result < 0 or mathematical_result > _U64_MASK
            )


def test_i64_unsigned_division_by_word_recipes_are_exact() -> None:
    _check_i64_unsigned_division_samples(_i64_unsigned_division_basis())


@pytest.mark.exhaustive
def test_i64_unsigned_division_by_word_recipes_match_seeded_stress_oracle() -> None:
    generator = random.Random(0xA1E2_D1A1)
    samples = [
        (generator.getrandbits(64), generator.randint(1, 2**32 - 1))
        for _ in range(4096)
    ]
    _check_i64_unsigned_division_samples(samples)


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
    # Exercise every count against each independent source bit and dense
    # complements. The additional random values cover interactions at every
    # split-word regime boundary without multiplying them by all 64 counts.
    basis_values = _i64_shift_basis()
    random_values = [lhs for lhs, _ in _i64_samples()[:256]]
    boundary_amounts = (0, 1, 30, 31, 32, 33, 62, 63)
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
        for value in basis_values:
            for amount in amounts:
                assert (
                    _evaluate_rule(rule, value, amount)
                    == reference(value, amount) & _U64_MASK
                )
        for value in random_values:
            for amount in boundary_amounts:
                if amount not in amounts:
                    continue
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


def test_i64_vector_bitwise_recipes_are_exact() -> None:
    rules = (
        (_rule(vector.vector_andi), int.__and__),
        (_rule(vector.vector_ori), int.__or__),
        (_rule(vector.vector_xori), int.__xor__),
    )
    for lhs_lanes, rhs_lanes in _pair_vector_samples():
        lhs_words = _pair_vector_words(lhs_lanes)
        rhs_words = _pair_vector_words(rhs_lanes)
        for rule, operation in rules:
            result = _evaluate_pair_vector_rule(rule, lhs_words, rhs_words)
            assert isinstance(result, tuple)
            expected = _pair_vector_words(
                tuple(
                    operation(lhs, rhs)
                    for lhs, rhs in zip(lhs_lanes, rhs_lanes, strict=True)
                )
            )
            assert result == expected


def test_i64_vector_comparison_recipes_are_exact() -> None:
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
        predicate: _rule(vector.vector_cmpi, predicate=predicate)
        for predicate in references
    }
    for lhs_lanes, rhs_lanes in _pair_vector_samples():
        lhs_words = _pair_vector_words(lhs_lanes)
        rhs_words = _pair_vector_words(rhs_lanes)
        for predicate, reference in references.items():
            result = _evaluate_pair_vector_rule(rules[predicate], lhs_words, rhs_words)
            assert isinstance(result, int)
            expected = sum(
                reference(lhs, rhs) << lane
                for lane, (lhs, rhs) in enumerate(
                    zip(lhs_lanes, rhs_lanes, strict=True)
                )
            )
            assert result & 0xFF == expected, (
                predicate,
                lhs_lanes,
                rhs_lanes,
                result,
                expected,
            )


def test_pair_vector_select_recipes_preserve_lane_bits() -> None:
    rules = [
        rule
        for rule in AIE2P_I64_RULES
        if isinstance(rule, DescriptorRule) and rule.source_op is vector.vector_select
    ]
    assert len(rules) == 8

    true_words = tuple(_u32(0x01234567 + word * 0x10203041) for word in range(16))
    false_words = tuple(_u32(0xFEDCBA98 - word * 0x01030507) for word in range(16))
    for rule in rules:
        result_guard = next(guard for guard in rule.guards if guard.field == "result")
        result_type = result_guard.type_pattern
        assert result_type is not None
        minimum_lanes = result_type.minimum_static_elements
        maximum_lanes = result_type.maximum_static_elements
        assert isinstance(minimum_lanes, int)
        assert isinstance(maximum_lanes, int)
        for lane_count in range(minimum_lanes, maximum_lanes + 1):
            for condition in range(256):
                # Bits outside the eight logical predicate positions may hold
                # arbitrary carrier state and must not affect selected lanes.
                result = _evaluate_vector_select_rule(
                    rule,
                    condition | 0xA5A5FF00,
                    true_words,
                    false_words,
                )
                for lane in range(lane_count):
                    expected = true_words if condition & (1 << lane) else false_words
                    assert (
                        result[2 * lane : 2 * lane + 2]
                        == expected[2 * lane : 2 * lane + 2]
                    )
