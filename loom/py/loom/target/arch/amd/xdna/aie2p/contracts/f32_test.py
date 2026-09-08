# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for exact AMD XDNA AIE2P binary32 contracts."""

from __future__ import annotations

import math
import random
import struct

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.target.arch.amd.xdna.aie2p.contracts.f32 import AIE2P_F32_RULES
from loom.target.arch.amd.xdna.aie2p.contracts.f32_compare import (
    AIE2P_F32_COMPARE_RULES,
)
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    ValueRef,
    compile_lower_rule_set,
)
from loom.target.contracts.kinds import SourceValueKind


def _u32(value: int) -> int:
    return value & 0xFFFFFFFF


def _s32(value: int) -> int:
    value = _u32(value)
    return value - (1 << 32) if value & 0x80000000 else value


def _leading_zeros(value: int) -> int:
    value = _u32(value)
    return 32 - value.bit_length()


def _logical_shift(value: int, amount: int) -> int:
    amount = _s32(amount)
    assert -31 <= amount <= 31
    if amount < 0:
        return _u32(value) >> -amount
    return _u32(value << amount)


def _value(
    ref: ValueRef,
    inputs: dict[str, int],
    temporaries: dict[str, int],
) -> int:
    if ref.kind is SourceValueKind.OPERAND:
        return inputs[ref.field]
    assert ref.kind is SourceValueKind.TEMPORARY
    return temporaries[ref.field]


def _evaluate_rule(rule: DescriptorRule, inputs: dict[str, int]) -> int:
    inputs = {name: _u32(value) for name, value in inputs.items()}
    temporaries: dict[str, int] = {}
    result: int | None = None
    for emit in rule.emit:
        descriptor_key = emit.descriptor.key.removeprefix("amd.xdna.aie2p.")
        result_ref = next(iter(emit.results.values()))
        if emit.form is DescriptorEmitForm.CONST:
            value = _u32(emit.immediates["i"])
        else:
            operands = {
                name: _value(ref, inputs, temporaries)
                for name, ref in emit.operands.items()
            }
            if descriptor_key == "add.i32.immediate":
                value = operands["s0"] + emit.immediates["imm"]
            elif descriptor_key == "add.i32":
                value = operands["s0"] + operands["s1"]
            elif descriptor_key == "sub.i32":
                value = operands["s0"] - operands["s1"]
            elif descriptor_key == "mul.i32":
                value = operands["s0"] * operands["s1"]
            elif descriptor_key == "and.i32":
                value = operands["s0"] & operands["s1"]
            elif descriptor_key == "or.i32":
                value = operands["s0"] | operands["s1"]
            elif descriptor_key == "xor.i32":
                value = operands["s0"] ^ operands["s1"]
            elif descriptor_key == "lshl.i32":
                value = _logical_shift(operands["s0"], operands["s1"])
            elif descriptor_key == "clz.i32":
                value = _leading_zeros(operands["s0"])
            elif descriptor_key == "cmp.eq.i32":
                value = int(operands["s0"] == operands["s1"])
            elif descriptor_key == "cmp.ult.i32":
                value = int(operands["s0"] < operands["s1"])
            elif descriptor_key == "cmp.uge.i32":
                value = int(operands["s0"] >= operands["s1"])
            elif descriptor_key == "cmp.slt.i32":
                value = int(_s32(operands["s0"]) < _s32(operands["s1"]))
            elif descriptor_key == "cmp.sge.i32":
                value = int(_s32(operands["s0"]) >= _s32(operands["s1"]))
            elif descriptor_key == "cmp.eqz.i32":
                value = int(operands["s0"] == 0)
            elif descriptor_key == "cmp.nez.i32":
                value = int(operands["s0"] != 0)
            elif descriptor_key == "select.zero.i32":
                value = operands["s0"] if operands["s2"] == 0 else operands["s1"]
            elif descriptor_key == "select.nonzero.i32":
                value = operands["s0"] if operands["s2"] != 0 else operands["s1"]
            elif descriptor_key == "convert.round-nearest.f32.to.signed.i32":
                source = _float_from_bits(operands["s0"])
                value = (
                    round(source)
                    if math.isfinite(source) and -(2**31) <= source < 2**31
                    else 0
                )
            elif descriptor_key == "convert.signed.i32.to.f32":
                assert _s32(operands["m"]) == 0
                value = _float_bits(float(_s32(operands["s0"])))
            else:
                raise AssertionError(f"unmodeled descriptor {descriptor_key}")
            value = _u32(value)

        if result_ref.kind is SourceValueKind.RESULT:
            result = value
        else:
            assert result_ref.kind is SourceValueKind.TEMPORARY
            temporaries[result_ref.field] = value
    assert result is not None
    return result


def _rule(report_key: str) -> DescriptorRule:
    candidates = [rule for rule in AIE2P_F32_RULES if rule.report_key == report_key]
    assert len(candidates) == 1
    return candidates[0]


def _evaluate_multiply(lhs: int, rhs: int) -> int:
    return _evaluate_rule(_rule("exact_binary32"), {"lhs": lhs, "rhs": rhs})


def _float_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", _u32(bits)))[0]


def _float_bits(value: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", value))[0]
    except OverflowError:
        return 0xFF800000 if value < 0 else 0x7F800000


def _reference_multiply(lhs: int, rhs: int) -> int:
    absolute_mask = 0x7FFFFFFF
    infinity = 0x7F800000
    quiet_bit = 0x00400000
    lhs_absolute = lhs & absolute_mask
    rhs_absolute = rhs & absolute_mask
    if lhs_absolute > infinity:
        return lhs | quiet_bit
    if rhs_absolute > infinity:
        return rhs | quiet_bit
    if (lhs_absolute == infinity and rhs_absolute == 0) or (
        rhs_absolute == infinity and lhs_absolute == 0
    ):
        return infinity | quiet_bit
    return _float_bits(_float_from_bits(lhs) * _float_from_bits(rhs))


def _reference_compare(predicate: str, lhs: int, rhs: int) -> int:
    infinity = 0x7F800000
    unordered = (lhs & 0x7FFFFFFF) > infinity or (rhs & 0x7FFFFFFF) > infinity
    lhs_value = _float_from_bits(lhs)
    rhs_value = _float_from_bits(rhs)
    ordered_relations = {
        "eq": lhs_value == rhs_value,
        "gt": lhs_value > rhs_value,
        "ge": lhs_value >= rhs_value,
        "lt": lhs_value < rhs_value,
        "le": lhs_value <= rhs_value,
        "ne": lhs_value != rhs_value,
    }
    relation = predicate[1:]
    if relation == "rd":
        return int(not unordered)
    if relation == "no":
        return int(unordered)
    if predicate.startswith("o"):
        return int(not unordered and ordered_relations[relation])
    return int(unordered or ordered_relations[relation])


def _is_f32_nan(bits: int) -> bool:
    return bits & 0x7FFFFFFF > 0x7F800000


def _reference_extremum(
    lhs: int,
    rhs: int,
    *,
    extremum: str,
    nan_policy: str,
) -> int:
    """Returns exact binary32 minimum/maximum bits for the requested policy."""

    quiet_bit = 0x00400000
    lhs_nan = _is_f32_nan(lhs)
    rhs_nan = _is_f32_nan(rhs)
    if nan_policy == "ieee":
        if lhs_nan:
            return lhs | quiet_bit
        if rhs_nan:
            return rhs | quiet_bit
    else:
        if lhs_nan:
            return lhs | quiet_bit if rhs_nan else rhs
        if rhs_nan:
            return lhs

    if (lhs | rhs) & 0x7FFFFFFF == 0:
        return lhs | rhs if extremum == "minimum" else lhs & rhs
    lhs_value = _float_from_bits(lhs)
    rhs_value = _float_from_bits(rhs)
    if extremum == "minimum":
        return lhs if lhs_value < rhs_value else rhs
    return rhs if lhs_value < rhs_value else lhs


def _reference_clamp(
    value: int,
    lower: int,
    upper: int,
    *,
    mode: str,
) -> int:
    if mode != "ordered":
        lower_bounded = _reference_extremum(
            value,
            lower,
            extremum="maximum",
            nan_policy=mode,
        )
        return _reference_extremum(
            lower_bounded,
            upper,
            extremum="minimum",
            nan_policy=mode,
        )

    lower_bounded = value
    if not (_is_f32_nan(value) or _is_f32_nan(lower)) and (
        _float_from_bits(value) < _float_from_bits(lower)
    ):
        lower_bounded = lower
    if not (_is_f32_nan(lower_bounded) or _is_f32_nan(upper)) and (
        _float_from_bits(lower_bounded) > _float_from_bits(upper)
    ):
        return upper
    return lower_bounded


def _round_binary32_integer(value: int, exponent: int) -> int:
    if not value:
        return 0
    sign = 0x80000000 if value < 0 else 0
    magnitude = abs(value)
    leading_bit = magnitude.bit_length() - 1
    unbiased_exponent = leading_bit + exponent

    def round_right(amount: int) -> int:
        if amount <= 0:
            return magnitude << -amount
        truncated = magnitude >> amount
        remainder = magnitude - (truncated << amount)
        halfway = 1 << (amount - 1)
        return truncated + int(
            remainder > halfway or (remainder == halfway and truncated & 1)
        )

    if unbiased_exponent < -126:
        subnormal = round_right(-149 - exponent)
        return sign | min(subnormal, 0x00800000)

    significand = round_right(leading_bit - 23)
    if significand == 0x01000000:
        significand >>= 1
        unbiased_exponent += 1
    if unbiased_exponent > 127:
        return sign | 0x7F800000
    exponent_bits = (unbiased_exponent + 127) << 23
    return sign | exponent_bits | (significand & 0x007FFFFF)


def _finite_binary32_term(bits: int) -> tuple[int, int]:
    sign = -1 if bits & 0x80000000 else 1
    magnitude = bits & 0x7FFFFFFF
    exponent = magnitude >> 23
    fraction = magnitude & 0x007FFFFF
    if exponent:
        return sign * (fraction | 0x00800000), exponent - 150
    return sign * fraction, -149


def _reference_fma(a: int, b: int, c: int) -> int:
    absolute_a = a & 0x7FFFFFFF
    absolute_b = b & 0x7FFFFFFF
    absolute_c = c & 0x7FFFFFFF
    infinity = 0x7F800000
    quiet_bit = 0x00400000
    product_sign = (a ^ b) & 0x80000000
    c_sign = c & 0x80000000
    if absolute_a > infinity:
        return a | quiet_bit
    if absolute_b > infinity:
        return b | quiet_bit
    if absolute_c > infinity:
        return c | quiet_bit

    a_infinity = absolute_a == infinity
    b_infinity = absolute_b == infinity
    c_infinity = absolute_c == infinity
    a_zero = absolute_a == 0
    b_zero = absolute_b == 0
    if (a_infinity and b_zero) or (b_infinity and a_zero):
        return infinity | quiet_bit
    if a_infinity or b_infinity:
        if c_infinity and product_sign != c_sign:
            return infinity | quiet_bit
        return product_sign | infinity
    if c_infinity:
        return c

    a_significand, a_exponent = _finite_binary32_term(a)
    b_significand, b_exponent = _finite_binary32_term(b)
    c_significand, c_exponent = _finite_binary32_term(c)
    product_significand = a_significand * b_significand
    product_exponent = a_exponent + b_exponent
    common_exponent = min(product_exponent, c_exponent)
    exact = (product_significand << (product_exponent - common_exponent)) + (
        c_significand << (c_exponent - common_exponent)
    )
    if not exact:
        product_zero = not product_significand
        if product_zero and not c_significand and product_sign == c_sign:
            return product_sign
        return 0
    return _round_binary32_integer(exact, common_exponent)


def _reference_integral(bits: int, *, ties_to_even: bool) -> int:
    if bits & 0x7FFFFFFF >= 0x4B000000:
        return bits
    value = _float_from_bits(bits)
    integral = round(value) if ties_to_even else math.trunc(value)
    result = _float_bits(float(integral))
    if integral == 0:
        result |= bits & 0x80000000
    return result


_EDGE_VALUES = (
    0x00000000,
    0x80000000,
    0x00000001,
    0x007FFFFF,
    0x00800000,
    0x00800001,
    0x3EFFFFFF,
    0x3F000000,
    0x3F000001,
    0x3F7FFFFF,
    0x3F800000,
    0x3F800001,
    0x3FFFFFFF,
    0x40000000,
    0x40000001,
    0x7F7FFFFE,
    0x7F7FFFFF,
    0x7F800000,
    0xFF800000,
    0x7F800001,
    0x7FC00000,
    0x7FFFFFFF,
    0xFF800001,
    0xFFFFFFFF,
)


def test_f32_programs_are_compact_descriptor_data() -> None:
    for rule in AIE2P_F32_RULES:
        rule.validate(AIE2P_CORE_DESCRIPTOR_SET)
        assert all(
            emit.form in (DescriptorEmitForm.CONST, DescriptorEmitForm.OP)
            for emit in rule.emit
        )
        assert all(
            (
                emit.form is DescriptorEmitForm.CONST
                if ".constant.i32" in emit.descriptor.key
                else emit.form is DescriptorEmitForm.OP
            )
            for emit in rule.emit
        )
        assert next(iter(rule.emit[-1].results.values())).kind is SourceValueKind.RESULT
    compiled = compile_lower_rule_set(
        ContractFragment(
            name="amd.xdna.aie2p.f32.multiply.test",
            descriptor_set=AIE2P_CORE_DESCRIPTOR_SET,
            cases=AIE2P_F32_RULES,
        ),
        dialect_ops={"scalar": ALL_SCALAR_OPS, "vector": ALL_VECTOR_OPS},
    )
    for compiled_rule, authored_index in zip(
        compiled.rules, compiled.authored_case_indices, strict=True
    ):
        authored_rule = AIE2P_F32_RULES[authored_index]
        emitted_steps = compiled.emits[
            compiled_rule.emit_start : compiled_rule.emit_start
            + compiled_rule.emit_count
        ]
        assert compiled_rule.report_key == authored_rule.report_key
        assert [step.descriptor for step in emitted_steps] == [
            step.descriptor for step in authored_rule.emit
        ]
    integer_multiply_counts = {
        rule.report_key: sum(
            emit.descriptor.key == "amd.xdna.aie2p.mul.i32" for emit in rule.emit
        )
        for rule in AIE2P_F32_RULES
    }
    assert integer_multiply_counts["exact_binary32"] == 4
    assert integer_multiply_counts["exact_binary32_fma"] == 4
    assert all(
        len(_rule(report_key).emit) <= maximum_emit_count
        for report_key, maximum_emit_count in (
            ("exact_binary32_minimum_ieee", 28),
            ("exact_binary32_maximum_ieee", 28),
            ("exact_binary32_minimum_number", 28),
            ("exact_binary32_maximum_number", 28),
            ("exact_binary32_clamp_ordered", 48),
            ("exact_binary32_clamp_number", 50),
            ("exact_binary32_clamp_ieee", 50),
        )
    )


def test_f32_bitwise_and_integral_programs_match_exact_oracles() -> None:
    absolute = _rule("exact_binary32_abs")
    negate = _rule("exact_binary32_neg")
    copysign = _rule("exact_binary32_copysign")
    roundeven = _rule("exact_binary32_roundeven")
    trunc = _rule("exact_binary32_trunc")
    vector_absolute = _rule("native_vector_binary32_abs")
    vector_negate = _rule("native_vector_binary32_neg")
    assert [emit.descriptor.key for emit in vector_absolute.emit] == [
        "amd.xdna.aie2p.constant.i32",
        "amd.xdna.aie2p.splat.i32x16",
        "amd.xdna.aie2p.and.bits512",
    ]
    assert [emit.descriptor.key for emit in vector_negate.emit] == [
        "amd.xdna.aie2p.constant.i32",
        "amd.xdna.aie2p.splat.i32x16",
        "amd.xdna.aie2p.or.bits512",
        "amd.xdna.aie2p.and.bits512",
        "amd.xdna.aie2p.sub.i32x16",
    ]
    values = set(_EDGE_VALUES)
    values.update(
        _float_bits(value)
        for value in (
            -8388607.5,
            -3.5,
            -2.5,
            -1.5,
            -0.75,
            -0.5,
            -0.25,
            0.25,
            0.5,
            0.75,
            1.5,
            2.5,
            3.5,
            8388607.5,
        )
    )
    random_source = random.Random(0xA1E2_1A7E)
    values.update(random_source.getrandbits(32) for _ in range(2048))
    for bits in values:
        assert _evaluate_rule(absolute, {"input": bits}) == (bits & 0x7FFFFFFF)
        assert _evaluate_rule(negate, {"input": bits}) == (bits ^ 0x80000000)
        assert _evaluate_rule(roundeven, {"input": bits}) == _reference_integral(
            bits, ties_to_even=True
        )
        assert _evaluate_rule(trunc, {"input": bits}) == _reference_integral(
            bits, ties_to_even=False
        )

    signs = (0x00000000, 0x80000000, 0x7FC00001, 0xFFC00001)
    for magnitude in values:
        for sign in signs:
            assert _evaluate_rule(
                copysign,
                {"lhs": magnitude, "rhs": sign},
            ) == ((magnitude & 0x7FFFFFFF) | (sign & 0x80000000))


def test_f32_multiply_program_matches_binary32_oracle() -> None:
    pairs = [(lhs, rhs) for lhs in _EDGE_VALUES for rhs in _EDGE_VALUES]
    random_source = random.Random(0xA1E2F32)
    pairs.extend(
        (random_source.getrandbits(32), random_source.getrandbits(32))
        for _ in range(2048)
    )
    for lhs, rhs in pairs:
        assert _evaluate_multiply(lhs, rhs) == _reference_multiply(lhs, rhs), (
            f"{lhs:08x} * {rhs:08x}"
        )


def test_f32_compare_programs_match_ieee_oracle() -> None:
    predicates = (
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
    rules_by_predicate = {}
    for rule in AIE2P_F32_COMPARE_RULES:
        predicate = rule.guards[0].enum_keyword
        assert predicate not in rules_by_predicate
        rules_by_predicate[predicate] = rule
    assert set(rules_by_predicate) == set(predicates)
    pairs = [(lhs, rhs) for lhs in _EDGE_VALUES for rhs in _EDGE_VALUES]
    random_source = random.Random(0xA1E2C32)
    pairs.extend(
        (random_source.getrandbits(32), random_source.getrandbits(32))
        for _ in range(1024)
    )
    for predicate in predicates:
        rule = rules_by_predicate[predicate]
        rule.validate(AIE2P_CORE_DESCRIPTOR_SET)
        assert rule.emit[-1].results["d0"].kind is SourceValueKind.RESULT
        for lhs, rhs in pairs:
            actual = _evaluate_rule(rule, {"lhs": lhs, "rhs": rhs})
            expected = _reference_compare(predicate, lhs, rhs)
            assert actual == expected, (
                f"{predicate}({lhs:08x}, {rhs:08x}): {actual} != {expected}"
            )


def test_f32_extremum_programs_match_exact_oracles() -> None:
    cases = (
        ("exact_binary32_minimum_ieee", "minimum", "ieee"),
        ("exact_binary32_maximum_ieee", "maximum", "ieee"),
        ("exact_binary32_minimum_number", "minimum", "number"),
        ("exact_binary32_maximum_number", "maximum", "number"),
    )
    pairs = [(lhs, rhs) for lhs in _EDGE_VALUES for rhs in _EDGE_VALUES]
    random_source = random.Random(0xA1E2_E87A)
    pairs.extend(
        (random_source.getrandbits(32), random_source.getrandbits(32))
        for _ in range(1024)
    )
    for report_key, extremum, nan_policy in cases:
        rule = _rule(report_key)
        for lhs, rhs in pairs:
            actual = _evaluate_rule(rule, {"lhs": lhs, "rhs": rhs})
            expected = _reference_extremum(
                lhs,
                rhs,
                extremum=extremum,
                nan_policy=nan_policy,
            )
            assert actual == expected, (
                f"{report_key}({lhs:08x}, {rhs:08x}): {actual:08x} != {expected:08x}"
            )


def test_f32_clamp_programs_match_exact_oracles() -> None:
    values = (
        0x00000000,
        0x80000000,
        0xBF800000,
        0x3F800000,
        0xFF800000,
        0x7F800000,
        0x7F800001,
        0x7FC00000,
        0xFF800001,
        0xFFC00000,
    )
    triples = [
        (value, lower, upper)
        for value in values
        for lower in values
        for upper in values
    ]
    random_source = random.Random(0xA1E2_C1A0)
    triples.extend(
        (
            random_source.getrandbits(32),
            random_source.getrandbits(32),
            random_source.getrandbits(32),
        )
        for _ in range(1024)
    )
    for mode in ("ordered", "number", "ieee"):
        rule = _rule(f"exact_binary32_clamp_{mode}")
        for value, lower, upper in triples:
            actual = _evaluate_rule(
                rule,
                {"value": value, "lower": lower, "upper": upper},
            )
            expected = _reference_clamp(value, lower, upper, mode=mode)
            assert actual == expected, (
                f"clamp<{mode}>({value:08x}, {lower:08x}, {upper:08x}): "
                f"{actual:08x} != {expected:08x}"
            )


def test_f32_fma_program_matches_single_rounding_oracle() -> None:
    special_values = (
        0x00000000,
        0x80000000,
        0x3F800000,
        0xBF800000,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7F800001,
        0x7FC00000,
    )
    triples = {
        (a, b, c)
        for a in _EDGE_VALUES
        for b in _EDGE_VALUES
        for c in special_values[:4]
    }
    triples.update(
        (a, b, c) for a in special_values for b in special_values for c in _EDGE_VALUES
    )
    random_source = random.Random(0xA1E2F3A)
    triples.update(
        (
            random_source.getrandbits(32),
            random_source.getrandbits(32),
            random_source.getrandbits(32),
        )
        for _ in range(2048)
    )
    fma_rule = _rule("exact_binary32_fma")
    for a, b, c in triples:
        actual = _evaluate_rule(fma_rule, {"a": a, "b": b, "c": c})
        expected = _reference_fma(a, b, c)
        assert actual == expected, (
            f"{a:08x} * {b:08x} + {c:08x}: {actual:08x} != {expected:08x}"
        )
