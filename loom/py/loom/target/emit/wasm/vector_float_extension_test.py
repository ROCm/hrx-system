# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import struct

import pytest

from loom.target.arch.wasm.descriptors import WASM_CORE_SIMD128_DESCRIPTOR_SET
from loom.target.contracts import (
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardKind,
    ValueRef,
    descriptor_by_key,
)
from loom.target.emit.wasm.vector_float_extension import float_extension_rules


def _descriptor(key):
    return descriptor_by_key(WASM_CORE_SIMD128_DESCRIPTOR_SET, key)


def _type_guard(field, type_pattern):
    return Guard.value_type(field, type_pattern)


def _rules() -> tuple[DescriptorRule, ...]:
    return float_extension_rules(_descriptor, _type_guard)


def _rule(source_element: str) -> DescriptorRule:
    report_key = f"wasm.float_extension.exact_{source_element.lower()}_to_f32"
    return next(rule for rule in _rules() if rule.report_key == report_key)


def _evaluate_lane(rule: DescriptorRule, input_bits: int) -> int:
    values = {ValueRef.operand("input"): input_bits}
    for emit in rule.emit:
        assert isinstance(emit, EmitDescriptorOp)
        key = emit.descriptor.key.removeprefix("wasm.")
        operands = {name: values[value] for name, value in emit.operands.items()}
        if key == "v128.const":
            result = emit.immediates["lo64"] & 0xFFFFFFFF
        elif key == "i32.const":
            result = emit.immediates["i32_value"] & 0xFFFFFFFF
        elif ".extend_low_" in key:
            result = operands["input"]
        elif key == "v128.and":
            result = operands["lhs"] & operands["rhs"]
        elif key == "v128.or":
            result = operands["lhs"] | operands["rhs"]
        elif key == "i32x4.add":
            result = operands["lhs"] + operands["rhs"]
        elif key == "i32x4.sub":
            result = operands["lhs"] - operands["rhs"]
        elif key == "i32x4.eq":
            result = 0xFFFFFFFF if operands["lhs"] == operands["rhs"] else 0
        elif key == "i32x4.lt_u":
            result = 0xFFFFFFFF if operands["lhs"] < operands["rhs"] else 0
        elif key == "i32x4.shl":
            result = operands["value"] << (operands["count"] & 31)
        elif key == "i32x4.shr_u":
            result = operands["value"] >> (operands["count"] & 31)
        else:
            assert key == "v128.bitselect"
            condition = operands["condition"]
            result = (operands["true_value"] & condition) | (
                operands["false_value"] & ~condition
            )
        values[next(iter(emit.results.values()))] = result & 0xFFFFFFFF
    return values[ValueRef.result("result")]


def _reference_f16(input_bits: int) -> int:
    sign = (input_bits & 0x8000) << 16
    exponent = (input_bits >> 10) & 0x1F
    fraction = input_bits & 0x03FF
    if exponent == 0:
        if fraction == 0:
            return sign
        unbiased_exponent = -14
        while fraction & 0x0400 == 0:
            fraction <<= 1
            unbiased_exponent -= 1
        fraction &= 0x03FF
        return sign | ((unbiased_exponent + 127) << 23) | (fraction << 13)
    if exponent == 0x1F:
        return sign | 0x7F800000 | (fraction << 13)
    return sign | ((exponent + 112) << 23) | (fraction << 13)


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _reference_fp8(
    input_bits: int,
    *,
    exponent_bit_count: int,
    mantissa_bit_count: int,
    has_infinity: bool,
) -> int:
    sign = input_bits & 0x80
    exponent_mask = (1 << exponent_bit_count) - 1
    exponent = (input_bits >> mantissa_bit_count) & exponent_mask
    mantissa_mask = (1 << mantissa_bit_count) - 1
    mantissa = input_bits & mantissa_mask
    if exponent == exponent_mask:
        if has_infinity:
            if mantissa:
                return 0x7FC00000
            return (sign << 24) | 0x7F800000
        if mantissa == mantissa_mask:
            return 0x7FC00000

    significand = mantissa if exponent == 0 else (1 << mantissa_bit_count) | mantissa
    if not significand:
        return sign << 24
    exponent_bias = (1 << (exponent_bit_count - 1)) - 1
    scale = (1 if exponent == 0 else exponent) - exponent_bias - mantissa_bit_count
    value = math.ldexp(float(significand), scale)
    return _f32_bits(-value if sign else value)


def test_rules_cover_the_complete_narrow_to_f32_packet_family():
    rules = _rules()

    assert {rule.report_key for rule in rules} == {
        "wasm.float_extension.exact_f8e4m3_to_f32",
        "wasm.float_extension.exact_f8e5m2_to_f32",
        "wasm.float_extension.exact_f16_to_f32",
        "wasm.float_extension.exact_bf16_to_f32",
    }
    for rule in rules:
        assert rule.source_op.name == "vector.extf"
        type_guards = {
            guard.field: guard.type_pattern
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_TYPE
        }
        source_type = type_guards["input"]
        result_type = type_guards["result"]
        assert source_type is not None
        assert result_type is not None
        assert source_type.minimum_lanes == 1
        assert source_type.maximum_lanes == 4
        assert result_type.elements == ("f32",)
        assert result_type.minimum_lanes == 1
        assert result_type.maximum_lanes == 4
        assert any(
            guard.kind is GuardKind.VALUE_STATIC_ELEMENT_COUNT_EQ
            for guard in rule.guards
        )
        assert all(
            "extract_lane" not in emit.descriptor.key
            and "replace_lane" not in emit.descriptor.key
            for emit in rule.emit
        )

    assert {rule.report_key: len(rule.emit) for rule in rules} == {
        "wasm.float_extension.exact_f8e4m3_to_f32": 53,
        "wasm.float_extension.exact_f8e5m2_to_f32": 47,
        "wasm.float_extension.exact_f16_to_f32": 69,
        "wasm.float_extension.exact_bf16_to_f32": 3,
    }


def test_bf16_widening_is_one_extension_and_one_shift():
    rule = _rule("bf16")

    assert [emit.descriptor.key for emit in rule.emit] == [
        "wasm.i32x4.extend_low_i16x8_u",
        "wasm.i32.const",
        "wasm.i32x4.shl",
    ]
    for input_bits in (0x0000, 0x0001, 0x3F80, 0x7F80, 0x7FC1, 0x8000, 0xFFFF):
        assert _evaluate_lane(rule, input_bits) == input_bits << 16


def test_fp8_widening_programs_match_every_encoding():
    for source_element, exponent_bits, mantissa_bits, has_infinity in (
        ("f8E4M3", 4, 3, False),
        ("f8E5M2", 5, 2, True),
    ):
        rule = _rule(source_element)
        for input_bits in range(1 << 8):
            assert _evaluate_lane(rule, input_bits) == _reference_fp8(
                input_bits,
                exponent_bit_count=exponent_bits,
                mantissa_bit_count=mantissa_bits,
                has_infinity=has_infinity,
            ), (source_element, hex(input_bits))


def test_f16_widening_program_matches_boundary_encodings():
    rule = _rule("f16")
    values = {
        0x0000,
        0x0001,
        0x0002,
        0x03FF,
        0x0400,
        0x3C00,
        0x7BFF,
        0x7C00,
        0x7C01,
        0x7E00,
        0x7FFF,
        0x8000,
        0x8001,
        0xBC00,
        0xFC00,
        0xFFFF,
    }
    for input_bits in values:
        assert _evaluate_lane(rule, input_bits) == _reference_f16(input_bits)


@pytest.mark.exhaustive
def test_f16_widening_program_matches_every_encoding():
    rule = _rule("f16")
    for input_bits in range(1 << 16):
        assert _evaluate_lane(rule, input_bits) == _reference_f16(input_bits), hex(
            input_bits
        )
