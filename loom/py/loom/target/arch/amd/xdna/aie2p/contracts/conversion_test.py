# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for exact AMD XDNA AIE2P scalar conversion contracts."""

from __future__ import annotations

import math
import random
import struct
from itertools import pairwise

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.contracts.conversion import (
    AIE2P_CONVERSION_RULES,
)
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    ValueAliasRule,
    ValueRef,
    compile_lower_rule_set,
)
from loom.target.low_descriptors import Constraint, ConstraintKind, OperandRole


def _u32(value: int) -> int:
    return value & 0xFFFFFFFF


def _s32(value: int) -> int:
    value = _u32(value)
    return value - (1 << 32) if value & 0x80000000 else value


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _bits_float(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", _u32(value)))[0]


def _evaluate(rule: DescriptorRule, input_value: int | tuple[int, int]) -> int:
    values: dict[ValueRef, int | tuple[int, int]] = {
        ValueRef.operand("input"): input_value,
    }
    for emit in rule.emit:
        if isinstance(emit, EmitRegisterSlice):
            source = values[emit.source]
            assert isinstance(source, tuple)
            values[emit.result] = source[emit.unit_offset]
            continue
        if isinstance(emit, EmitRegisterConcat):
            assert len(emit.sources) == 2
            low_word = values[emit.sources[0]]
            high_word = values[emit.sources[1]]
            assert isinstance(low_word, int)
            assert isinstance(high_word, int)
            values[emit.result] = (_u32(low_word), _u32(high_word))
            continue

        descriptor_key = emit.descriptor.key.removeprefix("amd.xdna.aie2p.")
        result_ref = next(iter(emit.results.values()))
        if emit.form is DescriptorEmitForm.CONST:
            value = _u32(emit.immediates["i"])
        else:
            operands = {name: values[ref] for name, ref in emit.operands.items()}
            assert all(isinstance(operand, int) for operand in operands.values())
            if descriptor_key == "extend.signed.i8":
                value = operands["s0"] & 0xFF
                if value & 0x80:
                    value |= 0xFFFFFF00
            elif descriptor_key == "extend.signed.i16":
                value = operands["s0"] & 0xFFFF
                if value & 0x8000:
                    value |= 0xFFFF0000
            elif descriptor_key == "extend.unsigned.i8":
                value = operands["s0"] & 0xFF
            elif descriptor_key == "extend.unsigned.i16":
                value = operands["s0"] & 0xFFFF
            elif descriptor_key == "convert.signed.i32.to.f32":
                assert operands["m"] == 0
                value = _float_bits(float(_s32(operands["s0"])))
            elif descriptor_key == "convert.round-nearest.f32.to.signed.i32":
                scale = _s32(operands["m"])
                value = round(math.ldexp(_bits_float(operands["s0"]), scale))
            elif descriptor_key == "and.i32":
                value = operands["s0"] & operands["s1"]
            elif descriptor_key == "or.i32":
                value = operands["s0"] | operands["s1"]
            elif descriptor_key == "add.i32":
                value = operands["s0"] + operands["s1"]
            elif descriptor_key == "add.i32.immediate":
                value = operands["s0"] + emit.immediates["imm"]
            elif descriptor_key == "sub.i32":
                value = operands["s0"] - operands["s1"]
            elif descriptor_key == "lshl.i32":
                shift = _s32(operands["s1"])
                assert -31 <= shift <= 31
                value = (
                    operands["s0"] << shift if shift >= 0 else operands["s0"] >> -shift
                )
            elif descriptor_key == "ashl.i32":
                shift = _s32(operands["s1"])
                assert -31 <= shift <= 31
                value = (
                    operands["s0"] << shift
                    if shift >= 0
                    else _s32(operands["s0"]) >> -shift
                )
            elif descriptor_key == "clz.i32":
                value = 32 - _u32(operands["s0"]).bit_length()
            elif descriptor_key == "cmp.eqz.i32":
                value = int(_u32(operands["s0"]) == 0)
            elif descriptor_key == "cmp.nez.i32":
                value = int(_u32(operands["s0"]) != 0)
            elif descriptor_key == "cmp.eq.i32":
                value = int(_u32(operands["s0"]) == _u32(operands["s1"]))
            elif descriptor_key == "cmp.slt.i32":
                value = int(_s32(operands["s0"]) < _s32(operands["s1"]))
            elif descriptor_key == "cmp.ult.i32":
                value = int(_u32(operands["s0"]) < _u32(operands["s1"]))
            elif descriptor_key == "cmp.uge.i32":
                value = int(_u32(operands["s0"]) >= _u32(operands["s1"]))
            elif descriptor_key == "select.nonzero.i32":
                value = operands["s0"] if operands["s2"] != 0 else operands["s1"]
            else:
                raise AssertionError(f"unmodeled descriptor {descriptor_key}")
            value = _u32(value)

        values[result_ref] = _u32(value)
    result = values[ValueRef.result("result")]
    if isinstance(result, tuple):
        return result[0] | (result[1] << 32)
    return result


def _reference_f16_to_f32(input_bits: int) -> int:
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


def _reference_fp8_to_f32(
    input_bits: int,
    *,
    exponent_bits: int,
    mantissa_bits: int,
    has_infinity: bool,
) -> int:
    sign = input_bits & 0x80
    exponent_mask = (1 << exponent_bits) - 1
    exponent = (input_bits >> mantissa_bits) & exponent_mask
    mantissa_mask = (1 << mantissa_bits) - 1
    mantissa = input_bits & mantissa_mask
    if exponent == exponent_mask:
        if has_infinity:
            if mantissa != 0:
                return 0x7FC00000
            return (sign << 24) | 0x7F800000
        if mantissa == mantissa_mask:
            return 0x7FC00000

    significand = mantissa if exponent == 0 else (1 << mantissa_bits) | mantissa
    if significand == 0:
        return sign << 24
    bias = (1 << (exponent_bits - 1)) - 1
    scale = (1 if exponent == 0 else exponent) - bias - mantissa_bits
    value = math.ldexp(float(significand), scale)
    return _float_bits(-value if sign else value)


def _reference_f32_to_bf16(input_bits: int) -> int:
    exponent = input_bits & 0x7F800000
    fraction = input_bits & 0x007FFFFF
    upper = input_bits >> 16
    if exponent == 0x7F800000 and fraction != 0:
        return upper | 0x0040
    return ((input_bits + 0x7FFF + (upper & 1)) & 0xFFFFFFFF) >> 16


def _round_unsigned_to_even(value: int, shift: int) -> int:
    truncated = value >> shift
    remainder = value & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    if remainder > halfway or (remainder == halfway and truncated & 1):
        truncated += 1
    return truncated


def _reference_f32_to_fp8(
    input_bits: int,
    *,
    exponent_bits: int,
    mantissa_bits: int,
    has_infinity: bool,
) -> int:
    sign = (input_bits >> 24) & 0x80
    exponent = (input_bits >> 23) & 0xFF
    fraction = input_bits & 0x007FFFFF
    exponent_mask = ((1 << exponent_bits) - 1) << mantissa_bits
    mantissa_mask = (1 << mantissa_bits) - 1
    nan_payload = exponent_mask | mantissa_mask
    maximum_finite = nan_payload - 1
    if exponent == 0xFF:
        if fraction != 0:
            return sign | nan_payload
        return sign | (exponent_mask if has_infinity else maximum_finite)
    if exponent == 0:
        return sign

    exponent_bias = (1 << (exponent_bits - 1)) - 1
    arithmetic_exponent = exponent - 127
    maximum_arithmetic_exponent = (1 << (exponent_bits - 1)) - int(has_infinity)
    if arithmetic_exponent > maximum_arithmetic_exponent:
        return sign | (exponent_mask if has_infinity else maximum_finite)

    if arithmetic_exponent + exponent_bias <= 0:
        shift = 23 - mantissa_bits - arithmetic_exponent + 1 - exponent_bias
        if shift < 0 or shift > 24:
            magnitude = 0
        else:
            magnitude = _round_unsigned_to_even(0x00800000 | fraction, shift)
    else:
        shift = 23 - mantissa_bits
        rounded_fraction = _round_unsigned_to_even(fraction, shift)
        if rounded_fraction > mantissa_mask:
            rounded_fraction = 0
            arithmetic_exponent += 1
        magnitude = (
            (arithmetic_exponent + exponent_bias) << mantissa_bits
        ) | rounded_fraction

    if not has_infinity and magnitude >= nan_payload:
        magnitude = maximum_finite
    return sign | magnitude


def _reference_f32_to_f16(input_bits: int) -> int:
    sign = (input_bits >> 16) & 0x8000
    nonsign = input_bits & 0x7FFFFFFF
    exponent = nonsign >> 23
    fraction = nonsign & 0x007FFFFF
    if exponent == 0xFF:
        if fraction == 0:
            return sign | 0x7C00
        return sign | 0x7C00 | (fraction >> 13) | 0x0200
    if exponent >= 143:
        return sign | 0x7C00
    if exponent >= 113:
        rebased = nonsign - 0x38000000
        return sign | _round_unsigned_to_even(rebased, 13)
    if exponent < 102:
        return sign
    significand = fraction | 0x00800000
    return sign | _round_unsigned_to_even(significand, 126 - exponent)


def _reference_integer_to_f16(value: int) -> int:
    sign = 0
    magnitude = value
    if value < 0:
        sign = 0x8000
        magnitude = -value
    if magnitude == 0:
        return sign
    exponent = magnitude.bit_length() - 1
    shift = max(exponent - 10, 0)
    significand = _round_unsigned_to_even(magnitude, shift) if shift else magnitude
    if significand == 0x0800:
        significand = 0x0400
        exponent += 1
    if exponent >= 16:
        return sign | 0x7C00
    significand <<= max(10 - exponent, 0)
    return sign | ((exponent + 15) << 10) | (significand & 0x03FF)


def _rule(report_key: str) -> DescriptorRule:
    return next(
        rule
        for rule in AIE2P_CONVERSION_RULES
        if isinstance(rule, DescriptorRule) and rule.report_key == report_key
    )


def _integer_conversion_case(
    source_op,
    input_element: str,
    result_element: str,
) -> DescriptorRule | ValueAliasRule:
    candidates = []
    for case in AIE2P_CONVERSION_RULES:
        if case.source_op is not source_op:
            continue
        guarded_types = {
            guard.field: guard.type_pattern.element
            for guard in case.guards
            if guard.type_pattern is not None
        }
        if guarded_types == {
            "input": input_element,
            "result": result_element,
        }:
            candidates.append(case)
    assert len(candidates) == 1
    return candidates[0]


def _evaluate_integer_conversion(
    case: DescriptorRule | ValueAliasRule,
    input_value: int,
    input_bits: int,
) -> int:
    if isinstance(case, ValueAliasRule):
        return input_value
    physical_input: int | tuple[int, int] = input_value
    if input_bits == 64:
        physical_input = (_u32(input_value), _u32(input_value >> 32))
    return _evaluate(case, physical_input)


def test_conversion_programs_are_valid_compact_descriptor_data() -> None:
    for rule in AIE2P_CONVERSION_RULES:
        rule.validate(AIE2P_CORE_DESCRIPTOR_SET)
    compiled = compile_lower_rule_set(
        ContractFragment(
            name="amd.xdna.aie2p.conversion.test",
            descriptor_set=AIE2P_CORE_DESCRIPTOR_SET,
            cases=AIE2P_CONVERSION_RULES,
        ),
        dialect_ops={"scalar": ALL_SCALAR_OPS, "vector": ALL_VECTOR_OPS},
    )
    assert all(
        rule.emit_count > 0 or rule.alias_ref_count == 1 for rule in compiled.rules
    )

    for report_key in (
        "native_signed_i8_to_binary32",
        "native_signed_i16_to_binary32",
        "native_signed_i32_to_binary32",
        "native_unsigned_i8_to_binary32",
        "native_unsigned_i16_to_binary32",
    ):
        rule = _rule(report_key)
        assert rule.emit[-2].descriptor.key == (
            "amd.xdna.aie2p.constant.i32.fx2flt-scale"
        )
        assert rule.emit[-1].descriptor.key == (
            "amd.xdna.aie2p.convert.signed.i32.to.f32"
        )


def test_integer_width_conversions_match_exact_oracles() -> None:
    conversions = (
        (scalar_conversion.scalar_extsi, 8, 16, True),
        (scalar_conversion.scalar_extsi, 8, 32, True),
        (scalar_conversion.scalar_extsi, 8, 64, True),
        (scalar_conversion.scalar_extsi, 16, 32, True),
        (scalar_conversion.scalar_extsi, 16, 64, True),
        (scalar_conversion.scalar_extsi, 32, 64, True),
        (scalar_conversion.scalar_extui, 8, 16, False),
        (scalar_conversion.scalar_extui, 8, 32, False),
        (scalar_conversion.scalar_extui, 8, 64, False),
        (scalar_conversion.scalar_extui, 16, 32, False),
        (scalar_conversion.scalar_extui, 16, 64, False),
        (scalar_conversion.scalar_extui, 32, 64, False),
        (scalar_conversion.scalar_trunci, 16, 8, False),
        (scalar_conversion.scalar_trunci, 32, 8, False),
        (scalar_conversion.scalar_trunci, 32, 16, False),
        (scalar_conversion.scalar_trunci, 64, 8, False),
        (scalar_conversion.scalar_trunci, 64, 16, False),
        (scalar_conversion.scalar_trunci, 64, 32, False),
    )
    random_source = random.Random(0xA1E2_1064)
    for source_op, input_bits, result_bits, signed in conversions:
        case = _integer_conversion_case(
            source_op,
            f"i{input_bits}",
            f"i{result_bits}",
        )
        input_mask = (1 << input_bits) - 1
        result_mask = (1 << result_bits) - 1
        values = [
            0,
            1,
            input_mask >> 1,
            1 << (input_bits - 1),
            input_mask,
        ]
        if input_bits == 8:
            values = list(range(1 << input_bits))
        else:
            values.extend(random_source.getrandbits(input_bits) for _ in range(2048))
        for input_value in values:
            reference = input_value
            if signed and input_value & (1 << (input_bits - 1)):
                reference -= 1 << input_bits
            actual = _evaluate_integer_conversion(
                case,
                input_value,
                input_bits,
            )
            expected = reference & result_mask
            assert actual & result_mask == expected, (
                source_op.name,
                input_bits,
                result_bits,
                input_value,
                actual,
                expected,
            )


def test_native_integer_conversion_preserves_scale_and_status_state() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_CORE_DESCRIPTOR_SET.descriptors
    }
    scale = descriptors["amd.xdna.aie2p.constant.i32.fx2flt-scale"]
    assert scale.operands[0].reg_alts[0].reg_class == "aie2p.ms2"
    assert scale.constraints[0].kind is ConstraintKind.REMATERIALIZABLE

    convert = descriptors["amd.xdna.aie2p.convert.signed.i32.to.f32"]
    assert [operand.field_name for operand in convert.operands] == [
        "d0",
        "s0",
        "m",
        "implicit_def_srfpcnvfx2fl",
        "implicit_use_crfpcnvfx2flmask",
    ]
    assert [operand.reg_alts[0].reg_class for operand in convert.operands] == [
        "aie2p.er",
        "aie2p.er",
        "aie2p.ms2",
        "aie2p.state.srfpcnvfx2fl",
        "aie2p.state.crfpcnvfx2flmask",
    ]
    assert convert.operands[0].role is OperandRole.RESULT
    assert [
        (operand.read_stage, operand.ready_stage) for operand in convert.operands
    ] == [
        (0, 2),
        (1, 0),
        (1, 0),
        (0, 1),
        (1, 0),
    ]
    reverse_scale = descriptors["amd.xdna.aie2p.constant.i32.flt2fx-scale"]
    assert reverse_scale.operands[0].reg_alts[0].reg_class == "aie2p.ms3"
    assert Constraint(ConstraintKind.REMATERIALIZABLE, 0) in reverse_scale.constraints

    reverse_convert = descriptors[
        "amd.xdna.aie2p.convert.round-nearest.f32.to.signed.i32"
    ]
    assert [operand.field_name for operand in reverse_convert.operands] == [
        "d0",
        "s0",
        "m",
        "implicit_def_srfpcnvfl2fx",
        "implicit_use_crfpcnvfl2fxmask",
    ]
    assert [operand.reg_alts[0].reg_class for operand in reverse_convert.operands] == [
        "aie2p.er",
        "aie2p.er",
        "aie2p.ms3",
        "aie2p.state.srfpcnvfl2fx",
        "aie2p.state.crfpcnvfl2fxmask",
    ]
    assert [
        (operand.read_stage, operand.ready_stage)
        for operand in reverse_convert.operands
    ] == [
        (0, 2),
        (1, 0),
        (1, 0),
        (0, 1),
        (1, 0),
    ]


def test_native_bfloat16_packet_conversions_preserve_exact_width_and_rounding() -> None:
    narrow = _rule("native_binary32x32_to_bfloat16x32")
    assert [emit.descriptor.key for emit in narrow.emit] == [
        "amd.xdna.aie2p.state.rounding.immediate",
        "amd.xdna.aie2p.convert.f32x32.to.bf16x32",
    ]
    assert narrow.emit[0].immediates == {"i": 12}

    widen = _rule("native_bfloat16x32_to_binary32x32")
    assert [emit.descriptor.key for emit in widen.emit] == [
        "amd.xdna.aie2p.convert.bf16x32.to.f32x32"
    ]


def test_native_integer_packet_conversions_are_compact_exact_rules() -> None:
    widening_shapes = (
        ("i16", "i32", 16, "2x.w-to-b", 0, True, 1, False),
        ("i32", "i64", 8, "2x.w-to-b", 1, True, 1, False),
        ("i8", "i32", 32, "4x.w-to-c", 0, True, 2, False),
        ("i16", "i64", 16, "4x.w-to-c", 1, True, 2, False),
        ("i16", "i32", 32, "2x.x-to-c", 0, False, 2, False),
        ("i32", "i64", 16, "2x.x-to-c", 1, False, 2, False),
        ("i8", "i32", 64, "4x.x-to-d", 0, False, 4, True),
        ("i16", "i64", 32, "4x.x-to-d", 1, False, 4, True),
    )
    for source_op, signedness in (
        (vector.vector_extui, "unsigned"),
        (vector.vector_extsi, "signed"),
    ):
        for (
            input_element,
            result_element,
            lane_count,
            physical_shape,
            ups_mode,
            sliced,
            accumulator_unit_count,
            direct_accumulator_result,
        ) in widening_shapes:
            rule = _rule(
                f"native_{signedness}_{input_element}x{lane_count}_to_"
                f"{result_element}x{lane_count}"
            )
            assert rule.source_op is source_op
            assert rule.descriptor.key == (
                f"amd.xdna.aie2p.widen.{physical_shape}.{signedness}.configured"
            )
            input_slices = [
                emit
                for emit in rule.emit
                if isinstance(emit, EmitRegisterSlice)
                and emit.result.field == "source_w"
            ]
            assert len(input_slices) == int(sliced)
            descriptor_keys = [
                emit.descriptor.key
                for emit in rule.emit
                if isinstance(emit, EmitDescriptorOp)
            ]
            assert descriptor_keys[:4] == [
                "amd.xdna.aie2p.constant.i32.shift",
                "amd.xdna.aie2p.state.saturation.immediate",
                "amd.xdna.aie2p.state.ups-mode.immediate",
                rule.descriptor.key,
            ]
            assert descriptor_keys[4:] == (
                []
                if direct_accumulator_result
                else ["amd.xdna.aie2p.move.accumulator512.to.vector512"]
                * accumulator_unit_count
            )
            set_ups_mode = next(
                emit
                for emit in rule.emit
                if isinstance(emit, EmitDescriptorOp)
                and emit.descriptor.key == "amd.xdna.aie2p.state.ups-mode.immediate"
            )
            assert set_ups_mode.immediates == {"i": ups_mode}

    for report_key, source_op, pack_size in (
        ("native_trunc_i16x32_to_i8x32", vector.vector_trunci, 1),
        ("native_trunc_i16x64_to_i8x64", vector.vector_trunci, 1),
        ("native_bitpack_i8x64_to_i4x64", vector.vector_bitpack, 0),
        ("native_bitpack_i8x128_to_i4x128", vector.vector_bitpack, 0),
    ):
        rule = _rule(report_key)
        assert rule.source_op is source_op
        assert [
            emit.descriptor.key
            for emit in rule.emit
            if isinstance(emit, EmitDescriptorOp)
        ] == [
            "amd.xdna.aie2p.state.saturation.immediate",
            "amd.xdna.aie2p.state.pack-size.immediate",
            rule.descriptor.key,
        ]
        assert rule.emit[0].immediates == {"i": 0}
        assert rule.emit[1].immediates == {"i": pack_size}


def test_binary32_to_integer_programs_match_truncation_oracles() -> None:
    signed = _rule("exact_binary32_to_signed_integer")
    unsigned_narrow = _rule("exact_binary32_to_narrow_unsigned_integer")
    unsigned_i32 = _rule("exact_binary32_to_unsigned_i32")

    signed_result_guard = next(
        guard for guard in signed.guards if guard.field == "result"
    )
    assert set(signed_result_guard.type_pattern.elements) == {"i8", "i16", "i32"}
    unsigned_result_guard = next(
        guard for guard in unsigned_narrow.guards if guard.field == "result"
    )
    assert set(unsigned_result_guard.type_pattern.elements) == {"i8", "i16"}

    signed_bits = {
        _float_bits(value)
        for value in (
            -(2**31),
            -65535.75,
            -32768.75,
            -255.75,
            -128.75,
            -2.5,
            -1.75,
            -1.5,
            -0.75,
            -0.5,
            -0.0,
            0.0,
            0.5,
            0.75,
            1.5,
            1.75,
            127.75,
            255.75,
            32767.75,
            65535.75,
            2**24 + 2,
            2**31 - 128,
        )
    }
    unsigned_bits = {
        _float_bits(value)
        for value in (
            0.0,
            0.5,
            0.75,
            1.5,
            1.75,
            127.75,
            255.75,
            32767.75,
            65535.75,
            2**24 + 2,
            2**31 - 128,
            2**31,
            2**31 + 256,
            3 * (2**30),
            2**32 - 256,
        )
    }
    random_source = random.Random(0xA1E2_F2C0)
    while len(signed_bits) < 1024:
        bits = random_source.getrandbits(32)
        value = _bits_float(bits)
        if math.isfinite(value) and -(2**31) <= value < 2**31:
            signed_bits.add(bits)
    while len(unsigned_bits) < 1024:
        bits = random_source.getrandbits(31)
        value = _bits_float(bits)
        if math.isfinite(value) and 0 <= value < 2**32:
            unsigned_bits.add(bits)

    for bits in signed_bits:
        value = _bits_float(bits)
        assert _s32(_evaluate(signed, bits)) == math.trunc(value)
    for bits in unsigned_bits:
        value = _bits_float(bits)
        expected = math.trunc(value)
        assert _evaluate(unsigned_i32, bits) == expected
        if value < 2**16:
            assert _evaluate(unsigned_narrow, bits) == expected


def test_integer_to_f32_programs_match_binary32_oracle() -> None:
    signed_i8 = _rule("native_signed_i8_to_binary32")
    unsigned_i8 = _rule("native_unsigned_i8_to_binary32")
    for bits in range(1 << 8):
        signed = bits - 256 if bits & 0x80 else bits
        assert _evaluate(signed_i8, bits) == _float_bits(float(signed))
        assert _evaluate(unsigned_i8, bits) == _float_bits(float(bits))

    signed_i16 = _rule("native_signed_i16_to_binary32")
    unsigned_i16 = _rule("native_unsigned_i16_to_binary32")
    for bits in range(1 << 16):
        assert _evaluate(signed_i16, bits) == _float_bits(
            float(bits - (1 << 16) if bits & 0x8000 else bits)
        )
        assert _evaluate(unsigned_i16, bits) == _float_bits(float(bits))

    signed_i32 = _rule("native_signed_i32_to_binary32")
    values = [
        0,
        1,
        -1,
        2**24 - 1,
        2**24,
        2**24 + 1,
        2**31 - 1,
        -(2**31),
    ]
    random_source = random.Random(0xA1E2C0DE)
    values.extend(_s32(random_source.getrandbits(32)) for _ in range(2048))
    for value in values:
        assert _evaluate(signed_i32, _u32(value)) == _float_bits(float(value))

    unsigned_i32 = _rule("exact_unsigned_i32_to_binary32")
    unsigned_values = [
        0,
        1,
        2**24 - 1,
        2**24,
        2**24 + 1,
        2**31 - 1,
        2**31,
        2**31 + 1,
        2**32 - 1,
    ]
    unsigned_values.extend(random_source.getrandbits(32) for _ in range(2048))
    for value in unsigned_values:
        assert _evaluate(unsigned_i32, value) == _float_bits(float(value))


def test_16bit_float_widening_programs_cover_complete_magnitudes() -> None:
    f16 = _rule("exact_f16_to_binary32")
    bf16 = _rule("exact_bf16_to_binary32")
    for bits in range(1 << 15):
        assert _evaluate(f16, bits) == _reference_f16_to_f32(bits)
    negative_payloads = {
        0x8000 | (exponent << 10) | mantissa
        for exponent in range(32)
        for mantissa in (0, 1, 2, 0x1FF, 0x200, 0x3FE, 0x3FF)
    }
    for bits in negative_payloads:
        assert _evaluate(f16, bits) == _reference_f16_to_f32(bits)
    for bits in range(1 << 16):
        assert _evaluate(bf16, bits) == bits << 16


def test_float8_widening_programs_match_exhaustive_oracles() -> None:
    formats = (
        ("f8e4m3", 4, 3, False),
        ("f8e5m2", 5, 2, True),
    )
    for source_name, exponent_bits, mantissa_bits, has_infinity in formats:
        f16 = _rule(f"exact_{source_name}_to_f16")
        bf16 = _rule(f"exact_{source_name}_to_bf16")
        f32 = _rule(f"exact_{source_name}_to_f32")
        for bits in range(1 << 8):
            f32_bits = _reference_fp8_to_f32(
                bits,
                exponent_bits=exponent_bits,
                mantissa_bits=mantissa_bits,
                has_infinity=has_infinity,
            )
            assert _evaluate(f32, bits) == f32_bits
            assert _evaluate(bf16, bits) == f32_bits >> 16
            actual_f16 = _evaluate(f16, bits)
            if f32_bits & 0x7FFFFFFF > 0x7F800000:
                assert actual_f16 & 0x7C00 == 0x7C00
                assert actual_f16 & 0x03FF != 0
            else:
                assert actual_f16 == _reference_f32_to_f16(f32_bits), (
                    source_name,
                    hex(bits),
                    hex(actual_f16),
                    hex(_reference_f32_to_f16(f32_bits)),
                )


def test_float8_narrowing_programs_match_rounding_oracles() -> None:
    formats = (
        ("f8e4m3", 4, 3, False, 0x7E),
        ("f8e5m2", 5, 2, True, 0x7B),
    )
    random_source = random.Random(0xA1E2_F008)
    for result_name, exponent_bits, mantissa_bits, has_infinity, max_finite in formats:
        f32_rule = _rule(f"exact_f32_to_{result_name}")
        values = {
            0x00000000,
            0x80000000,
            0x00000001,
            0x007FFFFF,
            0x00800000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7F800001,
            0x7FC00000,
            0x7FFFFFFF,
            0xFF800001,
            0xFFC00000,
            0xFFFFFFFF,
        }
        decoded_positive = [
            _reference_fp8_to_f32(
                bits,
                exponent_bits=exponent_bits,
                mantissa_bits=mantissa_bits,
                has_infinity=has_infinity,
            )
            for bits in range(max_finite + 1)
        ]
        for f32_bits in decoded_positive:
            values.update(
                {
                    _u32(f32_bits - 1),
                    f32_bits,
                    _u32(f32_bits + 1),
                    f32_bits | 0x80000000,
                }
            )
        for lhs_bits, rhs_bits in pairwise(decoded_positive):
            midpoint = (_bits_float(lhs_bits) + _bits_float(rhs_bits)) * 0.5
            midpoint_bits = _float_bits(midpoint)
            negative_midpoint_bits = _float_bits(-midpoint)
            values.update(
                {
                    _u32(midpoint_bits - 1),
                    midpoint_bits,
                    _u32(midpoint_bits + 1),
                    _u32(negative_midpoint_bits - 1),
                    negative_midpoint_bits,
                    _u32(negative_midpoint_bits + 1),
                }
            )
        values.update(random_source.getrandbits(32) for _ in range(1024))
        for bits in values:
            expected = _reference_f32_to_fp8(
                bits,
                exponent_bits=exponent_bits,
                mantissa_bits=mantissa_bits,
                has_infinity=has_infinity,
            )
            assert _evaluate(f32_rule, bits) == expected, hex(bits)

        f16_rule = _rule(f"exact_f16_to_{result_name}")
        bf16_rule = _rule(f"exact_bf16_to_{result_name}")
        source_payloads = {
            0x0000,
            0x8000,
            0x0001,
            0x03FF,
            0x0400,
            0x3C00,
            0x7BFF,
            0x7C00,
            0xFC00,
            0x7C01,
            0x7E00,
            0xFFFF,
        }
        source_payloads.update(random_source.getrandbits(16) for _ in range(256))
        for bits in source_payloads:
            f16_source = _reference_f16_to_f32(bits)
            bf16_source = bits << 16
            assert _evaluate(f16_rule, bits) == _reference_f32_to_fp8(
                f16_source,
                exponent_bits=exponent_bits,
                mantissa_bits=mantissa_bits,
                has_infinity=has_infinity,
            )
            assert _evaluate(bf16_rule, bits) == _reference_f32_to_fp8(
                bf16_source,
                exponent_bits=exponent_bits,
                mantissa_bits=mantissa_bits,
                has_infinity=has_infinity,
            )


def test_binary32_to_bfloat16_program_matches_rounding_oracle() -> None:
    rule = _rule("exact_binary32_to_bfloat16")
    values = {
        0x00000000,
        0x80000000,
        0x00000001,
        0x007FFFFF,
        0x00800000,
        0x3F7F7FFF,
        0x3F7F8000,
        0x3F7F8001,
        0x3F808000,
        0x7F7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7F800001,
        0x7FC00000,
        0x7FFFFFFF,
        0xFF800001,
        0xFFC00000,
        0xFFFFFFFF,
    }
    random_source = random.Random(0xA1E2_BF16)
    values.update(random_source.getrandbits(32) for _ in range(1024))
    for bits in values:
        assert _evaluate(rule, bits) == _reference_f32_to_bf16(bits)


def test_binary32_to_f16_program_matches_rounding_oracle() -> None:
    rule = _rule("exact_binary32_to_f16")
    values = {
        0x00000000,
        0x80000000,
        0x33000000,
        0x33000001,
        0x337FFFFF,
        0x33800000,
        0x387FC000,
        0x387FE000,
        0x387FFFFF,
        0x38800000,
        0x3F7FEFFF,
        0x3F7FF000,
        0x3F7FF001,
        0x477FDFFF,
        0x477FE000,
        0x477FFFFF,
        0x47800000,
        0x7F7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7F800001,
        0x7FC00000,
        0x7FFFFFFF,
        0xFF800001,
        0xFFC00000,
        0xFFFFFFFF,
    }
    # Exercise representative mantissa, exponent, carry, and subnormal
    # boundaries without redundantly evaluating every binary16 value through
    # the same already-exhaustive widening program.
    for exponent in range(32):
        for mantissa in (
            0,
            1,
            2,
            3,
            0x1FE,
            0x1FF,
            0x200,
            0x201,
            0x3FC,
            0x3FD,
            0x3FE,
            0x3FF,
        ):
            widened = _reference_f16_to_f32((exponent << 10) | mantissa)
            values.update(
                {
                    _u32(widened - 1),
                    widened,
                    _u32(widened + 1),
                }
            )
    random_source = random.Random(0xA1E2_F16)
    values.update(random_source.getrandbits(32) for _ in range(2048))
    for bits in values:
        assert _evaluate(rule, bits) == _reference_f32_to_f16(bits), hex(bits)


def test_integer_to_f16_programs_match_direct_rounding_oracle() -> None:
    signed_i8 = _rule("exact_signed_i8_to_f16")
    unsigned_i8 = _rule("exact_unsigned_i8_to_f16")
    for bits in range(1 << 8):
        signed = bits - (1 << 8) if bits & 0x80 else bits
        assert _evaluate(signed_i8, bits) == _reference_integer_to_f16(signed)
        assert _evaluate(unsigned_i8, bits) == _reference_integer_to_f16(bits)

    signed_i16 = _rule("exact_signed_i16_to_f16")
    unsigned_i16 = _rule("exact_unsigned_i16_to_f16")
    random_source = random.Random(0xA1E2_1F16)
    i16_values = {
        *range(1 << 8),
        0x03FF,
        0x0400,
        0x07FF,
        0x0800,
        0x0801,
        0x7FFF,
        0x8000,
        0x8001,
        0xF7FF,
        0xF800,
        0xF801,
        0xFFFF,
    }
    i16_values.update(random_source.getrandbits(16) for _ in range(1024))
    for bits in i16_values:
        signed = bits - (1 << 16) if bits & 0x8000 else bits
        assert _evaluate(signed_i16, bits) == _reference_integer_to_f16(signed)
        assert _evaluate(unsigned_i16, bits) == _reference_integer_to_f16(bits)

    signed_i32 = _rule("exact_signed_i32_to_f16")
    unsigned_i32 = _rule("exact_unsigned_i32_to_f16")
    values = {
        0,
        1,
        2047,
        2048,
        2049,
        65503,
        65504,
        65519,
        65520,
        65521,
        2**31 - 1,
        2**31,
        2**32 - 1,
    }
    values.update(random_source.getrandbits(32) for _ in range(1024))
    for bits in values:
        signed = _s32(bits)
        assert _evaluate(signed_i32, bits) == _reference_integer_to_f16(signed)
        assert _evaluate(unsigned_i32, bits) == _reference_integer_to_f16(bits)
