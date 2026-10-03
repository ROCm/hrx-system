# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for exact AMD XDNA AIE2P scalar conversion contracts."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable
from dataclasses import replace

import pytest

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amd.xdna.aie2p.contracts.conversion import (
    AIE2P_CONVERSION_RULES,
)
from loom.target.arch.amd.xdna.aie2p.contracts.packet_conversion import (
    FLOAT8_PACKET_FORMATS,
    FLOAT_PACKET_SOURCE_FORMATS,
    INTEGER_PACK_INSTRUCTIONS,
    INTEGER_PACK_RULE_SHAPES,
    INTEGER_SHIFT_RULE_SHAPES,
    INTEGER_TRUNCATION_INSTRUCTIONS,
    INTEGER_TRUNCATION_RULE_SHAPES,
    INTEGER_WIDEN_RULE_SHAPES,
    MXFP8_E4M3FN_E8M0_X8_SCHEMA,
    MXFP8_E4M3FN_E8M0_X32_SCHEMA,
    Float8PacketFormat,
    FloatPacketSourceFormat,
    IntegerPackInstruction,
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
    Guard,
    ValueAliasRule,
    ValueRef,
    Vector,
    compile_lower_rule_set,
)
from loom.target.low_descriptors import Constraint, ConstraintKind, OperandRole

_CANONICAL_BF16_NAN = 0x7FC0


def _u32(value: int) -> int:
    return value & 0xFFFFFFFF


def _s32(value: int) -> int:
    value = _u32(value)
    return value - (1 << 32) if value & 0x80000000 else value


def _s16(value: int) -> int:
    value &= 0xFFFF
    return value - (1 << 16) if value & 0x8000 else value


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


_MXFP8_PACKET_DESCRIPTOR_KEYS = frozenset(
    {
        "and.i32",
        "cmp.eq.i32",
        "cmp.eqz.i32",
        "cmp.ge.unsigned.i16x32.el.low32",
        "cmp.slt.i32",
        "cmp.ult.i32",
        "convert.floor.bf16x16.to.i32x16",
        "extract.i32.immediate",
        "lshl.i32",
        "multiply.i16x32.configured",
        "narrow.trunc.signed.i16x32",
        "select.i32x16",
        "select.mask.i32",
        "select.nonzero.i32",
        "sub.i32",
    }
)


def _evaluate_mxfp8_packet_descriptor(
    descriptor_key: str,
    operands: dict[str, int],
    immediates: dict[str, int],
    state: dict[str, int],
) -> int:
    """Evaluates one lane of an MXFP8-specific packet descriptor."""

    if descriptor_key == "extract.i32.immediate":
        assert immediates["idx"] == 0
        return operands["s1"] & 0xFFFFFFFF
    if descriptor_key == "and.i32":
        return operands["s0"] & operands["s1"]
    if descriptor_key == "sub.i32":
        return operands["s0"] - operands["s1"]
    if descriptor_key == "lshl.i32":
        shift = _s32(operands["s1"])
        assert -31 <= shift <= 31
        return operands["s0"] << shift if shift >= 0 else operands["s0"] >> -shift
    if descriptor_key == "cmp.slt.i32":
        return int(_s32(operands["s0"]) < _s32(operands["s1"]))
    if descriptor_key == "cmp.ult.i32":
        return int(_u32(operands["s0"]) < _u32(operands["s1"]))
    if descriptor_key == "cmp.eq.i32":
        return int(_u32(operands["s0"]) == _u32(operands["s1"]))
    if descriptor_key == "cmp.eqz.i32":
        return int(_u32(operands["s0"]) == 0)
    if descriptor_key == "select.nonzero.i32":
        return operands["s0"] if operands["s2"] else operands["s1"]
    if descriptor_key == "select.mask.i32":
        return immediates["imm"] if operands["s0"] else 0
    if descriptor_key == "cmp.ge.unsigned.i16x32.el.low32":
        return int((operands["s1"] & 0xFFFF) >= (operands["s2"] & 0xFFFF))
    if descriptor_key == "select.i32x16":
        return operands["s1"] if operands["sel"] else operands["s2"]
    if descriptor_key == "convert.floor.bf16x16.to.i32x16":
        bf16_value = _bits_float((operands["src"] & 0xFFFF) << 16)
        # The recipe's retained payload predicate replaces NaN lanes, so the
        # intermediate integer value is unobservable there.
        return (
            0
            if math.isnan(bf16_value)
            else math.floor(math.ldexp(bf16_value, operands["shft"]))
        )
    if descriptor_key == "multiply.i16x32.configured":
        assert operands["acc"] == 858
        return _s16(operands["s1"]) * _s16(operands["s2"])
    if descriptor_key == "narrow.trunc.signed.i16x32":
        assert state["rounding"] == 0
        assert state["srs-mode"] == 1
        assert state["saturation"] == 0
        return (_s32(operands["src"]) >> operands["su"]) & 0xFFFF
    raise AssertionError(f"unmodeled MXFP8 packet descriptor {descriptor_key}")


def _evaluate_packet_lane(
    rule: DescriptorRule, input_value: int, *, scale_value: int | None = None
) -> int:
    """Evaluates one replicated lane through a native packet program."""

    values: dict[ValueRef, int]
    if scale_value is None:
        values = {ValueRef.operand("input"): input_value}
    else:
        values = {
            ValueRef.operand("payload"): input_value,
            ValueRef.operand("auxiliary", element=0): scale_value,
        }
    state: dict[str, int] = {}
    for emit in rule.emit:
        if isinstance(emit, EmitRegisterSlice):
            values[emit.result] = values[emit.source]
            continue
        if isinstance(emit, EmitRegisterConcat):
            source_values = [values[source] for source in emit.sources]
            # This evaluator follows lane zero. Register concatenation places
            # the first source in the low lanes containing that observation.
            values[emit.result] = source_values[0]
            continue

        descriptor_key = emit.descriptor.key.removeprefix("amd.xdna.aie2p.")
        if not emit.results:
            assert descriptor_key.startswith("state.")
            state[descriptor_key.removeprefix("state.").removesuffix(".immediate")] = (
                emit.immediates["i"]
            )
            continue
        result_ref = next(iter(emit.results.values()))
        if emit.form is DescriptorEmitForm.CONST:
            value = emit.immediates["i"]
        else:
            operands = {name: values[ref] for name, ref in emit.operands.items()}
            if descriptor_key in _MXFP8_PACKET_DESCRIPTOR_KEYS:
                value = _evaluate_mxfp8_packet_descriptor(
                    descriptor_key, operands, emit.immediates, state
                )
            elif descriptor_key == "sub.i8x64":
                value = (operands["s1"] - operands["s2"]) & 0xFF
            elif descriptor_key == "shuffle.x.configured":
                control = operands["mod"]
                if control == 20:
                    value = (operands["s1"] & 0xFF) | ((operands["s2"] & 0xFF) << 8)
                else:
                    assert control in (0, 2, 4)
                    sublane_bits = 8 << (control // 2)
                    value = operands["s1"] & ((1 << sublane_bits) - 1)
            elif descriptor_key == "splat.i16x32":
                value = operands["src"] & 0xFFFF
            elif descriptor_key == "splat.i32x16":
                value = operands["src"] & 0xFFFFFFFF
            elif descriptor_key == "add.i16x32":
                value = (operands["s1"] + operands["s2"]) & 0xFFFF
            elif descriptor_key == "sub.i16x32":
                value = (operands["s1"] - operands["s2"]) & 0xFFFF
            elif descriptor_key == "sub.i32x16":
                value = (operands["s1"] - operands["s2"]) & 0xFFFFFFFF
            elif descriptor_key == "max.signed.i16x32":
                value = max(_s16(operands["s1"]), _s16(operands["s2"])) & 0xFFFF
            elif descriptor_key == "min.unsigned.i16x32":
                value = min(operands["s1"] & 0xFFFF, operands["s2"] & 0xFFFF)
            elif descriptor_key == "max.unsigned.i16x32":
                value = max(operands["s1"] & 0xFFFF, operands["s2"] & 0xFFFF)
            elif descriptor_key == "and.bits512":
                value = operands["s1"] & operands["s2"]
            elif descriptor_key == "or.bits512":
                value = operands["s1"] | operands["s2"]
            elif descriptor_key == "cmp.eqz.i16x32.el.low32":
                value = int((operands["s2"] & 0xFFFF) == 0)
            elif descriptor_key == "cmp.lt.unsigned.i16x32.el.low32":
                value = int((operands["s1"] & 0xFFFF) < (operands["s2"] & 0xFFFF))
            elif descriptor_key == "cmp.lt.unsigned.i32x16.el.low32":
                value = int(
                    (operands["s1"] & 0xFFFFFFFF) < (operands["s2"] & 0xFFFFFFFF)
                )
            elif descriptor_key == "cmp.ge.unsigned.i32x16.el.low32":
                value = int(
                    (operands["s1"] & 0xFFFFFFFF) >= (operands["s2"] & 0xFFFFFFFF)
                )
            elif descriptor_key == "predicate.complete.zero.high32":
                value = operands["storage"]
            elif descriptor_key == "select.i16x32.mask64":
                value = operands["s2"] if operands["sel"] else operands["s1"]
            elif descriptor_key.startswith("move."):
                value = operands["src"]
            elif descriptor_key.startswith("widen.2x."):
                assert state["saturation"] == 1
                assert state["ups-mode"] == 0
                assert operands["su"] == 0
                value = operands["src"] & 0xFFFF
            elif descriptor_key == "narrow.2x.b-to-w.unsigned.configured":
                assert state["saturation"] in (0, 1)
                assert state["rounding"] == 12
                assert state["srs-mode"] == 0
                value = _round_unsigned_to_even(
                    operands["src"] & 0xFFFFFFFF, operands["su"]
                )
                if state["saturation"]:
                    value = min(value, 0xFFFF)
            elif descriptor_key == "narrow.2x.b-to-w.signed.configured":
                assert state["saturation"] == 1
                assert state["rounding"] == 12
                assert state["srs-mode"] == 0
                rounded = _round_signed_to_even(_s32(operands["src"]), operands["su"])
                value = min(max(rounded, -(1 << 15)), (1 << 15) - 1) & 0xFFFF
            elif descriptor_key in (
                "pack.w.trunc.configured",
                "pack.x.trunc.configured",
            ):
                assert state["saturation"] == 0
                assert state["pack-size"] in (0, 1)
                output_bits = 4 << state["pack-size"]
                value = operands["src"] & ((1 << output_bits) - 1)
            elif descriptor_key in (
                "convert.bf16x16.to.f32x16",
                "convert.bf16x32.to.f32x32",
            ):
                value = (operands["src"] & 0xFFFF) << 16
            else:
                raise AssertionError(f"unmodeled packet descriptor {descriptor_key}")
        values[result_ref] = value
    return values[ValueRef.result("result")]


def _evaluate_integer_truncation_lanes(
    rule: DescriptorRule,
    input_values: tuple[int, ...],
    *,
    input_bits: int,
    result_bits: int,
    source_carrier_count: int,
) -> tuple[int, ...]:
    """Evaluates lane compaction across physical X-register boundaries."""

    unit_bits = 256
    unit_mask = (1 << unit_bits) - 1
    input_mask = (1 << input_bits) - 1
    packed_input = sum(
        (value & input_mask) << (lane * input_bits)
        for lane, value in enumerate(input_values)
    )
    values: dict[ValueRef, int | tuple[int, ...]] = {
        ValueRef.operand("input"): tuple(
            (packed_input >> (unit * unit_bits)) & unit_mask
            for unit in range(source_carrier_count * 2)
        )
    }

    for emit in rule.emit:
        if isinstance(emit, EmitRegisterSlice):
            source_units = values[emit.source]
            assert isinstance(source_units, tuple)
            result_units = source_units[
                emit.unit_offset : emit.unit_offset + emit.unit_count
            ]
            assert len(result_units) == emit.unit_count
            values[emit.result] = result_units
            continue

        assert isinstance(emit, EmitDescriptorOp)
        result_ref = next(iter(emit.results.values()))
        if emit.form is DescriptorEmitForm.CONST:
            values[result_ref] = emit.immediates["i"]
            continue

        assert emit.descriptor.key == "amd.xdna.aie2p.shuffle.x.configured"
        low_units = values[emit.operands["s1"]]
        high_units = values[emit.operands["s2"]]
        control = values[emit.operands["mod"]]
        assert isinstance(low_units, tuple) and len(low_units) == 2
        assert isinstance(high_units, tuple) and len(high_units) == 2
        assert isinstance(control, int) and control in (0, 2, 4)

        sublane_bits = 8 << (control // 2)
        sublane_mask = (1 << sublane_bits) - 1
        group_bits = sublane_bits * 2
        packed_result = 0
        result_offset = 0
        for source_units in (low_units, high_units):
            packed_source = source_units[0] | (source_units[1] << unit_bits)
            for source_offset in range(0, unit_bits * 2, group_bits):
                packed_result |= (
                    (packed_source >> source_offset) & sublane_mask
                ) << result_offset
                result_offset += sublane_bits
        assert result_offset == unit_bits * 2
        values[result_ref] = (
            packed_result & unit_mask,
            (packed_result >> unit_bits) & unit_mask,
        )

    result_units = values[ValueRef.result("result")]
    assert isinstance(result_units, tuple) and len(result_units) == 2
    packed_result = result_units[0] | (result_units[1] << unit_bits)
    result_mask = (1 << result_bits) - 1
    return tuple(
        (packed_result >> (lane * result_bits)) & result_mask
        for lane in range(len(input_values))
    )


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


def _reference_mxfp8_e4m3fn_e8m0_to_bf16(payload_bits: int, scale_bits: int) -> int:
    """Returns the exact BF16 bits for one MXFP8 E4M3FN/E8M0 lane."""

    payload_bf16 = (
        _reference_fp8_to_f32(
            payload_bits,
            exponent_bits=4,
            mantissa_bits=3,
            has_infinity=False,
        )
        >> 16
    )
    magnitude = payload_bf16 & 0x7FFF
    if scale_bits == 0xFF or magnitude == _CANONICAL_BF16_NAN:
        return _CANONICAL_BF16_NAN
    sign = payload_bf16 & 0x8000
    if magnitude == 0:
        return sign

    exponent = magnitude >> 7
    fraction = magnitude & 0x7F
    scaled_exponent = exponent + scale_bits - 254
    if scaled_exponent > 127:
        return sign | 0x7F80
    if scaled_exponent >= -126:
        return sign | ((scaled_exponent + 127) << 7) | fraction

    subnormal_shift = -(scaled_exponent + 126)
    subnormal = _round_unsigned_to_even(0x80 | fraction, subnormal_shift)
    return sign | subnormal


def _reference_f32_to_bf16(input_bits: int) -> int:
    exponent = input_bits & 0x7F800000
    fraction = input_bits & 0x007FFFFF
    upper = input_bits >> 16
    if exponent == 0x7F800000 and fraction != 0:
        return upper | 0x0040
    return ((input_bits + 0x7FFF + (upper & 1)) & 0xFFFFFFFF) >> 16


def _round_unsigned_to_even(value: int, shift: int) -> int:
    if shift == 0:
        return value
    truncated = value >> shift
    remainder = value & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    if remainder > halfway or (remainder == halfway and truncated & 1):
        truncated += 1
    return truncated


def _round_signed_to_even(value: int, shift: int) -> int:
    if shift == 0:
        return value
    truncated = value >> shift
    remainder = value - (truncated << shift)
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


def _f32_fp8_rounding_boundaries(
    fp8_format: Float8PacketFormat,
) -> set[int]:
    """Returns representative binary32 FP8 rounding neighborhoods."""

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
    maximum_finite = (
        fp8_format.special_payload - 1
        if fp8_format.has_infinity
        else fp8_format.nan_payload - 1
    )
    mantissa_mask = (1 << fp8_format.mantissa_bits) - 1
    maximum_normal_exponent = maximum_finite >> fp8_format.mantissa_bits
    representative_exponents = {
        1,
        max(1, maximum_normal_exponent // 2),
        max(1, maximum_normal_exponent - 1),
        maximum_normal_exponent,
    }
    transition_lower_payloads = {
        0,
        1,
        max(0, mantissa_mask - 1),
        mantissa_mask,
    }
    for exponent in representative_exponents:
        base = exponent << fp8_format.mantissa_bits
        transition_lower_payloads.update(
            {
                base,
                base + 1,
                base + max(0, mantissa_mask - 1),
                base + mantissa_mask,
            }
        )
    transition_lower_payloads = {
        payload for payload in transition_lower_payloads if payload < maximum_finite
    }

    for lower_payload in transition_lower_payloads:
        lhs_bits = _reference_fp8_to_f32(
            lower_payload,
            exponent_bits=7 - fp8_format.mantissa_bits,
            mantissa_bits=fp8_format.mantissa_bits,
            has_infinity=fp8_format.has_infinity,
        )
        rhs_bits = _reference_fp8_to_f32(
            lower_payload + 1,
            exponent_bits=7 - fp8_format.mantissa_bits,
            mantissa_bits=fp8_format.mantissa_bits,
            has_infinity=fp8_format.has_infinity,
        )
        for f32_bits in (lhs_bits, rhs_bits):
            for signed_bits in (f32_bits, f32_bits | 0x80000000):
                values.update(_encoding_neighbors(signed_bits, 32))
        midpoint = (_bits_float(lhs_bits) + _bits_float(rhs_bits)) * 0.5
        values.update(_encoding_neighbors(_float_bits(midpoint), 32))
        values.update(_encoding_neighbors(_float_bits(-midpoint), 32))

    maximum_bits = _reference_fp8_to_f32(
        maximum_finite,
        exponent_bits=7 - fp8_format.mantissa_bits,
        mantissa_bits=fp8_format.mantissa_bits,
        has_infinity=fp8_format.has_infinity,
    )
    previous_bits = _reference_fp8_to_f32(
        maximum_finite - 1,
        exponent_bits=7 - fp8_format.mantissa_bits,
        mantissa_bits=fp8_format.mantissa_bits,
        has_infinity=fp8_format.has_infinity,
    )
    maximum_value = _bits_float(maximum_bits)
    previous_value = _bits_float(previous_bits)
    overflow_midpoint = maximum_value + (maximum_value - previous_value) * 0.5
    values.update(_encoding_neighbors(_float_bits(overflow_midpoint), 32))
    values.update(_encoding_neighbors(_float_bits(-overflow_midpoint), 32))
    return values


def _encoding_neighbors(value: int, bit_width: int) -> set[int]:
    mask = (1 << bit_width) - 1
    return {(value - 1) & mask, value & mask, (value + 1) & mask}


def _float_encoding_boundaries(source_format: FloatPacketSourceFormat) -> set[int]:
    """Returns class and carry boundaries for one floating-point encoding."""

    fraction_mask = source_format.fraction_mask
    minimum_normal = source_format.hidden_bit
    infinity = source_format.infinity_bits
    magnitudes = {
        0,
        1,
        max(fraction_mask - 1, 0),
        fraction_mask,
        minimum_normal,
        minimum_normal + 1,
        infinity - 1,
        infinity,
        infinity + 1,
        infinity | fraction_mask,
    }
    return {
        magnitude | sign
        for magnitude in magnitudes
        for sign in (0, source_format.sign_bit)
    }


def _fp8_encoding_boundaries(fp8_format: Float8PacketFormat) -> set[int]:
    """Returns the zero, normal, finite, infinity, and NaN FP8 boundaries."""

    fraction_mask = (1 << fp8_format.mantissa_bits) - 1
    minimum_normal = 1 << fp8_format.mantissa_bits
    magnitudes = {
        0,
        1,
        max(fraction_mask - 1, 0),
        fraction_mask,
        minimum_normal,
        minimum_normal + 1,
        fp8_format.special_payload - 1,
        fp8_format.special_payload,
        fp8_format.special_payload + 1,
        fp8_format.nan_payload,
    }
    return {magnitude | sign for magnitude in magnitudes for sign in (0, 0x80)}


def _mxfp8_scale_boundaries(payload_bits: int) -> set[int]:
    """Returns E8M0 branch, normalization, overflow, and NaN boundaries."""

    scale_values = {0, 1, 2, 3, 8, 9, 10, 126, 127, 128, 254, 255}
    payload_bf16 = (
        _reference_fp8_to_f32(
            payload_bits,
            exponent_bits=4,
            mantissa_bits=3,
            has_infinity=False,
        )
        >> 16
    )
    magnitude = payload_bf16 & 0x7FFF
    if magnitude not in (0, _CANONICAL_BF16_NAN):
        exponent = magnitude >> 7
        for boundary in (128 - exponent, 382 - exponent):
            scale_values.update(range(boundary - 1, boundary + 2))
    return {scale for scale in scale_values if 0 <= scale <= 0xFF}


def _source_fp8_rounding_boundaries(
    source_format: FloatPacketSourceFormat,
    fp8_format: Float8PacketFormat,
) -> set[int]:
    """Maps representative FP8 boundaries into a source encoding."""

    f32_values = _f32_fp8_rounding_boundaries(fp8_format)
    if source_format.element == "f32":
        return f32_values
    encode = (
        _reference_f32_to_f16
        if source_format.element == "f16"
        else _reference_f32_to_bf16
    )
    values = _float_encoding_boundaries(source_format)
    for f32_bits in f32_values:
        values.update(_encoding_neighbors(encode(f32_bits), source_format.bit_width))
    return values


def _source_float_to_f32(
    source_format: FloatPacketSourceFormat, input_bits: int
) -> int:
    if source_format.element == "f16":
        return _reference_f16_to_f32(input_bits)
    if source_format.element == "bf16":
        return input_bits << 16
    return input_bits


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


def _reference_bf16_fptosi_recipe(input_bits: int) -> int:
    minimum = input_bits == 0xCF00
    negative = bool(input_bits & 0x8000)
    absolute_bits = 0 if minimum else input_bits & 0x7FFF
    magnitude = math.floor(_bits_float(absolute_bits << 16))
    result = -magnitude if negative else magnitude
    return -(2**31) if minimum else result


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


_INTEGER_WIDTH_CONVERSIONS = (
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


def _assert_integer_width_conversion(
    source_op: Op,
    input_bits: int,
    result_bits: int,
    signed: bool,
    values: Iterable[int],
) -> None:
    case = _integer_conversion_case(
        source_op,
        f"i{input_bits}",
        f"i{result_bits}",
    )
    result_mask = (1 << result_bits) - 1
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


def test_integer_width_conversions_match_boundary_oracles() -> None:
    for source_op, input_bits, result_bits, signed in _INTEGER_WIDTH_CONVERSIONS:
        input_mask = (1 << input_bits) - 1
        values = {
            0,
            1,
            input_mask >> 1,
            1 << (input_bits - 1),
            input_mask,
        }
        if result_bits < input_bits:
            values.update(_encoding_neighbors((1 << result_bits) - 1, input_bits))
            values.update(_encoding_neighbors(1 << result_bits, input_bits))
        _assert_integer_width_conversion(
            source_op, input_bits, result_bits, signed, values
        )


@pytest.mark.exhaustive
def test_8bit_integer_width_conversions_match_exhaustive_oracles() -> None:
    for source_op, input_bits, result_bits, signed in _INTEGER_WIDTH_CONVERSIONS:
        if input_bits == 8:
            _assert_integer_width_conversion(
                source_op,
                input_bits,
                result_bits,
                signed,
                range(1 << input_bits),
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

    partial_narrow = _rule("native_binary32x1-15_to_bfloat16x1-15")
    assert (
        Guard.value_type("input", Vector("f32", minimum_lanes=1, maximum_lanes=15))
        in partial_narrow.guards
    )
    assert (
        Guard.value_type("result", Vector("bf16", minimum_lanes=1, maximum_lanes=15))
        in partial_narrow.guards
    )
    assert [type(emit) for emit in partial_narrow.emit] == [
        EmitDescriptorOp,
        EmitDescriptorOp,
        EmitDescriptorOp,
        EmitRegisterSlice,
        EmitRegisterConcat,
    ]
    assert [
        emit.descriptor.key
        for emit in partial_narrow.emit
        if isinstance(emit, EmitDescriptorOp)
    ] == [
        "amd.xdna.aie2p.move.vector512.to.accumulator512",
        "amd.xdna.aie2p.state.rounding.immediate",
        "amd.xdna.aie2p.convert.f32x16.to.bf16x16",
    ]
    assert partial_narrow.emit[1].immediates == {"i": 12}

    partial_widen = _rule("native_bfloat16x1-15_to_binary32x1-15")
    assert (
        Guard.value_type("input", Vector("bf16", minimum_lanes=1, maximum_lanes=15))
        in partial_widen.guards
    )
    assert (
        Guard.value_type("result", Vector("f32", minimum_lanes=1, maximum_lanes=15))
        in partial_widen.guards
    )
    assert [type(emit) for emit in partial_widen.emit] == [
        EmitRegisterSlice,
        EmitDescriptorOp,
        EmitDescriptorOp,
    ]
    assert [
        emit.descriptor.key
        for emit in partial_widen.emit
        if isinstance(emit, EmitDescriptorOp)
    ] == [
        "amd.xdna.aie2p.convert.bf16x16.to.f32x16",
        "amd.xdna.aie2p.move.accumulator512.to.vector512",
    ]

    partial_wide_narrow = _rule("native_binary32x17-31_to_bfloat16x17-31")
    assert [type(emit) for emit in partial_wide_narrow.emit] == [
        EmitRegisterSlice,
        EmitDescriptorOp,
        EmitRegisterSlice,
        EmitDescriptorOp,
        EmitRegisterConcat,
        EmitDescriptorOp,
        EmitDescriptorOp,
    ]
    assert [
        emit.descriptor.key
        for emit in partial_wide_narrow.emit
        if isinstance(emit, EmitDescriptorOp)
    ] == [
        "amd.xdna.aie2p.move.vector512.to.accumulator512",
        "amd.xdna.aie2p.move.vector512.to.accumulator512",
        "amd.xdna.aie2p.state.rounding.immediate",
        "amd.xdna.aie2p.convert.f32x32.to.bf16x32",
    ]
    assert [partial_wide_narrow.emit[index].unit_offset for index in (0, 2)] == [
        0,
        2,
    ]

    partial_wide_widen = _rule("native_bfloat16x17-31_to_binary32x17-31")
    assert [type(emit) for emit in partial_wide_widen.emit] == [
        EmitDescriptorOp,
        EmitRegisterSlice,
        EmitDescriptorOp,
        EmitRegisterSlice,
        EmitDescriptorOp,
        EmitRegisterConcat,
    ]
    assert [
        emit.descriptor.key
        for emit in partial_wide_widen.emit
        if isinstance(emit, EmitDescriptorOp)
    ] == [
        "amd.xdna.aie2p.convert.bf16x32.to.f32x32",
        "amd.xdna.aie2p.move.accumulator512.to.vector512",
        "amd.xdna.aie2p.move.accumulator512.to.vector512",
    ]
    assert [partial_wide_widen.emit[index].unit_offset for index in (1, 3)] == [
        0,
        1,
    ]


def test_float8_packet_widening_covers_every_native_logical_width() -> None:
    lane_ranges = ("16", "32", "1-15", "17-31")
    for fp8_format in FLOAT8_PACKET_FORMATS:
        for lane_range in lane_ranges:
            for result_name in ("bfloat16", "binary32"):
                rule = _rule(
                    f"native_{fp8_format.report_name}x{lane_range}_to_"
                    f"{result_name}x{lane_range}"
                )
                descriptor_keys = [
                    emit.descriptor.key
                    for emit in rule.emit
                    if isinstance(emit, EmitDescriptorOp)
                ]
                assert descriptor_keys[:4] == [
                    "amd.xdna.aie2p.sub.i8x64",
                    "amd.xdna.aie2p.constant.i32.mova",
                    "amd.xdna.aie2p.shuffle.x.configured",
                    "amd.xdna.aie2p.shuffle.x.configured",
                ]
                assert all(
                    ".extract." not in key and ".insert." not in key
                    for key in descriptor_keys
                )


def test_mxfp8_decode_has_exact_group_schemas_and_shared_packet_program() -> None:
    rules = tuple(
        _rule(f"native_mxfp8_e4m3fn_e8m0x{lane_count}_to_bfloat16x{lane_count}")
        for lane_count in (8, 32)
    )
    schemas = (MXFP8_E4M3FN_E8M0_X8_SCHEMA, MXFP8_E4M3FN_E8M0_X32_SCHEMA)
    for lane_count, schema, rule in zip((8, 32), schemas, rules, strict=True):
        assert rule.source_op is vector.vector_decode
        assert rule.guards == (
            Guard.value_type("payload", Vector("f8E4M3", lanes=lane_count)),
            Guard.value_storage_operand_schema("schema", schema),
            Guard.operand_segment_count("auxiliary", 1),
            Guard.value_type("auxiliary", Vector("i32", lanes=1), element=0),
            Guard.value_type("result", Vector("bf16", lanes=lane_count)),
        )

    four_x8_group_schema = replace(
        MXFP8_E4M3FN_E8M0_X32_SCHEMA,
        scale_group_element_count=8,
        scale_group_shape=(8,),
    )
    assert four_x8_group_schema not in schemas
    assert rules[0].emit == rules[1].emit

    compiled = compile_lower_rule_set(
        ContractFragment(
            name="amd.xdna.aie2p.mxfp8.packet.test",
            descriptor_set=AIE2P_CORE_DESCRIPTOR_SET,
            cases=rules,
        ),
        dialect_ops={"vector": ALL_VECTOR_OPS},
    )
    assert len(compiled.emits) == len(rules[0].emit)
    assert compiled.rules[0].emit_start == compiled.rules[1].emit_start
    assert compiled.rules[0].emit_count == compiled.rules[1].emit_count

    descriptor_keys = [
        emit.descriptor.key
        for emit in rules[0].emit
        if isinstance(emit, EmitDescriptorOp)
    ]
    assert [key for key in descriptor_keys if ".extract." in key] == [
        "amd.xdna.aie2p.extract.i32.immediate"
    ]
    assert all(".insert." not in key for key in descriptor_keys)
    assert "amd.xdna.aie2p.convert.floor.bf16x16.to.i32x16" in descriptor_keys
    assert "amd.xdna.aie2p.multiply.i16x32.configured" in descriptor_keys


def _assert_mxfp8_packet_decode_matches_oracle(
    values: Iterable[tuple[int, int]],
    *,
    lane_count: int = 8,
) -> None:
    rule = _rule(f"native_mxfp8_e4m3fn_e8m0x{lane_count}_to_bfloat16x{lane_count}")
    for payload_bits, scale_bits in values:
        expected = _reference_mxfp8_e4m3fn_e8m0_to_bf16(payload_bits, scale_bits)
        actual = _evaluate_packet_lane(rule, payload_bits, scale_value=scale_bits)
        assert actual == expected, (
            hex(payload_bits),
            hex(scale_bits),
            hex(actual),
            hex(expected),
        )


def test_mxfp8_packet_decode_matches_boundary_oracles() -> None:
    fp8_format = FLOAT8_PACKET_FORMATS[0]
    payloads = _fp8_encoding_boundaries(fp8_format)
    for lane_count in (8, 32):
        _assert_mxfp8_packet_decode_matches_oracle(
            (
                (payload, scale)
                for payload in payloads
                for scale in _mxfp8_scale_boundaries(payload)
            ),
            lane_count=lane_count,
        )


@pytest.mark.exhaustive
def test_mxfp8_packet_decode_matches_exhaustive_oracles() -> None:
    _assert_mxfp8_packet_decode_matches_oracle(
        (payload, scale) for payload in range(1 << 8) for scale in range(1 << 8)
    )


def _assert_float8_packet_widening_matches_oracles(
    fp8_format: Float8PacketFormat, values: Iterable[int]
) -> None:
    bf16_rule = _rule(f"native_{fp8_format.report_name}x32_to_bfloat16x32")
    f32_rule = _rule(f"native_{fp8_format.report_name}x32_to_binary32x32")
    for bits in values:
        expected_f32 = _reference_fp8_to_f32(
            bits,
            exponent_bits=7 - fp8_format.mantissa_bits,
            mantissa_bits=fp8_format.mantissa_bits,
            has_infinity=fp8_format.has_infinity,
        )
        assert _evaluate_packet_lane(bf16_rule, bits) == (expected_f32 >> 16), (
            fp8_format.element,
            hex(bits),
            "bf16",
        )
        assert _evaluate_packet_lane(f32_rule, bits) == expected_f32, (
            fp8_format.element,
            hex(bits),
            "f32",
        )


def test_float8_packet_widening_matches_boundary_oracles() -> None:
    for fp8_format in FLOAT8_PACKET_FORMATS:
        _assert_float8_packet_widening_matches_oracles(
            fp8_format, _fp8_encoding_boundaries(fp8_format)
        )


@pytest.mark.exhaustive
def test_float8_packet_widening_matches_exhaustive_oracles() -> None:
    for fp8_format in FLOAT8_PACKET_FORMATS:
        _assert_float8_packet_widening_matches_oracles(fp8_format, range(1 << 8))


def test_float8_packet_narrowing_covers_every_native_logical_width() -> None:
    for source_format in FLOAT_PACKET_SOURCE_FORMATS:
        lane_ranges = (
            ("1-16", "17-32")
            if source_format.bit_width == 16
            else ("1-16", "17-31", "32")
        )
        for fp8_format in FLOAT8_PACKET_FORMATS:
            for lane_range in lane_ranges:
                rule = _rule(
                    f"native_{source_format.report_name}x{lane_range}_to_"
                    f"{fp8_format.report_name}x{lane_range}"
                )
                descriptor_keys = [
                    emit.descriptor.key
                    for emit in rule.emit
                    if isinstance(emit, EmitDescriptorOp)
                ]
                assert (
                    descriptor_keys.count("amd.xdna.aie2p.pack.w.trunc.configured") == 1
                )
                assert any(
                    key.startswith("amd.xdna.aie2p.narrow.2x.b-to-w")
                    for key in descriptor_keys
                )
                assert any(".widen.2x." in key for key in descriptor_keys) == (
                    source_format.bit_width == 16
                )
                assert all(
                    ".extract." not in key and ".insert." not in key
                    for key in descriptor_keys
                )


def _assert_float8_packet_narrowing_matches_oracle(
    source_format: FloatPacketSourceFormat,
    fp8_format: Float8PacketFormat,
    values: Iterable[int],
) -> None:
    lane_range = "17-32" if source_format.bit_width == 16 else "32"
    rule = _rule(
        f"native_{source_format.report_name}x{lane_range}_to_"
        f"{fp8_format.report_name}x{lane_range}"
    )
    for bits in values:
        expected = _reference_f32_to_fp8(
            _source_float_to_f32(source_format, bits),
            exponent_bits=7 - fp8_format.mantissa_bits,
            mantissa_bits=fp8_format.mantissa_bits,
            has_infinity=fp8_format.has_infinity,
        )
        assert _evaluate_packet_lane(rule, bits) == expected, (
            source_format.element,
            fp8_format.element,
            hex(bits),
        )


def test_float8_packet_narrowing_matches_rounding_boundaries() -> None:
    for source_format in FLOAT_PACKET_SOURCE_FORMATS:
        for fp8_format in FLOAT8_PACKET_FORMATS:
            _assert_float8_packet_narrowing_matches_oracle(
                source_format,
                fp8_format,
                _source_fp8_rounding_boundaries(source_format, fp8_format),
            )


@pytest.mark.exhaustive
def test_float8_packet_narrowing_matches_exhaustive_16bit_oracles() -> None:
    for source_format in FLOAT_PACKET_SOURCE_FORMATS:
        if source_format.bit_width != 16:
            continue
        for fp8_format in FLOAT8_PACKET_FORMATS:
            _assert_float8_packet_narrowing_matches_oracle(
                source_format,
                fp8_format,
                range(1 << 16),
            )


def _assert_bfloat16_fptosi_recipe(values: Iterable[int]) -> None:
    for input_bits in values:
        value = _bits_float(input_bits << 16)
        if not math.isfinite(value) or not -(2**31) <= value < 2**31:
            continue
        assert _reference_bf16_fptosi_recipe(input_bits) == math.trunc(value), hex(
            input_bits
        )


def test_bfloat16_packet_to_signed_i32_is_exact_at_domain_boundaries() -> None:
    rule = _rule("exact_bfloat16x16_to_signed_i32x16")
    assert rule.source_op is vector.vector_fptosi
    assert [
        emit.descriptor.key for emit in rule.emit if isinstance(emit, EmitDescriptorOp)
    ] == [
        "amd.xdna.aie2p.constant.i32",
        "amd.xdna.aie2p.splat.i16x32",
        "amd.xdna.aie2p.sub.i16x32",
        "amd.xdna.aie2p.cmp.eqz.i16x32.el.low32",
        "amd.xdna.aie2p.predicate.complete.zero.high32",
        "amd.xdna.aie2p.sub.i16x32",
        "amd.xdna.aie2p.cmp.lt.signed.i16x32.el.low32",
        "amd.xdna.aie2p.predicate.complete.zero.high32",
        "amd.xdna.aie2p.constant.i32",
        "amd.xdna.aie2p.splat.i16x32",
        "amd.xdna.aie2p.and.bits512",
        "amd.xdna.aie2p.select.i16x32.mask64",
        "amd.xdna.aie2p.constant.i32.shift",
        "amd.xdna.aie2p.convert.floor.bf16x16.to.i32x16",
        "amd.xdna.aie2p.sub.i32x16",
        "amd.xdna.aie2p.select.i32x16.mask64",
        "amd.xdna.aie2p.constant.i32",
        "amd.xdna.aie2p.splat.i32x16",
        "amd.xdna.aie2p.select.i32x16.mask64",
    ]
    assert isinstance(rule.emit[12], EmitRegisterSlice)
    assert rule.emit[12].unit_count == 1

    sanitize = rule.emit[11]
    assert isinstance(sanitize, EmitDescriptorOp)
    assert sanitize.operands == {
        "s1": ValueRef.temporary("absolute"),
        "s2": ValueRef.temporary("zero"),
        "sel": ValueRef.temporary("minimum"),
    }
    restore_sign = rule.emit[16]
    assert isinstance(restore_sign, EmitDescriptorOp)
    assert restore_sign.operands == {
        "s1": ValueRef.temporary("magnitude"),
        "s2": ValueRef.temporary("negative_magnitude"),
        "sel": ValueRef.temporary("negative"),
    }
    restore_minimum = rule.emit[19]
    assert isinstance(restore_minimum, EmitDescriptorOp)
    assert restore_minimum.operands == {
        "s1": ValueRef.temporary("signed_result"),
        "s2": ValueRef.temporary("minimum_i32"),
        "sel": ValueRef.temporary("minimum"),
    }

    boundary_values = set()
    for center in (
        0x0000,
        0x3F00,
        0x3F80,
        0x4000,
        0x4EFF,
        0x4F00,
        0x8000,
        0xBF00,
        0xBF80,
        0xC000,
        0xCEFF,
        0xCF00,
    ):
        boundary_values.update(_encoding_neighbors(center, 16))
    _assert_bfloat16_fptosi_recipe(boundary_values)


@pytest.mark.exhaustive
def test_bfloat16_packet_to_signed_i32_is_exact_over_defined_domain() -> None:
    _assert_bfloat16_fptosi_recipe(range(1 << 16))


def test_native_uniform_i32_shifts_cover_the_logical_carrier() -> None:
    covered_lane_counts: set[int] = set()
    for rule_shape in INTEGER_SHIFT_RULE_SHAPES:
        logical_lane_counts = set(
            range(rule_shape.minimum_lane_count, rule_shape.maximum_lane_count + 1)
        )
        assert covered_lane_counts.isdisjoint(logical_lane_counts)
        covered_lane_counts.update(logical_lane_counts)
    assert covered_lane_counts == set(range(1, 17))

    for source_op in (
        vector.vector_shli,
        vector.vector_shrui,
        vector.vector_shrsi,
    ):
        signedness = "signed" if source_op is vector.vector_shrsi else "unsigned"
        for rule_shape in INTEGER_SHIFT_RULE_SHAPES:
            report_key = (
                "native_"
                + source_op.name.removeprefix("vector.")
                + f"_i32x{rule_shape.report_lane_range}_uniform"
            )
            rule = _rule(report_key)
            assert rule.source_op is source_op
            assert rule.guards == (
                *(
                    Guard.value_type(field, rule_shape.vector_type)
                    for field in ("lhs", "rhs", "result")
                ),
                Guard.value_exact_i64("rhs"),
                Guard.value_i64_range("rhs", 0, 31),
            )
            assert [
                emit.descriptor.key
                for emit in rule.emit
                if isinstance(emit, EmitDescriptorOp)
            ] == [
                "amd.xdna.aie2p.constant.i32.shift",
                "amd.xdna.aie2p.constant.i32.shift",
                "amd.xdna.aie2p.state.saturation.immediate",
                "amd.xdna.aie2p.state.ups-mode.immediate",
                "amd.xdna.aie2p.state.srs-mode.immediate",
                "amd.xdna.aie2p.state.rounding.immediate",
                f"amd.xdna.aie2p.widen.2x.x-to-c.{signedness}.configured",
                f"amd.xdna.aie2p.narrow.2x.c-to-x.{signedness}.configured",
            ]


def test_native_integer_widening_covers_each_logical_carrier_interval() -> None:
    expected_lane_counts = {
        ("i8", "i32"): {*range(1, 33), 64},
        ("i16", "i32"): set(range(1, 33)),
        ("i16", "i64"): {*range(1, 17), 32},
        ("i32", "i64"): set(range(1, 17)),
    }
    covered_lane_counts = {key: set() for key in expected_lane_counts}
    for rule_shape in INTEGER_WIDEN_RULE_SHAPES:
        instruction = rule_shape.instruction
        key = (instruction.input_element, instruction.result_element)
        logical_lane_counts = set(
            range(rule_shape.minimum_lane_count, rule_shape.maximum_lane_count + 1)
        )
        assert covered_lane_counts[key].isdisjoint(logical_lane_counts)
        covered_lane_counts[key].update(logical_lane_counts)
    assert covered_lane_counts == expected_lane_counts

    for source_op, signedness in (
        (vector.vector_extui, "unsigned"),
        (vector.vector_extsi, "signed"),
    ):
        for rule_shape in INTEGER_WIDEN_RULE_SHAPES:
            instruction = rule_shape.instruction
            rule = _rule(rule_shape.report_key(signedness))
            assert rule.source_op is source_op
            assert rule.descriptor.key == (
                f"amd.xdna.aie2p.widen.{instruction.physical_shape}."
                f"{signedness}.configured"
            )
            assert rule.guards == (
                Guard.value_type("input", rule_shape.input_type),
                Guard.value_type("result", rule_shape.result_type),
            )
            input_slices = [
                emit
                for emit in rule.emit
                if isinstance(emit, EmitRegisterSlice)
                and emit.result.field == "source_w"
            ]
            assert len(input_slices) == int(instruction.slice_input)
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
                if (
                    instruction.direct_accumulator_result
                    and rule_shape.result_accumulator_unit_count
                    == instruction.accumulator_unit_count
                )
                else ["amd.xdna.aie2p.move.accumulator512.to.vector512"]
                * rule_shape.result_accumulator_unit_count
            )
            set_ups_mode = next(
                emit
                for emit in rule.emit
                if isinstance(emit, EmitDescriptorOp)
                and emit.descriptor.key == "amd.xdna.aie2p.state.ups-mode.immediate"
            )
            assert set_ups_mode.immediates == {"i": instruction.ups_mode}


def test_native_integer_packing_covers_each_logical_carrier_interval() -> None:
    expected_truncation_lanes = {
        ("i16", "i8"): set(range(1, 65)),
    }
    covered_truncation_lanes = {key: set() for key in expected_truncation_lanes}
    for rule_shape in INTEGER_PACK_RULE_SHAPES:
        instruction = rule_shape.instruction
        if instruction.bit_width is None:
            key = (instruction.input_element, instruction.result_element)
            logical_lane_counts = set(
                range(
                    rule_shape.minimum_lane_count,
                    rule_shape.maximum_lane_count + 1,
                )
            )
            assert covered_truncation_lanes[key].isdisjoint(logical_lane_counts)
            covered_truncation_lanes[key].update(logical_lane_counts)

        rule = _rule(rule_shape.report_key)
        assert rule.source_op is instruction.source_op
        assert rule.descriptor.key == (
            f"amd.xdna.aie2p.pack.{instruction.physical_width}.trunc.configured"
        )
        assert rule.guards[:2] == (
            Guard.value_type(instruction.source_field, rule_shape.input_type),
            Guard.value_type("result", rule_shape.result_type),
        )
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
        assert rule.emit[1].immediates == {"i": instruction.pack_size}
        assert sum(isinstance(emit, EmitRegisterSlice) for emit in rule.emit) == int(
            instruction.pad_result
        )
        assert sum(isinstance(emit, EmitRegisterConcat) for emit in rule.emit) == int(
            instruction.pad_result
        )
    assert covered_truncation_lanes == expected_truncation_lanes

    assert {
        rule_shape.instruction
        for rule_shape in INTEGER_PACK_RULE_SHAPES
        if (
            rule_shape.minimum_lane_count == rule_shape.instruction.native_lane_count
            and rule_shape.maximum_lane_count
            == rule_shape.instruction.native_lane_count
        )
    } == set(INTEGER_PACK_INSTRUCTIONS)


def test_native_integer_shuffle_truncation_covers_every_remaining_width() -> None:
    expected_lanes_and_controls = {
        ("i32", "i16"): (set(range(1, 33)), (2,)),
        ("i32", "i8"): (set(range(1, 33)), (2, 0)),
        ("i64", "i32"): (set(range(1, 17)), (4,)),
        ("i64", "i16"): (set(range(1, 17)), (4, 2)),
        ("i64", "i8"): (set(range(1, 17)), (4, 2, 0)),
    }
    covered_lanes = {key: set() for key in expected_lanes_and_controls}
    for rule_shape in INTEGER_TRUNCATION_RULE_SHAPES:
        instruction = rule_shape.instruction
        key = (instruction.input_element, instruction.result_element)
        assert instruction.shuffle_controls == expected_lanes_and_controls[key][1]
        logical_lane_counts = set(
            range(rule_shape.minimum_lane_count, rule_shape.maximum_lane_count + 1)
        )
        assert covered_lanes[key].isdisjoint(logical_lane_counts)
        covered_lanes[key].update(logical_lane_counts)

        rule = _rule(rule_shape.report_key)
        assert rule.source_op is vector.vector_trunci
        assert rule.descriptor.key == "amd.xdna.aie2p.shuffle.x.configured"
        assert rule.guards == (
            Guard.value_type("input", rule_shape.input_type),
            Guard.value_type("result", rule_shape.result_type),
        )
        descriptor_emits = tuple(
            emit for emit in rule.emit if isinstance(emit, EmitDescriptorOp)
        )
        assert [emit.descriptor.key for emit in descriptor_emits] == [
            descriptor_key
            for _ in instruction.shuffle_controls
            for descriptor_key in (
                "amd.xdna.aie2p.constant.i32.mova",
                "amd.xdna.aie2p.shuffle.x.configured",
            )
        ]
        assert [
            descriptor_emits[index].immediates["i"]
            for index in range(0, len(descriptor_emits), 2)
        ] == list(instruction.shuffle_controls)
        assert sum(isinstance(emit, EmitRegisterSlice) for emit in rule.emit) == (
            2 if rule_shape.source_carrier_count == 2 else 0
        )

    assert covered_lanes == {
        key: lanes for key, (lanes, _) in expected_lanes_and_controls.items()
    }
    assert {shape.instruction for shape in INTEGER_TRUNCATION_RULE_SHAPES} == set(
        INTEGER_TRUNCATION_INSTRUCTIONS
    )


def test_native_integer_shuffle_truncation_preserves_every_lane() -> None:
    for rule_shape in INTEGER_TRUNCATION_RULE_SHAPES:
        instruction = rule_shape.instruction
        input_bits = int(instruction.input_element[1:])
        result_bits = int(instruction.result_element[1:])
        input_mask = (1 << input_bits) - 1
        result_mask = (1 << result_bits) - 1
        rule = _rule(rule_shape.report_key)
        for lane_count in range(
            rule_shape.minimum_lane_count, rule_shape.maximum_lane_count + 1
        ):
            input_values = tuple(
                (
                    ((lane + 1) * 0x9E3779B97F4A7C15 ^ lane_count * 0xD1B54A32D192ED03)
                    & input_mask
                    & ~result_mask
                )
                | ((lane * 37 + lane_count * 11) & result_mask)
                for lane in range(lane_count)
            )
            assert _evaluate_integer_truncation_lanes(
                rule,
                input_values,
                input_bits=input_bits,
                result_bits=result_bits,
                source_carrier_count=rule_shape.source_carrier_count,
            ) == tuple(value & result_mask for value in input_values)


def test_integer_pack_rejects_unrepresentable_control_values() -> None:
    with pytest.raises(ValueError, match="one-bit crPackSize"):
        IntegerPackInstruction("i32", 16, "i16", None)


def test_native_integer_truncation_preserves_low_bits_at_boundaries() -> None:
    boundary_values = {
        "i16": (0, 1, 0x7F, 0x80, 0xFF, 0x100, 0x7FFF, 0x8000, 0xFFFF),
        "i32": (
            0,
            1,
            0x200,
            0x7FFF,
            0x8000,
            0xFFFF,
            0x10000,
            0x12345678,
            0x7FFFFFFF,
            0x80000000,
            0xFFFFFFFF,
        ),
        "i64": (
            0,
            1,
            0x100,
            0x10000,
            0x100000000,
            0x123456789ABCDEF0,
            0x7FFFFFFFFFFFFFFF,
            0x8000000000000000,
            0xFFFFFFFFFFFFFFFF,
        ),
    }
    for rule_shape in (*INTEGER_PACK_RULE_SHAPES, *INTEGER_TRUNCATION_RULE_SHAPES):
        instruction = rule_shape.instruction
        if isinstance(instruction, IntegerPackInstruction) and (
            instruction.bit_width is not None
        ):
            continue
        rule = _rule(rule_shape.report_key)
        result_bits = (
            instruction.output_element_bits
            if isinstance(instruction, IntegerPackInstruction)
            else int(instruction.result_element[1:])
        )
        result_mask = (1 << result_bits) - 1
        for input_value in boundary_values[instruction.input_element]:
            assert _evaluate_packet_lane(rule, input_value) == (
                input_value & result_mask
            )


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
    for bits in signed_bits:
        value = _bits_float(bits)
        assert _s32(_evaluate(signed, bits)) == math.trunc(value)
    for bits in unsigned_bits:
        value = _bits_float(bits)
        expected = math.trunc(value)
        assert _evaluate(unsigned_i32, bits) == expected
        if value < 2**16:
            assert _evaluate(unsigned_narrow, bits) == expected


def _assert_small_integer_to_f32_oracles(bit_width: int, values: Iterable[int]) -> None:
    signed_rule = _rule(f"native_signed_i{bit_width}_to_binary32")
    unsigned_rule = _rule(f"native_unsigned_i{bit_width}_to_binary32")
    sign_bit = 1 << (bit_width - 1)
    modulus = 1 << bit_width
    for bits in values:
        signed = bits - modulus if bits & sign_bit else bits
        assert _evaluate(signed_rule, bits) == _float_bits(float(signed))
        assert _evaluate(unsigned_rule, bits) == _float_bits(float(bits))


def test_integer_to_f32_programs_match_binary32_boundary_oracle() -> None:
    for bit_width in (8, 16):
        values = set()
        for center in (0, 1 << (bit_width - 1), (1 << bit_width) - 1):
            values.update(_encoding_neighbors(center, bit_width))
        _assert_small_integer_to_f32_oracles(bit_width, values)

    signed_i32 = _rule("native_signed_i32_to_binary32")
    values = {
        0,
        1,
        -1,
        2**24 - 1,
        2**24,
        2**24 + 1,
        2**31 - 1,
        -(2**31),
    }
    for exponent in range(24, 31):
        values.update(
            {
                (1 << exponent) - 1,
                1 << exponent,
                (1 << exponent) + 1,
                -((1 << exponent) - 1),
                -(1 << exponent),
                -((1 << exponent) + 1),
            }
        )
    for value in values:
        assert _evaluate(signed_i32, _u32(value)) == _float_bits(float(value))

    unsigned_i32 = _rule("exact_unsigned_i32_to_binary32")
    unsigned_values = {
        0,
        1,
        2**24 - 1,
        2**24,
        2**24 + 1,
        2**31 - 1,
        2**31,
        2**31 + 1,
        2**32 - 1,
    }
    for exponent in range(24, 32):
        unsigned_values.update(
            {
                (1 << exponent) - 1,
                1 << exponent,
                min((1 << exponent) + 1, (1 << 32) - 1),
            }
        )
    for value in unsigned_values:
        assert _evaluate(unsigned_i32, value) == _float_bits(float(value))


@pytest.mark.exhaustive
def test_small_integer_to_f32_programs_match_exhaustive_oracles() -> None:
    for bit_width in (8, 16):
        _assert_small_integer_to_f32_oracles(bit_width, range(1 << bit_width))


def _assert_16bit_float_widening_oracles(values: Iterable[int]) -> None:
    f16 = _rule("exact_f16_to_binary32")
    bf16 = _rule("exact_bf16_to_binary32")
    for bits in values:
        assert _evaluate(f16, bits) == _reference_f16_to_f32(bits)
        assert _evaluate(bf16, bits) == bits << 16


def test_16bit_float_widening_programs_match_boundary_oracles() -> None:
    values = set()
    for source_format in FLOAT_PACKET_SOURCE_FORMATS:
        if source_format.bit_width == 16:
            values.update(_float_encoding_boundaries(source_format))
    _assert_16bit_float_widening_oracles(values)


@pytest.mark.exhaustive
def test_16bit_float_widening_programs_match_exhaustive_oracles() -> None:
    _assert_16bit_float_widening_oracles(range(1 << 16))


def _assert_float8_widening_oracles(
    fp8_format: Float8PacketFormat, values: Iterable[int]
) -> None:
    source_name = fp8_format.report_name
    f16 = _rule(f"exact_{source_name}_to_f16")
    bf16 = _rule(f"exact_{source_name}_to_bf16")
    f32 = _rule(f"exact_{source_name}_to_f32")
    for bits in values:
        f32_bits = _reference_fp8_to_f32(
            bits,
            exponent_bits=7 - fp8_format.mantissa_bits,
            mantissa_bits=fp8_format.mantissa_bits,
            has_infinity=fp8_format.has_infinity,
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


def test_float8_widening_programs_match_boundary_oracles() -> None:
    for fp8_format in FLOAT8_PACKET_FORMATS:
        _assert_float8_widening_oracles(
            fp8_format, _fp8_encoding_boundaries(fp8_format)
        )


@pytest.mark.exhaustive
def test_float8_widening_programs_match_exhaustive_oracles() -> None:
    for fp8_format in FLOAT8_PACKET_FORMATS:
        _assert_float8_widening_oracles(fp8_format, range(1 << 8))


def test_float8_narrowing_programs_match_rounding_oracles() -> None:
    for fp8_format in FLOAT8_PACKET_FORMATS:
        for source_format in FLOAT_PACKET_SOURCE_FORMATS:
            rule = _rule(f"exact_{source_format.element}_to_{fp8_format.report_name}")
            for bits in _source_fp8_rounding_boundaries(source_format, fp8_format):
                expected = _reference_f32_to_fp8(
                    _source_float_to_f32(source_format, bits),
                    exponent_bits=7 - fp8_format.mantissa_bits,
                    mantissa_bits=fp8_format.mantissa_bits,
                    has_infinity=fp8_format.has_infinity,
                )
                assert _evaluate(rule, bits) == expected, (
                    source_format.element,
                    fp8_format.element,
                    hex(bits),
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
    for bits in values:
        assert _evaluate(rule, bits) == _reference_f32_to_f16(bits), hex(bits)


def _assert_small_integer_to_f16_oracles(bit_width: int, values: Iterable[int]) -> None:
    signed_rule = _rule(f"exact_signed_i{bit_width}_to_f16")
    unsigned_rule = _rule(f"exact_unsigned_i{bit_width}_to_f16")
    sign_bit = 1 << (bit_width - 1)
    modulus = 1 << bit_width
    for bits in values:
        signed = bits - modulus if bits & sign_bit else bits
        assert _evaluate(signed_rule, bits) == _reference_integer_to_f16(signed)
        assert _evaluate(unsigned_rule, bits) == _reference_integer_to_f16(bits)


def test_integer_to_f16_programs_match_direct_rounding_boundaries() -> None:
    i8_values = set()
    for center in (0x00, 0x7F, 0x80, 0xFF):
        i8_values.update(_encoding_neighbors(center, 8))
    _assert_small_integer_to_f16_oracles(8, i8_values)

    i16_values = {
        0x0000,
        0x0001,
        0x00FF,
        0x0100,
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
    _assert_small_integer_to_f16_oracles(16, i16_values)

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
    for bits in values:
        signed = _s32(bits)
        assert _evaluate(signed_i32, bits) == _reference_integer_to_f16(signed)
        assert _evaluate(unsigned_i32, bits) == _reference_integer_to_f16(bits)


@pytest.mark.exhaustive
def test_small_integer_to_f16_programs_match_exhaustive_oracles() -> None:
    for bit_width in (8, 16):
        _assert_small_integer_to_f16_oracles(bit_width, range(1 << bit_width))
