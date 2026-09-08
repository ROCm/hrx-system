# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact scalar and vector conversion contracts for AMD XDNA AIE2P."""

from __future__ import annotations

from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amd.xdna.aie2p.contracts.packet_conversion import (
    AIE2P_PACKET_CONVERSION_RULES,
)
from loom.target.arch.amd.xdna.aie2p.contracts.scalar_program import ScalarProgram
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    ContractEmit,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    Guard,
    Scalar,
    TypePattern,
    ValueAliasRule,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_I8 = Scalar("i8")
_I16 = Scalar("i16")
_I32 = Scalar("i32")
_I64 = Scalar("i64")
_F8E4M3 = Scalar("f8E4M3")
_F8E5M2 = Scalar("f8E5M2")
_F16 = Scalar("f16")
_BF16 = Scalar("bf16")
_F32 = Scalar("f32")
_BF16X16 = Vector("bf16", lanes=16)
_I32X16 = Vector("i32", lanes=16)
_NARROW_INTEGERS = ("i8", "i16")
_SIGNED_INTEGERS = (*_NARROW_INTEGERS, "i32")


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


def _exact_vector(element: str, element_count: int) -> TypePattern:
    return Vector(element, lanes=element_count)


def emit_round_nearest_f32_to_i32(
    program: ScalarProgram,
    input_value: ValueRef,
    scale: int,
    prefix: str,
) -> ValueRef:
    scale_value = program.constant(
        f"{prefix}_scale",
        scale,
        descriptor_key="amd.xdna.aie2p.constant.i32.flt2fx-scale",
    )
    return program.operation(
        f"{prefix}_rounded",
        "convert.round-nearest.f32.to.signed.i32",
        operands={"s0": input_value, "m": scale_value},
    )


def emit_truncate_f32_to_signed_i32(
    program: ScalarProgram,
    input_value: ValueRef,
    prefix: str,
    result_name: str | None,
) -> ValueRef:
    """Converts an in-range binary32 value toward zero exactly."""

    rounded = emit_round_nearest_f32_to_i32(program, input_value, 0, prefix)
    back_scale = program.constant(
        f"{prefix}_back_scale",
        0,
        descriptor_key="amd.xdna.aie2p.constant.i32.fx2flt-scale",
    )
    rounded_float = program.operation(
        f"{prefix}_rounded_float",
        "convert.signed.i32.to.f32",
        operands={"s0": rounded, "m": back_scale},
    )

    absolute_mask = program.constant(f"{prefix}_absolute_mask", 0x7FFFFFFF)
    input_absolute = program.binary(
        f"{prefix}_input_absolute", "and.i32", input_value, absolute_mask
    )
    rounded_absolute = program.binary(
        f"{prefix}_rounded_absolute", "and.i32", rounded_float, absolute_mask
    )
    rounded_away_from_zero = program.binary(
        f"{prefix}_rounded_away_from_zero",
        "cmp.ult.i32",
        input_absolute,
        rounded_absolute,
    )

    plus_one = program.add_immediate(f"{prefix}_plus_one", rounded, 1)
    minus_one = program.add_immediate(f"{prefix}_minus_one", rounded, -1)
    sign_mask = program.constant(f"{prefix}_sign_mask", -(2**31))
    sign = program.binary(f"{prefix}_sign", "and.i32", input_value, sign_mask)
    corrected = program.select(f"{prefix}_corrected", plus_one, minus_one, sign)
    return program.select(
        result_name,
        corrected,
        rounded,
        rounded_away_from_zero,
    )


def _f32_to_signed_integer_rule() -> DescriptorRule:
    program = ScalarProgram()
    emit_truncate_f32_to_signed_i32(
        program,
        ValueRef.operand("input"),
        "signed",
        None,
    )
    return DescriptorRule(
        source_op=scalar_conversion.scalar_fptosi,
        descriptor=program.emits[-1].descriptor,
        guards=(
            Guard.value_type("input", _F32),
            Guard.value_type("result", Scalar(_SIGNED_INTEGERS)),
        ),
        emit=tuple(program.emits),
        report_key="exact_binary32_to_signed_integer",
    )


def _f32_to_narrow_unsigned_integer_rule() -> DescriptorRule:
    program = ScalarProgram()
    emit_truncate_f32_to_signed_i32(
        program,
        ValueRef.operand("input"),
        "unsigned",
        None,
    )
    return DescriptorRule(
        source_op=scalar_conversion.scalar_fptoui,
        descriptor=program.emits[-1].descriptor,
        guards=(
            Guard.value_type("input", _F32),
            Guard.value_type("result", Scalar(_NARROW_INTEGERS)),
        ),
        emit=tuple(program.emits),
        report_key="exact_binary32_to_narrow_unsigned_integer",
    )


def _f32_to_unsigned_i32_rule() -> DescriptorRule:
    program = ScalarProgram()
    input_value = ValueRef.operand("input")
    low_result = emit_truncate_f32_to_signed_i32(
        program,
        input_value,
        "unsigned_low",
        "unsigned_low_truncated",
    )

    high_boundary = program.constant("unsigned_high_boundary", 0x4F000000)
    use_high_range = program.binary(
        "unsigned_use_high_range",
        "cmp.uge.i32",
        input_value,
        high_boundary,
    )
    rounded_half = emit_round_nearest_f32_to_i32(
        program, input_value, -1, "unsigned_high"
    )
    one = program.constant("unsigned_high_shift", 1)
    high_result = program.binary("unsigned_high_result", "lshl.i32", rounded_half, one)
    program.select(None, high_result, low_result, use_high_range)
    return DescriptorRule(
        source_op=scalar_conversion.scalar_fptoui,
        descriptor=program.emits[-1].descriptor,
        guards=(
            Guard.value_type("input", _F32),
            Guard.value_type("result", _I32),
        ),
        emit=tuple(program.emits),
        report_key="exact_binary32_to_unsigned_i32",
    )


def emit_signed_integer_to_f32(
    program: ScalarProgram,
    input_value: ValueRef,
    extend_descriptor_key: str | None,
    result_name: str | None,
) -> ValueRef:
    if extend_descriptor_key is not None:
        input_value = program.unary(
            "integer_extended",
            extend_descriptor_key.removeprefix("amd.xdna.aie2p."),
            input_value,
        )
    scale = program.constant(
        "integer_scale",
        0,
        descriptor_key="amd.xdna.aie2p.constant.i32.fx2flt-scale",
    )
    return program.operation(
        result_name,
        "convert.signed.i32.to.f32",
        operands={"s0": input_value, "m": scale},
    )


def _integer_to_f32_rule(
    source_op: Op,
    input_type: Scalar,
    extend_descriptor_key: str | None,
    report_key: str,
) -> DescriptorRule:
    program = ScalarProgram()
    emit_signed_integer_to_f32(
        program,
        ValueRef.operand("input"),
        extend_descriptor_key,
        None,
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=_descriptor("amd.xdna.aie2p.convert.signed.i32.to.f32"),
        guards=(
            Guard.value_type("input", input_type),
            Guard.value_type("result", _F32),
        ),
        emit=tuple(program.emits),
        report_key=report_key,
    )


def _emit_unsigned_i32_to_f32(
    program: ScalarProgram,
    input_value: ValueRef,
    result_name: str | None,
) -> ValueRef:
    zero = program.constant("integer_zero", 0)
    one = program.constant("integer_one", 1)
    negative_one = program.constant("integer_negative_one", -1)
    exponent_increment = program.constant("integer_exponent_increment", 1 << 23)

    high_bit = program.binary("integer_high_bit", "cmp.slt.i32", input_value, zero)
    half = program.binary("integer_half", "lshl.i32", input_value, negative_one)
    sticky = program.binary("integer_sticky", "and.i32", input_value, one)
    rounded_half = program.binary("integer_rounded_half", "or.i32", half, sticky)
    signed_source = program.select(
        "integer_signed_source", rounded_half, input_value, high_bit
    )

    scale = program.constant(
        "integer_scale",
        0,
        descriptor_key="amd.xdna.aie2p.constant.i32.fx2flt-scale",
    )
    converted = program.operation(
        "integer_converted",
        "convert.signed.i32.to.f32",
        operands={"s0": signed_source, "m": scale},
    )
    adjustment = program.select(
        "integer_adjustment", exponent_increment, zero, high_bit
    )
    return program.operation(
        result_name,
        "add.i32",
        operands={"s0": converted, "s1": adjustment},
    )


def _unsigned_i32_to_f32_rule() -> DescriptorRule:
    program = ScalarProgram()
    _emit_unsigned_i32_to_f32(program, ValueRef.operand("input"), None)
    return DescriptorRule(
        source_op=scalar_conversion.scalar_uitofp,
        descriptor=_descriptor("amd.xdna.aie2p.convert.signed.i32.to.f32"),
        guards=(
            Guard.value_type("input", _I32),
            Guard.value_type("result", _F32),
        ),
        emit=tuple(program.emits),
        report_key="exact_unsigned_i32_to_binary32",
    )


def _bf16_to_f32_rule() -> DescriptorRule:
    program = ScalarProgram()
    shift = program.constant("shift", 16)
    program.operation(
        None,
        "lshl.i32",
        operands={"s0": ValueRef.operand("input"), "s1": shift},
    )
    return DescriptorRule(
        source_op=scalar_conversion.scalar_extf,
        descriptor=_descriptor("amd.xdna.aie2p.lshl.i32"),
        guards=(
            Guard.value_type("input", _BF16),
            Guard.value_type("result", _F32),
        ),
        emit=tuple(program.emits),
        report_key="exact_bf16_to_binary32",
    )


def _f32_to_bf16_rule() -> DescriptorRule:
    """Rounds binary32 to bfloat16 while preserving every NaN as a NaN."""

    program = ScalarProgram()
    input_value = ValueRef.operand("input")
    shift = program.constant("shift", -16)
    upper = program.binary("upper", "lshl.i32", input_value, shift)
    one = program.constant("one", 1)
    upper_lsb = program.binary("upper_lsb", "and.i32", upper, one)
    round_bias = program.constant("round_bias", 0x7FFF)
    bias = program.binary("bias", "add.i32", round_bias, upper_lsb)
    rounded = program.binary("rounded", "add.i32", input_value, bias)
    finite_result = program.binary("finite_result", "lshl.i32", rounded, shift)

    exponent_mask = program.constant("exponent_mask", 0x7F800000)
    exponent = program.binary("exponent", "and.i32", input_value, exponent_mask)
    special_exponent = program.binary(
        "special_exponent", "cmp.eq.i32", exponent, exponent_mask
    )
    fraction_mask = program.constant("fraction_mask", 0x007FFFFF)
    fraction = program.binary("fraction", "and.i32", input_value, fraction_mask)
    nonzero_fraction = program.unary("nonzero_fraction", "cmp.nez.i32", fraction)
    is_nan = program.binary("is_nan", "and.i32", special_exponent, nonzero_fraction)
    quiet_nan_bit = program.constant("quiet_nan_bit", 0x0040)
    nan_result = program.binary("nan_result", "or.i32", upper, quiet_nan_bit)
    program.select(None, nan_result, finite_result, is_nan)

    return DescriptorRule(
        source_op=scalar_conversion.scalar_fptrunc,
        descriptor=program.emits[-1].descriptor,
        guards=(
            Guard.value_type("input", _F32),
            Guard.value_type("result", _BF16),
        ),
        emit=tuple(program.emits),
        report_key="exact_binary32_to_bfloat16",
    )


def emit_fp8_to_f32(
    program: ScalarProgram,
    input_value: ValueRef,
    *,
    mantissa_bits: int,
    exponent_bias: int,
    has_infinity: bool,
    prefix: str,
    result_name: str | None,
) -> ValueRef:
    """Emits an exact float8-to-binary32 widening into |program|."""

    def name(suffix: str) -> str:
        return f"{prefix}_{suffix}"

    mantissa_mask_value = (1 << mantissa_bits) - 1
    exponent_mask_value = ((1 << (8 - mantissa_bits - 1)) - 1) << mantissa_bits
    sign_mask = program.constant(name("sign_mask"), 0x80)
    payload_mask = program.constant(name("payload_mask"), 0x7F)
    mantissa_mask = program.constant(name("mantissa_mask"), mantissa_mask_value)
    exponent_mask = program.constant(name("exponent_mask"), exponent_mask_value)
    exponent_shift = program.constant(name("exponent_shift"), -mantissa_bits)
    sign_shift = program.constant(name("sign_shift"), 24)
    mantissa_shift = program.constant(name("mantissa_shift"), 23 - mantissa_bits)
    f32_exponent_shift = program.constant(name("f32_exponent_shift"), 23)
    source_mask = program.constant(name("source_mask"), 0xFF)

    source = program.binary(
        name("source"),
        "and.i32",
        input_value,
        source_mask,
    )
    sign = program.binary(name("sign"), "and.i32", source, sign_mask)
    sign = program.binary(name("positioned_sign"), "lshl.i32", sign, sign_shift)
    payload = program.binary(name("payload"), "and.i32", source, payload_mask)
    exponent = program.binary(
        name("encoded_exponent_bits"), "and.i32", source, exponent_mask
    )
    exponent = program.binary(
        name("encoded_exponent"), "lshl.i32", exponent, exponent_shift
    )
    mantissa = program.binary(name("mantissa"), "and.i32", source, mantissa_mask)

    normal_exponent_bias = program.constant(
        name("normal_exponent_bias"), 127 - exponent_bias
    )
    normal_exponent = program.binary(
        name("normal_exponent"), "add.i32", exponent, normal_exponent_bias
    )
    normal_exponent = program.binary(
        name("positioned_normal_exponent"),
        "lshl.i32",
        normal_exponent,
        f32_exponent_shift,
    )
    normal_mantissa = program.binary(
        name("positioned_normal_mantissa"),
        "lshl.i32",
        mantissa,
        mantissa_shift,
    )
    normal = program.binary(
        name("normal_unsigned"), "or.i32", normal_exponent, normal_mantissa
    )
    normal = program.binary(name("normal"), "or.i32", sign, normal)

    leading_zeros = program.unary(name("leading_zeros"), "clz.i32", mantissa)
    normalization_bias = program.constant(
        name("normalization_bias"), -(31 - mantissa_bits)
    )
    normalization_shift = program.binary(
        name("normalization_shift"),
        "add.i32",
        leading_zeros,
        normalization_bias,
    )
    normalized_mantissa = program.binary(
        name("normalized_mantissa"),
        "lshl.i32",
        mantissa,
        normalization_shift,
    )
    normalized_mantissa = program.binary(
        name("normalized_payload"),
        "and.i32",
        normalized_mantissa,
        mantissa_mask,
    )
    normalized_mantissa = program.binary(
        name("positioned_normalized_payload"),
        "lshl.i32",
        normalized_mantissa,
        mantissa_shift,
    )
    subnormal_exponent_bias = program.constant(
        name("subnormal_exponent_bias"), 128 - exponent_bias
    )
    subnormal_exponent = program.binary(
        name("subnormal_exponent"),
        "sub.i32",
        subnormal_exponent_bias,
        normalization_shift,
    )
    subnormal_exponent = program.binary(
        name("positioned_subnormal_exponent"),
        "lshl.i32",
        subnormal_exponent,
        f32_exponent_shift,
    )
    subnormal = program.binary(
        name("subnormal_unsigned"),
        "or.i32",
        subnormal_exponent,
        normalized_mantissa,
    )
    subnormal = program.binary(name("subnormal"), "or.i32", sign, subnormal)
    zero_exponent = program.select(name("zero_exponent"), subnormal, sign, mantissa)
    exponent_is_zero = program.unary(name("exponent_is_zero"), "cmp.eqz.i32", exponent)
    finite = program.select(name("finite"), zero_exponent, normal, exponent_is_zero)

    canonical_nan = program.constant(name("canonical_nan"), 0x7FC00000)
    if has_infinity:
        exponent_max = program.constant(
            name("exponent_max"), (1 << (8 - mantissa_bits - 1)) - 1
        )
        exponent_is_special = program.binary(
            name("exponent_is_special"),
            "cmp.eq.i32",
            exponent,
            exponent_max,
        )
        infinity = program.constant(name("infinity"), 0x7F800000)
        infinity = program.binary(name("signed_infinity"), "or.i32", sign, infinity)
        special = program.select(name("special"), canonical_nan, infinity, mantissa)
        return program.select(result_name, special, finite, exponent_is_special)

    nan_payload = program.constant(name("nan_payload"), 0x7F)
    is_nan = program.binary(name("is_nan"), "cmp.eq.i32", payload, nan_payload)
    return program.select(result_name, canonical_nan, finite, is_nan)


def _fp8_to_float_rule(
    input_type: Scalar,
    result_type: Scalar,
    *,
    mantissa_bits: int,
    exponent_bias: int,
    has_infinity: bool,
    report_key: str,
) -> DescriptorRule:
    program = ScalarProgram()
    if input_type == _F8E5M2 and result_type == _F16:
        source_mask = program.constant("fp8_source_mask", 0xFF)
        source = program.binary(
            "fp8_source", "and.i32", ValueRef.operand("input"), source_mask
        )
        shift = program.constant("fp8_f16_shift", 8)
        program.binary(None, "lshl.i32", source, shift)
        return DescriptorRule(
            source_op=scalar_conversion.scalar_extf,
            descriptor=program.emits[-1].descriptor,
            guards=(
                Guard.value_type("input", input_type),
                Guard.value_type("result", result_type),
            ),
            emit=tuple(program.emits),
            report_key=report_key,
        )

    widened = emit_fp8_to_f32(
        program,
        ValueRef.operand("input"),
        mantissa_bits=mantissa_bits,
        exponent_bias=exponent_bias,
        has_infinity=has_infinity,
        prefix="fp8",
        result_name=None if result_type == _F32 else "fp8_as_f32",
    )
    if result_type == _BF16:
        shift = program.constant("fp8_bf16_shift", -16)
        program.binary(None, "lshl.i32", widened, shift)
    elif result_type == _F16:
        emit_f32_to_f16(program, widened, None)
    return DescriptorRule(
        source_op=scalar_conversion.scalar_extf,
        descriptor=program.emits[-1].descriptor,
        guards=(
            Guard.value_type("input", input_type),
            Guard.value_type("result", result_type),
        ),
        emit=tuple(program.emits),
        report_key=report_key,
    )


def emit_f16_to_f32(
    program: ScalarProgram,
    input_value: ValueRef,
    result_name: str | None,
) -> ValueRef:
    """Emits exact binary16-to-binary32 widening into |program|."""

    sign_mask = program.constant("sign_mask", 0x8000)
    nonsign_mask = program.constant("nonsign_mask", 0x7FFF)
    exponent_mask = program.constant("exponent_mask", 0x7C00)
    fraction_mask = program.constant("fraction_mask", 0x03FF)
    shift_13 = program.constant("shift_13", 13)
    shift_16 = program.constant("shift_16", 16)
    shift_23 = program.constant("shift_23", 23)

    sign = program.binary("sign", "and.i32", input_value, sign_mask)
    sign = program.binary("positioned_sign", "lshl.i32", sign, shift_16)
    nonsign = program.binary("nonsign", "and.i32", input_value, nonsign_mask)
    exponent = program.binary("exponent", "and.i32", input_value, exponent_mask)
    fraction = program.binary("fraction", "and.i32", input_value, fraction_mask)
    positioned_payload = program.binary(
        "positioned_payload", "lshl.i32", nonsign, shift_13
    )
    normal_bias = program.constant("normal_bias", 0x38000000)
    normal_unsigned = program.binary(
        "normal_unsigned", "add.i32", positioned_payload, normal_bias
    )
    normal = program.binary("normal", "or.i32", sign, normal_unsigned)

    special_bias = program.constant("special_bias", 0x70000000)
    special_unsigned = program.binary(
        "special_unsigned", "add.i32", positioned_payload, special_bias
    )
    special = program.binary("special", "or.i32", sign, special_unsigned)

    leading_zeros = program.unary("leading_zeros", "clz.i32", fraction)
    normalization_bias = program.constant("normalization_bias", -21)
    normalization_shift = program.binary(
        "normalization_shift", "add.i32", leading_zeros, normalization_bias
    )
    normalized_fraction = program.binary(
        "normalized_fraction", "lshl.i32", fraction, normalization_shift
    )
    normalized_payload = program.binary(
        "normalized_payload", "and.i32", normalized_fraction, fraction_mask
    )
    normalized_payload = program.binary(
        "positioned_normalized_payload",
        "lshl.i32",
        normalized_payload,
        shift_13,
    )
    subnormal_exponent_bias = program.constant("subnormal_exponent_bias", 113)
    subnormal_exponent = program.binary(
        "subnormal_exponent",
        "sub.i32",
        subnormal_exponent_bias,
        normalization_shift,
    )
    subnormal_exponent = program.binary(
        "positioned_subnormal_exponent",
        "lshl.i32",
        subnormal_exponent,
        shift_23,
    )
    subnormal_unsigned = program.binary(
        "subnormal_unsigned",
        "or.i32",
        subnormal_exponent,
        normalized_payload,
    )
    subnormal = program.binary("subnormal", "or.i32", sign, subnormal_unsigned)
    zero_exponent = program.select("zero_exponent", subnormal, sign, fraction)

    exponent_is_zero = program.unary("exponent_is_zero", "cmp.eqz.i32", exponent)
    exponent_is_special = program.binary(
        "exponent_is_special", "cmp.eq.i32", exponent, exponent_mask
    )
    nonzero_exponent = program.select(
        "nonzero_exponent", special, normal, exponent_is_special
    )
    return program.select(
        result_name, zero_exponent, nonzero_exponent, exponent_is_zero
    )


def _f16_to_f32_rule() -> DescriptorRule:
    program = ScalarProgram()
    emit_f16_to_f32(program, ValueRef.operand("input"), None)

    return DescriptorRule(
        source_op=scalar_conversion.scalar_extf,
        descriptor=_descriptor("amd.xdna.aie2p.select.nonzero.i32"),
        guards=(
            Guard.value_type("input", _F16),
            Guard.value_type("result", _F32),
        ),
        emit=tuple(program.emits),
        report_key="exact_f16_to_binary32",
    )


def emit_f32_to_f16(
    program: ScalarProgram,
    input_value: ValueRef,
    result_name: str | None,
) -> ValueRef:
    """Emits exact binary32-to-binary16 narrowing into |program|."""

    sign_mask = program.constant("sign_mask", -(1 << 31))
    nonsign_mask = program.constant("nonsign_mask", 0x7FFFFFFF)
    fraction_mask = program.constant("fraction_mask", 0x007FFFFF)
    hidden_bit = program.constant("hidden_bit", 0x00800000)
    one = program.constant("one", 1)
    zero = program.constant("zero", 0)
    shift_13 = program.constant("shift_13", -13)
    shift_16 = program.constant("shift_16", -16)
    shift_23 = program.constant("shift_23", -23)

    sign = program.binary("sign", "and.i32", input_value, sign_mask)
    sign = program.binary("positioned_sign", "lshl.i32", sign, shift_16)
    nonsign = program.binary("nonsign", "and.i32", input_value, nonsign_mask)
    exponent = program.binary("exponent", "lshl.i32", nonsign, shift_23)
    fraction = program.binary("fraction", "and.i32", nonsign, fraction_mask)

    # Normal binary16 values are the rounded binary32 payload after rebiasing
    # the exponent. Adding the rounding bias before shifting also carries a
    # rounded maximum mantissa into the next exponent, including infinity.
    normal_exponent_bias = program.constant("normal_exponent_bias", 0x38000000)
    normal_payload = program.binary(
        "normal_payload", "sub.i32", nonsign, normal_exponent_bias
    )
    normal_truncated = program.binary(
        "normal_truncated", "lshl.i32", normal_payload, shift_13
    )
    normal_lsb = program.binary("normal_lsb", "and.i32", normal_truncated, one)
    normal_round_bias = program.constant("normal_round_bias", 0x0FFF)
    normal_round_bias = program.binary(
        "normal_tie_bias", "add.i32", normal_round_bias, normal_lsb
    )
    normal_rounded = program.binary(
        "normal_rounded", "add.i32", normal_payload, normal_round_bias
    )
    normal_result = program.binary(
        "normal_result", "lshl.i32", normal_rounded, shift_13
    )

    # A binary16 subnormal has an exponent-dependent shift. Clamp the exponent
    # before forming that shift because every descriptor in the straight-line
    # program executes even when a later select discards its result.
    subnormal_min_exponent = program.constant("subnormal_min_exponent", 102)
    normal_min_exponent = program.constant("normal_min_exponent", 113)
    subnormal_max_exponent = program.constant("subnormal_max_exponent", 112)
    below_subnormal = program.binary(
        "below_subnormal",
        "cmp.ult.i32",
        exponent,
        subnormal_min_exponent,
    )
    clamped_exponent = program.select(
        "low_clamped_exponent",
        subnormal_min_exponent,
        exponent,
        below_subnormal,
    )
    at_or_above_normal = program.binary(
        "at_or_above_normal",
        "cmp.uge.i32",
        clamped_exponent,
        normal_min_exponent,
    )
    clamped_exponent = program.select(
        "clamped_exponent",
        subnormal_max_exponent,
        clamped_exponent,
        at_or_above_normal,
    )
    shift_origin = program.constant("shift_origin", 126)
    negative_subnormal_shift = program.binary(
        "negative_subnormal_shift",
        "sub.i32",
        clamped_exponent,
        shift_origin,
    )
    subnormal_payload = program.binary(
        "subnormal_payload", "or.i32", fraction, hidden_bit
    )
    subnormal_truncated = program.binary(
        "subnormal_truncated",
        "lshl.i32",
        subnormal_payload,
        negative_subnormal_shift,
    )
    subnormal_lsb = program.binary("subnormal_lsb", "and.i32", subnormal_truncated, one)
    rounding_shift_origin = program.constant("rounding_shift_origin", 125)
    rounding_shift = program.binary(
        "rounding_shift",
        "sub.i32",
        rounding_shift_origin,
        clamped_exponent,
    )
    rounding_half = program.binary("rounding_half", "lshl.i32", one, rounding_shift)
    rounding_bias_base = program.binary(
        "rounding_bias_base", "sub.i32", rounding_half, one
    )
    subnormal_round_bias = program.binary(
        "subnormal_round_bias", "add.i32", rounding_bias_base, subnormal_lsb
    )
    subnormal_rounded = program.binary(
        "subnormal_rounded", "add.i32", subnormal_payload, subnormal_round_bias
    )
    subnormal_result = program.binary(
        "subnormal_result",
        "lshl.i32",
        subnormal_rounded,
        negative_subnormal_shift,
    )
    subnormal_result = program.select(
        "subnormal_or_zero", zero, subnormal_result, below_subnormal
    )
    below_normal = program.binary(
        "below_normal", "cmp.ult.i32", exponent, normal_min_exponent
    )
    finite_result = program.select(
        "finite_narrow_result",
        subnormal_result,
        normal_result,
        below_normal,
    )

    infinity = program.constant("infinity", 0x7C00)
    overflow_exponent = program.constant("overflow_exponent", 143)
    overflows = program.binary("overflows", "cmp.uge.i32", exponent, overflow_exponent)
    finite_result = program.select("finite_result", infinity, finite_result, overflows)

    # Preserve the high payload bits and force the quiet bit so even a NaN
    # represented solely by discarded low payload bits remains a binary16 NaN.
    nan_payload = program.binary("nan_payload", "lshl.i32", fraction, shift_13)
    quiet_nan_bit = program.constant("quiet_nan_bit", 0x0200)
    nan_payload = program.binary(
        "quiet_nan_payload", "or.i32", nan_payload, quiet_nan_bit
    )
    nan_result = program.binary("nan_result", "or.i32", infinity, nan_payload)
    is_nan = program.unary("is_nan", "cmp.nez.i32", fraction)
    special_result = program.select("special_result", nan_result, infinity, is_nan)
    special_exponent = program.constant("special_exponent", 255)
    is_special = program.binary("is_special", "cmp.eq.i32", exponent, special_exponent)
    magnitude = program.select("magnitude", special_result, finite_result, is_special)
    return program.binary(result_name, "or.i32", sign, magnitude)


def _f32_to_f16_rule() -> DescriptorRule:
    """Rounds binary32 to IEEE binary16 with exact special-value behavior."""

    program = ScalarProgram()
    emit_f32_to_f16(program, ValueRef.operand("input"), None)

    return DescriptorRule(
        source_op=scalar_conversion.scalar_fptrunc,
        descriptor=program.emits[-1].descriptor,
        guards=(
            Guard.value_type("input", _F32),
            Guard.value_type("result", _F16),
        ),
        emit=tuple(program.emits),
        report_key="exact_binary32_to_f16",
    )


def emit_f32_to_fp8(
    program: ScalarProgram,
    input_value: ValueRef,
    *,
    mantissa_bits: int,
    exponent_bias: int,
    has_infinity: bool,
    prefix: str,
    result_name: str | None,
) -> ValueRef:
    """Emits exact RNE binary32-to-float8 narrowing into |program|."""

    def name(suffix: str) -> str:
        return f"{prefix}_{suffix}"

    sign_mask = program.constant(name("sign_mask"), -(1 << 31))
    nonsign_mask = program.constant(name("nonsign_mask"), 0x7FFFFFFF)
    fraction_mask = program.constant(name("fraction_mask"), 0x007FFFFF)
    hidden_bit = program.constant(name("hidden_bit"), 0x00800000)
    one = program.constant(name("one"), 1)
    zero = program.constant(name("zero"), 0)
    sign_shift = program.constant(name("sign_shift"), -24)
    exponent_shift = program.constant(name("exponent_shift"), -23)
    mantissa_shift_value = 23 - mantissa_bits
    mantissa_shift = program.constant(name("mantissa_shift"), -mantissa_shift_value)

    sign = program.binary(name("sign"), "and.i32", input_value, sign_mask)
    sign = program.binary(name("positioned_sign"), "lshl.i32", sign, sign_shift)
    nonsign = program.binary(name("nonsign"), "and.i32", input_value, nonsign_mask)
    exponent = program.binary(name("exponent"), "lshl.i32", nonsign, exponent_shift)
    fraction = program.binary(name("fraction"), "and.i32", nonsign, fraction_mask)

    # Round the normal destination mantissa before combining it with the
    # rebased exponent. A mantissa carry then increments the exponent without
    # a separate branch or predicate.
    normal_truncated = program.binary(
        name("normal_truncated"), "lshl.i32", fraction, mantissa_shift
    )
    normal_lsb = program.binary(name("normal_lsb"), "and.i32", normal_truncated, one)
    normal_round_bias = program.constant(
        name("normal_round_bias"), (1 << (mantissa_shift_value - 1)) - 1
    )
    normal_round_bias = program.binary(
        name("normal_tie_bias"), "add.i32", normal_round_bias, normal_lsb
    )
    normal_rounded = program.binary(
        name("normal_rounded"), "add.i32", fraction, normal_round_bias
    )
    normal_mantissa = program.binary(
        name("normal_mantissa"), "lshl.i32", normal_rounded, mantissa_shift
    )
    normal_exponent_bias = program.constant(
        name("normal_exponent_bias"), exponent_bias - 127
    )
    normal_exponent = program.binary(
        name("normal_exponent"), "add.i32", exponent, normal_exponent_bias
    )
    destination_exponent_shift = program.constant(
        name("destination_exponent_shift"), mantissa_bits
    )
    normal_exponent = program.binary(
        name("positioned_normal_exponent"),
        "lshl.i32",
        normal_exponent,
        destination_exponent_shift,
    )
    normal = program.binary(name("normal"), "add.i32", normal_exponent, normal_mantissa)

    # Destination subnormals use an exponent-dependent shift. Clamp the source
    # exponent before evaluating the straight-line rounding program so every
    # dynamic shift remains defined even when the result is later discarded.
    minimum_rounding_exponent_value = 127 - mantissa_bits - exponent_bias
    minimum_rounding_exponent = program.constant(
        name("minimum_rounding_exponent"), minimum_rounding_exponent_value
    )
    normal_minimum_exponent = program.constant(
        name("normal_minimum_exponent"), 128 - exponent_bias
    )
    subnormal_maximum_exponent = program.constant(
        name("subnormal_maximum_exponent"), 127 - exponent_bias
    )
    below_subnormal = program.binary(
        name("below_subnormal"),
        "cmp.ult.i32",
        exponent,
        minimum_rounding_exponent,
    )
    clamped_exponent = program.select(
        name("low_clamped_exponent"),
        minimum_rounding_exponent,
        exponent,
        below_subnormal,
    )
    at_or_above_normal = program.binary(
        name("at_or_above_normal"),
        "cmp.uge.i32",
        clamped_exponent,
        normal_minimum_exponent,
    )
    clamped_exponent = program.select(
        name("clamped_exponent"),
        subnormal_maximum_exponent,
        clamped_exponent,
        at_or_above_normal,
    )
    subnormal_shift_origin_value = 151 - mantissa_bits - exponent_bias
    subnormal_shift_origin = program.constant(
        name("subnormal_shift_origin"), subnormal_shift_origin_value
    )
    negative_subnormal_shift = program.binary(
        name("negative_subnormal_shift"),
        "sub.i32",
        clamped_exponent,
        subnormal_shift_origin,
    )
    significand = program.binary(name("significand"), "or.i32", fraction, hidden_bit)
    subnormal_truncated = program.binary(
        name("subnormal_truncated"),
        "lshl.i32",
        significand,
        negative_subnormal_shift,
    )
    subnormal_lsb = program.binary(
        name("subnormal_lsb"), "and.i32", subnormal_truncated, one
    )
    rounding_shift_origin = program.constant(
        name("rounding_shift_origin"), subnormal_shift_origin_value - 1
    )
    rounding_shift = program.binary(
        name("rounding_shift"),
        "sub.i32",
        rounding_shift_origin,
        clamped_exponent,
    )
    rounding_half = program.binary(
        name("rounding_half"), "lshl.i32", one, rounding_shift
    )
    rounding_bias_base = program.binary(
        name("rounding_bias_base"), "sub.i32", rounding_half, one
    )
    subnormal_round_bias = program.binary(
        name("subnormal_round_bias"),
        "add.i32",
        rounding_bias_base,
        subnormal_lsb,
    )
    subnormal_rounded = program.binary(
        name("subnormal_rounded"),
        "add.i32",
        significand,
        subnormal_round_bias,
    )
    subnormal = program.binary(
        name("subnormal"),
        "lshl.i32",
        subnormal_rounded,
        negative_subnormal_shift,
    )
    subnormal = program.select(
        name("subnormal_or_zero"), zero, subnormal, below_subnormal
    )
    below_normal = program.binary(
        name("below_normal"),
        "cmp.ult.i32",
        exponent,
        normal_minimum_exponent,
    )
    magnitude = program.select(
        name("finite_unclamped"), subnormal, normal, below_normal
    )

    nan_payload_value = (1 << (8 - 1)) - 1
    nan_payload = program.constant(name("nan_payload"), nan_payload_value)
    if has_infinity:
        infinity_value = ((1 << (8 - mantissa_bits - 1)) - 1) << mantissa_bits
        infinity = program.constant(name("infinity"), infinity_value)
        overflow_exponent = program.constant(
            name("overflow_exponent"),
            128 + (1 << (8 - mantissa_bits - 2)) - 1,
        )
        overflows = program.binary(
            name("overflows"),
            "cmp.uge.i32",
            exponent,
            overflow_exponent,
        )
        magnitude = program.select(name("finite"), infinity, magnitude, overflows)
        special_magnitude = program.select(
            name("special_magnitude"), nan_payload, infinity, fraction
        )
    else:
        maximum_finite = program.constant(name("maximum_finite"), nan_payload_value - 1)
        uses_reserved_payload = program.binary(
            name("uses_reserved_payload"),
            "cmp.uge.i32",
            magnitude,
            nan_payload,
        )
        magnitude = program.select(
            name("finite"), maximum_finite, magnitude, uses_reserved_payload
        )
        special_magnitude = program.select(
            name("special_magnitude"), nan_payload, maximum_finite, fraction
        )

    special_exponent = program.constant(name("special_exponent"), 255)
    is_special = program.binary(
        name("is_special"), "cmp.eq.i32", exponent, special_exponent
    )
    magnitude = program.select(
        name("magnitude"), special_magnitude, magnitude, is_special
    )
    return program.binary(result_name, "or.i32", sign, magnitude)


def emit_f16_to_f8e5m2(
    program: ScalarProgram,
    input_value: ValueRef,
    result_name: str | None,
) -> ValueRef:
    """Emits exact RNE binary16-to-E5M2 narrowing into |program|."""

    nonsign_mask = program.constant("f16_e5m2_nonsign_mask", 0x7FFF)
    sign_mask = program.constant("f16_e5m2_sign_mask", 0x8000)
    fraction_mask = program.constant("f16_e5m2_fraction_mask", 0x03FF)
    exponent_mask = program.constant("f16_e5m2_exponent_mask", 0x7C00)
    one = program.constant("f16_e5m2_one", 1)
    shift = program.constant("f16_e5m2_shift", -8)

    sign = program.binary("f16_e5m2_sign", "and.i32", input_value, sign_mask)
    sign = program.binary("f16_e5m2_positioned_sign", "lshl.i32", sign, shift)
    nonsign = program.binary("f16_e5m2_nonsign", "and.i32", input_value, nonsign_mask)
    truncated = program.binary("f16_e5m2_truncated", "lshl.i32", nonsign, shift)
    truncated_lsb = program.binary("f16_e5m2_truncated_lsb", "and.i32", truncated, one)
    round_bias = program.constant("f16_e5m2_round_bias", 0x7F)
    round_bias = program.binary(
        "f16_e5m2_tie_bias", "add.i32", round_bias, truncated_lsb
    )
    rounded = program.binary("f16_e5m2_rounded", "add.i32", nonsign, round_bias)
    finite = program.binary("f16_e5m2_finite", "lshl.i32", rounded, shift)

    fraction = program.binary(
        "f16_e5m2_fraction", "and.i32", input_value, fraction_mask
    )
    is_nan = program.unary("f16_e5m2_is_nan", "cmp.nez.i32", fraction)
    infinity = program.constant("f16_e5m2_infinity", 0x7C)
    nan = program.constant("f16_e5m2_nan", 0x7F)
    special = program.select("f16_e5m2_special", nan, infinity, is_nan)
    exponent = program.binary(
        "f16_e5m2_exponent", "and.i32", input_value, exponent_mask
    )
    is_special = program.binary(
        "f16_e5m2_is_special", "cmp.eq.i32", exponent, exponent_mask
    )
    magnitude = program.select("f16_e5m2_magnitude", special, finite, is_special)
    return program.binary(result_name, "or.i32", sign, magnitude)


def _float_to_fp8_rule(
    input_type: Scalar,
    result_type: Scalar,
    *,
    mantissa_bits: int,
    exponent_bias: int,
    has_infinity: bool,
    report_key: str,
) -> DescriptorRule:
    program = ScalarProgram()
    if input_type == _F16 and result_type == _F8E5M2:
        emit_f16_to_f8e5m2(program, ValueRef.operand("input"), None)
    else:
        widened = ValueRef.operand("input")
        if input_type == _BF16:
            shift = program.constant("source_bf16_shift", 16)
            widened = program.binary("source_as_f32", "lshl.i32", widened, shift)
        elif input_type == _F16:
            widened = emit_f16_to_f32(program, widened, "source_as_f32")
        emit_f32_to_fp8(
            program,
            widened,
            mantissa_bits=mantissa_bits,
            exponent_bias=exponent_bias,
            has_infinity=has_infinity,
            prefix="fp8",
            result_name=None,
        )
    return DescriptorRule(
        source_op=scalar_conversion.scalar_fptrunc,
        descriptor=program.emits[-1].descriptor,
        guards=(
            Guard.value_type("input", input_type),
            Guard.value_type("result", result_type),
        ),
        emit=tuple(program.emits),
        report_key=report_key,
    )


def _integer_to_f16_rule(
    source_op: Op,
    input_type: Scalar,
    extend_descriptor_key: str | None,
    report_key: str,
) -> DescriptorRule:
    """Converts an integer to binary16 without a double-rounding ambiguity."""

    # Every integer in binary16's finite range is represented exactly by
    # binary32. Any integer that binary32 could round first is already beyond
    # binary16's finite rounding boundary and therefore maps to infinity.
    program = ScalarProgram()
    converted = emit_signed_integer_to_f32(
        program,
        ValueRef.operand("input"),
        extend_descriptor_key,
        "integer_as_f32",
    )
    emit_f32_to_f16(program, converted, None)
    return DescriptorRule(
        source_op=source_op,
        descriptor=program.emits[-1].descriptor,
        guards=(
            Guard.value_type("input", input_type),
            Guard.value_type("result", _F16),
        ),
        emit=tuple(program.emits),
        report_key=report_key,
    )


def _unsigned_i32_to_f16_rule() -> DescriptorRule:
    program = ScalarProgram()
    converted = _emit_unsigned_i32_to_f32(
        program,
        ValueRef.operand("input"),
        "integer_as_f32",
    )
    emit_f32_to_f16(program, converted, None)
    return DescriptorRule(
        source_op=scalar_conversion.scalar_uitofp,
        descriptor=program.emits[-1].descriptor,
        guards=(
            Guard.value_type("input", _I32),
            Guard.value_type("result", _F16),
        ),
        emit=tuple(program.emits),
        report_key="exact_unsigned_i32_to_f16",
    )


def _bf16x16_to_signed_i32x16_rule() -> DescriptorRule:
    """Truncates every defined BF16 input to signed i32 exactly."""

    emits: list[ContractEmit] = []

    def constant(
        result_name: str,
        value: int,
        descriptor_key: str = "constant.i32",
    ) -> ValueRef:
        result = ValueRef.temporary(result_name)
        emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor(f"amd.xdna.aie2p.{descriptor_key}"),
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
                immediates={"i": value},
                form=DescriptorEmitForm.CONST,
            )
        )
        return result

    def operation(
        result_name: str | None,
        descriptor_key: str,
        result_field: str,
        *,
        immediates: dict[str, int] | None = None,
        **operands: ValueRef,
    ) -> ValueRef:
        result = (
            ValueRef.result("result")
            if result_name is None
            else ValueRef.temporary(result_name)
        )
        emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor(f"amd.xdna.aie2p.{descriptor_key}"),
                operands=operands,
                results={result_field: result},
                result_types=(
                    None
                    if result_name is None
                    else {result_field: DescriptorResultType()}
                ),
                immediates={} if immediates is None else immediates,
                form=DescriptorEmitForm.OP,
            )
        )
        return result

    # VFLOOR implements floor, while vector.fptosi requires truncation toward
    # zero. Convert a nonnegative magnitude and restore the sign afterward.
    # The sole defined input whose magnitude is not representable in signed
    # i32 is -2^31; sanitize that lane before VFLOOR and restore it last. This
    # keeps the native conversion in range for every defined source value.
    input_value = ValueRef.operand("input")
    minimum_scalar = constant("minimum_bf16_scalar", 0xCF00)
    minimum_bf16 = operation("minimum_bf16", "splat.i16x32", "dst", src=minimum_scalar)
    minimum_difference = operation(
        "minimum_difference",
        "sub.i16x32",
        "d",
        s1=input_value,
        s2=minimum_bf16,
    )
    minimum_low = operation(
        "minimum_low", "cmp.eqz.i16x32.el.low32", "cmp", s2=minimum_difference
    )
    minimum = operation(
        "minimum",
        "predicate.complete.zero.high32",
        "dst",
        immediates={"i": 0},
        storage=minimum_low,
    )
    zero = operation("zero", "sub.i16x32", "d", s1=minimum_bf16, s2=minimum_bf16)
    negative_low = operation(
        "negative_low",
        "cmp.lt.signed.i16x32.el.low32",
        "cmp",
        s1=input_value,
        s2=zero,
    )
    negative = operation(
        "negative",
        "predicate.complete.zero.high32",
        "dst",
        immediates={"i": 0},
        storage=negative_low,
    )
    absolute_mask_scalar = constant("absolute_mask_scalar", 0x7FFF)
    absolute_mask = operation(
        "absolute_mask", "splat.i16x32", "dst", src=absolute_mask_scalar
    )
    absolute = operation(
        "absolute", "and.bits512", "d", s1=input_value, s2=absolute_mask
    )
    safe_absolute = operation(
        "safe_absolute",
        "select.i16x32.mask64",
        "d",
        s1=absolute,
        s2=zero,
        sel=minimum,
    )
    safe_absolute_w = ValueRef.temporary("safe_absolute_w")
    emits.append(
        EmitRegisterSlice(
            source=safe_absolute,
            result=safe_absolute_w,
            unit_count=1,
        )
    )
    shift = constant("shift", 0, "constant.i32.shift")
    magnitude = operation(
        "magnitude",
        "convert.floor.bf16x16.to.i32x16",
        "dst",
        src=safe_absolute_w,
        shft=shift,
    )
    negative_magnitude = operation(
        "negative_magnitude", "sub.i32x16", "d", s1=zero, s2=magnitude
    )
    signed_result = operation(
        "signed_result",
        "select.i32x16.mask64",
        "d",
        s1=magnitude,
        s2=negative_magnitude,
        sel=negative,
    )
    minimum_i32_scalar = constant("minimum_i32_scalar", -(2**31))
    minimum_i32 = operation(
        "minimum_i32", "splat.i32x16", "dst", src=minimum_i32_scalar
    )
    operation(
        None,
        "select.i32x16.mask64",
        "d",
        s1=signed_result,
        s2=minimum_i32,
        sel=minimum,
    )

    return DescriptorRule(
        source_op=vector.vector_fptosi,
        descriptor=emits[-1].descriptor,
        guards=(
            Guard.value_type("input", _BF16X16),
            Guard.value_type("result", _I32X16),
        ),
        emit=tuple(emits),
        report_key="exact_bfloat16x16_to_signed_i32x16",
    )


def _integer_extend_rule(
    source_op: Op,
    input_type: Scalar,
    result_type: Scalar,
    operation: str,
) -> DescriptorRule:
    program = ScalarProgram()
    program.unary(None, operation, ValueRef.operand("input"))
    return DescriptorRule(
        source_op=source_op,
        descriptor=program.emits[-1].descriptor,
        guards=(
            Guard.value_type("input", input_type),
            Guard.value_type("result", result_type),
        ),
        emit=tuple(program.emits),
    )


def _integer_truncate_alias_rule(
    input_type: Scalar,
    result_type: Scalar,
) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=scalar_conversion.scalar_trunci,
        source=ValueRef.operand("input"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_type("input", input_type),
            Guard.value_type("result", result_type),
        ),
    )


def _integer_to_i64_rule(
    source_op: Op,
    input_type: Scalar,
    extend_operation: str | None,
    *,
    signed: bool,
) -> DescriptorRule:
    program = ScalarProgram()
    low_word = ValueRef.operand("input")
    if extend_operation is not None:
        low_word = program.unary("low_word", extend_operation, low_word)
    if signed:
        sign_shift = program.constant("sign_shift", -31)
        high_word = program.binary("high_word", "ashl.i32", low_word, sign_shift)
    else:
        high_word = program.constant("high_word", 0)
    return DescriptorRule(
        source_op=source_op,
        descriptor=program.emits[-1].descriptor,
        guards=(
            Guard.value_type("input", input_type),
            Guard.value_type("result", _I64),
        ),
        emit=(
            *program.emits,
            EmitRegisterConcat(
                sources=(low_word, high_word),
                result=ValueRef.result("result"),
            ),
        ),
    )


def _i64_truncate_rule(result_type: Scalar) -> DescriptorRule:
    return DescriptorRule(
        source_op=scalar_conversion.scalar_trunci,
        guards=(
            Guard.value_type("input", _I64),
            Guard.value_type("result", result_type),
        ),
        emit=(
            EmitRegisterSlice(
                source=ValueRef.operand("input"),
                result=ValueRef.result("result"),
            ),
        ),
    )


AIE2P_CONVERSION_RULES = (
    *AIE2P_PACKET_CONVERSION_RULES,
    *(
        _integer_extend_rule(source_op, input_type, result_type, operation)
        for source_op, input_type, result_type, operation in (
            (
                scalar_conversion.scalar_extsi,
                _I8,
                _I16,
                "extend.signed.i8",
            ),
            (
                scalar_conversion.scalar_extsi,
                _I8,
                _I32,
                "extend.signed.i8",
            ),
            (
                scalar_conversion.scalar_extsi,
                _I16,
                _I32,
                "extend.signed.i16",
            ),
            (
                scalar_conversion.scalar_extui,
                _I8,
                _I16,
                "extend.unsigned.i8",
            ),
            (
                scalar_conversion.scalar_extui,
                _I8,
                _I32,
                "extend.unsigned.i8",
            ),
            (
                scalar_conversion.scalar_extui,
                _I16,
                _I32,
                "extend.unsigned.i16",
            ),
        )
    ),
    *(
        _integer_truncate_alias_rule(input_type, result_type)
        for input_type, result_type in (
            (_I16, _I8),
            (_I32, _I8),
            (_I32, _I16),
        )
    ),
    *(
        _integer_to_i64_rule(
            source_op,
            input_type,
            extend_operation,
            signed=signed,
        )
        for source_op, input_type, extend_operation, signed in (
            (
                scalar_conversion.scalar_extsi,
                _I8,
                "extend.signed.i8",
                True,
            ),
            (
                scalar_conversion.scalar_extsi,
                _I16,
                "extend.signed.i16",
                True,
            ),
            (scalar_conversion.scalar_extsi, _I32, None, True),
            (
                scalar_conversion.scalar_extui,
                _I8,
                "extend.unsigned.i8",
                False,
            ),
            (
                scalar_conversion.scalar_extui,
                _I16,
                "extend.unsigned.i16",
                False,
            ),
            (scalar_conversion.scalar_extui, _I32, None, False),
        )
    ),
    *(_i64_truncate_rule(result_type) for result_type in (_I8, _I16, _I32)),
    _integer_to_f32_rule(
        scalar_conversion.scalar_sitofp,
        _I8,
        "amd.xdna.aie2p.extend.signed.i8",
        "native_signed_i8_to_binary32",
    ),
    _integer_to_f32_rule(
        scalar_conversion.scalar_sitofp,
        _I16,
        "amd.xdna.aie2p.extend.signed.i16",
        "native_signed_i16_to_binary32",
    ),
    _integer_to_f32_rule(
        scalar_conversion.scalar_sitofp,
        _I32,
        None,
        "native_signed_i32_to_binary32",
    ),
    _integer_to_f32_rule(
        scalar_conversion.scalar_uitofp,
        _I8,
        "amd.xdna.aie2p.extend.unsigned.i8",
        "native_unsigned_i8_to_binary32",
    ),
    _integer_to_f32_rule(
        scalar_conversion.scalar_uitofp,
        _I16,
        "amd.xdna.aie2p.extend.unsigned.i16",
        "native_unsigned_i16_to_binary32",
    ),
    _unsigned_i32_to_f32_rule(),
    _integer_to_f16_rule(
        scalar_conversion.scalar_sitofp,
        _I8,
        "amd.xdna.aie2p.extend.signed.i8",
        "exact_signed_i8_to_f16",
    ),
    _integer_to_f16_rule(
        scalar_conversion.scalar_sitofp,
        _I16,
        "amd.xdna.aie2p.extend.signed.i16",
        "exact_signed_i16_to_f16",
    ),
    _integer_to_f16_rule(
        scalar_conversion.scalar_sitofp,
        _I32,
        None,
        "exact_signed_i32_to_f16",
    ),
    _integer_to_f16_rule(
        scalar_conversion.scalar_uitofp,
        _I8,
        "amd.xdna.aie2p.extend.unsigned.i8",
        "exact_unsigned_i8_to_f16",
    ),
    _integer_to_f16_rule(
        scalar_conversion.scalar_uitofp,
        _I16,
        "amd.xdna.aie2p.extend.unsigned.i16",
        "exact_unsigned_i16_to_f16",
    ),
    _unsigned_i32_to_f16_rule(),
    _f32_to_signed_integer_rule(),
    _f32_to_narrow_unsigned_integer_rule(),
    _f32_to_unsigned_i32_rule(),
    _bf16_to_f32_rule(),
    _f32_to_bf16_rule(),
    _f16_to_f32_rule(),
    _f32_to_f16_rule(),
    *(
        _fp8_to_float_rule(
            input_type,
            result_type,
            mantissa_bits=mantissa_bits,
            exponent_bias=exponent_bias,
            has_infinity=has_infinity,
            report_key=(
                f"exact_{input_type.element.lower()}_to_{result_type.element.lower()}"
            ),
        )
        for input_type, mantissa_bits, exponent_bias, has_infinity in (
            (_F8E4M3, 3, 7, False),
            (_F8E5M2, 2, 15, True),
        )
        for result_type in (_F16, _BF16, _F32)
    ),
    *(
        _float_to_fp8_rule(
            input_type,
            result_type,
            mantissa_bits=mantissa_bits,
            exponent_bias=exponent_bias,
            has_infinity=has_infinity,
            report_key=(
                f"exact_{input_type.element.lower()}_to_{result_type.element.lower()}"
            ),
        )
        for result_type, mantissa_bits, exponent_bias, has_infinity in (
            (_F8E4M3, 3, 7, False),
            (_F8E5M2, 2, 15, True),
        )
        for input_type in (_F16, _BF16, _F32)
    ),
    _bf16x16_to_signed_i32x16_rule(),
)
