# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact scalar float-narrowing Wasm contract rules."""

from collections.abc import Callable

from loom.dialect.scalar import conversion as scalar_conversion
from loom.target.contracts import (
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    TypePattern,
    ValueRef,
)
from loom.target.emit.float_narrowing import (
    BF16_FORMAT,
    BF16_SOURCE_FORMAT,
    F8E4M3_FORMAT,
    F8E5M2_FORMAT,
    F16_FORMAT,
    F16_SOURCE_FORMAT,
    F32_FORMAT,
    F64_FORMAT,
    BinaryFloatFormat,
    FloatNarrowingDescriptors,
    IntegerNarrowingDescriptors,
    NarrowFloatFormat,
    NarrowFloatSubnormalRounding,
    build_f32_to_bf16_emits,
    build_float_to_narrow_float_emits,
    build_integer_bits_to_narrow_float_emits,
    build_representable_f32_to_bf16_emits,
)
from loom.target.low_descriptors import Descriptor

_F8E4M3 = Scalar("f8E4M3")
_F8E5M2 = Scalar("f8E5M2")
_F16 = Scalar("f16")
_BF16 = Scalar("bf16")
_F32 = Scalar("f32")
_F64 = Scalar("f64")


def _integer_narrowing_descriptors(
    descriptor_lookup: Callable[[str], Descriptor],
    integer_bit_width: int,
) -> IntegerNarrowingDescriptors:
    integer_name = f"i{integer_bit_width}"
    return IntegerNarrowingDescriptors(
        integer_bit_width=integer_bit_width,
        integer_constant=descriptor_lookup(f"wasm.{integer_name}.const"),
        integer_add=descriptor_lookup(f"wasm.{integer_name}.add"),
        integer_subtract=descriptor_lookup(f"wasm.{integer_name}.sub"),
        integer_shift_left=descriptor_lookup(f"wasm.{integer_name}.shl"),
        integer_shift_right_logical=descriptor_lookup(f"wasm.{integer_name}.shr_u"),
        integer_bitwise_and=descriptor_lookup(f"wasm.{integer_name}.and"),
        integer_bitwise_or=descriptor_lookup(f"wasm.{integer_name}.or"),
        integer_less_than_nonnegative=descriptor_lookup(f"wasm.{integer_name}.lt_u"),
        integer_greater_than_equal_nonnegative=descriptor_lookup(
            f"wasm.{integer_name}.ge_u"
        ),
        integer_greater_than_nonnegative=descriptor_lookup(f"wasm.{integer_name}.gt_u"),
        integer_select=descriptor_lookup(f"wasm.{integer_name}.select"),
    )


def _f32_narrowing_descriptors(
    descriptor_lookup: Callable[[str], Descriptor],
) -> FloatNarrowingDescriptors:
    return FloatNarrowingDescriptors(
        integer=_integer_narrowing_descriptors(descriptor_lookup, 32),
        float_constant=descriptor_lookup("wasm.f32.const"),
        float_add=descriptor_lookup("wasm.f32.add"),
        reinterpret_float_as_integer=descriptor_lookup("wasm.i32.reinterpret_f32"),
        reinterpret_integer_as_float=descriptor_lookup("wasm.f32.reinterpret_i32"),
    )


def _f32_to_bf16_rule(
    descriptors: FloatNarrowingDescriptors,
    type_guard: Callable[[str, TypePattern], Guard],
) -> DescriptorRule:
    emits = build_f32_to_bf16_emits(
        descriptors,
        ValueRef.operand("input"),
        ValueRef.result("result"),
    )
    return DescriptorRule(
        source_op=scalar_conversion.scalar_fptrunc,
        descriptor=emits[-1].descriptor,
        guards=(
            type_guard("input", _F32),
            type_guard("result", _BF16),
        ),
        emit=emits,
        report_key="exact_binary32_to_bfloat16",
    )


def _representable_f32_to_bf16_rule(
    descriptors: FloatNarrowingDescriptors,
    type_guard: Callable[[str, TypePattern], Guard],
    *,
    preserve_nan: bool,
) -> DescriptorRule:
    emits = build_representable_f32_to_bf16_emits(
        descriptors,
        ValueRef.operand("input"),
        ValueRef.result("result"),
        preserve_nan=preserve_nan,
    )
    return DescriptorRule(
        source_op=scalar_conversion.scalar_fptrunc,
        descriptor=emits[-1].descriptor,
        guards=(
            type_guard("input", _F32),
            type_guard("result", _BF16),
            *((Guard.value_not_nan("input"),) if not preserve_nan else ()),
            Guard.value_float_exactly_representable_as("input", "result"),
        ),
        emit=emits,
        report_key=(
            "exactly_representable_binary32_to_bfloat16"
            + ("_preserve_nan" if preserve_nan else "")
        ),
    )


def _f32_to_narrow_float_rule(
    descriptors: FloatNarrowingDescriptors,
    type_guard: Callable[[str, TypePattern], Guard],
    result_type: TypePattern,
    narrow_format: NarrowFloatFormat,
    report_key: str,
) -> DescriptorRule:
    emits = build_float_to_narrow_float_emits(
        descriptors,
        F32_FORMAT,
        narrow_format,
        ValueRef.operand("input"),
        ValueRef.result("result"),
        subnormal_rounding=NarrowFloatSubnormalRounding.RNE_FLOAT_ADD,
    )
    return DescriptorRule(
        source_op=scalar_conversion.scalar_fptrunc,
        descriptor=emits[-1].descriptor,
        guards=(
            type_guard("input", _F32),
            type_guard("result", result_type),
        ),
        emit=emits,
        report_key=report_key,
    )


def _f64_to_narrow_float_rule(
    descriptor_lookup: Callable[[str], Descriptor],
    descriptors: IntegerNarrowingDescriptors,
    type_guard: Callable[[str, TypePattern], Guard],
    result_type: TypePattern,
    narrow_format: NarrowFloatFormat,
    result_name: str,
) -> DescriptorRule:
    reinterpret_descriptor = descriptor_lookup("wasm.i64.reinterpret_f64")
    emits = [
        EmitDescriptorOp(
            descriptor=reinterpret_descriptor,
            operands={"input": ValueRef.operand("input")},
            results={"dst": ValueRef.temporary("input_bits")},
            result_types={"dst": DescriptorResultType()},
        )
    ]
    emits.extend(
        build_integer_bits_to_narrow_float_emits(
            descriptors,
            F64_FORMAT,
            narrow_format,
            ValueRef.temporary("input_bits"),
            ValueRef.temporary("carrier"),
            subnormal_rounding=NarrowFloatSubnormalRounding.INTEGER,
        )
    )
    wrap_descriptor = descriptor_lookup("wasm.i32.wrap_i64")
    emits.append(
        EmitDescriptorOp(
            descriptor=wrap_descriptor,
            operands={"input": ValueRef.temporary("carrier")},
            results={"dst": ValueRef.result("result")},
        )
    )
    return DescriptorRule(
        source_op=scalar_conversion.scalar_fptrunc,
        descriptor=wrap_descriptor,
        guards=(
            type_guard("input", _F64),
            type_guard("result", result_type),
        ),
        emit=tuple(emits),
        report_key=f"exact_binary64_to_{result_name}",
    )


def _integer_carrier_to_float8_rule(
    descriptors: IntegerNarrowingDescriptors,
    type_guard: Callable[[str, TypePattern], Guard],
    source_type: TypePattern,
    source_format: BinaryFloatFormat,
    source_name: str,
    result_type: TypePattern,
    narrow_format: NarrowFloatFormat,
    result_name: str,
) -> DescriptorRule:
    if (
        source_format.exponent_bias
        >= narrow_format.exponent_bias + narrow_format.mantissa_bits + 1
    ):
        subnormal_rounding = NarrowFloatSubnormalRounding.INTEGER
    else:
        subnormal_rounding = NarrowFloatSubnormalRounding.INTEGER_SOURCE_SUBNORMALS
    emits = build_integer_bits_to_narrow_float_emits(
        descriptors,
        source_format,
        narrow_format,
        ValueRef.operand("input"),
        ValueRef.result("result"),
        subnormal_rounding=subnormal_rounding,
    )
    return DescriptorRule(
        source_op=scalar_conversion.scalar_fptrunc,
        descriptor=emits[-1].descriptor,
        guards=(
            type_guard("input", source_type),
            type_guard("result", result_type),
        ),
        emit=emits,
        report_key=f"exact_{source_name}_to_{result_name}",
    )


def float_narrowing_rules(
    descriptor_lookup: Callable[[str], Descriptor],
    type_guard: Callable[[str, TypePattern], Guard],
) -> tuple[DescriptorRule, ...]:
    """Returns the complete exact scalar float-narrowing matrix for Wasm."""

    f32_descriptors = _f32_narrowing_descriptors(descriptor_lookup)
    i32_descriptors = f32_descriptors.integer
    i64_descriptors = _integer_narrowing_descriptors(descriptor_lookup, 64)
    rules = [
        _representable_f32_to_bf16_rule(
            f32_descriptors, type_guard, preserve_nan=False
        ),
        _representable_f32_to_bf16_rule(f32_descriptors, type_guard, preserve_nan=True),
        _f32_to_bf16_rule(f32_descriptors, type_guard),
        _f32_to_narrow_float_rule(
            f32_descriptors,
            type_guard,
            _F16,
            F16_FORMAT,
            "exact_binary32_to_f16",
        ),
        _f32_to_narrow_float_rule(
            f32_descriptors,
            type_guard,
            _F8E4M3,
            F8E4M3_FORMAT,
            "exact_binary32_to_f8e4m3",
        ),
        _f32_to_narrow_float_rule(
            f32_descriptors,
            type_guard,
            _F8E5M2,
            F8E5M2_FORMAT,
            "exact_binary32_to_f8e5m2",
        ),
    ]
    rules.extend(
        _f64_to_narrow_float_rule(
            descriptor_lookup,
            i64_descriptors,
            type_guard,
            result_type,
            narrow_format,
            result_name,
        )
        for result_type, narrow_format, result_name in (
            (_F16, F16_FORMAT, "f16"),
            (_BF16, BF16_FORMAT, "bfloat16"),
            (_F8E4M3, F8E4M3_FORMAT, "f8e4m3"),
            (_F8E5M2, F8E5M2_FORMAT, "f8e5m2"),
        )
    )
    rules.extend(
        _integer_carrier_to_float8_rule(
            i32_descriptors,
            type_guard,
            source_type,
            source_format,
            source_name,
            result_type,
            narrow_format,
            result_name,
        )
        for source_type, source_format, source_name in (
            (_F16, F16_SOURCE_FORMAT, "binary16"),
            (_BF16, BF16_SOURCE_FORMAT, "bfloat16"),
        )
        for result_type, narrow_format, result_name in (
            (_F8E4M3, F8E4M3_FORMAT, "f8e4m3"),
            (_F8E5M2, F8E5M2_FORMAT, "f8e5m2"),
        )
    )
    return tuple(rules)
