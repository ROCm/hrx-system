# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact scalar conversion contracts for AMD XDNA AIE2P."""

from __future__ import annotations

from collections.abc import Mapping

from loom.dialect.scalar import conversion as scalar_conversion
from loom.dsl import Op
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    ValueRef,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_I8 = Scalar("i8")
_I16 = Scalar("i16")
_I32 = Scalar("i32")
_F16 = Scalar("f16")
_BF16 = Scalar("bf16")
_F32 = Scalar("f32")
_SHORT_MIN = -1024
_SHORT_MAX = 1023


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


class _ConversionProgram:
    """Builds one compact descriptor program for a scalar conversion."""

    def __init__(self) -> None:
        self.emits: list[EmitDescriptorOp] = []

    def constant(
        self,
        result_name: str,
        value: int,
        *,
        shift_register: bool = False,
    ) -> ValueRef:
        if shift_register:
            descriptor_key = "amd.xdna.aie2p.constant.i32.fx2flt-scale"
        elif _SHORT_MIN <= value <= _SHORT_MAX:
            descriptor_key = "amd.xdna.aie2p.constant.i32.short"
        else:
            descriptor_key = "amd.xdna.aie2p.constant.i32"
        result = ValueRef.temporary(result_name)
        self.emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor(descriptor_key),
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
                immediates={"i": value},
                form=DescriptorEmitForm.CONST,
            )
        )
        return result

    def operation(
        self,
        result_name: str | None,
        operation: str,
        operands: Mapping[str, ValueRef],
        *,
        copy_operands: tuple[str, ...] = (),
    ) -> ValueRef:
        result = (
            ValueRef.result("result")
            if result_name is None
            else ValueRef.temporary(result_name)
        )
        self.emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor(f"amd.xdna.aie2p.{operation}"),
                operands=operands,
                results={"d0": result},
                result_types=(
                    None if result_name is None else {"d0": DescriptorResultType()}
                ),
                form=DescriptorEmitForm.OP,
                copy_operands=copy_operands,
            )
        )
        return result

    def unary(self, result_name: str, operation: str, operand: ValueRef) -> ValueRef:
        return self.operation(result_name, operation, {"s0": operand})

    def binary(
        self,
        result_name: str,
        operation: str,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> ValueRef:
        return self.operation(result_name, operation, {"s0": lhs, "s1": rhs})

    def select(
        self,
        result_name: str | None,
        true_value: ValueRef,
        false_value: ValueRef,
        condition: ValueRef,
    ) -> ValueRef:
        return self.operation(
            result_name,
            "select.nonzero.i32",
            {"s0": true_value, "s1": false_value, "s2": condition},
            copy_operands=("s2",),
        )


def _integer_to_f32_rule(
    source_op: Op,
    input_type: Scalar,
    extend_descriptor_key: str | None,
    report_key: str,
) -> DescriptorRule:
    program = _ConversionProgram()
    input_value = ValueRef.operand("input")
    if extend_descriptor_key is not None:
        input_value = program.unary(
            "extended",
            extend_descriptor_key.removeprefix("amd.xdna.aie2p."),
            input_value,
        )
    scale = program.constant("scale", 0, shift_register=True)
    convert = _descriptor("amd.xdna.aie2p.convert.signed.i32.to.f32")
    program.emits.append(
        EmitDescriptorOp(
            descriptor=convert,
            operands={"s0": input_value, "m": scale},
            results={"d0": ValueRef.result("result")},
            form=DescriptorEmitForm.OP,
        )
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=convert,
        guards=(
            Guard.value_type("input", input_type),
            Guard.value_type("result", _F32),
        ),
        emit=tuple(program.emits),
        report_key=report_key,
    )


def _unsigned_i32_to_f32_rule() -> DescriptorRule:
    program = _ConversionProgram()
    input_value = ValueRef.operand("input")
    zero = program.constant("zero", 0)
    one = program.constant("one", 1)
    negative_one = program.constant("negative_one", -1)
    exponent_increment = program.constant("exponent_increment", 1 << 23)

    high_bit = program.binary("high_bit", "cmp.slt.i32", input_value, zero)
    half = program.binary("half", "lshl.i32", input_value, negative_one)
    sticky = program.binary("sticky", "and.i32", input_value, one)
    rounded_half = program.binary("rounded_half", "or.i32", half, sticky)
    signed_source = program.select("signed_source", rounded_half, input_value, high_bit)

    scale = program.constant("scale", 0, shift_register=True)
    converted = program.operation(
        "converted",
        "convert.signed.i32.to.f32",
        {"s0": signed_source, "m": scale},
    )
    adjustment = program.select("adjustment", exponent_increment, zero, high_bit)
    program.operation(
        None,
        "add.i32",
        {"s0": converted, "s1": adjustment},
    )

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
    program = _ConversionProgram()
    shift = program.constant("shift", 16)
    program.operation(
        None,
        "lshl.i32",
        {"s0": ValueRef.operand("input"), "s1": shift},
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


def _f16_to_f32_rule() -> DescriptorRule:
    program = _ConversionProgram()
    input_value = ValueRef.operand("input")

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
    program.select(None, zero_exponent, nonzero_exponent, exponent_is_zero)

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


AIE2P_CONVERSION_RULES = (
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
    _bf16_to_f32_rule(),
    _f16_to_f32_rule(),
)
