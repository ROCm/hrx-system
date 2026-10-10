# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact packed narrow-float widening for Wasm SIMD128."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from loom.dialect.vector import defs as vector
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    TypePattern,
    ValueRef,
    Vector,
)
from loom.target.low_descriptors import Descriptor

_I32 = Scalar("i32")
_F32_PACKET = Vector("f32", minimum_lanes=1, maximum_lanes=4)


@dataclass(frozen=True, slots=True)
class _NarrowFloatFormat:
    element: str
    source_bit_count: int
    mantissa_bit_count: int
    exponent_bit_count: int
    exponent_bias: int
    has_infinity: bool


_F8E4M3 = _NarrowFloatFormat("f8E4M3", 8, 3, 4, 7, False)
_F8E5M2 = _NarrowFloatFormat("f8E5M2", 8, 2, 5, 15, True)
_F16 = _NarrowFloatFormat("f16", 16, 10, 5, 15, True)


def _repeated_i32_word(value: int) -> int:
    lane_bits = value & 0xFFFFFFFF
    return lane_bits | (lane_bits << 32)


class _PackedI32Program:
    """Builds one straight-line i32x4 descriptor program."""

    def __init__(self, descriptor_lookup: Callable[[str], Descriptor]) -> None:
        self._descriptor_lookup = descriptor_lookup
        self.emits: list[EmitDescriptorOp] = []
        self._vector_constants: dict[int, ValueRef] = {}
        self._scalar_constants: dict[int, ValueRef] = {}

    def _result(self, result_name: str | None) -> ValueRef:
        return (
            ValueRef.result("result")
            if result_name is None
            else ValueRef.temporary(result_name)
        )

    def constant(self, result_name: str, value: int) -> ValueRef:
        if existing := self._vector_constants.get(value):
            return existing
        result = self._result(result_name)
        word = _repeated_i32_word(value)
        self.emits.append(
            EmitDescriptorOp(
                descriptor=self._descriptor_lookup("wasm.v128.const"),
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
                immediates={"lo64": word, "hi64": word},
                form=DescriptorEmitForm.CONST,
            )
        )
        self._vector_constants[value] = result
        return result

    def scalar_constant(self, result_name: str, value: int) -> ValueRef:
        if existing := self._scalar_constants.get(value):
            return existing
        result = self._result(result_name)
        self.emits.append(
            EmitDescriptorOp(
                descriptor=self._descriptor_lookup("wasm.i32.const"),
                results={"dst": result},
                result_types={"dst": _I32},
                immediates={"i32_value": value},
                form=DescriptorEmitForm.CONST,
            )
        )
        self._scalar_constants[value] = result
        return result

    def operation(
        self,
        result_name: str | None,
        descriptor_key: str,
        operands: Mapping[str, ValueRef],
    ) -> ValueRef:
        result = self._result(result_name)
        self.emits.append(
            EmitDescriptorOp(
                descriptor=self._descriptor_lookup(descriptor_key),
                operands=operands,
                results={"dst": result},
                result_types=(
                    None if result_name is None else {"dst": DescriptorResultType()}
                ),
                form=DescriptorEmitForm.OP,
            )
        )
        return result

    def binary(
        self,
        result_name: str | None,
        descriptor_key: str,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> ValueRef:
        return self.operation(
            result_name,
            descriptor_key,
            {"lhs": lhs, "rhs": rhs},
        )

    def shift(
        self,
        result_name: str | None,
        descriptor_key: str,
        value: ValueRef,
        count: ValueRef,
    ) -> ValueRef:
        return self.operation(
            result_name,
            descriptor_key,
            {"value": value, "count": count},
        )

    def select(
        self,
        result_name: str | None,
        true_value: ValueRef,
        false_value: ValueRef,
        condition: ValueRef,
    ) -> ValueRef:
        return self.operation(
            result_name,
            "wasm.v128.bitselect",
            {
                "true_value": true_value,
                "false_value": false_value,
                "condition": condition,
            },
        )


def _extend_to_i32_lanes(
    program: _PackedI32Program,
    source_bit_count: int,
) -> ValueRef:
    current = ValueRef.operand("input")
    bit_count = source_bit_count
    while bit_count < 32:
        next_bit_count = bit_count * 2
        current = program.operation(
            f"input_i{next_bit_count}",
            (
                f"wasm.i{next_bit_count}x{128 // next_bit_count}.extend_low_"
                f"i{bit_count}x{128 // bit_count}_u"
            ),
            {"input": current},
        )
        bit_count = next_bit_count
    return current


def _normalize_mantissa(
    program: _PackedI32Program,
    mantissa: ValueRef,
    mantissa_bit_count: int,
    prefix: str,
) -> tuple[ValueRef, ValueRef]:
    """Normalizes nonzero mantissas and returns shift counts and payloads."""

    shift_count = program.constant(f"{prefix}_initial_shift", 1)
    current = mantissa
    # Move each leading one to the source format's implicit-bit position. A
    # descending binary search needs log2(mantissa width) packed stages, then
    # one unconditional shift crosses the implicit-bit boundary.
    amount = 1 << ((mantissa_bit_count - 1).bit_length() - 1)
    stage = 0
    scalar_one = None
    while amount:
        threshold = 1 << (mantissa_bit_count - amount)
        threshold_value = program.constant(f"{prefix}_threshold{stage}", threshold)
        needs_shift = program.binary(
            f"{prefix}_needs_shift{stage}",
            "wasm.i32x4.lt_u",
            current,
            threshold_value,
        )
        scalar_amount = program.scalar_constant(f"{prefix}_scalar_shift{stage}", amount)
        if amount == 1:
            scalar_one = scalar_amount
        candidate = program.shift(
            f"{prefix}_candidate{stage}",
            "wasm.i32x4.shl",
            current,
            scalar_amount,
        )
        current = program.select(
            f"{prefix}_shifted{stage}", candidate, current, needs_shift
        )
        amount_value = program.constant(f"{prefix}_shift{stage}", amount)
        increment = program.binary(
            f"{prefix}_increment{stage}",
            "wasm.v128.and",
            needs_shift,
            amount_value,
        )
        shift_count = program.binary(
            f"{prefix}_shift_count{stage}",
            "wasm.i32x4.add",
            shift_count,
            increment,
        )
        amount >>= 1
        stage += 1

    assert scalar_one is not None
    normalized = program.shift(
        f"{prefix}_normalized",
        "wasm.i32x4.shl",
        current,
        scalar_one,
    )
    return shift_count, normalized


def _emit_float_widening(
    program: _PackedI32Program,
    input_bits: ValueRef,
    source_format: _NarrowFloatFormat,
) -> None:
    mantissa_bit_count = source_format.mantissa_bit_count
    exponent_max = (1 << source_format.exponent_bit_count) - 1
    mantissa_mask_value = (1 << mantissa_bit_count) - 1

    sign_mask = program.constant("sign_mask", 1 << (source_format.source_bit_count - 1))
    sign_shift = program.scalar_constant(
        "sign_shift", 32 - source_format.source_bit_count
    )
    sign = program.binary("sign", "wasm.v128.and", input_bits, sign_mask)
    sign = program.shift("positioned_sign", "wasm.i32x4.shl", sign, sign_shift)

    mantissa_mask = program.constant("mantissa_mask", mantissa_mask_value)
    mantissa = program.binary("mantissa", "wasm.v128.and", input_bits, mantissa_mask)
    exponent_shift = program.scalar_constant("exponent_shift", mantissa_bit_count)
    exponent = program.shift(
        "encoded_exponent", "wasm.i32x4.shr_u", input_bits, exponent_shift
    )
    exponent_mask = program.constant("exponent_mask", exponent_max)
    exponent = program.binary(
        "masked_exponent", "wasm.v128.and", exponent, exponent_mask
    )

    normal_bias = program.constant("normal_bias", 127 - source_format.exponent_bias)
    normal_exponent = program.binary(
        "normal_exponent", "wasm.i32x4.add", exponent, normal_bias
    )
    f32_exponent_shift = program.scalar_constant("f32_exponent_shift", 23)
    normal_exponent = program.shift(
        "positioned_normal_exponent",
        "wasm.i32x4.shl",
        normal_exponent,
        f32_exponent_shift,
    )
    f32_mantissa_shift = program.scalar_constant(
        "f32_mantissa_shift", 23 - mantissa_bit_count
    )
    normal_mantissa = program.shift(
        "positioned_normal_mantissa",
        "wasm.i32x4.shl",
        mantissa,
        f32_mantissa_shift,
    )
    normal = program.binary(
        "normal_unsigned", "wasm.v128.or", normal_exponent, normal_mantissa
    )
    normal = program.binary("normal", "wasm.v128.or", sign, normal)

    normalization_shift, normalized_mantissa = _normalize_mantissa(
        program, mantissa, mantissa_bit_count, "subnormal"
    )
    normalized_mantissa = program.binary(
        "subnormal_payload",
        "wasm.v128.and",
        normalized_mantissa,
        mantissa_mask,
    )
    normalized_mantissa = program.shift(
        "positioned_subnormal_payload",
        "wasm.i32x4.shl",
        normalized_mantissa,
        f32_mantissa_shift,
    )
    subnormal_bias = program.constant(
        "subnormal_bias", 128 - source_format.exponent_bias
    )
    subnormal_exponent = program.binary(
        "subnormal_exponent",
        "wasm.i32x4.sub",
        subnormal_bias,
        normalization_shift,
    )
    subnormal_exponent = program.shift(
        "positioned_subnormal_exponent",
        "wasm.i32x4.shl",
        subnormal_exponent,
        f32_exponent_shift,
    )
    subnormal = program.binary(
        "subnormal_unsigned",
        "wasm.v128.or",
        subnormal_exponent,
        normalized_mantissa,
    )
    subnormal = program.binary("subnormal", "wasm.v128.or", sign, subnormal)

    zero = program.constant("zero", 0)
    mantissa_is_zero = program.binary(
        "mantissa_is_zero", "wasm.i32x4.eq", mantissa, zero
    )
    zero_or_subnormal = program.select(
        "zero_or_subnormal", sign, subnormal, mantissa_is_zero
    )
    exponent_is_zero = program.binary(
        "exponent_is_zero", "wasm.i32x4.eq", exponent, zero
    )
    finite = program.select("finite", zero_or_subnormal, normal, exponent_is_zero)

    if source_format.has_infinity:
        special_exponent = program.constant("special_exponent", exponent_max)
        exponent_is_special = program.binary(
            "exponent_is_special",
            "wasm.i32x4.eq",
            exponent,
            special_exponent,
        )
        infinity_bits = program.constant("infinity_bits", 0x7F800000)
        infinity = program.binary("infinity", "wasm.v128.or", sign, infinity_bits)
        if source_format.source_bit_count == 16:
            special_payload = program.shift(
                "special_payload",
                "wasm.i32x4.shl",
                mantissa,
                f32_mantissa_shift,
            )
            special = program.binary(
                "special", "wasm.v128.or", infinity, special_payload
            )
        else:
            canonical_nan = program.constant("canonical_nan", 0x7FC00000)
            special = program.select(
                "special", infinity, canonical_nan, mantissa_is_zero
            )
        program.select(None, special, finite, exponent_is_special)
    else:
        payload_mask = program.constant("payload_mask", 0x7F)
        payload = program.binary("payload", "wasm.v128.and", input_bits, payload_mask)
        nan_payload = program.constant("nan_payload", 0x7F)
        is_nan = program.binary("is_nan", "wasm.i32x4.eq", payload, nan_payload)
        canonical_nan = program.constant("canonical_nan", 0x7FC00000)
        program.select(None, canonical_nan, finite, is_nan)


def _float_widening_rule(
    descriptor_lookup: Callable[[str], Descriptor],
    type_guard: Callable[[str, TypePattern], Guard],
    source_format: _NarrowFloatFormat,
) -> DescriptorRule:
    program = _PackedI32Program(descriptor_lookup)
    input_bits = _extend_to_i32_lanes(program, source_format.source_bit_count)
    _emit_float_widening(program, input_bits, source_format)
    source_type = Vector(
        source_format.element,
        minimum_lanes=1,
        maximum_lanes=4,
    )
    return DescriptorRule(
        source_op=vector.vector_extf,
        descriptor=program.emits[-1].descriptor,
        guards=(
            type_guard("input", source_type),
            type_guard("result", _F32_PACKET),
            Guard.value_static_element_count_eq("input", "result"),
        ),
        emit=tuple(program.emits),
        report_key=(
            f"wasm.float_extension.exact_{source_format.element.lower()}_to_f32"
        ),
    )


def _bf16_widening_rule(
    descriptor_lookup: Callable[[str], Descriptor],
    type_guard: Callable[[str, TypePattern], Guard],
) -> DescriptorRule:
    program = _PackedI32Program(descriptor_lookup)
    input_bits = _extend_to_i32_lanes(program, 16)
    shift = program.scalar_constant("shift", 16)
    program.shift(None, "wasm.i32x4.shl", input_bits, shift)
    return DescriptorRule(
        source_op=vector.vector_extf,
        descriptor=program.emits[-1].descriptor,
        guards=(
            type_guard("input", Vector("bf16", minimum_lanes=1, maximum_lanes=4)),
            type_guard("result", _F32_PACKET),
            Guard.value_static_element_count_eq("input", "result"),
        ),
        emit=tuple(program.emits),
        report_key="wasm.float_extension.exact_bf16_to_f32",
    )


def float_extension_rules(
    descriptor_lookup: Callable[[str], Descriptor],
    type_guard: Callable[[str, TypePattern], Guard],
) -> tuple[DescriptorRule, ...]:
    """Returns exact physical-packet narrow-float widening rules."""

    return (
        *(
            _float_widening_rule(descriptor_lookup, type_guard, source_format)
            for source_format in (_F8E4M3, _F8E5M2, _F16)
        ),
        _bf16_widening_rule(descriptor_lookup, type_guard),
    )
