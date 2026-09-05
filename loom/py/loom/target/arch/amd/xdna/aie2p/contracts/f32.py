# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact binary32 source-to-Low contracts for AMD XDNA AIE2P."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from loom.dialect.scalar import arithmetic as scalar_arithmetic
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

_F32 = Scalar("f32")
_SHORT_MIN = -1024
_SHORT_MAX = 1023


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


class _F32MultiplyProgram:
    """Builds one compact descriptor program for an exact binary32 product."""

    def __init__(self) -> None:
        self.emits: list[EmitDescriptorOp] = []

    def constant(self, result_name: str, value: int) -> ValueRef:
        descriptor_key = (
            "amd.xdna.aie2p.constant.i32.short"
            if _SHORT_MIN <= value <= _SHORT_MAX
            else "amd.xdna.aie2p.constant.i32"
        )
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

    def unary(self, result_name: str, operation: str, operand: ValueRef) -> ValueRef:
        return self.operation(
            result_name,
            operation,
            operands={"s0": operand},
        )

    def binary(
        self,
        result_name: str,
        operation: str,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> ValueRef:
        return self.operation(
            result_name,
            operation,
            operands={"s0": lhs, "s1": rhs},
        )

    def add_immediate(
        self, result_name: str, operand: ValueRef, immediate: int
    ) -> ValueRef:
        return self.operation(
            result_name,
            "add.i32.immediate",
            operands={"s0": operand},
            immediates={"imm": immediate},
        )

    def select(
        self,
        result_name: str | None,
        true_value: ValueRef,
        false_value: ValueRef,
        condition: ValueRef,
        *,
        when_zero: bool = False,
    ) -> ValueRef:
        return self.operation(
            result_name,
            "select.zero.i32" if when_zero else "select.nonzero.i32",
            operands={"s0": true_value, "s1": false_value, "s2": condition},
            copy_operands=("s2",),
        )

    def operation(
        self,
        result_name: str | None,
        operation: str,
        *,
        operands: Mapping[str, ValueRef],
        immediates: Mapping[str, int] | None = None,
        copy_operands: Sequence[str] = (),
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
                immediates={} if immediates is None else immediates,
                form=DescriptorEmitForm.OP,
                copy_operands=copy_operands,
            )
        )
        return result


def _f32_multiply_emits() -> tuple[EmitDescriptorOp, ...]:
    program = _F32MultiplyProgram()

    # Keep constants local to the algorithm phase that consumes them. AIE2P
    # scalar registers are unspillable, and materializing a cheap constant
    # again is preferable to keeping it live across the whole product.
    absolute_mask = program.constant("absolute_mask", 0x7FFFFFFF)
    classify_sign_bit = program.constant("classify_sign_bit", -(2**31))
    normalize_zero = program.constant("normalize_zero", 0)
    normalize_shift_left_23 = program.constant("normalize_shift_left_23", 23)
    normalize_shift_right_23 = program.constant("normalize_shift_right_23", -23)
    normalize_implicit_bit = program.constant("normalize_implicit_bit", 0x00800000)

    lhs = ValueRef.operand("lhs")
    rhs = ValueRef.operand("rhs")
    lhs_absolute = program.binary("lhs_absolute", "and.i32", lhs, absolute_mask)
    rhs_absolute = program.binary("rhs_absolute", "and.i32", rhs, absolute_mask)
    sign_xor = program.binary("sign_xor", "xor.i32", lhs, rhs)
    product_sign = program.binary(
        "product_sign", "and.i32", sign_xor, classify_sign_bit
    )

    def normalize_operand(prefix: str, absolute: ValueRef) -> tuple[ValueRef, ValueRef]:
        exponent = program.binary(
            f"{prefix}_exponent",
            "lshl.i32",
            absolute,
            normalize_shift_right_23,
        )
        exponent_bits = program.binary(
            f"{prefix}_exponent_bits",
            "lshl.i32",
            exponent,
            normalize_shift_left_23,
        )
        significand = program.binary(
            f"{prefix}_significand", "sub.i32", absolute, exponent_bits
        )
        leading_zeros = program.unary(f"{prefix}_leading_zeros", "clz.i32", significand)
        normalization_shift = program.add_immediate(
            f"{prefix}_normalization_shift", leading_zeros, -8
        )
        selected_shift = program.select(
            f"{prefix}_selected_shift",
            normalization_shift,
            normalize_zero,
            exponent,
            when_zero=True,
        )
        shifted_significand = program.binary(
            f"{prefix}_shifted_significand",
            "lshl.i32",
            significand,
            selected_shift,
        )
        normalized_significand = program.binary(
            f"{prefix}_normalized_significand",
            "or.i32",
            shifted_significand,
            normalize_implicit_bit,
        )
        negative_shift = program.binary(
            f"{prefix}_negative_shift",
            "sub.i32",
            normalize_zero,
            normalization_shift,
        )
        subnormal_exponent = program.add_immediate(
            f"{prefix}_subnormal_exponent", negative_shift, 1
        )
        effective_exponent = program.select(
            f"{prefix}_effective_exponent",
            subnormal_exponent,
            exponent,
            exponent,
            when_zero=True,
        )
        return normalized_significand, effective_exponent

    lhs_significand, lhs_exponent = normalize_operand("lhs", lhs_absolute)
    rhs_significand, rhs_exponent = normalize_operand("rhs", rhs_absolute)

    # Four 16x16 products recover the exact 48-bit significand product. The
    # high/low pair is then aligned like compiler-rt's wide binary32 product.
    mask16 = program.constant("product_mask16", 0xFFFF)
    product_shift_left_8 = program.constant("product_shift_left_8", 8)
    product_shift_left_16 = program.constant("product_shift_left_16", 16)
    product_shift_right_16 = program.constant("product_shift_right_16", -16)
    product_shift_right_24 = program.constant("product_shift_right_24", -24)
    lhs_low = program.binary("lhs_low", "and.i32", lhs_significand, mask16)
    lhs_high = program.binary(
        "lhs_high", "lshl.i32", lhs_significand, product_shift_right_16
    )
    rhs_low = program.binary("rhs_low", "and.i32", rhs_significand, mask16)
    rhs_high = program.binary(
        "rhs_high", "lshl.i32", rhs_significand, product_shift_right_16
    )
    low_low = program.binary("low_low", "mul.i32", lhs_low, rhs_low)
    low_high = program.binary("low_high", "mul.i32", lhs_low, rhs_high)
    high_low = program.binary("high_low", "mul.i32", lhs_high, rhs_low)
    cross = program.binary("cross", "add.i32", low_high, high_low)
    shifted_cross = program.binary(
        "shifted_cross", "lshl.i32", cross, product_shift_left_16
    )
    product_low_unaligned = program.binary(
        "product_low_unaligned", "add.i32", low_low, shifted_cross
    )
    product_carry = program.binary(
        "product_carry", "cmp.ult.i32", product_low_unaligned, low_low
    )
    high_high = program.binary("high_high", "mul.i32", lhs_high, rhs_high)
    cross_high = program.binary("cross_high", "lshl.i32", cross, product_shift_right_16)
    product_high_base = program.binary(
        "product_high_base", "add.i32", high_high, cross_high
    )
    product_high_unaligned = program.binary(
        "product_high_unaligned", "add.i32", product_high_base, product_carry
    )
    aligned_high_base = program.binary(
        "aligned_high_base",
        "lshl.i32",
        product_high_unaligned,
        product_shift_left_8,
    )
    aligned_low_high = program.binary(
        "aligned_low_high",
        "lshl.i32",
        product_low_unaligned,
        product_shift_right_24,
    )
    product_high = program.binary(
        "product_high", "or.i32", aligned_high_base, aligned_low_high
    )
    product_low = program.binary(
        "product_low", "lshl.i32", product_low_unaligned, product_shift_left_8
    )

    negative_exponent_bias = program.constant("negative_exponent_bias", -127)
    product_implicit_bit = program.constant("product_implicit_bit", 0x00800000)
    product_one = program.constant("product_one", 1)
    product_shift_right_31 = program.constant("product_shift_right_31", -31)
    exponent_sum = program.binary("exponent_sum", "add.i32", lhs_exponent, rhs_exponent)
    product_exponent = program.binary(
        "product_exponent", "add.i32", exponent_sum, negative_exponent_bias
    )
    wide_product = program.binary(
        "wide_product", "and.i32", product_high, product_implicit_bit
    )
    exponent_increment = program.unary(
        "exponent_increment", "cmp.nez.i32", wide_product
    )
    normalized_exponent = program.binary(
        "normalized_exponent", "add.i32", product_exponent, exponent_increment
    )
    shifted_high_base = program.binary(
        "shifted_high_base", "lshl.i32", product_high, product_one
    )
    shifted_low_high = program.binary(
        "shifted_low_high", "lshl.i32", product_low, product_shift_right_31
    )
    shifted_high = program.binary(
        "shifted_high", "or.i32", shifted_high_base, shifted_low_high
    )
    shifted_low = program.binary("shifted_low", "lshl.i32", product_low, product_one)
    normalized_high = program.select(
        "normalized_high", product_high, shifted_high, wide_product
    )
    normalized_low = program.select(
        "normalized_low", product_low, shifted_low, wide_product
    )

    # Shift the 64-bit aligned product into the subnormal range with one of
    # three branchless cases. Masked counts keep every issued scalar shift in
    # the hardware-defined [-31, 31] range, including unselected cases.
    underflow_zero = program.constant("underflow_zero", 0)
    underflow_mask31 = program.constant("underflow_mask31", 31)
    negative_exponent = program.binary(
        "negative_exponent", "sub.i32", underflow_zero, normalized_exponent
    )
    underflow_shift = program.add_immediate("underflow_shift", negative_exponent, 1)
    shift_mod = program.binary(
        "shift_mod", "and.i32", underflow_shift, underflow_mask31
    )
    negative_shift_mod = program.binary(
        "negative_shift_mod", "sub.i32", underflow_zero, shift_mod
    )
    inverse_shift_mod = program.binary(
        "inverse_shift_mod", "and.i32", negative_shift_mod, underflow_mask31
    )
    low_shifted_right = program.binary(
        "low_shifted_right", "lshl.i32", normalized_low, negative_shift_mod
    )
    high_shifted_left = program.binary(
        "high_shifted_left", "lshl.i32", normalized_high, inverse_shift_mod
    )
    low_discarded = program.binary(
        "low_discarded", "lshl.i32", normalized_low, inverse_shift_mod
    )
    low_case_sticky = program.unary("low_case_sticky", "cmp.nez.i32", low_discarded)
    low_case_high = program.binary(
        "low_case_high", "lshl.i32", normalized_high, negative_shift_mod
    )
    low_case_base = program.binary(
        "low_case_base", "or.i32", low_shifted_right, high_shifted_left
    )
    low_case_low = program.binary(
        "low_case_low", "or.i32", low_case_base, low_case_sticky
    )
    high_discarded = program.binary(
        "high_discarded", "lshl.i32", normalized_high, inverse_shift_mod
    )
    selected_high_discarded = program.select(
        "selected_high_discarded", high_discarded, underflow_zero, shift_mod
    )
    high_case_discarded = program.binary(
        "high_case_discarded", "or.i32", normalized_low, selected_high_discarded
    )
    high_case_sticky = program.unary(
        "high_case_sticky", "cmp.nez.i32", high_case_discarded
    )
    high_case_low = program.binary(
        "high_case_low", "or.i32", low_case_high, high_case_sticky
    )
    all_case_discarded = program.binary(
        "all_case_discarded", "or.i32", normalized_high, normalized_low
    )
    all_case_low = program.unary("all_case_low", "cmp.nez.i32", all_case_discarded)
    shift_minus_32 = program.add_immediate("shift_minus_32", underflow_shift, -32)
    shift_lt32 = program.binary(
        "shift_lt32", "cmp.slt.i32", shift_minus_32, underflow_zero
    )
    shift_minus_64 = program.add_immediate("shift_minus_64", underflow_shift, -64)
    shift_lt64 = program.binary(
        "shift_lt64", "cmp.slt.i32", shift_minus_64, underflow_zero
    )
    high_or_all_low = program.select(
        "high_or_all_low", high_case_low, all_case_low, shift_lt64
    )
    underflow_low = program.select(
        "underflow_low", low_case_low, high_or_all_low, shift_lt32
    )
    underflow_high = program.select(
        "underflow_high", low_case_high, underflow_zero, shift_lt32
    )

    finite_implicit_bit = program.constant("finite_implicit_bit", 0x00800000)
    finite_shift_left_23 = program.constant("finite_shift_left_23", 23)
    finite_zero = program.constant("finite_zero", 0)
    significand_mask = program.add_immediate(
        "significand_mask", finite_implicit_bit, -1
    )
    normal_significand = program.binary(
        "normal_significand", "and.i32", normalized_high, significand_mask
    )
    normal_exponent = program.binary(
        "normal_exponent",
        "lshl.i32",
        normalized_exponent,
        finite_shift_left_23,
    )
    normal_high = program.binary(
        "normal_high", "or.i32", normal_significand, normal_exponent
    )
    is_underflow = program.binary(
        "is_underflow", "cmp.sge.i32", finite_zero, normalized_exponent
    )
    finite_high = program.select(
        "finite_high", underflow_high, normal_high, is_underflow
    )
    finite_low = program.select(
        "finite_low", underflow_low, normalized_low, is_underflow
    )

    round_half = program.constant("round_half", -(2**31))
    round_one = program.constant("round_one", 1)
    greater_than_half = program.binary(
        "greater_than_half", "cmp.ult.i32", round_half, finite_low
    )
    exactly_half = program.binary("exactly_half", "cmp.eq.i32", finite_low, round_half)
    result_lsb = program.binary("result_lsb", "and.i32", finite_high, round_one)
    round_tie = program.binary("round_tie", "and.i32", exactly_half, result_lsb)
    round_up = program.binary("round_up", "or.i32", greater_than_half, round_tie)
    signed_high = program.binary("signed_high", "or.i32", finite_high, product_sign)
    finite_result = program.binary("finite_result", "add.i32", signed_high, round_up)

    infinity = program.constant("infinity", 0x7F800000)
    negative_infinity_exponent = program.constant("negative_infinity_exponent", -255)
    special_zero = program.constant("special_zero", 0)
    quiet_bit = program.constant("quiet_bit", 0x00400000)
    signed_infinity = program.binary(
        "signed_infinity", "or.i32", infinity, product_sign
    )
    exponent_minus_255 = program.binary(
        "exponent_minus_255",
        "add.i32",
        normalized_exponent,
        negative_infinity_exponent,
    )
    is_overflow = program.binary(
        "is_overflow", "cmp.sge.i32", exponent_minus_255, special_zero
    )
    result = program.select(
        "overflow_selected", signed_infinity, finite_result, is_overflow
    )

    lhs_zero = program.unary("lhs_zero", "cmp.eqz.i32", lhs_absolute)
    rhs_zero = program.unary("rhs_zero", "cmp.eqz.i32", rhs_absolute)
    either_zero = program.binary("either_zero", "or.i32", lhs_zero, rhs_zero)
    result = program.select("zero_selected", product_sign, result, either_zero)

    canonical_nan = program.binary("canonical_nan", "or.i32", infinity, quiet_bit)
    lhs_infinity = program.binary("lhs_infinity", "cmp.eq.i32", lhs_absolute, infinity)
    rhs_infinity = program.binary("rhs_infinity", "cmp.eq.i32", rhs_absolute, infinity)
    lhs_infinity_result = program.select(
        "lhs_infinity_result", canonical_nan, signed_infinity, rhs_zero
    )
    rhs_infinity_result = program.select(
        "rhs_infinity_result", canonical_nan, signed_infinity, lhs_zero
    )
    result = program.select(
        "rhs_infinity_selected", rhs_infinity_result, result, rhs_infinity
    )
    result = program.select(
        "lhs_infinity_selected", lhs_infinity_result, result, lhs_infinity
    )

    lhs_nan = program.binary("lhs_nan", "cmp.ult.i32", infinity, lhs_absolute)
    rhs_nan = program.binary("rhs_nan", "cmp.ult.i32", infinity, rhs_absolute)
    quiet_lhs = program.binary("quiet_lhs", "or.i32", lhs, quiet_bit)
    quiet_rhs = program.binary("quiet_rhs", "or.i32", rhs, quiet_bit)
    result = program.select("rhs_nan_selected", quiet_rhs, result, rhs_nan)
    program.select(None, quiet_lhs, result, lhs_nan)

    return tuple(program.emits)


_F32_MULTIPLY_EMITS = _f32_multiply_emits()

AIE2P_F32_RULES = (
    DescriptorRule(
        source_op=scalar_arithmetic.scalar_mulf,
        descriptor=_descriptor("amd.xdna.aie2p.select.nonzero.i32"),
        guards=tuple(
            Guard.value_type(field, _F32) for field in ("lhs", "rhs", "result")
        ),
        emit=_F32_MULTIPLY_EMITS,
        report_key="exact_binary32",
    ),
)
