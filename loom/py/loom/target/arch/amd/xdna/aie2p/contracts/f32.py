# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact binary32 source-to-Low contracts for AMD XDNA AIE2P."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import math as scalar_math
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amd.xdna.aie2p.contracts.conversion import (
    emit_round_nearest_f32_to_i32,
    emit_signed_integer_to_f32,
    emit_truncate_f32_to_signed_i32,
)
from loom.target.arch.amd.xdna.aie2p.contracts.scalar_program import (
    ScalarProgram as _F32Program,
)
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
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_F32 = Scalar("f32")
_F32_VECTOR = Vector("f32", minimum_static_elements=1, maximum_static_elements=16)
_ExtremumKind = Literal["minimum", "maximum"]
_NaNPolicy = Literal["number", "ieee"]


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


def _f32_order_constants(
    program: _F32Program,
    prefix: str,
) -> tuple[ValueRef, ValueRef, ValueRef, ValueRef, ValueRef, ValueRef]:
    """Returns shared constants for binary32 ordering programs."""

    return (
        program.constant(f"{prefix}_absolute_mask", 0x7FFFFFFF),
        program.constant(f"{prefix}_sign_mask", -(2**31)),
        program.constant(f"{prefix}_all_bits", -1),
        program.constant(f"{prefix}_one", 1),
        program.constant(f"{prefix}_infinity", 0x7F800000),
        program.constant(f"{prefix}_quiet_bit", 0x00400000),
    )


def _emit_f32_order_state(
    program: _F32Program,
    lhs: ValueRef,
    rhs: ValueRef,
    constants: tuple[ValueRef, ValueRef, ValueRef, ValueRef, ValueRef, ValueRef],
    prefix: str,
) -> tuple[ValueRef, ValueRef, ValueRef, ValueRef]:
    """Returns lhs-NaN, rhs-NaN, both-zero, and raw lhs-less state."""

    absolute_mask, sign_mask, all_bits, _, infinity, _ = constants
    lhs_absolute = program.binary(
        f"{prefix}_lhs_absolute", "and.i32", lhs, absolute_mask
    )
    rhs_absolute = program.binary(
        f"{prefix}_rhs_absolute", "and.i32", rhs, absolute_mask
    )
    lhs_nan = program.binary(f"{prefix}_lhs_nan", "cmp.ult.i32", infinity, lhs_absolute)
    rhs_nan = program.binary(f"{prefix}_rhs_nan", "cmp.ult.i32", infinity, rhs_absolute)
    absolute_union = program.binary(
        f"{prefix}_absolute_union", "or.i32", lhs_absolute, rhs_absolute
    )
    both_zero = program.unary(f"{prefix}_both_zero", "cmp.eqz.i32", absolute_union)

    keys = []
    for operand_name, operand in (("lhs", lhs), ("rhs", rhs)):
        sign = program.binary(
            f"{prefix}_{operand_name}_sign", "and.i32", operand, sign_mask
        )
        negative_key = program.binary(
            f"{prefix}_{operand_name}_negative_key",
            "xor.i32",
            operand,
            all_bits,
        )
        positive_key = program.binary(
            f"{prefix}_{operand_name}_positive_key",
            "xor.i32",
            operand,
            sign_mask,
        )
        keys.append(
            program.select(
                f"{prefix}_{operand_name}_key",
                negative_key,
                positive_key,
                sign,
            )
        )
    lhs_key, rhs_key = keys
    lhs_less = program.binary(f"{prefix}_lhs_less", "cmp.ult.i32", lhs_key, rhs_key)
    return lhs_nan, rhs_nan, both_zero, lhs_less


def _emit_f32_ordered_less(
    program: _F32Program,
    lhs: ValueRef,
    rhs: ValueRef,
    constants: tuple[ValueRef, ValueRef, ValueRef, ValueRef, ValueRef, ValueRef],
    prefix: str,
) -> ValueRef:
    """Emits an ordered binary32 less-than comparison."""

    lhs_nan, rhs_nan, both_zero, raw_less = _emit_f32_order_state(
        program, lhs, rhs, constants, prefix
    )
    one = constants[3]
    unordered = program.binary(f"{prefix}_unordered", "or.i32", lhs_nan, rhs_nan)
    ordered = program.binary(f"{prefix}_ordered", "xor.i32", unordered, one)
    nonzero_pair = program.binary(f"{prefix}_nonzero_pair", "xor.i32", both_zero, one)
    ordered_less = program.binary(
        f"{prefix}_ordered_less", "and.i32", raw_less, ordered
    )
    return program.binary(f"{prefix}_result", "and.i32", ordered_less, nonzero_pair)


def emit_f32_extremum(
    program: _F32Program,
    lhs: ValueRef,
    rhs: ValueRef,
    *,
    extremum: _ExtremumKind,
    nan_policy: _NaNPolicy,
    prefix: str,
    result_name: str | None,
    constants: tuple[ValueRef, ValueRef, ValueRef, ValueRef, ValueRef, ValueRef]
    | None = None,
) -> ValueRef:
    """Emits an exact binary32 minimum or maximum into |program|."""

    if extremum not in ("minimum", "maximum"):
        raise ValueError(f"unsupported binary32 extremum {extremum}")
    if nan_policy not in ("number", "ieee"):
        raise ValueError(f"unsupported binary32 NaN policy {nan_policy}")
    if constants is None:
        constants = _f32_order_constants(program, f"{prefix}_order")

    lhs_nan, rhs_nan, both_zero, lhs_less = _emit_f32_order_state(
        program, lhs, rhs, constants, prefix
    )
    quiet_bit = constants[5]
    if extremum == "minimum":
        ordered = program.select(f"{prefix}_ordered", lhs, rhs, lhs_less)
        signed_zero = program.binary(f"{prefix}_signed_zero", "or.i32", lhs, rhs)
    else:
        ordered = program.select(f"{prefix}_ordered", rhs, lhs, lhs_less)
        signed_zero = program.binary(f"{prefix}_signed_zero", "and.i32", lhs, rhs)
    finite = program.select(f"{prefix}_finite", signed_zero, ordered, both_zero)

    quiet_lhs = program.binary(f"{prefix}_quiet_lhs", "or.i32", lhs, quiet_bit)
    if nan_policy == "number":
        lhs_or_finite = program.select(
            f"{prefix}_rhs_nan_selected", lhs, finite, rhs_nan
        )
        lhs_nan_value = program.select(
            f"{prefix}_both_nan_selected", quiet_lhs, rhs, rhs_nan
        )
        return program.select(result_name, lhs_nan_value, lhs_or_finite, lhs_nan)

    quiet_rhs = program.binary(f"{prefix}_quiet_rhs", "or.i32", rhs, quiet_bit)
    rhs_or_finite = program.select(
        f"{prefix}_rhs_nan_selected", quiet_rhs, finite, rhs_nan
    )
    return program.select(result_name, quiet_lhs, rhs_or_finite, lhs_nan)


def _f32_extremum_rule(
    source_op: Op,
    extremum: _ExtremumKind,
    nan_policy: _NaNPolicy,
) -> DescriptorRule:
    program = _F32Program()
    emit_f32_extremum(
        program,
        ValueRef.operand("lhs"),
        ValueRef.operand("rhs"),
        extremum=extremum,
        nan_policy=nan_policy,
        prefix="extremum",
        result_name=None,
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=program.emits[-1].descriptor,
        guards=tuple(
            Guard.value_type(field, _F32) for field in ("lhs", "rhs", "result")
        ),
        emit=tuple(program.emits),
        report_key=f"exact_binary32_{extremum}_{nan_policy}",
    )


def _f32_clamp_rule(mode: Literal["ordered", "number", "ieee"]) -> DescriptorRule:
    program = _F32Program()
    value = ValueRef.operand("value")
    lower = ValueRef.operand("lower")
    upper = ValueRef.operand("upper")
    constants = _f32_order_constants(program, "clamp_order")
    if mode == "ordered":
        below_lower = _emit_f32_ordered_less(
            program, value, lower, constants, "lower_bound"
        )
        lower_bounded = program.select("lower_bounded", lower, value, below_lower)
        above_upper = _emit_f32_ordered_less(
            program, upper, lower_bounded, constants, "upper_bound"
        )
        program.select(None, upper, lower_bounded, above_upper)
    else:
        lower_bounded = emit_f32_extremum(
            program,
            value,
            lower,
            extremum="maximum",
            nan_policy=mode,
            prefix="lower_bound",
            result_name="lower_bounded",
            constants=constants,
        )
        emit_f32_extremum(
            program,
            lower_bounded,
            upper,
            extremum="minimum",
            nan_policy=mode,
            prefix="upper_bound",
            result_name=None,
            constants=constants,
        )
    return DescriptorRule(
        source_op=scalar_arithmetic.scalar_clampf,
        descriptor=program.emits[-1].descriptor,
        guards=(
            Guard.enum_attr_equals("mode", mode),
            *(
                Guard.value_type(field, _F32)
                for field in ("value", "lower", "upper", "result")
            ),
        ),
        emit=tuple(program.emits),
        report_key=f"exact_binary32_clamp_{mode}",
    )


def _normalize_f32_operands(
    program: _F32Program,
    operands: Sequence[tuple[str, ValueRef]],
    constants: tuple[ValueRef, ValueRef, ValueRef, ValueRef],
) -> tuple[tuple[ValueRef, ValueRef], ...]:
    """Returns normalized 24-bit significands and biased exponents."""
    (
        normalize_zero,
        normalize_shift_left_23,
        normalize_shift_right_23,
        normalize_implicit_bit,
    ) = constants

    normalized: list[tuple[ValueRef, ValueRef]] = []
    for prefix, absolute in operands:
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
        normalized.append((normalized_significand, effective_exponent))
    return tuple(normalized)


def _multiply_f32_significands(
    program: _F32Program,
    lhs_significand: ValueRef,
    rhs_significand: ValueRef,
    constants: tuple[ValueRef, ValueRef, ValueRef],
) -> tuple[ValueRef, ValueRef]:
    """Returns the high and low limbs of an exact 24x24-bit product."""
    mask16, product_shift_left_16, product_shift_right_16 = constants
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
    product_low = program.binary(
        "product_low_unaligned", "add.i32", low_low, shifted_cross
    )
    product_carry = program.binary("product_carry", "cmp.ult.i32", product_low, low_low)
    high_high = program.binary("high_high", "mul.i32", lhs_high, rhs_high)
    cross_high = program.binary("cross_high", "lshl.i32", cross, product_shift_right_16)
    product_high_base = program.binary(
        "product_high_base", "add.i32", high_high, cross_high
    )
    product_high = program.binary(
        "product_high_unaligned", "add.i32", product_high_base, product_carry
    )
    return product_high, product_low


def _select_u64(
    program: _F32Program,
    prefix: str,
    true_value: tuple[ValueRef, ValueRef],
    false_value: tuple[ValueRef, ValueRef],
    condition: ValueRef,
) -> tuple[ValueRef, ValueRef]:
    return (
        program.select(f"{prefix}_high", true_value[0], false_value[0], condition),
        program.select(f"{prefix}_low", true_value[1], false_value[1], condition),
    )


def _add_u64(
    program: _F32Program,
    prefix: str,
    lhs: tuple[ValueRef, ValueRef],
    rhs: tuple[ValueRef, ValueRef],
) -> tuple[ValueRef, ValueRef]:
    low = program.binary(f"{prefix}_low", "add.i32", lhs[1], rhs[1])
    carry = program.binary(f"{prefix}_carry", "cmp.ult.i32", low, lhs[1])
    high_base = program.binary(f"{prefix}_high_base", "add.i32", lhs[0], rhs[0])
    high = program.binary(f"{prefix}_high", "add.i32", high_base, carry)
    return high, low


def _subtract_u64(
    program: _F32Program,
    prefix: str,
    lhs: tuple[ValueRef, ValueRef],
    rhs: tuple[ValueRef, ValueRef],
) -> tuple[ValueRef, ValueRef]:
    borrow = program.binary(f"{prefix}_borrow", "cmp.ult.i32", lhs[1], rhs[1])
    low = program.binary(f"{prefix}_low", "sub.i32", lhs[1], rhs[1])
    high_base = program.binary(f"{prefix}_high_base", "sub.i32", lhs[0], rhs[0])
    high = program.binary(f"{prefix}_high", "sub.i32", high_base, borrow)
    return high, low


def _u64_less_than(
    program: _F32Program,
    prefix: str,
    lhs: tuple[ValueRef, ValueRef],
    rhs: tuple[ValueRef, ValueRef],
) -> ValueRef:
    high_less = program.binary(f"{prefix}_high_less", "cmp.ult.i32", lhs[0], rhs[0])
    high_equal = program.binary(f"{prefix}_high_equal", "cmp.eq.i32", lhs[0], rhs[0])
    low_less = program.binary(f"{prefix}_low_less", "cmp.ult.i32", lhs[1], rhs[1])
    equal_high_low_less = program.binary(
        f"{prefix}_equal_high_low_less", "and.i32", high_equal, low_less
    )
    return program.binary(f"{prefix}_less", "or.i32", high_less, equal_high_low_less)


def _shift_right_one_u64(
    program: _F32Program,
    prefix: str,
    value: tuple[ValueRef, ValueRef],
) -> tuple[ValueRef, ValueRef]:
    shift_right_one = program.constant(f"{prefix}_shift_right_one", -1)
    shift_left_31 = program.constant(f"{prefix}_shift_left_31", 31)
    one = program.constant(f"{prefix}_one", 1)
    high = program.binary(f"{prefix}_high", "lshl.i32", value[0], shift_right_one)
    low_base = program.binary(
        f"{prefix}_low_base", "lshl.i32", value[1], shift_right_one
    )
    high_cross = program.binary(
        f"{prefix}_high_cross", "lshl.i32", value[0], shift_left_31
    )
    low_crossed = program.binary(
        f"{prefix}_low_crossed", "or.i32", low_base, high_cross
    )
    sticky = program.binary(f"{prefix}_sticky", "and.i32", value[1], one)
    low = program.binary(f"{prefix}_low", "or.i32", low_crossed, sticky)
    return high, low


def _shift_right_jam_u64(
    program: _F32Program,
    prefix: str,
    value: tuple[ValueRef, ValueRef],
    amount: ValueRef,
) -> tuple[ValueRef, ValueRef]:
    """Logically shifts a two-limb value right and jams discarded bits."""
    zero = program.constant(f"{prefix}_zero", 0)
    mask31 = program.constant(f"{prefix}_mask31", 31)
    shift_mod = program.binary(f"{prefix}_shift_mod", "and.i32", amount, mask31)
    negative_shift_mod = program.binary(
        f"{prefix}_negative_shift_mod", "sub.i32", zero, shift_mod
    )
    inverse_shift_mod = program.binary(
        f"{prefix}_inverse_shift_mod", "and.i32", negative_shift_mod, mask31
    )
    shift_mod_nonzero = program.unary(
        f"{prefix}_shift_mod_nonzero", "cmp.nez.i32", shift_mod
    )

    low_case_high = program.binary(
        f"{prefix}_low_case_high",
        "lshl.i32",
        value[0],
        negative_shift_mod,
    )
    low_case_low_base = program.binary(
        f"{prefix}_low_case_low_base",
        "lshl.i32",
        value[1],
        negative_shift_mod,
    )
    low_case_cross_unmasked = program.binary(
        f"{prefix}_low_case_cross_unmasked",
        "lshl.i32",
        value[0],
        inverse_shift_mod,
    )
    low_case_cross = program.select(
        f"{prefix}_low_case_cross",
        low_case_cross_unmasked,
        zero,
        shift_mod_nonzero,
    )
    low_case_crossed = program.binary(
        f"{prefix}_low_case_crossed",
        "or.i32",
        low_case_low_base,
        low_case_cross,
    )
    low_case_discarded_unmasked = program.binary(
        f"{prefix}_low_case_discarded_unmasked",
        "lshl.i32",
        value[1],
        inverse_shift_mod,
    )
    low_case_discarded = program.select(
        f"{prefix}_low_case_discarded",
        low_case_discarded_unmasked,
        zero,
        shift_mod_nonzero,
    )
    low_case_sticky = program.unary(
        f"{prefix}_low_case_sticky", "cmp.nez.i32", low_case_discarded
    )
    low_case_low = program.binary(
        f"{prefix}_low_case_low",
        "or.i32",
        low_case_crossed,
        low_case_sticky,
    )

    high_case_base = program.binary(
        f"{prefix}_high_case_base",
        "lshl.i32",
        value[0],
        negative_shift_mod,
    )
    high_case_discarded_unmasked = program.binary(
        f"{prefix}_high_case_discarded_unmasked",
        "lshl.i32",
        value[0],
        inverse_shift_mod,
    )
    high_case_high_discarded = program.select(
        f"{prefix}_high_case_high_discarded",
        high_case_discarded_unmasked,
        zero,
        shift_mod_nonzero,
    )
    high_case_discarded = program.binary(
        f"{prefix}_high_case_discarded",
        "or.i32",
        value[1],
        high_case_high_discarded,
    )
    high_case_sticky = program.unary(
        f"{prefix}_high_case_sticky", "cmp.nez.i32", high_case_discarded
    )
    high_case_low = program.binary(
        f"{prefix}_high_case_low",
        "or.i32",
        high_case_base,
        high_case_sticky,
    )

    all_case_bits = program.binary(
        f"{prefix}_all_case_bits", "or.i32", value[0], value[1]
    )
    all_case_low = program.unary(f"{prefix}_all_case_low", "cmp.nez.i32", all_case_bits)
    shift_minus_32 = program.add_immediate(f"{prefix}_shift_minus_32", amount, -32)
    shift_lt32 = program.binary(
        f"{prefix}_shift_lt32", "cmp.slt.i32", shift_minus_32, zero
    )
    shift_minus_64 = program.add_immediate(f"{prefix}_shift_minus_64", amount, -64)
    shift_lt64 = program.binary(
        f"{prefix}_shift_lt64", "cmp.slt.i32", shift_minus_64, zero
    )
    high_or_all_low = program.select(
        f"{prefix}_high_or_all_low",
        high_case_low,
        all_case_low,
        shift_lt64,
    )
    low = program.select(f"{prefix}_low", low_case_low, high_or_all_low, shift_lt32)
    high = program.select(f"{prefix}_high", low_case_high, zero, shift_lt32)
    return high, low


def _shift_left_u64(
    program: _F32Program,
    prefix: str,
    value: tuple[ValueRef, ValueRef],
    amount: ValueRef,
) -> tuple[ValueRef, ValueRef]:
    """Logically shifts a two-limb value left by an amount in [0, 63]."""
    zero = program.constant(f"{prefix}_zero", 0)
    mask31 = program.constant(f"{prefix}_mask31", 31)
    shift_mod = program.binary(f"{prefix}_shift_mod", "and.i32", amount, mask31)
    negative_shift_mod = program.binary(
        f"{prefix}_negative_shift_mod", "sub.i32", zero, shift_mod
    )
    inverse_shift_mod = program.binary(
        f"{prefix}_inverse_shift_mod", "and.i32", negative_shift_mod, mask31
    )
    negative_inverse_shift_mod = program.binary(
        f"{prefix}_negative_inverse_shift_mod",
        "sub.i32",
        zero,
        inverse_shift_mod,
    )
    shift_mod_nonzero = program.unary(
        f"{prefix}_shift_mod_nonzero", "cmp.nez.i32", shift_mod
    )
    low_case_high_base = program.binary(
        f"{prefix}_low_case_high_base", "lshl.i32", value[0], shift_mod
    )
    low_case_cross_unmasked = program.binary(
        f"{prefix}_low_case_cross_unmasked",
        "lshl.i32",
        value[1],
        negative_inverse_shift_mod,
    )
    low_case_cross = program.select(
        f"{prefix}_low_case_cross",
        low_case_cross_unmasked,
        zero,
        shift_mod_nonzero,
    )
    low_case_high = program.binary(
        f"{prefix}_low_case_high",
        "or.i32",
        low_case_high_base,
        low_case_cross,
    )
    low_case_low = program.binary(
        f"{prefix}_low_case_low", "lshl.i32", value[1], shift_mod
    )
    high_case_high = program.binary(
        f"{prefix}_high_case_high", "lshl.i32", value[1], shift_mod
    )
    shift_minus_32 = program.add_immediate(f"{prefix}_shift_minus_32", amount, -32)
    shift_lt32 = program.binary(
        f"{prefix}_shift_lt32", "cmp.slt.i32", shift_minus_32, zero
    )
    high = program.select(f"{prefix}_high", low_case_high, high_case_high, shift_lt32)
    low = program.select(f"{prefix}_low", low_case_low, zero, shift_lt32)
    return high, low


def _count_leading_zeros_u64(
    program: _F32Program,
    prefix: str,
    value: tuple[ValueRef, ValueRef],
) -> ValueRef:
    high_nonzero = program.unary(f"{prefix}_high_nonzero", "cmp.nez.i32", value[0])
    high_count = program.unary(f"{prefix}_high_count", "clz.i32", value[0])
    low_count = program.unary(f"{prefix}_low_count", "clz.i32", value[1])
    low_count_with_high_limb = program.add_immediate(
        f"{prefix}_low_count_with_high_limb", low_count, 32
    )
    return program.select(
        f"{prefix}_count",
        high_count,
        low_count_with_high_limb,
        high_nonzero,
    )


def emit_f32_multiply(
    *,
    lhs: ValueRef | None = None,
    rhs: ValueRef | None = None,
    result_name: str | None = None,
    temporary_prefix: str = "",
) -> tuple[EmitDescriptorOp, ...]:
    """Builds an exact binary32 multiply descriptor program."""

    program = _F32Program(temporary_prefix)
    lhs = ValueRef.operand("lhs") if lhs is None else lhs
    rhs = ValueRef.operand("rhs") if rhs is None else rhs

    # Keep constants local to the algorithm phase that consumes them. AIE2P
    # scalar registers are unspillable, and materializing a cheap constant
    # again is preferable to keeping it live across the whole product.
    absolute_mask = program.constant("absolute_mask", 0x7FFFFFFF)
    classify_sign_bit = program.constant("classify_sign_bit", -(2**31))
    normalize_zero = program.constant("normalize_zero", 0)
    normalize_shift_left_23 = program.constant("normalize_shift_left_23", 23)
    normalize_shift_right_23 = program.constant("normalize_shift_right_23", -23)
    normalize_implicit_bit = program.constant("normalize_implicit_bit", 0x00800000)
    lhs_absolute = program.binary("lhs_absolute", "and.i32", lhs, absolute_mask)
    rhs_absolute = program.binary("rhs_absolute", "and.i32", rhs, absolute_mask)
    sign_xor = program.binary("sign_xor", "xor.i32", lhs, rhs)
    product_sign = program.binary(
        "product_sign", "and.i32", sign_xor, classify_sign_bit
    )

    (lhs_significand, lhs_exponent), (rhs_significand, rhs_exponent) = (
        _normalize_f32_operands(
            program,
            (("lhs", lhs_absolute), ("rhs", rhs_absolute)),
            (
                normalize_zero,
                normalize_shift_left_23,
                normalize_shift_right_23,
                normalize_implicit_bit,
            ),
        )
    )

    # Four 16x16 products recover the exact 48-bit significand product. The
    # high/low pair is then aligned like compiler-rt's wide binary32 product.
    mask16 = program.constant("product_mask16", 0xFFFF)
    product_shift_left_8 = program.constant("product_shift_left_8", 8)
    product_shift_left_16 = program.constant("product_shift_left_16", 16)
    product_shift_right_16 = program.constant("product_shift_right_16", -16)
    product_shift_right_24 = program.constant("product_shift_right_24", -24)
    product_high_unaligned, product_low_unaligned = _multiply_f32_significands(
        program,
        lhs_significand,
        rhs_significand,
        (mask16, product_shift_left_16, product_shift_right_16),
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
    program.select(result_name, quiet_lhs, result, lhs_nan)

    return tuple(program.emits)


def _f32_abs_rule() -> DescriptorRule:
    program = _F32Program()
    absolute_mask = program.constant("absolute_mask", 0x7FFFFFFF)
    program.binary(
        None,
        "and.i32",
        ValueRef.operand("input"),
        absolute_mask,
    )
    return DescriptorRule(
        source_op=scalar_arithmetic.scalar_absf,
        descriptor=program.emits[-1].descriptor,
        guards=tuple(Guard.value_type(field, _F32) for field in ("input", "result")),
        emit=tuple(program.emits),
        report_key="exact_binary32_abs",
    )


def _f32_neg_rule() -> DescriptorRule:
    program = _F32Program()
    sign_mask = program.constant("sign_mask", -(1 << 31))
    program.binary(
        None,
        "xor.i32",
        ValueRef.operand("input"),
        sign_mask,
    )
    return DescriptorRule(
        source_op=scalar_arithmetic.scalar_negf,
        descriptor=program.emits[-1].descriptor,
        guards=tuple(Guard.value_type(field, _F32) for field in ("input", "result")),
        emit=tuple(program.emits),
        report_key="exact_binary32_neg",
    )


def _vector_f32_abs_rule() -> DescriptorRule:
    constant = _descriptor("amd.xdna.aie2p.constant.i32")
    splat = _descriptor("amd.xdna.aie2p.splat.i32x16")
    bitwise_and = _descriptor("amd.xdna.aie2p.and.bits512")
    return DescriptorRule(
        source_op=vector.vector_absf,
        descriptor=bitwise_and,
        guards=tuple(
            Guard.value_type(field, _F32_VECTOR) for field in ("input", "result")
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=constant,
                results={"dst": ValueRef.temporary("absolute_mask_scalar")},
                result_types={"dst": DescriptorResultType()},
                immediates={"i": 0x7FFFFFFF},
                form=DescriptorEmitForm.CONST,
            ),
            EmitDescriptorOp(
                descriptor=splat,
                operands={"src": ValueRef.temporary("absolute_mask_scalar")},
                results={"dst": ValueRef.temporary("absolute_mask")},
                result_types={"dst": ValueRef.operand("input")},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=bitwise_and,
                operands={
                    "s1": ValueRef.operand("input"),
                    "s2": ValueRef.temporary("absolute_mask"),
                },
                results={"d": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
        report_key="native_vector_binary32_abs",
    )


def _vector_f32_neg_rule() -> DescriptorRule:
    constant = _descriptor("amd.xdna.aie2p.constant.i32")
    splat = _descriptor("amd.xdna.aie2p.splat.i32x16")
    bitwise_or = _descriptor("amd.xdna.aie2p.or.bits512")
    bitwise_and = _descriptor("amd.xdna.aie2p.and.bits512")
    subtract = _descriptor("amd.xdna.aie2p.sub.i32x16")
    return DescriptorRule(
        source_op=vector.vector_negf,
        descriptor=subtract,
        guards=tuple(
            Guard.value_type(field, _F32_VECTOR) for field in ("input", "result")
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=constant,
                results={"dst": ValueRef.temporary("sign_mask_scalar")},
                result_types={"dst": DescriptorResultType()},
                immediates={"i": -(1 << 31)},
                form=DescriptorEmitForm.CONST,
            ),
            EmitDescriptorOp(
                descriptor=splat,
                operands={"src": ValueRef.temporary("sign_mask_scalar")},
                results={"dst": ValueRef.temporary("sign_mask")},
                result_types={"dst": ValueRef.operand("input")},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=bitwise_or,
                operands={
                    "s1": ValueRef.operand("input"),
                    "s2": ValueRef.temporary("sign_mask"),
                },
                results={"d": ValueRef.temporary("union")},
                result_types={"d": ValueRef.operand("input")},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=bitwise_and,
                operands={
                    "s1": ValueRef.operand("input"),
                    "s2": ValueRef.temporary("sign_mask"),
                },
                results={"d": ValueRef.temporary("intersection")},
                result_types={"d": ValueRef.operand("input")},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=subtract,
                operands={
                    "s1": ValueRef.temporary("union"),
                    "s2": ValueRef.temporary("intersection"),
                },
                results={"d": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
        report_key="native_vector_binary32_neg",
    )


def _f32_copysign_rule() -> DescriptorRule:
    program = _F32Program()
    absolute_mask = program.constant("absolute_mask", 0x7FFFFFFF)
    sign_mask = program.constant("sign_mask", -(1 << 31))
    magnitude = program.binary(
        "magnitude", "and.i32", ValueRef.operand("lhs"), absolute_mask
    )
    sign = program.binary("sign", "and.i32", ValueRef.operand("rhs"), sign_mask)
    program.binary(None, "or.i32", magnitude, sign)
    return DescriptorRule(
        source_op=scalar_arithmetic.scalar_copysignf,
        descriptor=program.emits[-1].descriptor,
        guards=tuple(
            Guard.value_type(field, _F32) for field in ("lhs", "rhs", "result")
        ),
        emit=tuple(program.emits),
        report_key="exact_binary32_copysign",
    )


def _emit_roundeven_f32_to_i32(program: _F32Program, input_value: ValueRef) -> ValueRef:
    return emit_round_nearest_f32_to_i32(
        program,
        input_value,
        0,
        "roundeven",
    )


def _emit_trunc_f32_to_i32(program: _F32Program, input_value: ValueRef) -> ValueRef:
    return emit_truncate_f32_to_signed_i32(
        program,
        input_value,
        "trunc",
        "truncated_i32",
    )


def _f32_integral_rule(
    source_op: Op,
    integer_emitter: Callable[[_F32Program, ValueRef], ValueRef],
    report_key: str,
) -> DescriptorRule:
    """Builds an exact integral-valued f32 operation."""

    program = _F32Program()
    input_value = ValueRef.operand("input")
    absolute_mask = program.constant("absolute_mask", 0x7FFFFFFF)
    integral_limit = program.constant("integral_limit", 0x4B000000)
    absolute = program.binary("absolute", "and.i32", input_value, absolute_mask)
    already_integral = program.binary(
        "already_integral", "cmp.uge.i32", absolute, integral_limit
    )
    integer_result = integer_emitter(program, input_value)
    rounded_result = emit_signed_integer_to_f32(
        program,
        integer_result,
        None,
        "integral_f32",
    )
    zero_result = program.unary("zero_result", "cmp.eqz.i32", integer_result)
    sign_mask = program.constant("sign_mask", -(1 << 31))
    input_sign = program.binary("input_sign", "and.i32", input_value, sign_mask)
    signed_zero = program.binary("signed_zero", "or.i32", rounded_result, input_sign)
    rounded_result = program.select(
        "signed_zero_result", signed_zero, rounded_result, zero_result
    )
    program.select(None, input_value, rounded_result, already_integral)
    return DescriptorRule(
        source_op=source_op,
        descriptor=program.emits[-1].descriptor,
        guards=tuple(Guard.value_type(field, _F32) for field in ("input", "result")),
        emit=tuple(program.emits),
        report_key=report_key,
    )


def _f32_arcp_div_rule() -> DescriptorRule:
    program = _F32Program()
    reciprocal = program.unary(
        "reciprocal",
        "reciprocal.f32",
        ValueRef.operand("rhs"),
    )
    emits = (
        *program.emits,
        *emit_f32_multiply(
            lhs=ValueRef.operand("lhs"),
            rhs=reciprocal,
            temporary_prefix="quotient_",
        ),
    )
    return DescriptorRule(
        source_op=scalar_arithmetic.scalar_divf,
        descriptor=program.emits[0].descriptor,
        guards=(
            *(Guard.value_type(field, _F32) for field in ("lhs", "rhs", "result")),
            Guard.instance_flags_has_all("fastmath", "arcp"),
        ),
        emit=emits,
        report_key="approximate_binary32_reciprocal_multiply",
    )


def _f32_fma_emits() -> tuple[EmitDescriptorOp, ...]:
    program = _F32Program()

    absolute_mask = program.constant("absolute_mask", 0x7FFFFFFF)
    sign_mask = program.constant("sign_mask", -(2**31))
    zero = program.constant("zero", 0)
    disabled_exponent = program.constant("disabled_exponent", -512)
    infinity = program.constant("infinity", 0x7F800000)
    quiet_bit = program.constant("quiet_bit", 0x00400000)
    normalize_zero = program.constant("normalize_zero", 0)
    normalize_shift_left_23 = program.constant("normalize_shift_left_23", 23)
    normalize_shift_right_23 = program.constant("normalize_shift_right_23", -23)
    normalize_implicit_bit = program.constant("normalize_implicit_bit", 0x00800000)

    a = ValueRef.operand("a")
    b = ValueRef.operand("b")
    c = ValueRef.operand("c")
    absolute_a = program.binary("absolute_a", "and.i32", a, absolute_mask)
    absolute_b = program.binary("absolute_b", "and.i32", b, absolute_mask)
    absolute_c = program.binary("absolute_c", "and.i32", c, absolute_mask)
    product_sign_xor = program.binary("product_sign_xor", "xor.i32", a, b)
    product_sign = program.binary(
        "product_sign", "and.i32", product_sign_xor, sign_mask
    )
    c_sign = program.binary("c_sign", "and.i32", c, sign_mask)

    (
        (a_significand, a_exponent),
        (b_significand, b_exponent),
        (c_significand, c_exponent),
    ) = _normalize_f32_operands(
        program,
        (
            ("a", absolute_a),
            ("b", absolute_b),
            ("c", absolute_c),
        ),
        (
            normalize_zero,
            normalize_shift_left_23,
            normalize_shift_right_23,
            normalize_implicit_bit,
        ),
    )
    product_mask16 = program.constant("product_mask16", 0xFFFF)
    product_shift_left_16 = program.constant("product_shift_left_16", 16)
    product_shift_right_16 = program.constant("product_shift_right_16", -16)
    product_high, product_low = _multiply_f32_significands(
        program,
        a_significand,
        b_significand,
        (product_mask16, product_shift_left_16, product_shift_right_16),
    )

    # Normalize the exact 48-bit product and the 24-bit addend into a shared
    # two-limb fixed-point domain whose leading bit is bit 62. This leaves 39
    # low bits below the final binary32 significand for exact cancellation and
    # one final round-to-nearest-even operation.
    product_wide_mask = program.constant("product_wide_mask", 0x00008000)
    shift_left_7 = program.constant("shift_left_7", 7)
    shift_left_15 = program.constant("shift_left_15", 15)
    shift_left_16 = program.constant("shift_left_16", 16)
    shift_right_16 = program.constant("shift_right_16", -16)
    shift_right_17 = program.constant("shift_right_17", -17)
    product_wide_bits = program.binary(
        "product_wide_bits", "and.i32", product_high, product_wide_mask
    )
    product_wide = program.unary("product_wide", "cmp.nez.i32", product_wide_bits)
    product_high_shift15_base = program.binary(
        "product_high_shift15_base",
        "lshl.i32",
        product_high,
        shift_left_15,
    )
    product_high_shift15_cross = program.binary(
        "product_high_shift15_cross",
        "lshl.i32",
        product_low,
        shift_right_17,
    )
    product_high_shift15 = program.binary(
        "product_high_shift15",
        "or.i32",
        product_high_shift15_base,
        product_high_shift15_cross,
    )
    product_low_shift15 = program.binary(
        "product_low_shift15", "lshl.i32", product_low, shift_left_15
    )
    product_high_shift16_base = program.binary(
        "product_high_shift16_base",
        "lshl.i32",
        product_high,
        shift_left_16,
    )
    product_high_shift16_cross = program.binary(
        "product_high_shift16_cross",
        "lshl.i32",
        product_low,
        shift_right_16,
    )
    product_high_shift16 = program.binary(
        "product_high_shift16",
        "or.i32",
        product_high_shift16_base,
        product_high_shift16_cross,
    )
    product_low_shift16 = program.binary(
        "product_low_shift16", "lshl.i32", product_low, shift_left_16
    )
    normalized_product = _select_u64(
        program,
        "normalized_product",
        (product_high_shift15, product_low_shift15),
        (product_high_shift16, product_low_shift16),
        product_wide,
    )
    product_exponent_sum = program.binary(
        "product_exponent_sum", "add.i32", a_exponent, b_exponent
    )
    product_exponent_base = program.add_immediate(
        "product_exponent_base", product_exponent_sum, -127
    )
    product_exponent = program.binary(
        "product_exponent", "add.i32", product_exponent_base, product_wide
    )

    a_zero = program.unary("a_zero", "cmp.eqz.i32", absolute_a)
    b_zero = program.unary("b_zero", "cmp.eqz.i32", absolute_b)
    c_zero = program.unary("c_zero", "cmp.eqz.i32", absolute_c)
    product_zero = program.binary("product_zero", "or.i32", a_zero, b_zero)
    product_exponent = program.select(
        "nonzero_product_exponent",
        disabled_exponent,
        product_exponent,
        product_zero,
    )
    normalized_product = _select_u64(
        program,
        "nonzero_product",
        (zero, zero),
        normalized_product,
        product_zero,
    )
    normalized_c_high = program.binary(
        "normalized_c_high", "lshl.i32", c_significand, shift_left_7
    )
    normalized_c = _select_u64(
        program,
        "nonzero_c",
        (zero, zero),
        (normalized_c_high, zero),
        c_zero,
    )
    c_exponent = program.select(
        "nonzero_c_exponent", disabled_exponent, c_exponent, c_zero
    )

    product_exponent_less = program.binary(
        "product_exponent_less", "cmp.slt.i32", product_exponent, c_exponent
    )
    product_exponent_equal = program.binary(
        "product_exponent_equal", "cmp.eq.i32", product_exponent, c_exponent
    )
    product_significand_less = _u64_less_than(
        program, "product_significand", normalized_product, normalized_c
    )
    equal_exponent_product_less = program.binary(
        "equal_exponent_product_less",
        "and.i32",
        product_exponent_equal,
        product_significand_less,
    )
    product_less = program.binary(
        "product_less",
        "or.i32",
        product_exponent_less,
        equal_exponent_product_less,
    )
    large_significand = _select_u64(
        program,
        "large_significand",
        normalized_c,
        normalized_product,
        product_less,
    )
    small_significand = _select_u64(
        program,
        "small_significand",
        normalized_product,
        normalized_c,
        product_less,
    )
    large_exponent = program.select(
        "large_exponent", c_exponent, product_exponent, product_less
    )
    small_exponent = program.select(
        "small_exponent", product_exponent, c_exponent, product_less
    )
    large_sign = program.select("large_sign", c_sign, product_sign, product_less)
    exponent_difference = program.binary(
        "exponent_difference", "sub.i32", large_exponent, small_exponent
    )
    aligned_small = _shift_right_jam_u64(
        program,
        "aligned_small",
        small_significand,
        exponent_difference,
    )

    sum_significand = _add_u64(
        program, "sum_significand", large_significand, aligned_small
    )
    sum_overflow_bits = program.binary(
        "sum_overflow_bits", "and.i32", sum_significand[0], sign_mask
    )
    sum_overflow = program.unary("sum_overflow", "cmp.nez.i32", sum_overflow_bits)
    shifted_sum = _shift_right_one_u64(program, "shifted_sum", sum_significand)
    normalized_sum = _select_u64(
        program,
        "normalized_sum",
        shifted_sum,
        sum_significand,
        sum_overflow,
    )
    incremented_sum_exponent = program.add_immediate(
        "incremented_sum_exponent", large_exponent, 1
    )
    sum_exponent = program.select(
        "sum_exponent",
        incremented_sum_exponent,
        large_exponent,
        sum_overflow,
    )

    difference_significand = _subtract_u64(
        program,
        "difference_significand",
        large_significand,
        aligned_small,
    )
    difference_bits = program.binary(
        "difference_bits",
        "or.i32",
        difference_significand[0],
        difference_significand[1],
    )
    difference_zero = program.unary("difference_zero", "cmp.eqz.i32", difference_bits)
    difference_leading_zeros = _count_leading_zeros_u64(
        program, "difference", difference_significand
    )
    difference_shift = program.add_immediate(
        "difference_shift", difference_leading_zeros, -1
    )
    normalized_difference = _shift_left_u64(
        program,
        "normalized_difference",
        difference_significand,
        difference_shift,
    )
    difference_exponent = program.binary(
        "difference_exponent", "sub.i32", large_exponent, difference_shift
    )

    same_sign = program.binary("same_sign", "cmp.eq.i32", product_sign, c_sign)
    opposite_sign = program.unary("opposite_sign", "cmp.eqz.i32", same_sign)
    exact_cancellation = program.binary(
        "exact_cancellation", "and.i32", opposite_sign, difference_zero
    )
    normalized_result = _select_u64(
        program,
        "normalized_result",
        normalized_sum,
        normalized_difference,
        same_sign,
    )
    result_exponent = program.select(
        "result_exponent", sum_exponent, difference_exponent, same_sign
    )
    arithmetic_sign = program.select(
        "arithmetic_sign", product_sign, large_sign, same_sign
    )

    # Move results below the normal range into the subnormal fixed-point
    # domain, then perform the operation's sole rounding step.
    is_underflow = program.binary("is_underflow", "cmp.sge.i32", zero, result_exponent)
    negative_result_exponent = program.binary(
        "negative_result_exponent", "sub.i32", zero, result_exponent
    )
    underflow_shift = program.add_immediate(
        "underflow_shift", negative_result_exponent, 1
    )
    selected_underflow_shift = program.select(
        "selected_underflow_shift", underflow_shift, zero, is_underflow
    )
    rounded_domain_result = _shift_right_jam_u64(
        program,
        "rounded_domain_result",
        normalized_result,
        selected_underflow_shift,
    )
    one = program.constant("one", 1)
    pack_exponent = program.select("pack_exponent", one, result_exponent, is_underflow)

    shift_right_7 = program.constant("shift_right_7", -7)
    half_mask = program.constant("half_mask", 0x00000040)
    below_half_mask = program.constant("below_half_mask", 0x0000003F)
    rounded_carry_mask = program.constant("rounded_carry_mask", 0x01000000)
    truncated = program.binary(
        "truncated",
        "lshl.i32",
        rounded_domain_result[0],
        shift_right_7,
    )
    half_bits = program.binary(
        "half_bits", "and.i32", rounded_domain_result[0], half_mask
    )
    half = program.unary("half", "cmp.nez.i32", half_bits)
    below_half_high = program.binary(
        "below_half_high",
        "and.i32",
        rounded_domain_result[0],
        below_half_mask,
    )
    below_half_bits = program.binary(
        "below_half_bits",
        "or.i32",
        below_half_high,
        rounded_domain_result[1],
    )
    below_half = program.unary("below_half", "cmp.nez.i32", below_half_bits)
    truncated_lsb = program.binary("truncated_lsb", "and.i32", truncated, one)
    round_tie_or_greater = program.binary(
        "round_tie_or_greater", "or.i32", below_half, truncated_lsb
    )
    round_up = program.binary("round_up", "and.i32", half, round_tie_or_greater)
    rounded = program.binary("rounded", "add.i32", truncated, round_up)
    rounded_carry_bits = program.binary(
        "rounded_carry_bits", "and.i32", rounded, rounded_carry_mask
    )
    rounded_carry = program.unary("rounded_carry", "cmp.nez.i32", rounded_carry_bits)
    rounded_shifted = program.binary(
        "rounded_shifted", "lshl.i32", rounded, program.constant("minus_one", -1)
    )
    packed_significand = program.select(
        "packed_significand", rounded_shifted, rounded, rounded_carry
    )
    pack_exponent = program.binary(
        "rounded_pack_exponent", "add.i32", pack_exponent, rounded_carry
    )
    exponent_minus_255 = program.add_immediate(
        "exponent_minus_255", pack_exponent, -255
    )
    overflow = program.binary("overflow", "cmp.sge.i32", exponent_minus_255, zero)
    pack_exponent_minus_one = program.add_immediate(
        "pack_exponent_minus_one", pack_exponent, -1
    )
    exponent_bits = program.binary(
        "exponent_bits",
        "lshl.i32",
        pack_exponent_minus_one,
        program.constant("shift_left_23", 23),
    )
    finite_magnitude = program.binary(
        "finite_magnitude", "add.i32", exponent_bits, packed_significand
    )
    magnitude = program.select("magnitude", infinity, finite_magnitude, overflow)
    magnitude = program.select(
        "cancellation_magnitude", zero, magnitude, exact_cancellation
    )
    finite_sign = program.select(
        "finite_sign", zero, arithmetic_sign, exact_cancellation
    )
    finite_result = program.binary("finite_result", "or.i32", finite_sign, magnitude)

    # IEEE special values bypass the fixed-point datapath. NaN operands are
    # quieted with deterministic source-order precedence; invalid infinity
    # combinations use the canonical quiet NaN.
    a_nan = program.binary("a_nan", "cmp.ult.i32", infinity, absolute_a)
    b_nan = program.binary("b_nan", "cmp.ult.i32", infinity, absolute_b)
    c_nan = program.binary("c_nan", "cmp.ult.i32", infinity, absolute_c)
    a_infinity = program.binary("a_infinity", "cmp.eq.i32", absolute_a, infinity)
    b_infinity = program.binary("b_infinity", "cmp.eq.i32", absolute_b, infinity)
    c_infinity = program.binary("c_infinity", "cmp.eq.i32", absolute_c, infinity)
    a_infinity_b_zero = program.binary(
        "a_infinity_b_zero", "and.i32", a_infinity, b_zero
    )
    b_infinity_a_zero = program.binary(
        "b_infinity_a_zero", "and.i32", b_infinity, a_zero
    )
    invalid_product = program.binary(
        "invalid_product", "or.i32", a_infinity_b_zero, b_infinity_a_zero
    )
    product_infinity = program.binary(
        "product_infinity", "or.i32", a_infinity, b_infinity
    )
    infinity_sign_difference = program.binary(
        "infinity_sign_difference", "xor.i32", product_sign, c_sign
    )
    infinity_signs_differ = program.unary(
        "infinity_signs_differ", "cmp.nez.i32", infinity_sign_difference
    )
    opposite_infinities = program.binary(
        "opposite_infinities",
        "and.i32",
        c_infinity,
        infinity_signs_differ,
    )
    canonical_nan = program.binary("canonical_nan", "or.i32", infinity, quiet_bit)
    signed_product_infinity = program.binary(
        "signed_product_infinity", "or.i32", product_sign, infinity
    )
    product_infinity_result = program.select(
        "product_infinity_result",
        canonical_nan,
        signed_product_infinity,
        opposite_infinities,
    )
    result = program.select("c_infinity_selected", c, finite_result, c_infinity)
    result = program.select(
        "product_infinity_selected",
        product_infinity_result,
        result,
        product_infinity,
    )
    result = program.select(
        "invalid_product_selected", canonical_nan, result, invalid_product
    )
    quiet_c = program.binary("quiet_c", "or.i32", c, quiet_bit)
    result = program.select("c_nan_selected", quiet_c, result, c_nan)
    quiet_b = program.binary("quiet_b", "or.i32", b, quiet_bit)
    result = program.select("b_nan_selected", quiet_b, result, b_nan)
    quiet_a = program.binary("quiet_a", "or.i32", a, quiet_bit)
    program.select(None, quiet_a, result, a_nan)

    return tuple(program.emits)


_F32_MULTIPLY_EMITS = emit_f32_multiply()
_F32_FMA_EMITS = _f32_fma_emits()

AIE2P_F32_RULES = (
    _f32_abs_rule(),
    _f32_neg_rule(),
    _vector_f32_abs_rule(),
    _vector_f32_neg_rule(),
    _f32_copysign_rule(),
    _f32_arcp_div_rule(),
    _f32_extremum_rule(scalar_arithmetic.scalar_minimumf, "minimum", "ieee"),
    _f32_extremum_rule(scalar_arithmetic.scalar_maximumf, "maximum", "ieee"),
    _f32_extremum_rule(scalar_arithmetic.scalar_minnumf, "minimum", "number"),
    _f32_extremum_rule(scalar_arithmetic.scalar_maxnumf, "maximum", "number"),
    _f32_clamp_rule("ordered"),
    _f32_clamp_rule("number"),
    _f32_clamp_rule("ieee"),
    _f32_integral_rule(
        scalar_math.scalar_roundevenf,
        _emit_roundeven_f32_to_i32,
        "exact_binary32_roundeven",
    ),
    _f32_integral_rule(
        scalar_math.scalar_truncf,
        _emit_trunc_f32_to_i32,
        "exact_binary32_trunc",
    ),
    DescriptorRule(
        source_op=scalar_arithmetic.scalar_mulf,
        descriptor=_descriptor("amd.xdna.aie2p.select.nonzero.i32"),
        guards=tuple(
            Guard.value_type(field, _F32) for field in ("lhs", "rhs", "result")
        ),
        emit=_F32_MULTIPLY_EMITS,
        report_key="exact_binary32",
    ),
    DescriptorRule(
        source_op=scalar_math.scalar_fmaf,
        descriptor=_descriptor("amd.xdna.aie2p.select.nonzero.i32"),
        guards=tuple(
            Guard.value_type(field, _F32) for field in ("a", "b", "c", "result")
        ),
        emit=_F32_FMA_EMITS,
        report_key="exact_binary32_fma",
    ),
)
