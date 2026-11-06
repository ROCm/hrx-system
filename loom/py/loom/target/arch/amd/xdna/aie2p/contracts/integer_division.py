# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact unsigned constant quotient and remainder for AIE2P scalar words."""

from loom.dialect.index import defs as index
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dsl import Op
from loom.target.arch.amd.xdna.aie2p.contracts.scalar_program import ScalarProgram
from loom.target.arch.amd.xdna.aie2p.core_descriptors import AIE2P_CORE_DESCRIPTOR_SET
from loom.target.contracts import (
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    TypePattern,
    UnsignedDivisorMagicKind,
    ValueProject,
    ValueRef,
    descriptor_by_key,
)

_I32 = Scalar("i32")
_INDEX = Scalar("index")
_LHS = ValueRef.operand("lhs")
_RHS = ValueRef.operand("rhs")


def append_runtime_unsigned_division_steps(
    emits: list[EmitDescriptorOp],
    *,
    initial_state: ValueRef,
    initial_remainder: ValueRef,
    divisor: ValueRef,
    temporary_prefix: str,
    final_remainder: ValueRef | None = None,
    full_remainder: bool = False,
    return_quotient: bool = False,
) -> tuple[ValueRef, ValueRef, ValueRef | None]:
    """Appends one word, retaining a continued remainder's overflow bit."""
    divide_step = descriptor_by_key(
        AIE2P_CORE_DESCRIPTOR_SET, "amd.xdna.aie2p.divide.step.unsigned.i32"
    )
    state = initial_state
    remainder = initial_remainder
    program = ScalarProgram(temporary_prefix)
    quotient_correction = None
    if full_remainder:
        right31 = program.constant("right31", -31)
        if return_quotient:
            left1 = program.constant("left1", 1)
            quotient_correction = program.constant("quotient_correction_0", 0)
    for step in range(1, 33):
        next_remainder = (
            final_remainder
            if step == 32 and final_remainder is not None and not full_remainder
            else ValueRef.temporary(f"{temporary_prefix}remainder_{step}")
        )
        next_state = ValueRef.temporary(f"{temporary_prefix}state_{step}")
        if full_remainder:
            carry = program.binary(f"carry_{step}", "lshl.i32", remainder, right31)
        program.emits.append(
            EmitDescriptorOp(
                descriptor=divide_step,
                operands={"sd": state, "s0": remainder, "s1": divisor},
                results={"d0": next_remainder, "sd_out": next_state},
                result_types={
                    "d0": DescriptorResultType(),
                    "sd_out": DescriptorResultType(),
                },
            )
        )
        state = next_state
        remainder = next_remainder
        if full_remainder:
            # DIVS shifts within 32 bits. On overflow the shifted low word is
            # below the divisor, so DIVS does not subtract it. Subtracting it
            # here restores the exact remainder modulo 2^32. The missing
            # quotient bits can be accumulated outside r31: they cannot reach
            # its dividend MSB during these 32 steps.
            correction = program.binary(
                f"remainder_correction_{step}", "mul.i32", carry, divisor
            )
            corrected_remainder = (
                final_remainder
                if step == 32 and final_remainder is not None
                else program.temporary(f"corrected_remainder_{step}")
            )
            program.emits.append(
                EmitDescriptorOp(
                    descriptor=descriptor_by_key(
                        AIE2P_CORE_DESCRIPTOR_SET, "amd.xdna.aie2p.sub.i32"
                    ),
                    operands={"s0": remainder, "s1": correction},
                    results={"d0": corrected_remainder},
                    result_types={"d0": DescriptorResultType()},
                )
            )
            remainder = corrected_remainder
            if return_quotient:
                quotient_correction = program.binary(
                    f"shifted_quotient_correction_{step}",
                    "lshl.i32",
                    quotient_correction,
                    left1,
                )
                quotient_correction = program.binary(
                    f"quotient_correction_{step}", "or.i32", quotient_correction, carry
                )
    emits.extend(program.emits)
    return state, remainder, quotient_correction


def _magic_rule(
    source_op: Op,
    type_pattern: TypePattern,
    *,
    kind: UnsignedDivisorMagicKind,
    remainder: bool,
) -> DescriptorRule:
    program = ScalarProgram()
    magic = program.constant(
        "magic", ValueProject.u32_divisor_magic_multiplier_as_i32("lhs", "rhs")
    )
    mask = program.constant("mask16", 0xFFFF)
    right16 = program.constant("right16", -16)
    low = program.binary("numerator_low", "and.i32", _LHS, mask)
    high = program.binary("numerator_high", "lshl.i32", _LHS, right16)
    magic_low = program.binary("magic_low", "and.i32", magic, mask)
    magic_high = program.binary("magic_high", "lshl.i32", magic, right16)

    # Each product is at most (2**16 - 1)**2. Adding the previous high
    # halfword remains below 2**32, so no intermediate carry is discarded.
    word_zero = program.binary("word_zero", "mul.i32", low, magic_low)
    carry_zero = program.binary("carry_zero", "lshl.i32", word_zero, right16)
    middle = program.multiply_add("middle", carry_zero, high, magic_low)
    middle_low = program.binary("middle_low", "and.i32", middle, mask)
    next_word = program.multiply_add("next_word", middle_low, low, magic_high)
    middle_high = program.binary("middle_high", "lshl.i32", middle, right16)
    product_high = program.multiply_add("product_high", middle_high, high, magic_high)
    next_high = program.binary("next_high", "lshl.i32", next_word, right16)
    quotient = program.binary(
        "raw_quotient" if kind.has_shift or remainder else None,
        "add.i32",
        product_high,
        next_high,
    )
    if kind.has_add:
        right_one = program.constant("right_one", -1)
        difference = program.binary("difference", "sub.i32", _LHS, quotient)
        half = program.binary("half_difference", "lshl.i32", difference, right_one)
        quotient = program.binary("adjusted_quotient", "add.i32", half, quotient)
    if kind.has_shift:
        shift = program.constant(
            "post_shift",
            ValueProject.u32_divisor_magic_shift("lhs", "rhs", product_bit_width=32),
            descriptor_key="amd.xdna.aie2p.constant.i32.short",
        )
        zero = program.constant("zero", 0)
        right_shift = program.binary("right_shift", "sub.i32", zero, shift)
        quotient = program.binary(
            "quotient" if remainder else None, "lshl.i32", quotient, right_shift
        )
    if remainder:
        product = program.binary("quotient_product", "mul.i32", quotient, _RHS)
        program.binary(None, "sub.i32", _LHS, product)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor_by_key(
            AIE2P_CORE_DESCRIPTOR_SET, "amd.xdna.aie2p.mul.i32"
        ),
        priority=1,
        guards=(
            *(
                Guard.value_type(field, type_pattern)
                for field in ("lhs", "rhs", "result")
            ),
            Guard.value_exact_i64("rhs"),
            Guard.value_i64_range("rhs", 2, (1 << 31) - 1),
            Guard.value_u32_divisor_magic_kind("lhs", "rhs", kind),
        ),
        emit=tuple(program.emits),
    )


def _high_bit_rule(source_op: Op, *, remainder: bool) -> DescriptorRule:
    program = ScalarProgram()
    below_divisor = program.binary("below_divisor", "cmp.ult.i32", _LHS, _RHS)
    if remainder:
        difference = program.binary("difference", "sub.i32", _LHS, _RHS)
        program.select(None, _LHS, difference, below_divisor)
    else:
        one = program.constant("one", 1)
        program.binary(None, "xor.i32", below_divisor, one)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor_by_key(
            AIE2P_CORE_DESCRIPTOR_SET, "amd.xdna.aie2p.cmp.ult.i32"
        ),
        priority=1,
        guards=(
            *(Guard.value_type(field, _I32) for field in ("lhs", "rhs", "result")),
            Guard.value_exact_i64("rhs"),
            Guard.value_i64_range("rhs", -(1 << 31), -1),
        ),
        emit=tuple(program.emits),
    )


AIE2P_INTEGER_DIVISION_RULES = (
    *(
        _magic_rule(source_op, type_pattern, kind=kind, remainder=remainder)
        for source_op, type_pattern, remainder in (
            (index.index_div, _INDEX, False),
            (index.index_rem, _INDEX, True),
            (scalar_arithmetic.scalar_divui, _I32, False),
            (scalar_arithmetic.scalar_remui, _I32, True),
        )
        for kind in UnsignedDivisorMagicKind
    ),
    _high_bit_rule(scalar_arithmetic.scalar_divui, remainder=False),
    _high_bit_rule(scalar_arithmetic.scalar_remui, remainder=True),
)
