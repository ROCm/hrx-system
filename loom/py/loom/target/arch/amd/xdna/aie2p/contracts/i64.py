# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar-pair i64 and f64 contracts for AMD XDNA AIE2P."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scf import defs as scf
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amd.xdna.aie2p.contracts.integer_division import (
    append_runtime_unsigned_division_steps,
)
from loom.target.arch.amd.xdna.aie2p.contracts.packet_program import PairPacketProgram
from loom.target.arch.amd.xdna.aie2p.contracts.scalar_program import ScalarProgram
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    AttrProject,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    Guard,
    Scalar,
    TypePattern,
    ValueProject,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_I1 = Scalar("i1")
_I32 = Scalar("i32")
_I64 = Scalar("i64")
_F64 = Scalar("f64")
_INDEX = Scalar("index")
_I64_VECTOR = Vector("i64", minimum_static_elements=1, maximum_static_elements=8)
_F64_VECTOR = Vector("f64", minimum_static_elements=1, maximum_static_elements=8)
_I64_PREDICATE_VECTOR = Vector(
    "i1", minimum_static_elements=1, maximum_static_elements=8
)


class _PairVectorConstantCarrier(Enum):
    NATIVE = "native"
    WIDE = "wide"
    ACCUMULATOR = "accumulator"


# Each stage dilates the active predicate bits into the low bit of progressively
# wider groups. The final common step copies each bit into the adjacent 32-bit
# payload-word position consumed by VSEL.32.
_PAIR_VECTOR_SELECT_RANGES = (
    (1, 1, ()),
    (2, 2, ((1, 0x5),)),
    (3, 4, ((2, 0x33), (1, 0x55))),
    (5, 8, ((4, 0x0F0F), (2, 0x3333), (1, 0x5555))),
)

AIE2P_PAIR_VECTOR_TYPES = (_I64_VECTOR, _F64_VECTOR)
AIE2P_PAIR_SCALAR_TYPES = (_I64, _F64)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


def _typed_guards(
    fields: Sequence[str], type_pattern: TypePattern
) -> tuple[Guard, ...]:
    return tuple(Guard.value_type(field, type_pattern) for field in fields)


def _split_pair(
    source: ValueRef, prefix: str
) -> tuple[tuple[EmitRegisterSlice, EmitRegisterSlice], ValueRef, ValueRef]:
    low = ValueRef.temporary(f"{prefix}_low")
    high = ValueRef.temporary(f"{prefix}_high")
    return (
        (
            EmitRegisterSlice(source=source, result=low, unit_count=1),
            EmitRegisterSlice(
                source=source,
                result=high,
                unit_offset=1,
                unit_count=1,
            ),
        ),
        low,
        high,
    )


def _concat_pair(low: ValueRef, high: ValueRef) -> EmitRegisterConcat:
    return EmitRegisterConcat(
        sources=(low, high),
        result=ValueRef.result("result"),
    )


def _pair_bitwise_rule(source_op: Op, operation: str) -> DescriptorRule:
    lhs_emits, lhs_low, lhs_high = _split_pair(ValueRef.operand("lhs"), "lhs")
    rhs_emits, rhs_low, rhs_high = _split_pair(ValueRef.operand("rhs"), "rhs")
    program = ScalarProgram()
    result_low = program.binary("result_low", operation, lhs_low, rhs_low)
    result_high = program.binary("result_high", operation, lhs_high, rhs_high)
    return DescriptorRule(
        source_op=source_op,
        descriptor=_descriptor(f"amd.xdna.aie2p.{operation}"),
        guards=_typed_guards(("lhs", "rhs", "result"), _I64),
        emit=(
            *lhs_emits,
            *rhs_emits,
            *program.emits,
            _concat_pair(result_low, result_high),
        ),
    )


def _pair_vector_binary_rule(source_op: Op, descriptor_key: str) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_typed_guards(("lhs", "rhs", "result"), _I64_VECTOR),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "s1": ValueRef.operand("lhs"),
                    "s2": ValueRef.operand("rhs"),
                },
                results={"d": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _pair_vector_xor_rule() -> DescriptorRule:
    bitwise_or = _descriptor("amd.xdna.aie2p.or.bits512")
    bitwise_and = _descriptor("amd.xdna.aie2p.and.bits512")
    subtract = _descriptor("amd.xdna.aie2p.sub.i32x16")
    return DescriptorRule(
        source_op=vector.vector_xori,
        descriptor=subtract,
        guards=_typed_guards(("lhs", "rhs", "result"), _I64_VECTOR),
        emit=(
            EmitDescriptorOp(
                descriptor=bitwise_or,
                operands={
                    "s1": ValueRef.operand("lhs"),
                    "s2": ValueRef.operand("rhs"),
                },
                results={"d": ValueRef.temporary("union")},
                result_types={"d": ValueRef.operand("lhs")},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=bitwise_and,
                operands={
                    "s1": ValueRef.operand("lhs"),
                    "s2": ValueRef.operand("rhs"),
                },
                results={"d": ValueRef.temporary("intersection")},
                result_types={"d": ValueRef.operand("lhs")},
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
    )


def _pair_add_sub_rule(
    source_op: Op,
    low_operation: str,
    high_operation: str,
) -> DescriptorRule:
    lhs_emits, lhs_low, lhs_high = _split_pair(ValueRef.operand("lhs"), "lhs")
    rhs_emits, rhs_low, rhs_high = _split_pair(ValueRef.operand("rhs"), "rhs")
    low_descriptor = _descriptor(f"amd.xdna.aie2p.{low_operation}")
    high_descriptor = _descriptor(f"amd.xdna.aie2p.{high_operation}")
    result_low = ValueRef.temporary("result_low")
    result_high = ValueRef.temporary("result_high")
    carry_low = ValueRef.temporary("carry_low")
    carry_high = ValueRef.temporary("carry_high")
    return DescriptorRule(
        source_op=source_op,
        descriptor=high_descriptor,
        guards=_typed_guards(("lhs", "rhs", "result"), _I64),
        emit=(
            *lhs_emits,
            *rhs_emits,
            EmitDescriptorOp(
                descriptor=low_descriptor,
                operands={"s0": lhs_low, "s1": rhs_low},
                results={"d0": result_low, "carry_out": carry_low},
                result_types={
                    "d0": DescriptorResultType(),
                    "carry_out": DescriptorResultType(),
                },
            ),
            EmitDescriptorOp(
                descriptor=high_descriptor,
                operands={"s0": lhs_high, "s1": rhs_high, "carry_in": carry_low},
                results={"d0": result_high, "carry_out": carry_high},
                result_types={
                    "d0": DescriptorResultType(),
                    "carry_out": DescriptorResultType(),
                },
            ),
            _concat_pair(result_low, result_high),
        ),
    )


def _pair_constant_emits(
    low_bits: ValueProject,
    high_bits: ValueProject,
    result: ValueRef,
    result_type: TypePattern | None,
) -> tuple[EmitDescriptorOp, EmitDescriptorOp, EmitRegisterConcat]:
    descriptor = _descriptor("amd.xdna.aie2p.constant.i32")
    low = ValueRef.temporary("constant_low")
    high = ValueRef.temporary("constant_high")
    return (
        EmitDescriptorOp(
            descriptor=descriptor,
            results={"dst": low},
            result_types={"dst": DescriptorResultType()},
            immediates={"i": low_bits},
            form=DescriptorEmitForm.CONST,
        ),
        EmitDescriptorOp(
            descriptor=descriptor,
            results={"dst": high},
            result_types={"dst": DescriptorResultType()},
            immediates={"i": high_bits},
            form=DescriptorEmitForm.CONST,
        ),
        EmitRegisterConcat(
            sources=(low, high),
            result=result,
            result_type=result_type,
        ),
    )


def _pair_constant_rule(
    result_type: TypePattern,
    attr_kind: str,
    low_bits: ValueProject,
    high_bits: ValueProject,
    exact_guard: Guard,
) -> DescriptorRule:
    descriptor = _descriptor("amd.xdna.aie2p.constant.i32")
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", attr_kind),
            Guard.value_type("result", result_type),
            exact_guard,
        ),
        emit=_pair_constant_emits(
            low_bits,
            high_bits,
            ValueRef.result("result"),
            None,
        ),
    )


def _pair_vector_constant_rule(
    scalar_type: TypePattern,
    result_type: TypePattern,
    source_op: Op,
    low_bits: ValueProject,
    high_bits: ValueProject,
    exact_guard: Guard,
    carrier: _PairVectorConstantCarrier,
) -> DescriptorRule:
    descriptor = _descriptor("amd.xdna.aie2p.splat.i64x8")
    packet = (
        ValueRef.result("result")
        if carrier is _PairVectorConstantCarrier.NATIVE
        else ValueRef.temporary("packet")
    )
    scalar = ValueRef.temporary("scalar")
    emits = list(_pair_constant_emits(low_bits, high_bits, scalar, scalar_type))
    emits.append(
        EmitDescriptorOp(
            descriptor=descriptor,
            operands={"src": scalar},
            results={"dst": packet},
            result_types={"dst": DescriptorResultType()},
            form=DescriptorEmitForm.OP,
        )
    )
    if carrier is _PairVectorConstantCarrier.WIDE:
        emits.append(
            EmitRegisterConcat(
                sources=(packet, packet),
                result=ValueRef.result("result"),
            )
        )
    elif carrier is _PairVectorConstantCarrier.ACCUMULATOR:
        accumulator_unit = ValueRef.temporary("accumulator_unit")
        emits.extend(
            (
                EmitDescriptorOp(
                    descriptor=_descriptor(
                        "amd.xdna.aie2p.move.vector512.to.accumulator512"
                    ),
                    operands={"src": packet},
                    results={"dst": accumulator_unit},
                    result_types={"dst": DescriptorResultType()},
                    form=DescriptorEmitForm.OP,
                ),
                EmitRegisterConcat(
                    sources=(accumulator_unit,) * 4,
                    result=ValueRef.result("result"),
                ),
            )
        )
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.value_type("result", result_type),
            exact_guard,
        ),
        emit=tuple(emits),
    )


def _pair_vector_constant_rules() -> tuple[DescriptorRule, ...]:
    integer_words = ValueProject.exact_i64_i32_word
    float_words = ValueProject.float_as_f64_i32_word
    integer_exact = Guard.value_exact_i64
    float_exact = Guard.value_exact_float
    rules = []
    for scalar_type, native_type, element_type, source_op, project, exact_guard in (
        (
            _I64,
            _I64_VECTOR,
            "i64",
            vector.vector_splati,
            integer_words,
            integer_exact,
        ),
        (
            _F64,
            _F64_VECTOR,
            "f64",
            vector.vector_splatf,
            float_words,
            float_exact,
        ),
    ):
        for result_type, carrier in (
            (native_type, _PairVectorConstantCarrier.NATIVE),
            (
                Vector(
                    element_type,
                    minimum_static_elements=9,
                    maximum_static_elements=16,
                ),
                _PairVectorConstantCarrier.WIDE,
            ),
        ):
            rules.append(
                _pair_vector_constant_rule(
                    scalar_type,
                    result_type,
                    source_op,
                    project("result", word_index=0),
                    project("result", word_index=1),
                    exact_guard("result"),
                    carrier,
                )
            )
    rules.append(
        _pair_vector_constant_rule(
            _I64,
            Vector("i64", lanes=32),
            vector.vector_splati,
            integer_words("result", word_index=0),
            integer_words("result", word_index=1),
            integer_exact("result"),
            _PairVectorConstantCarrier.ACCUMULATOR,
        )
    )
    return tuple(rules)


def _pair_multiply_rule() -> DescriptorRule:
    lhs_emits, lhs_low, lhs_high = _split_pair(ValueRef.operand("lhs"), "lhs")
    rhs_emits, rhs_low, rhs_high = _split_pair(ValueRef.operand("rhs"), "rhs")
    program = ScalarProgram()
    mask16 = program.constant("mask16", 0xFFFF)
    shift_right_16 = program.constant("shift_right_16", -16)

    lhs_low16 = program.binary("lhs_low16", "and.i32", lhs_low, mask16)
    lhs_high16 = program.binary("lhs_high16", "lshl.i32", lhs_low, shift_right_16)
    rhs_low16 = program.binary("rhs_low16", "and.i32", rhs_low, mask16)
    rhs_high16 = program.binary("rhs_high16", "lshl.i32", rhs_low, shift_right_16)

    low_product = program.binary("result_low", "mul.i32", lhs_low, rhs_low)
    word_zero = program.binary("word_zero", "mul.i32", lhs_low16, rhs_low16)
    word_zero_high = program.binary(
        "word_zero_high", "lshl.i32", word_zero, shift_right_16
    )
    middle = program.multiply_add("middle", word_zero_high, lhs_high16, rhs_low16)
    middle_low = program.binary("middle_low", "and.i32", middle, mask16)
    next_word = program.multiply_add("next_word", middle_low, lhs_low16, rhs_high16)
    middle_high = program.binary("middle_high", "lshl.i32", middle, shift_right_16)
    high_product = program.multiply_add(
        "high_product", middle_high, lhs_high16, rhs_high16
    )
    next_word_high = program.binary(
        "next_word_high", "lshl.i32", next_word, shift_right_16
    )
    high_with_carry = program.binary(
        "high_with_carry", "add.i32", high_product, next_word_high
    )
    high_with_lhs_cross = program.multiply_add(
        "high_with_lhs_cross", high_with_carry, lhs_low, rhs_high
    )
    result_high = program.multiply_add(
        "result_high", high_with_lhs_cross, lhs_high, rhs_low
    )

    return DescriptorRule(
        source_op=scalar_arithmetic.scalar_muli,
        descriptor=_descriptor("amd.xdna.aie2p.madd.i32"),
        guards=_typed_guards(("lhs", "rhs", "result"), _I64),
        emit=(
            *lhs_emits,
            *rhs_emits,
            *program.emits,
            _concat_pair(low_product, result_high),
        ),
    )


def _pair_unsigned_remainder_by_word_rule() -> DescriptorRule:
    """Continues the native divider state across both dividend words."""
    lhs_emits, lhs_low, lhs_high = _split_pair(ValueRef.operand("lhs"), "lhs")
    divisor = ValueRef.temporary("divisor")
    zero = ValueRef.temporary("zero")
    move_to_state = _descriptor("amd.xdna.aie2p.move.to.division-state")
    divide_step = _descriptor("amd.xdna.aie2p.divide.step.unsigned.i32")
    zero_descriptor = _descriptor("amd.xdna.aie2p.constant.i32.short")
    emits: list[EmitRegisterSlice | EmitDescriptorOp | EmitRegisterConcat] = [
        *lhs_emits,
        EmitRegisterSlice(source=ValueRef.operand("rhs"), result=divisor, unit_count=1),
        EmitDescriptorOp(
            descriptor=zero_descriptor,
            results={"dst": zero},
            result_types={"dst": DescriptorResultType()},
            immediates={"i": 0},
            form=DescriptorEmitForm.CONST,
        ),
    ]

    remainder = zero
    for word_name, word in (("high", lhs_high), ("low", lhs_low)):
        state = ValueRef.temporary(f"{word_name}_division_state_0")
        emits.append(
            EmitDescriptorOp(
                descriptor=move_to_state,
                operands={"src": word},
                results={"dst": state},
                result_types={"dst": DescriptorResultType()},
                form=DescriptorEmitForm.OP,
            )
        )
        state, remainder = append_runtime_unsigned_division_steps(
            emits,
            initial_state=state,
            initial_remainder=remainder,
            divisor=divisor,
            temporary_prefix=f"{word_name}_division_",
        )
    emits.append(_concat_pair(remainder, zero))
    return DescriptorRule(
        source_op=scalar_arithmetic.scalar_remui,
        descriptor=divide_step,
        priority=1,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), _I64),
            Guard.value_i64_range("rhs", 1, 2**32 - 1),
        ),
        emit=tuple(emits),
    )


def _pair_bounded_left_shift_rule(word_index: int) -> DescriptorRule:
    """Uses retained count ranges to select one split-word shift regime."""
    lhs_emits, lhs_low, lhs_high = _split_pair(ValueRef.operand("lhs"), "lhs")
    shift_count = ValueRef.temporary("shift_count")
    shift_count_emit = EmitRegisterSlice(
        source=ValueRef.operand("rhs"),
        result=shift_count,
        unit_count=1,
    )
    program = ScalarProgram()
    cross_count = program.add_immediate("cross_count", shift_count, -32)
    if word_index == 0:
        result_low = program.binary("result_low", "lshl.i32", lhs_low, shift_count)
        high_base = program.binary("high_base", "lshl.i32", lhs_high, shift_count)
        # This regime excludes zero: LSHL by -32 wraps to a zero-bit shift.
        low_cross = program.binary("low_cross", "lshl.i32", lhs_low, cross_count)
        result_high = program.binary("result_high", "or.i32", high_base, low_cross)
    else:
        result_low = program.constant("result_low", 0)
        result_high = program.binary("result_high", "lshl.i32", lhs_low, cross_count)
    return DescriptorRule(
        source_op=scalar_bitwise.scalar_shli,
        descriptor=_descriptor("amd.xdna.aie2p.lshl.i32"),
        priority=1,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), _I64),
            Guard.value_i64_range("rhs", max(1, word_index * 32), word_index * 32 + 31),
        ),
        emit=(
            *lhs_emits,
            shift_count_emit,
            *program.emits,
            _concat_pair(result_low, result_high),
        ),
    )


def _pair_left_shift_rule() -> DescriptorRule:
    lhs_emits, lhs_low, lhs_high = _split_pair(ValueRef.operand("lhs"), "lhs")
    shift_count = ValueRef.temporary("shift_count")
    shift_count_emit = EmitRegisterSlice(
        source=ValueRef.operand("rhs"),
        result=shift_count,
        unit_count=1,
    )
    program = ScalarProgram()
    zero = program.constant("zero", 0)
    mask31 = program.constant("mask31", 31)
    word_bits = program.constant("word_bits", 32)
    masked_count = program.binary("masked_count", "and.i32", shift_count, mask31)
    # A right shift by 32 cannot be represented by one LSHL: its magnitude
    # wraps to zero. Splitting the carry shift into 1 + (31 - count) preserves
    # a zero carry for count zero without a predicate or fixed select register.
    right_one = program.constant("right_one", -1)
    low_upper = program.binary("low_upper", "lshl.i32", lhs_low, right_one)
    cross_count = program.add_immediate("cross_count", masked_count, -31)
    low_if_small = program.binary("low_if_small", "lshl.i32", lhs_low, masked_count)
    high_base = program.binary("high_base", "lshl.i32", lhs_high, masked_count)
    low_cross = program.binary("low_cross", "lshl.i32", low_upper, cross_count)
    high_if_small = program.binary("high_if_small", "or.i32", high_base, low_cross)
    high_if_large = program.binary("high_if_large", "lshl.i32", lhs_low, masked_count)
    is_small = program.binary("is_small", "cmp.ult.i32", shift_count, word_bits)
    result_low = program.select("result_low", low_if_small, zero, is_small)
    result_high = program.select("result_high", high_if_small, high_if_large, is_small)
    return DescriptorRule(
        source_op=scalar_bitwise.scalar_shli,
        descriptor=_descriptor("amd.xdna.aie2p.lshl.i32"),
        guards=_typed_guards(("lhs", "rhs", "result"), _I64),
        emit=(
            *lhs_emits,
            shift_count_emit,
            *program.emits,
            _concat_pair(result_low, result_high),
        ),
    )


def _pair_right_shift_rule(
    source_op: Op, high_shift: str, count_range: tuple[int, int] | None
) -> DescriptorRule:
    program = ScalarProgram()
    lhs_emits, lhs_low, lhs_high = _split_pair(ValueRef.operand("lhs"), "lhs")
    shift_count = program.temporary("shift_count")
    if count_range == (32, 63):
        word_bits = program.constant("word_bits", 32)
        right_count = program.binary("right_count", "sub.i32", word_bits, shift_count)
    else:
        zero = program.constant("zero", 0)
        within_word_count = shift_count
        if count_range is None:
            mask31 = program.constant("mask31", 31)
            within_word_count = program.binary(
                "within_word_count", "and.i32", shift_count, mask31
            )
        right_count = program.binary("right_count", "sub.i32", zero, within_word_count)
    shifted_high = program.binary("shifted_high", high_shift, lhs_high, right_count)
    if count_range == (32, 63):
        result_low = shifted_high
    else:
        if count_range == (1, 31):
            cross_source = lhs_high
            cross_count = program.add_immediate("cross_count", right_count, 32)
        else:
            # A single left shift by 32 wraps to zero bits. Splitting the
            # carry into 1 + (31-count) makes count zero preserve the low word.
            one = program.constant("one", 1)
            cross_source = program.binary("high_lower", "lshl.i32", lhs_high, one)
            cross_count = program.add_immediate("cross_count", right_count, 31)
        high_cross = program.binary("high_cross", "lshl.i32", cross_source, cross_count)
        low_base = program.binary("low_base", "lshl.i32", lhs_low, right_count)
        result_low = program.binary("low_if_small", "or.i32", low_base, high_cross)
    result_high = shifted_high
    if count_range is None or count_range == (32, 63):
        if source_op is scalar_bitwise.scalar_shrsi:
            sign_count = program.constant("sign_count", -31)
            fill = program.binary("sign_fill", "ashl.i32", lhs_high, sign_count)
        else:
            fill = program.constant("zero", 0) if count_range == (32, 63) else zero
        if count_range == (32, 63):
            result_high = fill
        else:
            word_bits = program.constant("word_bits", 32)
            is_small = program.binary("is_small", "cmp.ult.i32", shift_count, word_bits)
            result_low = program.select(
                "result_low", result_low, shifted_high, is_small
            )
            result_high = program.select("result_high", shifted_high, fill, is_small)
    return DescriptorRule(
        source_op=source_op,
        descriptor=_descriptor(f"amd.xdna.aie2p.{high_shift}"),
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), _I64),
            *((Guard.value_i64_range("rhs", *count_range),) if count_range else ()),
        ),
        priority=2 if count_range == (1, 31) else 1 if count_range else 0,
        emit=(
            *lhs_emits,
            EmitRegisterSlice(
                source=ValueRef.operand("rhs"), result=shift_count, unit_count=1
            ),
            *program.emits,
            _concat_pair(result_low, result_high),
        ),
    )


def _pair_less(
    program: ScalarProgram,
    prefix: str,
    lhs: tuple[ValueRef, ValueRef],
    rhs: tuple[ValueRef, ValueRef],
    *,
    signed: bool,
    result_name: str | None,
) -> ValueRef:
    high_operation = "cmp.slt.i32" if signed else "cmp.ult.i32"
    high_less = program.binary(f"{prefix}_high_less", high_operation, lhs[1], rhs[1])
    high_equal = program.binary(f"{prefix}_high_equal", "cmp.eq.i32", lhs[1], rhs[1])
    low_less = program.binary(f"{prefix}_low_less", "cmp.ult.i32", lhs[0], rhs[0])
    tied_low_less = program.binary(
        f"{prefix}_tied_low_less", "and.i32", high_equal, low_less
    )
    return program.binary(result_name, "or.i32", high_less, tied_low_less)


def _pair_compare_rule(predicate: str) -> DescriptorRule:
    lhs_emits, lhs_low, lhs_high = _split_pair(ValueRef.operand("lhs"), "lhs")
    rhs_emits, rhs_low, rhs_high = _split_pair(ValueRef.operand("rhs"), "rhs")
    lhs = (lhs_low, lhs_high)
    rhs = (rhs_low, rhs_high)
    program = ScalarProgram()
    if predicate in ("eq", "ne"):
        comparison_operation = "cmp.eq.i32" if predicate == "eq" else "cmp.ne.i32"
        combine_operation = "and.i32" if predicate == "eq" else "or.i32"
        low = program.binary("low_comparison", comparison_operation, lhs_low, rhs_low)
        high = program.binary(
            "high_comparison", comparison_operation, lhs_high, rhs_high
        )
        program.binary(None, combine_operation, low, high)
    else:
        signed = predicate.startswith("s")
        swap_operands = predicate in ("sgt", "sge", "ugt", "uge")
        inclusive = predicate in ("sle", "sge", "ule", "uge")
        less_lhs, less_rhs = (rhs, lhs) if swap_operands else (lhs, rhs)
        less = _pair_less(
            program,
            "ordered",
            less_lhs,
            less_rhs,
            signed=signed,
            result_name="less" if inclusive else None,
        )
        if inclusive:
            low_equal = program.binary("low_equal", "cmp.eq.i32", lhs_low, rhs_low)
            high_equal = program.binary("high_equal", "cmp.eq.i32", lhs_high, rhs_high)
            equal = program.binary("equal", "and.i32", low_equal, high_equal)
            program.binary(None, "or.i32", less, equal)
    return DescriptorRule(
        source_op=scalar_comparison.scalar_cmpi,
        descriptor=program.emits[-1].descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            Guard.value_type("lhs", _I64),
            Guard.value_type("rhs", _I64),
            Guard.value_type("result", _I1),
        ),
        emit=(*lhs_emits, *rhs_emits, *program.emits),
    )


def _pair_vector_deinterleave_emits() -> tuple[
    tuple[EmitDescriptorOp, ...],
    tuple[ValueRef, ValueRef],
    tuple[ValueRef, ValueRef],
]:
    program = PairPacketProgram()
    lhs_words = program.split_i64_words(ValueRef.operand("lhs"), "lhs")
    rhs_words = program.split_i64_words(ValueRef.operand("rhs"), "rhs")
    return tuple(program.emits), lhs_words, rhs_words


def _pair_vector_word_equality(
    prefix: str,
    lhs: ValueRef,
    rhs: ValueRef,
) -> tuple[tuple[EmitDescriptorOp, EmitDescriptorOp], ValueRef]:
    subtract = _descriptor("amd.xdna.aie2p.sub.i32x16")
    compare = _descriptor("amd.xdna.aie2p.cmp.eqz.i32x16.el.low32")
    difference = ValueRef.temporary(f"{prefix}_word_difference")
    result = ValueRef.temporary(f"{prefix}_word_equal")
    return (
        (
            EmitDescriptorOp(
                descriptor=subtract,
                operands={"s1": lhs, "s2": rhs},
                results={"d": difference},
                result_types={"d": DescriptorResultType()},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=compare,
                operands={"s2": difference},
                results={"cmp": result},
                result_types={"cmp": DescriptorResultType()},
                form=DescriptorEmitForm.OP,
            ),
        ),
        result,
    )


def _pair_vector_word_compare(
    prefix: str,
    lhs: ValueRef,
    rhs: ValueRef,
    *,
    relation: str,
    signed: bool,
) -> tuple[EmitDescriptorOp, ValueRef]:
    signedness = "signed" if signed else "unsigned"
    compare = _descriptor(f"amd.xdna.aie2p.cmp.{relation}.{signedness}.i32x16.el.low32")
    result = ValueRef.temporary(f"{prefix}_word_{relation}")
    return (
        EmitDescriptorOp(
            descriptor=compare,
            operands={"s1": lhs, "s2": rhs},
            results={"cmp": result},
            result_types={"cmp": DescriptorResultType()},
            form=DescriptorEmitForm.OP,
        ),
        result,
    )


def _pair_vector_predicate_binary(
    operation: str,
    prefix: str,
    lhs: ValueRef,
    rhs: ValueRef,
) -> tuple[EmitDescriptorOp, ValueRef]:
    descriptor = _descriptor(f"amd.xdna.aie2p.predicate.{operation}.low32")
    result = ValueRef.temporary(prefix)
    return (
        EmitDescriptorOp(
            descriptor=descriptor,
            operands={"s0": lhs, "s1": rhs},
            results={"d0": result},
            result_types={"d0": DescriptorResultType()},
            form=DescriptorEmitForm.OP,
        ),
        result,
    )


def _pair_vector_compare_rule(predicate: str) -> DescriptorRule:
    deinterleave_emits, lhs_words, rhs_words = _pair_vector_deinterleave_emits()
    emits: list[EmitDescriptorOp] = list(deinterleave_emits)

    if predicate == "eq":
        low_equal_emits, low_equal = _pair_vector_word_equality(
            "low", lhs_words[0], rhs_words[0]
        )
        high_equal_emits, high_equal = _pair_vector_word_equality(
            "high", lhs_words[1], rhs_words[1]
        )
        emits.extend((*low_equal_emits, *high_equal_emits))
        combine, result_bits = _pair_vector_predicate_binary(
            "and", "pair_equal", low_equal, high_equal
        )
        emits.append(combine)
    elif predicate == "ne":
        differences: list[ValueRef] = []
        for word, lhs_word, rhs_word in zip(
            ("low", "high"), lhs_words, rhs_words, strict=True
        ):
            for direction, compare_lhs, compare_rhs in (
                ("forward", lhs_word, rhs_word),
                ("reverse", rhs_word, lhs_word),
            ):
                compare, difference = _pair_vector_word_compare(
                    f"{word}_{direction}",
                    compare_lhs,
                    compare_rhs,
                    relation="lt",
                    signed=False,
                )
                emits.append(compare)
                differences.append(difference)
        result_bits = differences[0]
        for index, difference in enumerate(differences[1:], start=1):
            combine, result_bits = _pair_vector_predicate_binary(
                "or", f"different_{index}", result_bits, difference
            )
            emits.append(combine)
    else:
        signed = predicate.startswith("s")
        swap_operands = predicate in ("sgt", "sge", "ugt", "uge")
        inclusive = predicate in ("sle", "sge", "ule", "uge")
        ordered_lhs, ordered_rhs = (
            (rhs_words, lhs_words) if swap_operands else (lhs_words, rhs_words)
        )
        # Compare the word pairs lexicographically. The high word carries the
        # i64 sign; the low word is always unsigned. Inclusive predicates use
        # low_rhs >= low_lhs only when the high words are equal.
        high_equal_emits, high_equal = _pair_vector_word_equality(
            "high", ordered_lhs[1], ordered_rhs[1]
        )
        emits.extend(high_equal_emits)
        high_compare, high_less = _pair_vector_word_compare(
            "high",
            ordered_lhs[1],
            ordered_rhs[1],
            relation="lt",
            signed=signed,
        )
        emits.append(high_compare)
        low_compare, low_ordered = _pair_vector_word_compare(
            "low",
            ordered_rhs[0] if inclusive else ordered_lhs[0],
            ordered_lhs[0] if inclusive else ordered_rhs[0],
            relation="ge" if inclusive else "lt",
            signed=False,
        )
        emits.append(low_compare)
        combine, tied_low_ordered = _pair_vector_predicate_binary(
            "and", "tied_low_ordered", high_equal, low_ordered
        )
        emits.append(combine)
        combine, result_bits = _pair_vector_predicate_binary(
            "or", "pair_ordered", high_less, tied_low_ordered
        )
        emits.append(combine)

    complete = _descriptor("amd.xdna.aie2p.predicate.complete.zero.high32")
    emits.append(
        EmitDescriptorOp(
            descriptor=complete,
            operands={"storage": result_bits},
            results={"dst": ValueRef.result("result")},
            immediates={"i": 0},
            form=DescriptorEmitForm.OP,
        )
    )
    return DescriptorRule(
        source_op=vector.vector_cmpi,
        descriptor=complete,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            Guard.value_type("lhs", _I64_VECTOR),
            Guard.value_type("rhs", _I64_VECTOR),
            Guard.value_type("result", _I64_PREDICATE_VECTOR),
        ),
        emit=tuple(emits),
    )


def _pair_select_rule(type_pattern: TypePattern) -> DescriptorRule:
    true_emits, true_low, true_high = _split_pair(
        ValueRef.operand("true_value"), "true"
    )
    false_emits, false_low, false_high = _split_pair(
        ValueRef.operand("false_value"), "false"
    )
    program = ScalarProgram()
    result_low = program.select(
        "result_low",
        true_low,
        false_low,
        ValueRef.operand("condition"),
    )
    result_high = program.select(
        "result_high",
        true_high,
        false_high,
        ValueRef.operand("condition"),
    )
    return DescriptorRule(
        source_op=scf.scf_select,
        descriptor=_descriptor("amd.xdna.aie2p.select.nonzero.i32"),
        guards=(
            Guard.value_type("condition", _I1),
            *_typed_guards(("true_value", "false_value", "result"), type_pattern),
        ),
        emit=(
            *true_emits,
            *false_emits,
            *program.emits,
            _concat_pair(result_low, result_high),
        ),
    )


def _pair_vector_select_rule(
    element_type: str,
    minimum_lanes: int,
    maximum_lanes: int,
    spread_stages: Sequence[tuple[int, int]],
) -> DescriptorRule:
    value_type = Vector(
        element_type,
        minimum_static_elements=minimum_lanes,
        maximum_static_elements=maximum_lanes,
    )
    condition_type = Vector(
        "i1",
        minimum_static_elements=minimum_lanes,
        maximum_static_elements=maximum_lanes,
    )
    program = ScalarProgram()
    constants: dict[int, ValueRef] = {}

    def constant(value: int) -> ValueRef:
        if value not in constants:
            constants[value] = program.constant(f"selector_constant_{value:x}", value)
        return constants[value]

    # Predicates pack one bit per logical 64-bit lane, while VSEL.32 consumes
    # one bit per physical 32-bit payload word. Mask undefined predicate bits,
    # dilate each active bit into an even position, then copy it to the adjacent
    # odd position. This selects both words of every i64/f64 lane together.
    selector = program.binary(
        "selector_active",
        "predicate.mask.low32",
        ValueRef.operand("condition"),
        constant((1 << maximum_lanes) - 1),
    )
    for stage_index, (shift, mask) in enumerate(spread_stages):
        shifted = program.binary(
            f"selector_spread_{stage_index}_shifted",
            "lshl.i32",
            selector,
            constant(shift),
        )
        combined = program.binary(
            f"selector_spread_{stage_index}_combined",
            "or.i32",
            selector,
            shifted,
        )
        selector = program.binary(
            f"selector_spread_{stage_index}",
            "and.i32",
            combined,
            constant(mask),
        )
    shifted = program.binary(
        "selector_odd",
        "lshl.i32",
        selector,
        constant(1),
    )
    hardware_selector = program.binary(
        "hardware_selector",
        "or.i32",
        selector,
        shifted,
    )
    select = _descriptor("amd.xdna.aie2p.select.i32x16")
    return DescriptorRule(
        source_op=vector.vector_select,
        descriptor=select,
        guards=(
            Guard.value_type("condition", condition_type),
            *_typed_guards(("true_value", "false_value", "result"), value_type),
        ),
        emit=(
            *program.emits,
            EmitDescriptorOp(
                descriptor=select,
                operands={
                    # VSEL chooses s1 for zero and s2 for one.
                    "s1": ValueRef.operand("false_value"),
                    "s2": ValueRef.operand("true_value"),
                    "sel": hardware_selector,
                },
                results={"d": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
                copy_operands=("sel",),
            ),
        ),
    )


def _pair_extract_rule(
    vector_type: TypePattern,
    scalar_type: TypePattern,
    *,
    dynamic: bool,
) -> DescriptorRule:
    descriptor = _descriptor(
        "amd.xdna.aie2p.extract.i64.register"
        if dynamic
        else "amd.xdna.aie2p.extract.i64.immediate"
    )
    if dynamic:
        index_guards = (
            Guard.value_type("indices", _INDEX),
            Guard.operand_segment_count("indices", 1),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range(
                "static_indices",
                element=0,
                minimum=-(2**63),
                maximum=-(2**63),
            ),
        )
        emit = EmitDescriptorOp(
            descriptor=descriptor,
            operands={
                "s1": ValueRef.operand("source"),
                "idx": ValueRef.operand("indices"),
            },
            results={"dst": ValueRef.result("result")},
            form=DescriptorEmitForm.OP,
        )
    else:
        index_guards = (
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range(
                "static_indices", element=0, minimum=0, maximum=7
            ),
        )
        emit = EmitDescriptorOp(
            descriptor=descriptor,
            operands={"s1": ValueRef.operand("source")},
            results={"dst": ValueRef.result("result")},
            immediates={
                "idx": AttrProject.i64_array_element("static_indices", element=0)
            },
            form=DescriptorEmitForm.OP,
        )
    return DescriptorRule(
        source_op=vector.vector_extract,
        descriptor=descriptor,
        guards=(
            Guard.value_type("source", vector_type),
            Guard.value_type("result", scalar_type),
            *index_guards,
        ),
        emit=(emit,),
    )


def _pair_insert_rule(
    scalar_type: TypePattern,
    vector_type: TypePattern,
    *,
    dynamic: bool,
    zero: bool = False,
) -> DescriptorRule:
    descriptor = _descriptor(
        "amd.xdna.aie2p.insert.i64.zero"
        if zero
        else "amd.xdna.aie2p.insert.i64.register"
    )
    index_emits: tuple[EmitDescriptorOp, ...] = ()
    if dynamic:
        index_guards = (
            Guard.value_type("indices", _INDEX),
            Guard.operand_segment_count("indices", 1),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range(
                "static_indices",
                element=0,
                minimum=-(2**63),
                maximum=-(2**63),
            ),
        )
        index = ValueRef.operand("indices")
    elif zero:
        index_guards = (
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range(
                "static_indices", element=0, minimum=0, maximum=0
            ),
        )
        index = None
    else:
        index_guards = (
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range(
                "static_indices", element=0, minimum=1, maximum=7
            ),
        )
        index = ValueRef.temporary("index")
        constant = _descriptor("amd.xdna.aie2p.constant.i32.short")
        index_emits = (
            EmitDescriptorOp(
                descriptor=constant,
                results={"dst": index},
                result_types={"dst": DescriptorResultType()},
                immediates={
                    "i": AttrProject.i64_array_element("static_indices", element=0)
                },
                form=DescriptorEmitForm.CONST,
            ),
        )
    operands = {
        "s1": ValueRef.operand("dest"),
        "src": ValueRef.operand("value"),
    }
    if index is not None:
        operands["idx"] = index
    return DescriptorRule(
        source_op=vector.vector_insert,
        descriptor=descriptor,
        guards=(
            Guard.value_type("value", scalar_type),
            Guard.value_type("dest", vector_type),
            Guard.value_type("result", vector_type),
            *index_guards,
        ),
        emit=(
            *index_emits,
            EmitDescriptorOp(
                descriptor=descriptor,
                operands=operands,
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
                copy_operands=(() if zero else ("idx",)),
            ),
        ),
    )


def _pair_splat_rule(
    scalar_type: TypePattern,
    vector_type: TypePattern,
) -> DescriptorRule:
    descriptor = _descriptor("amd.xdna.aie2p.splat.i64x8")
    return DescriptorRule(
        source_op=vector.vector_splat,
        descriptor=descriptor,
        guards=(
            Guard.value_type("scalar", scalar_type),
            Guard.value_type("result", vector_type),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"src": ValueRef.operand("scalar")},
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


AIE2P_I64_RULES = (
    _pair_constant_rule(
        _I64,
        "i64",
        ValueProject.exact_i64_i32_word("result", word_index=0),
        ValueProject.exact_i64_i32_word("result", word_index=1),
        Guard.value_exact_i64("result"),
    ),
    _pair_constant_rule(
        _F64,
        "f64",
        ValueProject.float_as_f64_i32_word("result", word_index=0),
        ValueProject.float_as_f64_i32_word("result", word_index=1),
        Guard.value_exact_float("result"),
    ),
    *_pair_vector_constant_rules(),
    _pair_vector_binary_rule(vector.vector_andi, "amd.xdna.aie2p.and.bits512"),
    _pair_vector_binary_rule(vector.vector_ori, "amd.xdna.aie2p.or.bits512"),
    _pair_vector_xor_rule(),
    _pair_bitwise_rule(scalar_bitwise.scalar_andi, "and.i32"),
    _pair_bitwise_rule(scalar_bitwise.scalar_ori, "or.i32"),
    _pair_bitwise_rule(scalar_bitwise.scalar_xori, "xor.i32"),
    _pair_add_sub_rule(
        scalar_arithmetic.scalar_addi,
        "add.carry_out.i32",
        "add.carry.i32",
    ),
    _pair_unsigned_remainder_by_word_rule(),
    _pair_multiply_rule(),
    *(_pair_bounded_left_shift_rule(word_index) for word_index in range(2)),
    _pair_left_shift_rule(),
    *(
        _pair_right_shift_rule(source_op, high_shift, count_range)
        for source_op, high_shift in (
            (scalar_bitwise.scalar_shrui, "lshl.i32"),
            (scalar_bitwise.scalar_shrsi, "ashl.i32"),
        )
        for count_range in ((1, 31), (0, 31), (32, 63), None)
    ),
    _pair_add_sub_rule(
        scalar_arithmetic.scalar_subi,
        "sub.borrow_out.i32",
        "sub.borrow.i32",
    ),
    *(
        _pair_compare_rule(predicate)
        for predicate in (
            "eq",
            "ne",
            "slt",
            "sle",
            "sgt",
            "sge",
            "ult",
            "ule",
            "ugt",
            "uge",
        )
    ),
    *(
        _pair_vector_compare_rule(predicate)
        for predicate in (
            "eq",
            "ne",
            "slt",
            "sle",
            "sgt",
            "sge",
            "ult",
            "ule",
            "ugt",
            "uge",
        )
    ),
    _pair_select_rule(_I64),
    _pair_select_rule(_F64),
    *(
        _pair_vector_select_rule(
            element_type,
            minimum_lanes,
            maximum_lanes,
            spread_stages,
        )
        for element_type in ("i64", "f64")
        for minimum_lanes, maximum_lanes, spread_stages in (_PAIR_VECTOR_SELECT_RANGES)
    ),
    *(
        rule
        for vector_type, scalar_type in (
            (_I64_VECTOR, _I64),
            (_F64_VECTOR, _F64),
        )
        for rule in (
            _pair_extract_rule(vector_type, scalar_type, dynamic=False),
            _pair_extract_rule(vector_type, scalar_type, dynamic=True),
            _pair_insert_rule(scalar_type, vector_type, dynamic=False, zero=True),
            _pair_insert_rule(scalar_type, vector_type, dynamic=False),
            _pair_insert_rule(scalar_type, vector_type, dynamic=True),
        )
    ),
    *(
        _pair_splat_rule(scalar_type, vector_type)
        for scalar_type, vector_type in (
            (_I64, _I64_VECTOR),
            (_F64, _F64_VECTOR),
        )
    ),
)
