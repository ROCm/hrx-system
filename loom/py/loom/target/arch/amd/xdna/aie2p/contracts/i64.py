# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar-pair i64 and f64 contracts for AMD XDNA AIE2P."""

from __future__ import annotations

from collections.abc import Sequence

from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scf import defs as scf
from loom.dialect.vector import defs as vector
from loom.dsl import Op
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


def _pair_add_sub_rule(
    source_op: Op,
    low_operation: str,
    high_operation: str,
) -> DescriptorRule:
    lhs_emits, lhs_low, lhs_high = _split_pair(ValueRef.operand("lhs"), "lhs")
    rhs_emits, rhs_low, rhs_high = _split_pair(ValueRef.operand("rhs"), "rhs")
    program = ScalarProgram()
    result_low = program.binary("result_low", low_operation, lhs_low, rhs_low)
    result_high = program.binary("result_high", high_operation, lhs_high, rhs_high)
    return DescriptorRule(
        source_op=source_op,
        descriptor=_descriptor(f"amd.xdna.aie2p.{high_operation}"),
        guards=_typed_guards(("lhs", "rhs", "result"), _I64),
        emit=(
            *lhs_emits,
            *rhs_emits,
            *program.emits,
            _concat_pair(result_low, result_high),
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
    low = ValueRef.temporary("constant_low")
    high = ValueRef.temporary("constant_high")
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", attr_kind),
            Guard.value_type("result", result_type),
            exact_guard,
        ),
        emit=(
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
            _concat_pair(low, high),
        ),
    )


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
    cross_count = program.add_immediate("cross_count", masked_count, -32)
    low_if_small = program.binary("low_if_small", "lshl.i32", lhs_low, masked_count)
    high_base = program.binary("high_base", "lshl.i32", lhs_high, masked_count)
    low_cross = program.binary("low_cross", "lshl.i32", lhs_low, cross_count)
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
    _pair_bitwise_rule(scalar_bitwise.scalar_andi, "and.i32"),
    _pair_bitwise_rule(scalar_bitwise.scalar_ori, "or.i32"),
    _pair_bitwise_rule(scalar_bitwise.scalar_xori, "xor.i32"),
    _pair_add_sub_rule(
        scalar_arithmetic.scalar_addi,
        "add.i32",
        "add.carry.i32",
    ),
    _pair_multiply_rule(),
    _pair_left_shift_rule(),
    _pair_add_sub_rule(
        scalar_arithmetic.scalar_subi,
        "sub.i32",
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
    _pair_select_rule(_I64),
    _pair_select_rule(_F64),
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
