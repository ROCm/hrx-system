# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar integer and bounded bitstream instructions."""

from __future__ import annotations

import enum
from typing import NamedTuple

from iree.vm.bytecode.spec.isa import (
    FailureCase,
    FieldRole,
    FieldRuleUse,
    Instruction,
    InstructionFamily,
    InstructionField,
    RecordRule,
)
from iree.vm.bytecode.spec.isa.core.rules import FieldRule, RecordRuleKind
from iree.vm.bytecode.spec.schema import (
    I16,
    U8,
    Field,
    NumericKind,
    NumericTable,
    NumericValue,
)
from iree.vm.bytecode.spec.version import CORE_0


class IntegerBinaryOperation(enum.Enum):
    ADD = "add"
    SUB = "sub"
    MUL = "mul"


class IntegerBinarySemantics(NamedTuple):
    operation: IntegerBinaryOperation
    bit_width: int


class _BitstreamOperation(enum.Enum):
    PACK = "pack"
    UNPACK_UNSIGNED = "unpack.u"
    UNPACK_SIGNED = "unpack.s"


class _BitstreamShape(enum.IntEnum):
    PACK = 0
    UNPACK = 1


INTEGER_COMPARE_SELECTOR = NumericTable(
    "integer.compare",
    U8,
    NumericKind.SELECTOR,
    tuple(
        NumericValue(name, value, CORE_0, summary)
        for name, value, summary in (
            ("eq", 0, "True when both selected-width bit patterns are equal."),
            ("ne", 1, "True when both selected-width bit patterns differ."),
            ("slt", 2, "True when left is signed less than right."),
            ("sle", 3, "True when left is signed less than or equal to right."),
            ("sgt", 4, "True when left is signed greater than right."),
            ("sge", 5, "True when left is signed greater than or equal to right."),
            ("ult", 6, "True when left is unsigned less than right."),
            ("ule", 7, "True when left is unsigned less than or equal to right."),
            ("ugt", 8, "True when left is unsigned greater than right."),
            ("uge", 9, "True when left is unsigned greater than or equal to right."),
        )
    ),
    CORE_0,
    "Selects equality, signed-order, or unsigned-order comparison.",
)


INTEGER_FAMILY = InstructionFamily(
    name="integer",
    since=CORE_0,
    summary="Scalar integer and bounded bitstream operations.",
    contract=(
        "Integer operations interpret the low 32 bits or complete 64 bits as exact "
        "two's-complement bit patterns; signedness belongs to an operation, not a "
        "register. Every i32, s32, or u32 result clears the high cell half and every "
        "boolean result is canonical complete-cell zero or one. Ordinary arithmetic "
        "wraps modulo its width. Implementations use unsigned host arithmetic, guarded "
        "signed division, explicit sign extension, and total shift helpers so no "
        "architectural result inherits host signed overflow, invalid-shift, or "
        "implementation-defined right-shift behavior. Dynamic shift and rotate counts "
        "are reduced modulo width. Every operation reads all inputs before publishing "
        "a result, permitting arbitrary source and destination aliasing. Except for "
        "division by zero and signed quotient overflow, these operations are infallible "
        "after verification. They have no ref ownership effect and never suspend."
    ),
)


def _field(
    name: str,
    encoding,
    summary: str,
    role: FieldRole,
    rule,
    *,
    element_count: int = 1,
) -> InstructionField:
    if not isinstance(rule, FieldRuleUse):
        rule = FieldRuleUse(rule)
    return InstructionField(Field(name, encoding, summary, element_count), role, rule)


def _value(name: str, role: FieldRole, summary: str) -> InstructionField:
    return _field(name, U8, summary, role, FieldRule.REGISTER_VALUE)


def _padding(*, element_count: int = 1) -> InstructionField:
    return _field(
        "zero_padding_u8",
        U8,
        "Canonical zero padding.",
        FieldRole.PADDING,
        FieldRule.ZERO,
        element_count=element_count,
    )


def _binary_fields() -> tuple[InstructionField, ...]:
    return (
        _value(
            "destination_v8",
            FieldRole.RESULT,
            "Value-register ordinal receiving the result.",
        ),
        _value("left_v8", FieldRole.OPERAND, "Left value-register ordinal."),
        _value("right_v8", FieldRole.OPERAND, "Right value-register ordinal."),
    )


def _result_contract(bit_width: int, result: str) -> str:
    suffix = " and clears the high cell half." if bit_width == 32 else "."
    return f"{result}{suffix}"


_SOURCE_BINARY_OPERATION = {
    "integer.add": IntegerBinaryOperation.ADD,
    "integer.sub": IntegerBinaryOperation.SUB,
    "integer.mul": IntegerBinaryOperation.MUL,
}


class _BinaryDefinition(NamedTuple):
    opcode: int
    mnemonic: str
    summary: str
    result: str
    expression: str


def _binary(definition: _BinaryDefinition) -> Instruction:
    opcode, mnemonic, summary, result, expression = definition
    bit_width = 32 if mnemonic.endswith("32") else 64
    operation = _SOURCE_BINARY_OPERATION.get(mnemonic.rsplit(".", 1)[0])
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=INTEGER_FAMILY,
        summary=summary,
        fields=_binary_fields(),
        semantics=(
            IntegerBinarySemantics(operation, bit_width)
            if operation is not None
            else None
        ),
        behavior=(
            f"Reads both operands before computing the selected {bit_width}-bit result."
        ),
        success=(result,),
        assembly=f"%v<destination> = {mnemonic} %v<left>, %v<right>",
        pseudocode=(
            f"left = read_w(left_v8, {bit_width});\n"
            f"right = read_w(right_v8, {bit_width});\n"
            f"write_w(destination_v8, {expression}, {bit_width});\n"
            "pc = pc + 4;"
        ),
    )


_REGULAR_BINARY_DEFINITIONS = (
    _BinaryDefinition(
        0x40,
        "integer.add.i32",
        "Adds low 32-bit patterns modulo 2^32.",
        "destination_v8 receives the low 32 sum bits and clears its high half.",
        "left + right",
    ),
    _BinaryDefinition(
        0x41,
        "integer.add.i64",
        "Adds complete 64-bit patterns modulo 2^64.",
        "destination_v8 receives the low 64 sum bits.",
        "left + right",
    ),
    _BinaryDefinition(
        0x42,
        "integer.sub.i32",
        "Subtracts low 32-bit patterns modulo 2^32.",
        "destination_v8 receives the low 32 difference bits and clears its high half.",
        "left - right",
    ),
    _BinaryDefinition(
        0x43,
        "integer.sub.i64",
        "Subtracts complete 64-bit patterns modulo 2^64.",
        "destination_v8 receives the low 64 difference bits.",
        "left - right",
    ),
    _BinaryDefinition(
        0x44,
        "integer.mul.i32",
        "Multiplies low 32-bit patterns and retains the low 32 product bits.",
        "destination_v8 receives the low 32 product bits and clears its high half.",
        "left * right",
    ),
    _BinaryDefinition(
        0x45,
        "integer.mul.i64",
        "Multiplies complete 64-bit patterns and retains the low 64 product bits.",
        "destination_v8 receives the low 64 product bits.",
        "left * right",
    ),
    _BinaryDefinition(
        0x52,
        "integer.min.s32",
        "Selects the lesser signed two's-complement 32-bit operand.",
        "destination_v8 receives the selected low bits and clears its high half.",
        "signed_w(left, 32) <= signed_w(right, 32) ? left : right",
    ),
    _BinaryDefinition(
        0x53,
        "integer.min.s64",
        "Selects the lesser signed two's-complement 64-bit operand.",
        "destination_v8 receives the selected complete operand bits.",
        "signed_w(left, 64) <= signed_w(right, 64) ? left : right",
    ),
    _BinaryDefinition(
        0x54,
        "integer.min.u32",
        "Selects the lesser unsigned 32-bit operand.",
        "destination_v8 receives the selected low bits and clears its high half.",
        "left <= right ? left : right",
    ),
    _BinaryDefinition(
        0x55,
        "integer.min.u64",
        "Selects the lesser unsigned 64-bit operand.",
        "destination_v8 receives the selected complete operand bits.",
        "left <= right ? left : right",
    ),
    _BinaryDefinition(
        0x56,
        "integer.max.s32",
        "Selects the greater signed two's-complement 32-bit operand.",
        "destination_v8 receives the selected low bits and clears its high half.",
        "signed_w(left, 32) >= signed_w(right, 32) ? left : right",
    ),
    _BinaryDefinition(
        0x57,
        "integer.max.s64",
        "Selects the greater signed two's-complement 64-bit operand.",
        "destination_v8 receives the selected complete operand bits.",
        "signed_w(left, 64) >= signed_w(right, 64) ? left : right",
    ),
    _BinaryDefinition(
        0x58,
        "integer.max.u32",
        "Selects the greater unsigned 32-bit operand.",
        "destination_v8 receives the selected low bits and clears its high half.",
        "left >= right ? left : right",
    ),
    _BinaryDefinition(
        0x59,
        "integer.max.u64",
        "Selects the greater unsigned 64-bit operand.",
        "destination_v8 receives the selected complete operand bits.",
        "left >= right ? left : right",
    ),
    _BinaryDefinition(
        0x5A,
        "integer.and.i32",
        "Computes bitwise AND over the low 32 bits.",
        "destination_v8 receives the AND bits and clears its high half.",
        "left & right",
    ),
    _BinaryDefinition(
        0x5B,
        "integer.and.i64",
        "Computes bitwise AND over all 64 bits.",
        "destination_v8 receives the complete 64-bit AND result.",
        "left & right",
    ),
    _BinaryDefinition(
        0x5C,
        "integer.or.i32",
        "Computes bitwise OR over the low 32 bits.",
        "destination_v8 receives the OR bits and clears its high half.",
        "left | right",
    ),
    _BinaryDefinition(
        0x5D,
        "integer.or.i64",
        "Computes bitwise OR over all 64 bits.",
        "destination_v8 receives the complete 64-bit OR result.",
        "left | right",
    ),
    _BinaryDefinition(
        0x5E,
        "integer.xor.i32",
        "Computes bitwise XOR over the low 32 bits.",
        "destination_v8 receives the XOR bits and clears its high half.",
        "left ^ right",
    ),
    _BinaryDefinition(
        0x5F,
        "integer.xor.i64",
        "Computes bitwise XOR over all 64 bits.",
        "destination_v8 receives the complete 64-bit XOR result.",
        "left ^ right",
    ),
    _BinaryDefinition(
        0x60,
        "integer.shift.left.i32",
        "Shifts low 32 bits left by the count's low five bits.",
        "destination_v8 receives the shifted bits and clears its high half.",
        "left << (right & 31)",
    ),
    _BinaryDefinition(
        0x61,
        "integer.shift.left.i64",
        "Shifts all 64 bits left by the count's low six bits.",
        "destination_v8 receives the complete shifted result.",
        "left << (right & 63)",
    ),
    _BinaryDefinition(
        0x62,
        "integer.shift.right.s32",
        "Sign-fills low 32 bits right by the count's low five bits.",
        "destination_v8 receives the arithmetic shift and clears its high half.",
        "asr_w(left, right & 31, 32)",
    ),
    _BinaryDefinition(
        0x63,
        "integer.shift.right.s64",
        "Sign-fills all 64 bits right by the count's low six bits.",
        "destination_v8 receives the complete arithmetic shift.",
        "asr_w(left, right & 63, 64)",
    ),
    _BinaryDefinition(
        0x64,
        "integer.shift.right.u32",
        "Logically shifts low 32 bits right by the count's low five bits.",
        "destination_v8 receives the logical shift and clears its high half.",
        "left >> (right & 31)",
    ),
    _BinaryDefinition(
        0x65,
        "integer.shift.right.u64",
        "Logically shifts all 64 bits right by the count's low six bits.",
        "destination_v8 receives the complete logical shift.",
        "left >> (right & 63)",
    ),
    _BinaryDefinition(
        0x66,
        "integer.rotate.left.i32",
        "Rotates low 32 bits left by the count's low five bits.",
        "destination_v8 receives the rotated bits and clears its high half.",
        "rotl_w(left, right & 31, 32)",
    ),
    _BinaryDefinition(
        0x67,
        "integer.rotate.left.i64",
        "Rotates all 64 bits left by the count's low six bits.",
        "destination_v8 receives the complete rotated result.",
        "rotl_w(left, right & 63, 64)",
    ),
    _BinaryDefinition(
        0x68,
        "integer.rotate.right.i32",
        "Rotates low 32 bits right by the count's low five bits.",
        "destination_v8 receives the rotated bits and clears its high half.",
        "rotr_w(left, right & 31, 32)",
    ),
    _BinaryDefinition(
        0x69,
        "integer.rotate.right.i64",
        "Rotates all 64 bits right by the count's low six bits.",
        "destination_v8 receives the complete rotated result.",
        "rotr_w(left, right & 63, 64)",
    ),
)

_REGULAR_BINARY_INSTRUCTIONS = tuple(map(_binary, _REGULAR_BINARY_DEFINITIONS))


class _DivisionKind(enum.Enum):
    SIGNED_QUOTIENT = ("div", "s")
    UNSIGNED_QUOTIENT = ("div", "u")
    SIGNED_REMAINDER = ("rem", "s")
    UNSIGNED_REMAINDER = ("rem", "u")


def _division(opcode: int, kind: _DivisionKind, bit_width: int) -> Instruction:
    operation, signedness = kind.value
    is_signed = signedness == "s"
    is_remainder = operation == "rem"
    mnemonic = f"integer.{operation}.{signedness}{bit_width}"
    result_name = "remainder" if is_remainder else "quotient"
    interpretation = "signed" if is_signed else "unsigned"
    read = (
        f"signed_w(read_w({{register}}, {bit_width}), {bit_width})"
        if is_signed
        else f"read_w({{register}}, {bit_width})"
    )
    if is_remainder:
        expression = (
            f"left == INT{bit_width}_MIN && right == -1\n"
            "    ? 0 : signed_remainder(left, right)"
            if is_signed
            else "left % right"
        )
    else:
        expression = (
            "signed_quotient_toward_zero(left, right)" if is_signed else "left / right"
        )
    if is_signed and is_remainder:
        behavior = (
            "Interprets both operands as signed two's-complement values. The "
            f"remainder has the dividend's sign, and INT{bit_width}_MIN % -1 is "
            "exactly zero."
        )
    elif is_signed:
        behavior = (
            "Interprets both operands as signed two's-complement values and rounds "
            "the quotient toward zero."
        )
    else:
        behavior = f"Interprets both operands as unsigned {bit_width}-bit values."
    failures = [
        FailureCase(
            "invalid_argument",
            "right_v8 is zero at the selected width.",
            "destination_v8 and all other VM state remain unchanged.",
        )
    ]
    if is_signed and not is_remainder:
        failures.append(
            FailureCase(
                "out_of_range",
                f"The operands are INT{bit_width}_MIN and -1.",
                "destination_v8 and all other VM state remain unchanged.",
            )
        )
    overflow_check = (
        f"if (left == INT{bit_width}_MIN && right == -1)\n"
        "  fail(out_of_range, no_message);\n"
        if is_signed and not is_remainder
        else ""
    )
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=INTEGER_FAMILY,
        summary=f"Computes the {interpretation} {bit_width}-bit {result_name}.",
        fields=_binary_fields(),
        semantics=None,
        behavior=behavior,
        success=(
            _result_contract(
                bit_width,
                f"destination_v8 receives the selected-width {result_name}",
            ),
        ),
        assembly=f"%v<destination> = {mnemonic} %v<left>, %v<right>",
        pseudocode=(
            f"left = {read.format(register='left_v8')};\n"
            f"right = {read.format(register='right_v8')};\n"
            "if (right == 0) fail(invalid_argument, no_message);\n"
            f"{overflow_check}"
            f"write_w(destination_v8, {expression}, {bit_width});\n"
            "pc = pc + 4;"
        ),
        preconditions=(
            "The selected-width divisor is nonzero."
            + (
                f" Signed quotient also excludes INT{bit_width}_MIN / -1."
                if is_signed and not is_remainder
                else ""
            ),
        ),
        failures=tuple(failures),
    )


_DIVISION_INSTRUCTIONS = tuple(
    _division(opcode, kind, bit_width)
    for opcode, kind, bit_width in (
        (0x46, _DivisionKind.SIGNED_QUOTIENT, 32),
        (0x47, _DivisionKind.SIGNED_QUOTIENT, 64),
        (0x48, _DivisionKind.UNSIGNED_QUOTIENT, 32),
        (0x49, _DivisionKind.UNSIGNED_QUOTIENT, 64),
        (0x4A, _DivisionKind.SIGNED_REMAINDER, 32),
        (0x4B, _DivisionKind.SIGNED_REMAINDER, 64),
        (0x4C, _DivisionKind.UNSIGNED_REMAINDER, 32),
        (0x4D, _DivisionKind.UNSIGNED_REMAINDER, 64),
    )
)


class _UnaryDefinition(NamedTuple):
    opcode: int
    mnemonic: str
    summary: str
    result: str
    expression: str


def _unary(definition: _UnaryDefinition) -> Instruction:
    opcode, mnemonic, summary, result, expression = definition
    bit_width = 32 if mnemonic.endswith("32") else 64
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=INTEGER_FAMILY,
        summary=summary,
        fields=(
            _value(
                "destination_v8",
                FieldRole.RESULT,
                "Value-register ordinal receiving the result.",
            ),
            _value("source_v8", FieldRole.OPERAND, "Source value-register ordinal."),
            _padding(),
        ),
        semantics=None,
        behavior=f"Reads source_v8 and computes the selected {bit_width}-bit result.",
        success=(result,),
        assembly=f"%v<destination> = {mnemonic} %v<source>",
        pseudocode=(
            f"bits = read_w(source_v8, {bit_width});\n"
            f"write_w(destination_v8, {expression}, {bit_width});\n"
            "pc = pc + 4;"
        ),
    )


_UNARY_DEFINITIONS = (
    _UnaryDefinition(
        0x4E,
        "integer.neg.i32",
        "Computes two's-complement negation modulo 2^32.",
        "destination_v8 receives the modular negation and clears its high half.",
        "0 - bits",
    ),
    _UnaryDefinition(
        0x4F,
        "integer.neg.i64",
        "Computes two's-complement negation modulo 2^64.",
        "destination_v8 receives the complete modular negation.",
        "0 - bits",
    ),
    _UnaryDefinition(
        0x50,
        "integer.abs.s32",
        "Computes modular signed 32-bit absolute value.",
        "destination_v8 receives abs(source); INT32_MIN retains its bits.",
        "bit_is_set(bits, 31) ? 0 - bits : bits",
    ),
    _UnaryDefinition(
        0x51,
        "integer.abs.s64",
        "Computes modular signed 64-bit absolute value.",
        "destination_v8 receives abs(source); INT64_MIN retains its bits.",
        "bit_is_set(bits, 63) ? 0 - bits : bits",
    ),
    _UnaryDefinition(
        0x6A,
        "integer.count.leading.zeros.i32",
        "Counts leading zeroes in the low 32 bits.",
        "destination_v8 receives 32 for zero or the exact count with high bits clear.",
        "bits == 0 ? 32 : count_leading_zeros(bits, 32)",
    ),
    _UnaryDefinition(
        0x6B,
        "integer.count.leading.zeros.i64",
        "Counts leading zeroes in all 64 bits.",
        "destination_v8 receives 64 for zero or the exact count.",
        "bits == 0 ? 64 : count_leading_zeros(bits, 64)",
    ),
    _UnaryDefinition(
        0x6C,
        "integer.count.trailing.zeros.i32",
        "Counts trailing zeroes in the low 32 bits.",
        "destination_v8 receives 32 for zero or the exact count with high bits clear.",
        "bits == 0 ? 32 : count_trailing_zeros(bits, 32)",
    ),
    _UnaryDefinition(
        0x6D,
        "integer.count.trailing.zeros.i64",
        "Counts trailing zeroes in all 64 bits.",
        "destination_v8 receives 64 for zero or the exact count.",
        "bits == 0 ? 64 : count_trailing_zeros(bits, 64)",
    ),
    _UnaryDefinition(
        0x6E,
        "integer.popcount.i32",
        "Counts one bits in the low 32 bits.",
        "destination_v8 receives the population count with high bits clear.",
        "population_count(bits, 32)",
    ),
    _UnaryDefinition(
        0x6F,
        "integer.popcount.i64",
        "Counts one bits in all 64 bits.",
        "destination_v8 receives the population count.",
        "population_count(bits, 64)",
    ),
)

_UNARY_INSTRUCTIONS = tuple(map(_unary, _UNARY_DEFINITIONS))


def _compare(opcode: int, bit_width: int) -> Instruction:
    mnemonic = f"integer.compare.i{bit_width}"
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=INTEGER_FAMILY,
        summary=(
            f"Evaluates one equality, signed-order, or unsigned-order i{bit_width} "
            "predicate."
        ),
        fields=(
            *_binary_fields(),
            _field(
                "predicate_u8",
                U8,
                "Closed integer comparison selector.",
                FieldRole.IMMEDIATE,
                FieldRuleUse(FieldRule.SELECTOR, data=INTEGER_COMPARE_SELECTOR),
            ),
            _padding(element_count=3),
        ),
        semantics=None,
        behavior=(
            f"Compares the selected low {bit_width} bits using equality, signed "
            "ordering, or unsigned ordering selected by predicate_u8."
        ),
        success=(
            "destination_v8 receives canonical complete-cell zero or one for the "
            "selected predicate.",
        ),
        assembly=f"%v<destination> = {mnemonic} %v<left>, %v<right> {{predicate}}",
        pseudocode=(
            f"left = read_w(left_v8, {bit_width});\n"
            f"right = read_w(right_v8, {bit_width});\n"
            "values[destination_v8] = canonical_bool(evaluate_integer_predicate(\n"
            f"    predicate_u8, left, right, {bit_width}));\n"
            "pc = pc + 8;"
        ),
    )


_COMPARE_INSTRUCTIONS = (_compare(0x70, 32), _compare(0x71, 64))


def _lea(opcode: int, bit_width: int) -> Instruction:
    mnemonic = f"integer.lea.i{bit_width}"
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=INTEGER_FAMILY,
        summary=(
            f"Combines base, scaled index, and signed offset modulo 2^{bit_width}."
        ),
        fields=(
            _value(
                "destination_v8",
                FieldRole.RESULT,
                "Value-register ordinal receiving the result.",
            ),
            _value("base_v8", FieldRole.OPERAND, "Base value-register ordinal."),
            _value("index_v8", FieldRole.OPERAND, "Index value-register ordinal."),
            _field(
                "scale_u8",
                U8,
                "Arbitrary unsigned scale including zero and non-powers-of-two.",
                FieldRole.IMMEDIATE,
                FieldRule.ANY_BITS,
            ),
            _padding(),
            _field(
                "offset_i16",
                I16,
                "Signed little-endian two's-complement affine offset.",
                FieldRole.IMMEDIATE,
                FieldRule.ANY_BITS,
            ),
        ),
        semantics=None,
        behavior=(
            f"Computes base + index * scale_u8 + sign_extend(offset_i16) modulo "
            f"2^{bit_width}. It creates no pointer authority and reports no arithmetic "
            "overflow."
        ),
        success=(
            _result_contract(
                bit_width,
                f"destination_v8 receives the modulo-2^{bit_width} affine result",
            ),
        ),
        assembly=(
            f"%v<destination> = {mnemonic} %v<base>, %v<index> {{scale, offset}}"
        ),
        pseudocode=(
            f"base = read_w(base_v8, {bit_width});\n"
            f"index = read_w(index_v8, {bit_width});\n"
            "offset = sext_u64(load_le(16, &record[6]), 16);\n"
            "write_w(destination_v8, base + index * scale_u8 + offset, "
            f"{bit_width});\n"
            "pc = pc + 8;"
        ),
    )


_LEA_INSTRUCTIONS = (_lea(0x72, 32), _lea(0x73, 64))


def _ceildiv(opcode: int, bit_width: int) -> Instruction:
    mnemonic = f"integer.ceildiv.pow2.u{bit_width}"
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=INTEGER_FAMILY,
        summary=(
            f"Computes exact unsigned u{bit_width} ceiling division by a power of two."
        ),
        fields=(
            _value(
                "destination_v8",
                FieldRole.RESULT,
                "Value-register ordinal receiving the result.",
            ),
            _value("source_v8", FieldRole.OPERAND, "Source value-register ordinal."),
            _field(
                "log2_u8",
                U8,
                "Base-two logarithm of the nonzero divisor.",
                FieldRole.IMMEDIATE,
                FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(0, bit_width - 1)),
            ),
        ),
        semantics=None,
        behavior=(
            f"Computes exact mathematical ceil(unsigned(source) / 2^log2) over the "
            f"complete u{bit_width} domain without a potentially overflowing addition."
        ),
        success=(
            _result_contract(
                bit_width,
                f"destination_v8 receives the exact u{bit_width} quotient",
            ),
        ),
        assembly=f"%v<destination> = {mnemonic} %v<source> {{log2}}",
        pseudocode=(
            f"source = read_w(source_v8, {bit_width});\n"
            "mask = low_mask(log2_u8);\n"
            "result = (source >> log2_u8) + ((source & mask) != 0 ? 1 : 0);\n"
            f"write_w(destination_v8, result, {bit_width});\n"
            "pc = pc + 4;"
        ),
    )


_CEILDIV_INSTRUCTIONS = (_ceildiv(0x74, 32), _ceildiv(0x75, 64))

INTEGER_BITSTREAM_MAXIMUM_BIT_COUNT = 64
_INTEGER_CARRIER_WIDTHS = (8, 16, 32, 64)


def _bitstream(opcode: int, operation: _BitstreamOperation) -> Instruction:
    mnemonic = f"integer.bitstream.{operation.value}"
    is_pack = operation == _BitstreamOperation.PACK
    shape = _BitstreamShape.PACK if is_pack else _BitstreamShape.UNPACK
    if is_pack:
        summary = "Packs low-width source fields into consecutive result carriers."
        behavior = (
            "Concatenates each source carrier's low field_width_u8 bits into one "
            "at-most-64-bit stream and writes consecutive result_width_u8 carriers. "
            "Source bits above field_width_u8 are ignored."
        )
        shape_contract = (
            "field_width_u8 is no greater than source_width_u8 and source_count_u8 * "
            "field_width_u8 equals result_count_u8 * result_width_u8 at no more than "
            "64 bits."
        )
        capture_loop = (
            "field = read_carrier(source_base_v8 + i, source_width_u8) &\n"
            "      low_mask(field_width_u8);\n"
            "  stream |= field << (i * field_width_u8);"
        )
        result_loop = (
            "lane = (stream >> (i * result_width_u8)) &\n"
            "      low_mask(result_width_u8);\n"
            "  write_carrier(result_base_v8 + i, lane, result_width_u8);"
        )
    else:
        extension = (
            "sign-extends"
            if operation == _BitstreamOperation.UNPACK_SIGNED
            else "zero-extends"
        )
        summary = f"Unpacks consecutive fields and {extension} each result."
        behavior = (
            "Concatenates complete low-width source carriers, extracts consecutive "
            f"fields, and {extension} each into a result carrier."
        )
        shape_contract = (
            "field_width_u8 is no greater than result_width_u8 and source_count_u8 * "
            "source_width_u8 equals result_count_u8 * field_width_u8 at no more than "
            "64 bits."
        )
        capture_loop = (
            "lane = read_carrier(source_base_v8 + i, source_width_u8);\n"
            "  stream |= lane << (i * source_width_u8);"
        )
        if operation == _BitstreamOperation.UNPACK_SIGNED:
            result_loop = (
                "field = (stream >> (i * field_width_u8)) &\n"
                "      low_mask(field_width_u8);\n"
                "  extended = sign_extend_within(\n"
                "      field, field_width_u8, result_width_u8);\n"
                "  write_carrier(result_base_v8 + i, extended, result_width_u8);"
            )
        else:
            result_loop = (
                "field = (stream >> (i * field_width_u8)) &\n"
                "      low_mask(field_width_u8);\n"
                "  write_carrier(result_base_v8 + i, field, result_width_u8);"
            )
    fields = (
        _value(
            "result_base_v8",
            FieldRole.RESULT,
            "Base of a nonempty consecutive result-register range.",
        ),
        _value(
            "source_base_v8",
            FieldRole.OPERAND,
            "Base of a nonempty consecutive source-register range.",
        ),
        _field(
            "field_width_u8",
            U8,
            "Logical field width in bits.",
            FieldRole.IMMEDIATE,
            FieldRuleUse(
                FieldRule.ALLOWED_RANGE,
                values=(1, INTEGER_BITSTREAM_MAXIMUM_BIT_COUNT),
            ),
        ),
        _field(
            "source_count_u8",
            U8,
            "Nonzero consecutive source-register count.",
            FieldRole.IMMEDIATE,
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(1, 255)),
        ),
        _field(
            "result_count_u8",
            U8,
            "Nonzero consecutive result-register count.",
            FieldRole.IMMEDIATE,
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(1, 255)),
        ),
        _field(
            "source_width_u8",
            U8,
            "Source carrier width in bits.",
            FieldRole.IMMEDIATE,
            FieldRuleUse(FieldRule.ALLOWED_VALUES, values=_INTEGER_CARRIER_WIDTHS),
        ),
        _field(
            "result_width_u8",
            U8,
            "Result carrier width in bits.",
            FieldRole.IMMEDIATE,
            FieldRuleUse(FieldRule.ALLOWED_VALUES, values=_INTEGER_CARRIER_WIDTHS),
        ),
    )
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=INTEGER_FAMILY,
        summary=summary,
        fields=fields,
        semantics=None,
        behavior=behavior,
        success=(
            "Source element zero occupies the least-significant stream region. The "
            "complete source stream is captured before any result write, so source "
            "and result ranges may overlap arbitrarily. Every result cell clears bits "
            "above its declared carrier width.",
        ),
        assembly=(
            f"%v<result_base>... = {mnemonic} %v<source_base>... "
            "{field_width, source_count, result_count, source_width, result_width}"
        ),
        pseudocode=(
            "stream = 0;\n"
            "for (i = 0; i < source_count_u8; ++i) {\n"
            f"  {capture_loop}\n"
            "}\n"
            "for (i = 0; i < result_count_u8; ++i) {\n"
            f"  {result_loop}\n"
            "}\n"
            "pc = pc + 8;"
        ),
        rules=(
            RecordRule(
                RecordRuleKind.VALUE_REGISTER_RANGE,
                ("result_base_v8", "result_count_u8"),
                summary="The complete nonempty result-register range must fit.",
            ),
            RecordRule(
                RecordRuleKind.VALUE_REGISTER_RANGE,
                ("source_base_v8", "source_count_u8"),
                summary="The complete nonempty source-register range must fit.",
            ),
            RecordRule(
                RecordRuleKind.INTEGER_BITSTREAM_SHAPE,
                (
                    "field_width_u8",
                    "source_count_u8",
                    "result_count_u8",
                    "source_width_u8",
                    "result_width_u8",
                ),
                values=(shape, INTEGER_BITSTREAM_MAXIMUM_BIT_COUNT),
                summary=shape_contract,
            ),
        ),
    )


_BITSTREAM_INSTRUCTIONS = (
    _bitstream(0x76, _BitstreamOperation.PACK),
    _bitstream(0x77, _BitstreamOperation.UNPACK_UNSIGNED),
    _bitstream(0x78, _BitstreamOperation.UNPACK_SIGNED),
)

_INSTRUCTIONS_BY_OPCODE = {
    instruction.opcode: instruction
    for instruction in (
        *_REGULAR_BINARY_INSTRUCTIONS,
        *_DIVISION_INSTRUCTIONS,
        *_UNARY_INSTRUCTIONS,
        *_COMPARE_INSTRUCTIONS,
        *_LEA_INSTRUCTIONS,
        *_CEILDIV_INSTRUCTIONS,
        *_BITSTREAM_INSTRUCTIONS,
    )
}
INTEGER_INSTRUCTIONS = tuple(
    _INSTRUCTIONS_BY_OPCODE[opcode] for opcode in sorted(_INSTRUCTIONS_BY_OPCODE)
)
