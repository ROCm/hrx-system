# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core 0.0 integer instructions."""

from __future__ import annotations

from model.isa import (
    FailureCase,
    Instruction,
    InstructionFamily,
    InstructionFieldRole,
    InstructionSemantics,
)
from model.isa.declarations import (
    core_instruction,
    instruction_field,
    value_register,
    zero_padding,
)
from model.isa.selectors import SELECTOR_TABLES_BY_NAME
from model.isa.validation import (
    ALLOWED_RANGE,
    ALLOWED_VALUES,
    ANY_BITS,
    INTEGER_BITSTREAM_SHAPE,
    SELECTOR,
    VALUE_REGISTER_RANGE,
)
from model.schema import I16, U8, EntityReference, FieldReference, RuleUse
from model.specification import CORE_0

FAMILY = InstructionFamily(
    entity_id="core.family.integer",
    since=CORE_0,
    summary="Wrapping scalar integer and bounded bitstream operations.",
    dependencies=("core.contract.machine",),
    document_order=8,
    normative_text=(
        "Integer operations interpret low 32 bits or complete 64 bits as exact "
        "two's-complement bit patterns; signedness belongs to an operation, not "
        "a register. Every i32/s32/u32 result clears the high cell half and "
        "every boolean is complete-cell zero or one. Ordinary arithmetic wraps "
        "modulo its width. Implementations use unsigned host arithmetic, guarded "
        "signed division, explicit sign extension, and total shift helpers: no "
        "architectural result inherits C signed overflow, invalid-shift, or "
        "implementation-defined right-shift behavior. Dynamic shift/rotate "
        "counts are reduced modulo width. All operands are read before any "
        "destination is published, permitting arbitrary source/destination "
        "aliasing. Except for division by zero and signed quotient overflow, "
        "verified operations are infallible. No operation accesses refs or suspends."
    ),
)


def _value(name: str, offset: int, role: InstructionFieldRole):
    return value_register(
        name,
        offset,
        role,
        f"{role.value.replace('_', ' ').capitalize()} value-register ordinal.",
    )


def _common_semantics(
    *,
    description: str,
    verification: tuple[str, ...],
    success: tuple[str, ...],
    assembly: tuple[str, ...],
    pseudocode: str,
    byte_length: int,
    preconditions: tuple[str, ...] = (),
    failures: tuple[FailureCase, ...] = (),
):
    return InstructionSemantics(
        description=description,
        verification=verification,
        preconditions=preconditions,
        success=(
            *success,
            f"The program counter advances by {byte_length} bytes.",
        ),
        failures=failures,
        ownership=(
            "All inputs are read before destination publication; no ref "
            "ownership state is accessed or changed.",
        ),
        assembly=assembly,
        pseudocode=pseudocode,
    )


def _binary(
    *,
    opcode: int,
    mnemonic: str,
    description: str,
    result: str,
    expression: str,
) -> Instruction:
    width = 32 if mnemonic.endswith("32") else 64
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=description,
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=4,
        family_id=FAMILY.entity_id,
        fields=(
            _value("dst_v8", 1, InstructionFieldRole.RESULT),
            _value("lhs_v8", 2, InstructionFieldRole.OPERAND),
            _value("rhs_v8", 3, InstructionFieldRole.OPERAND),
        ),
        semantics=_common_semantics(
            description=description,
            verification=(
                "dst_v8, lhs_v8, and rhs_v8 must be valid value-register ordinals.",
            ),
            success=(result,),
            assembly=(f"%v<dst> = {mnemonic} %v<lhs>, %v<rhs>",),
            pseudocode=(
                f"lhs = read_w(lhs_v8, {width});\n"
                f"rhs = read_w(rhs_v8, {width});\n"
                f"write_w(dst_v8, {expression}, {width});\n"
                "pc = pc + 4;"
            ),
            byte_length=4,
        ),
    )


_REGULAR_BINARY_DEFINITIONS = (
    (
        0x40,
        "integer.add.i32",
        "Adds low 32-bit patterns modulo 2^32.",
        "dst_v8 receives the low 32 sum bits and clears its high 32 bits.",
        "lhs + rhs",
    ),
    (
        0x41,
        "integer.add.i64",
        "Adds complete 64-bit patterns modulo 2^64.",
        "dst_v8 receives the low 64 sum bits.",
        "lhs + rhs",
    ),
    (
        0x42,
        "integer.sub.i32",
        "Subtracts low 32-bit patterns modulo 2^32.",
        "dst_v8 receives the low 32 difference bits and clears its high half.",
        "lhs - rhs",
    ),
    (
        0x43,
        "integer.sub.i64",
        "Subtracts complete 64-bit patterns modulo 2^64.",
        "dst_v8 receives the low 64 difference bits.",
        "lhs - rhs",
    ),
    (
        0x44,
        "integer.mul.i32",
        "Multiplies low 32-bit patterns and retains the low 32 product bits.",
        "dst_v8 receives the low 32 product bits and clears its high half.",
        "lhs * rhs",
    ),
    (
        0x45,
        "integer.mul.i64",
        "Multiplies complete 64-bit patterns and retains the low 64 product bits.",
        "dst_v8 receives the low 64 product bits.",
        "lhs * rhs",
    ),
    (
        0x52,
        "integer.min.s32",
        "Selects the lesser signed two's-complement 32-bit operand.",
        "dst_v8 receives the selected low 32 bits and clears its high half.",
        "signed_w(lhs, 32) <= signed_w(rhs, 32) ? lhs : rhs",
    ),
    (
        0x53,
        "integer.min.s64",
        "Selects the lesser signed two's-complement 64-bit operand.",
        "dst_v8 receives the selected complete operand bits.",
        "signed_w(lhs, 64) <= signed_w(rhs, 64) ? lhs : rhs",
    ),
    (
        0x54,
        "integer.min.u32",
        "Selects the lesser unsigned 32-bit operand.",
        "dst_v8 receives the selected low 32 bits and clears its high half.",
        "lhs <= rhs ? lhs : rhs",
    ),
    (
        0x55,
        "integer.min.u64",
        "Selects the lesser unsigned 64-bit operand.",
        "dst_v8 receives the selected complete operand bits.",
        "lhs <= rhs ? lhs : rhs",
    ),
    (
        0x56,
        "integer.max.s32",
        "Selects the greater signed two's-complement 32-bit operand.",
        "dst_v8 receives the selected low 32 bits and clears its high half.",
        "signed_w(lhs, 32) >= signed_w(rhs, 32) ? lhs : rhs",
    ),
    (
        0x57,
        "integer.max.s64",
        "Selects the greater signed two's-complement 64-bit operand.",
        "dst_v8 receives the selected complete operand bits.",
        "signed_w(lhs, 64) >= signed_w(rhs, 64) ? lhs : rhs",
    ),
    (
        0x58,
        "integer.max.u32",
        "Selects the greater unsigned 32-bit operand.",
        "dst_v8 receives the selected low 32 bits and clears its high half.",
        "lhs >= rhs ? lhs : rhs",
    ),
    (
        0x59,
        "integer.max.u64",
        "Selects the greater unsigned 64-bit operand.",
        "dst_v8 receives the selected complete operand bits.",
        "lhs >= rhs ? lhs : rhs",
    ),
    (
        0x5A,
        "integer.and.i32",
        "Computes bitwise AND over the low 32 bits.",
        "dst_v8 receives the low 32 AND bits and clears its high half.",
        "lhs & rhs",
    ),
    (
        0x5B,
        "integer.and.i64",
        "Computes bitwise AND over all 64 bits.",
        "dst_v8 receives the complete 64-bit AND result.",
        "lhs & rhs",
    ),
    (
        0x5C,
        "integer.or.i32",
        "Computes bitwise OR over the low 32 bits.",
        "dst_v8 receives the low 32 OR bits and clears its high half.",
        "lhs | rhs",
    ),
    (
        0x5D,
        "integer.or.i64",
        "Computes bitwise OR over all 64 bits.",
        "dst_v8 receives the complete 64-bit OR result.",
        "lhs | rhs",
    ),
    (
        0x5E,
        "integer.xor.i32",
        "Computes bitwise XOR over the low 32 bits.",
        "dst_v8 receives the low 32 XOR bits and clears its high half.",
        "lhs ^ rhs",
    ),
    (
        0x5F,
        "integer.xor.i64",
        "Computes bitwise XOR over all 64 bits.",
        "dst_v8 receives the complete 64-bit XOR result.",
        "lhs ^ rhs",
    ),
    (
        0x60,
        "integer.shift.left.i32",
        "Shifts low 32 bits left by the count's low five bits.",
        "dst_v8 receives the width-limited shifted bits and clears its high half.",
        "lhs << (rhs & 31)",
    ),
    (
        0x61,
        "integer.shift.left.i64",
        "Shifts all 64 bits left by the count's low six bits.",
        "dst_v8 receives the width-limited shifted bits.",
        "lhs << (rhs & 63)",
    ),
    (
        0x62,
        "integer.shift.right.s32",
        "Sign-fills low 32 bits right by the count's low five bits.",
        "dst_v8 receives the exact arithmetic shift and clears its high half.",
        "asr_w(lhs, rhs & 31, 32)",
    ),
    (
        0x63,
        "integer.shift.right.s64",
        "Sign-fills all 64 bits right by the count's low six bits.",
        "dst_v8 receives the exact arithmetic shift.",
        "asr_w(lhs, rhs & 63, 64)",
    ),
    (
        0x64,
        "integer.shift.right.u32",
        "Logically shifts low 32 bits right by the count's low five bits.",
        "dst_v8 receives the logical shift and clears its high half.",
        "lhs >> (rhs & 31)",
    ),
    (
        0x65,
        "integer.shift.right.u64",
        "Logically shifts all 64 bits right by the count's low six bits.",
        "dst_v8 receives the complete logical shift.",
        "lhs >> (rhs & 63)",
    ),
    (
        0x66,
        "integer.rotate.left.i32",
        "Rotates low 32 bits left by the count's low five bits.",
        "dst_v8 receives the rotated low 32 bits and clears its high half.",
        "rotl_w(lhs, rhs & 31, 32)",
    ),
    (
        0x67,
        "integer.rotate.left.i64",
        "Rotates all 64 bits left by the count's low six bits.",
        "dst_v8 receives the complete rotated result.",
        "rotl_w(lhs, rhs & 63, 64)",
    ),
    (
        0x68,
        "integer.rotate.right.i32",
        "Rotates low 32 bits right by the count's low five bits.",
        "dst_v8 receives the rotated low 32 bits and clears its high half.",
        "rotr_w(lhs, rhs & 31, 32)",
    ),
    (
        0x69,
        "integer.rotate.right.i64",
        "Rotates all 64 bits right by the count's low six bits.",
        "dst_v8 receives the complete rotated result.",
        "rotr_w(lhs, rhs & 63, 64)",
    ),
)

_REGULAR_BINARY_INSTRUCTIONS = tuple(
    _binary(
        opcode=opcode,
        mnemonic=mnemonic,
        description=description,
        result=result,
        expression=expression,
    )
    for opcode, mnemonic, description, result, expression in _REGULAR_BINARY_DEFINITIONS
)


def _division(
    *, opcode: int, mnemonic: str, signed: bool, remainder: bool
) -> Instruction:
    width = 32 if mnemonic.endswith("32") else 64
    operation = "remainder" if remainder else "quotient"
    interpretation = "signed" if signed else "unsigned"
    read = (
        f"signed_w(read_w({{register}}, {width}), {width})"
        if signed
        else f"read_w({{register}}, {width})"
    )
    if remainder:
        expression = (
            f"lhs == INT{width}_MIN && rhs == -1 ? 0 : signed_remainder(lhs, rhs)"
            if signed
            else "lhs % rhs"
        )
    else:
        expression = "signed_quotient_toward_zero(lhs, rhs)" if signed else "lhs / rhs"
    failures = [
        FailureCase(
            "invalid_argument",
            "rhs_v8 is zero at the selected width.",
            "dst_v8 and all other VM state remain unchanged.",
        )
    ]
    if signed and not remainder:
        failures.append(
            FailureCase(
                "out_of_range",
                f"The operands are INT{width}_MIN and -1.",
                "dst_v8 and all other VM state remain unchanged.",
            )
        )
    special = (
        f" INT{width}_MIN % -1 is defined as zero." if signed and remainder else ""
    )
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=f"Computes the {interpretation} {width}-bit {operation}.",
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=4,
        family_id=FAMILY.entity_id,
        fields=(
            _value("dst_v8", 1, InstructionFieldRole.RESULT),
            _value("lhs_v8", 2, InstructionFieldRole.OPERAND),
            _value("rhs_v8", 3, InstructionFieldRole.OPERAND),
        ),
        semantics=_common_semantics(
            description=(
                f"Computes the {interpretation} {width}-bit {operation}."
                + (
                    " Signed quotient rounds toward zero."
                    if signed and not remainder
                    else ""
                )
                + (
                    " Signed remainder has the dividend's sign."
                    if signed and remainder
                    else ""
                )
                + special
            ),
            verification=(
                "dst_v8, lhs_v8, and rhs_v8 must be valid value-register ordinals.",
            ),
            preconditions=(
                "The selected-width divisor must be nonzero."
                + (
                    f" Signed quotient also excludes INT{width}_MIN / -1."
                    if signed and not remainder
                    else ""
                ),
            ),
            success=(
                f"dst_v8 receives the selected-width {operation}; a 32-bit "
                "result clears the high cell half.",
            ),
            failures=tuple(failures),
            assembly=(f"%v<dst> = {mnemonic} %v<lhs>, %v<rhs>",),
            pseudocode=(
                f"lhs = {read.format(register='lhs_v8')};\n"
                f"rhs = {read.format(register='rhs_v8')};\n"
                "if (rhs == 0) fail(invalid_argument);\n"
                + (
                    f"if (lhs == INT{width}_MIN && rhs == -1) fail(out_of_range);\n"
                    if signed and not remainder
                    else ""
                )
                + f"write_w(dst_v8, {expression}, {width});\n"
                "pc = pc + 4;"
            ),
            byte_length=4,
        ),
    )


_DIVISION_INSTRUCTIONS = (
    _division(opcode=0x46, mnemonic="integer.div.s32", signed=True, remainder=False),
    _division(opcode=0x47, mnemonic="integer.div.s64", signed=True, remainder=False),
    _division(opcode=0x48, mnemonic="integer.div.u32", signed=False, remainder=False),
    _division(opcode=0x49, mnemonic="integer.div.u64", signed=False, remainder=False),
    _division(opcode=0x4A, mnemonic="integer.rem.s32", signed=True, remainder=True),
    _division(opcode=0x4B, mnemonic="integer.rem.s64", signed=True, remainder=True),
    _division(opcode=0x4C, mnemonic="integer.rem.u32", signed=False, remainder=True),
    _division(opcode=0x4D, mnemonic="integer.rem.u64", signed=False, remainder=True),
)


def _unary(
    *,
    opcode: int,
    mnemonic: str,
    description: str,
    result: str,
    expression: str,
) -> Instruction:
    width = 32 if mnemonic.endswith("32") else 64
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=description,
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=4,
        family_id=FAMILY.entity_id,
        fields=(
            _value("dst_v8", 1, InstructionFieldRole.RESULT),
            _value("src_v8", 2, InstructionFieldRole.OPERAND),
            zero_padding("zero_padding_u8", 3, 1),
        ),
        semantics=_common_semantics(
            description=description,
            verification=(
                "dst_v8 and src_v8 must be valid value-register ordinals and "
                "zero_padding_u8 must equal zero.",
            ),
            success=(result,),
            assembly=(f"%v<dst> = {mnemonic} %v<src>",),
            pseudocode=(
                f"bits = read_w(src_v8, {width});\n"
                f"write_w(dst_v8, {expression}, {width});\n"
                "pc = pc + 4;"
            ),
            byte_length=4,
        ),
    )


_UNARY_DEFINITIONS = (
    (
        0x4E,
        "integer.neg.i32",
        "Computes two's-complement negation modulo 2^32.",
        "dst_v8 receives the modular negation and clears its high cell half.",
        "0 - bits",
    ),
    (
        0x4F,
        "integer.neg.i64",
        "Computes two's-complement negation modulo 2^64.",
        "dst_v8 receives the complete modular negation.",
        "0 - bits",
    ),
    (
        0x50,
        "integer.abs.s32",
        "Computes modular signed 32-bit absolute value.",
        "dst_v8 receives abs(src); INT32_MIN retains its bit pattern.",
        "bit_is_set(bits, 31) ? 0 - bits : bits",
    ),
    (
        0x51,
        "integer.abs.s64",
        "Computes modular signed 64-bit absolute value.",
        "dst_v8 receives abs(src); INT64_MIN retains its bit pattern.",
        "bit_is_set(bits, 63) ? 0 - bits : bits",
    ),
    (
        0x6A,
        "integer.count.leading.zeros.i32",
        "Counts leading zeroes in the low 32 bits.",
        "dst_v8 receives 32 for zero or the exact count, with high bits clear.",
        "bits == 0 ? 32 : count_leading_zeros(bits, 32)",
    ),
    (
        0x6B,
        "integer.count.leading.zeros.i64",
        "Counts leading zeroes in all 64 bits.",
        "dst_v8 receives 64 for zero or the exact count.",
        "bits == 0 ? 64 : count_leading_zeros(bits, 64)",
    ),
    (
        0x6C,
        "integer.count.trailing.zeros.i32",
        "Counts trailing zeroes in the low 32 bits.",
        "dst_v8 receives 32 for zero or the exact count, with high bits clear.",
        "bits == 0 ? 32 : count_trailing_zeros(bits, 32)",
    ),
    (
        0x6D,
        "integer.count.trailing.zeros.i64",
        "Counts trailing zeroes in all 64 bits.",
        "dst_v8 receives 64 for zero or the exact count.",
        "bits == 0 ? 64 : count_trailing_zeros(bits, 64)",
    ),
    (
        0x6E,
        "integer.popcount.i32",
        "Counts one bits in the low 32 bits.",
        "dst_v8 receives the population count with high cell bits clear.",
        "population_count(bits, 32)",
    ),
    (
        0x6F,
        "integer.popcount.i64",
        "Counts one bits in all 64 bits.",
        "dst_v8 receives the population count.",
        "population_count(bits, 64)",
    ),
)

_UNARY_INSTRUCTIONS = tuple(
    _unary(
        opcode=opcode,
        mnemonic=mnemonic,
        description=description,
        result=result,
        expression=expression,
    )
    for opcode, mnemonic, description, result, expression in _UNARY_DEFINITIONS
)


def _comparison(*, opcode: int, width: int) -> Instruction:
    mnemonic = f"integer.compare.i{width}"
    table = SELECTOR_TABLES_BY_NAME["integer.compare"]
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=f"Evaluates one signed, unsigned, or equality i{width} predicate.",
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=8,
        family_id=FAMILY.entity_id,
        fields=(
            _value("dst_v8", 1, InstructionFieldRole.RESULT),
            _value("lhs_v8", 2, InstructionFieldRole.OPERAND),
            _value("rhs_v8", 3, InstructionFieldRole.OPERAND),
            _field_selector("predicate_u8", 4, table.entity_id),
            zero_padding("zero_padding_u8", 5, 3),
        ),
        semantics=_common_semantics(
            description=(
                f"Compares the low {width} bits using eq, ne, signed lt/le/gt/ge, "
                "or unsigned lt/le/gt/ge selected by predicate_u8."
            ),
            verification=(
                "Every register must be valid, predicate_u8 must be an assigned "
                "integer.compare selector, and every padding byte must be zero.",
            ),
            success=(
                "dst_v8 receives canonical complete-cell zero or one for the "
                "selected predicate.",
            ),
            assembly=(f"%v<dst> = {mnemonic} %v<lhs>, %v<rhs> {{predicate}}",),
            pseudocode=(
                f"lhs = read_w(lhs_v8, {width});\n"
                f"rhs = read_w(rhs_v8, {width});\n"
                "values[dst_v8] = canonical_bool(evaluate_integer_predicate(\n"
                f"    predicate_u8, lhs, rhs, {width}));\n"
                "pc = pc + 8;"
            ),
            byte_length=8,
        ),
    )


def _field_selector(name: str, offset: int, table_id: str):
    return instruction_field(
        name,
        offset,
        U8.entity_id,
        InstructionFieldRole.IMMEDIATE,
        "Closed operation selector.",
        (RuleUse(SELECTOR.entity_id, (EntityReference(table_id),)),),
    )


INTEGER_COMPARE_I32 = _comparison(opcode=0x70, width=32)
INTEGER_COMPARE_I64 = _comparison(opcode=0x71, width=64)


def _lea(*, opcode: int, width: int) -> Instruction:
    mnemonic = f"integer.lea.i{width}"
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=f"Combines base, scaled index, and signed offset modulo 2^{width}.",
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=8,
        family_id=FAMILY.entity_id,
        fields=(
            _value("dst_v8", 1, InstructionFieldRole.RESULT),
            _value("base_v8", 2, InstructionFieldRole.OPERAND),
            _value("index_v8", 3, InstructionFieldRole.OPERAND),
            instruction_field(
                "scale_u8",
                4,
                U8.entity_id,
                InstructionFieldRole.IMMEDIATE,
                "Arbitrary unsigned scale including zero and non-powers-of-two.",
                (RuleUse(ANY_BITS.entity_id),),
            ),
            zero_padding("zero_padding_u8", 5, 1),
            instruction_field(
                "offset_i16",
                6,
                I16.entity_id,
                InstructionFieldRole.IMMEDIATE,
                "Signed little-endian two's-complement affine offset.",
                (RuleUse(ANY_BITS.entity_id),),
            ),
        ),
        semantics=_common_semantics(
            description=(
                f"Computes base + index * scale_u8 + sign_extend(offset_i16) "
                f"modulo 2^{width}. It creates no pointer authority and reports "
                "no arithmetic overflow."
            ),
            verification=(
                "All registers must be valid and zero_padding_u8 zero; every "
                "scale and offset bit pattern is valid.",
            ),
            success=(
                f"dst_v8 receives the modulo-{width}-bit affine result; an i32 "
                "result clears the high cell half.",
            ),
            assembly=(f"%v<dst> = {mnemonic} %v<base>, %v<index> {{scale, offset}}",),
            pseudocode=(
                f"base = read_w(base_v8, {width});\n"
                f"index = read_w(index_v8, {width});\n"
                "offset = sext_u64(load_le(16, &record[6]), 16);\n"
                f"write_w(dst_v8, base + index * scale_u8 + offset, {width});\n"
                "pc = pc + 8;"
            ),
            byte_length=8,
        ),
    )


INTEGER_LEA_I32 = _lea(opcode=0x72, width=32)
INTEGER_LEA_I64 = _lea(opcode=0x73, width=64)


def _ceildiv(*, opcode: int, width: int) -> Instruction:
    mnemonic = f"integer.ceildiv.pow2.u{width}"
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=f"Computes exact unsigned u{width} ceiling division by a power of two.",
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=4,
        family_id=FAMILY.entity_id,
        fields=(
            _value("dst_v8", 1, InstructionFieldRole.RESULT),
            _value("src_v8", 2, InstructionFieldRole.OPERAND),
            instruction_field(
                "log2_u8",
                3,
                U8.entity_id,
                InstructionFieldRole.IMMEDIATE,
                "Base-two logarithm of the nonzero divisor.",
                (RuleUse(ALLOWED_RANGE.entity_id, (0, width - 1)),),
            ),
        ),
        semantics=_common_semantics(
            description=(
                f"Computes exact mathematical ceil(unsigned(src) / 2^log2) over "
                f"the complete u{width} domain without an overflowing addition."
            ),
            verification=(
                f"dst_v8 and src_v8 must be valid and log2_u8 must be in 0..{width - 1}.",
            ),
            success=(
                f"dst_v8 receives the exact u{width} quotient; a u32 result "
                "clears the high cell half.",
            ),
            assembly=(f"%v<dst> = {mnemonic} %v<src> {{log2}}",),
            pseudocode=(
                f"source = read_w(src_v8, {width});\n"
                "mask = low_mask(log2_u8);\n"
                "result = (source >> log2_u8) +\n"
                "    ((source & mask) != 0 ? 1 : 0);\n"
                f"write_w(dst_v8, result, {width});\n"
                "pc = pc + 4;"
            ),
            byte_length=4,
        ),
    )


INTEGER_CEILDIV_POW2_U32 = _ceildiv(opcode=0x74, width=32)
INTEGER_CEILDIV_POW2_U64 = _ceildiv(opcode=0x75, width=64)


def _bitstream(*, opcode: int, mnemonic: str, mode: str, signed: bool) -> Instruction:
    carrier_values = (8, 16, 32, 64)
    fields = (
        _value("result_base_v8", 1, InstructionFieldRole.RESULT),
        _value("source_base_v8", 2, InstructionFieldRole.OPERAND),
        instruction_field(
            "field_width_u8",
            3,
            U8.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Logical field width in bits.",
            (RuleUse(ALLOWED_RANGE.entity_id, (1, 64)),),
        ),
        instruction_field(
            "source_count_u8",
            4,
            U8.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Nonzero consecutive source-register count.",
            (RuleUse(ALLOWED_RANGE.entity_id, (1, 255)),),
        ),
        instruction_field(
            "result_count_u8",
            5,
            U8.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Nonzero consecutive result-register count.",
            (RuleUse(ALLOWED_RANGE.entity_id, (1, 255)),),
        ),
        instruction_field(
            "source_width_u8",
            6,
            U8.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Source carrier width in bits.",
            (RuleUse(ALLOWED_VALUES.entity_id, (carrier_values,)),),
        ),
        instruction_field(
            "result_width_u8",
            7,
            U8.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Result carrier width in bits.",
            (RuleUse(ALLOWED_VALUES.entity_id, (carrier_values,)),),
        ),
    )
    constraints = (
        RuleUse(
            VALUE_REGISTER_RANGE.entity_id,
            (FieldReference("result_base_v8"), FieldReference("result_count_u8")),
        ),
        RuleUse(
            VALUE_REGISTER_RANGE.entity_id,
            (FieldReference("source_base_v8"), FieldReference("source_count_u8")),
        ),
        RuleUse(
            INTEGER_BITSTREAM_SHAPE.entity_id,
            (
                mode,
                FieldReference("field_width_u8"),
                FieldReference("source_count_u8"),
                FieldReference("result_count_u8"),
                FieldReference("source_width_u8"),
                FieldReference("result_width_u8"),
            ),
        ),
    )
    if mode == "pack":
        description = (
            "Concatenates each source carrier's low field_width bits into one "
            "at-most-64-bit stream and writes consecutive result-width carriers."
        )
        shape = (
            "field_width <= source_width and source_count * field_width must "
            "equal result_count * result_width and be at most 64."
        )
        capture_loop = (
            "field = read_carrier(source_base_v8 + i, source_width_u8) &\n"
            "    low_mask(field_width_u8);\n"
            "  stream |= field << (i * field_width_u8);"
        )
        result_loop = (
            "field = (stream >> (i * result_width_u8)) &\n"
            "      low_mask(result_width_u8);\n"
            "  write_carrier(result_base_v8 + i, field, result_width_u8);"
        )
    else:
        extension = "sign-extends" if signed else "zero-extends"
        description = (
            "Concatenates complete low-width source carriers, extracts "
            f"field_width-bit fields, and {extension} each into a result carrier."
        )
        shape = (
            "field_width <= result_width and source_count * source_width must "
            "equal result_count * field_width and be at most 64."
        )
        capture_loop = (
            "lane = read_carrier(source_base_v8 + i, source_width_u8);\n"
            "  stream |= lane << (i * source_width_u8);"
        )
        if signed:
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
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=description,
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=8,
        family_id=FAMILY.entity_id,
        fields=fields,
        constraints=constraints,
        semantics=_common_semantics(
            description=description,
            verification=(
                "Both nonempty register ranges must fit and both carrier widths "
                "must be 8, 16, 32, or 64.",
                shape,
            ),
            success=(
                "Register zero contributes the least-significant stream region. "
                "The complete source stream is captured before any result write, "
                "so source and result ranges may overlap arbitrarily. Every "
                "result cell clears bits above its carrier width.",
            ),
            assembly=(
                f"%v<result_base>... = {mnemonic} %v<source_base>... "
                "{field_width, source_count, result_count, source_width, result_width}",
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
            byte_length=8,
        ),
    )


INTEGER_BITSTREAM_PACK = _bitstream(
    opcode=0x76,
    mnemonic="integer.bitstream.pack",
    mode="pack",
    signed=False,
)
INTEGER_BITSTREAM_UNPACK_U = _bitstream(
    opcode=0x77,
    mnemonic="integer.bitstream.unpack.u",
    mode="unpack",
    signed=False,
)
INTEGER_BITSTREAM_UNPACK_S = _bitstream(
    opcode=0x78,
    mnemonic="integer.bitstream.unpack.s",
    mode="unpack",
    signed=True,
)

_INSTRUCTIONS_BY_OPCODE = {
    instruction.opcode: instruction
    for instruction in (
        *_REGULAR_BINARY_INSTRUCTIONS,
        *_DIVISION_INSTRUCTIONS,
        *_UNARY_INSTRUCTIONS,
        INTEGER_COMPARE_I32,
        INTEGER_COMPARE_I64,
        INTEGER_LEA_I32,
        INTEGER_LEA_I64,
        INTEGER_CEILDIV_POW2_U32,
        INTEGER_CEILDIV_POW2_U64,
        INTEGER_BITSTREAM_PACK,
        INTEGER_BITSTREAM_UNPACK_U,
        INTEGER_BITSTREAM_UNPACK_S,
    )
}
INSTRUCTIONS = tuple(
    _INSTRUCTIONS_BY_OPCODE[opcode] for opcode in sorted(_INSTRUCTIONS_BY_OPCODE)
)
ENTITIES = (FAMILY, *INSTRUCTIONS)
