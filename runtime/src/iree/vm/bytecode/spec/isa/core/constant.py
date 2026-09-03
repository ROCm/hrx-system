# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Inline and module-pool value constants."""

from iree.vm.bytecode.spec.isa import (
    FieldRole,
    FieldRuleUse,
    Instruction,
    InstructionFamily,
    InstructionField,
)
from iree.vm.bytecode.spec.isa.core.rules import FieldRule
from iree.vm.bytecode.spec.schema import I16, U8, U16, U32, Field
from iree.vm.bytecode.spec.version import CORE_0

CONSTANT_FAMILY = InstructionFamily(
    name="constant",
    since=CORE_0,
    summary="Inline and module-pool value constants.",
    contract=(
        "Constant instructions define complete 64-bit value cells. An i32 suffix "
        "preserves or loads exactly 32 bits and clears the high cell half; an i64 "
        "suffix defines all 64 bits. constant.s16 sign-extends its encoded two's-"
        "complement immediate. The module pool is an eight-byte-aligned array of raw "
        "64-bit cells with at most 65,536 entries. Execution is infallible after "
        "module verification and never suspends."
    ),
)


def _field(
    name: str, encoding, summary: str, role: FieldRole, rule
) -> InstructionField:
    if not isinstance(rule, FieldRuleUse):
        rule = FieldRuleUse(rule)
    return InstructionField(Field(name, encoding, summary), role, rule)


def _destination() -> InstructionField:
    return _field(
        "destination_v8",
        U8,
        "Value-register ordinal receiving the constant.",
        FieldRole.RESULT,
        FieldRule.REGISTER_VALUE,
    )


def _padding() -> InstructionField:
    return _field(
        "zero_padding_u16",
        U16,
        "Canonical zero padding.",
        FieldRole.PADDING,
        FieldRule.ZERO,
    )


CONSTANT_ZERO = Instruction(
    opcode=0x18,
    mnemonic="constant.zero",
    since=CORE_0,
    family=CONSTANT_FAMILY,
    summary="Clears a complete value-register cell.",
    fields=(_destination(), _padding()),
    semantics=None,
    behavior="Writes zero to all 64 bits of destination_v8.",
    success=("destination_v8 becomes the complete 64-bit zero pattern.",),
    assembly="%v<destination> = constant.zero",
    pseudocode="values[destination_v8] = 0;\npc = pc + 4;",
)

CONSTANT_S16 = Instruction(
    opcode=0x19,
    mnemonic="constant.s16",
    since=CORE_0,
    family=CONSTANT_FAMILY,
    summary="Sign-extends a 16-bit immediate into a value cell.",
    fields=(
        _destination(),
        _field(
            "immediate_i16",
            I16,
            "Signed little-endian two's-complement immediate.",
            FieldRole.IMMEDIATE,
            FieldRule.ANY_BITS,
        ),
    ),
    semantics=None,
    behavior=(
        "Sign-extends the encoded 16-bit immediate through all 64 bits of the "
        "destination cell."
    ),
    success=("destination_v8 receives the 64-bit sign extension.",),
    assembly="%v<destination> = constant.s16 <immediate>",
    pseudocode=(
        "values[destination_v8] = sext_u64(load_le(16, &record[2]), 16);\npc = pc + 4;"
    ),
)

CONSTANT_I32 = Instruction(
    opcode=0x1A,
    mnemonic="constant.i32",
    since=CORE_0,
    family=CONSTANT_FAMILY,
    summary="Loads an inline 32-bit pattern into a value cell.",
    fields=(
        _destination(),
        _padding(),
        _field(
            "bits_u32",
            U32,
            "Arbitrary little-endian low 32 bits.",
            FieldRole.IMMEDIATE,
            FieldRule.ANY_BITS,
        ),
    ),
    semantics=None,
    behavior=("Writes bits_u32 into the low cell half and clears the high cell half."),
    success=("destination_v8 receives bits_u32 zero-extended to 64 bits.",),
    assembly="%v<destination> = constant.i32 <bits>",
    pseudocode="values[destination_v8] = load_le(32, &record[4]);\npc = pc + 8;",
)

CONSTANT_I64 = Instruction(
    opcode=0x1B,
    mnemonic="constant.i64",
    since=CORE_0,
    family=CONSTANT_FAMILY,
    summary="Loads an inline 64-bit pattern into a value cell.",
    fields=(
        _destination(),
        _padding(),
        _field(
            "bits_low_u32",
            U32,
            "Arbitrary little-endian low 32 bits.",
            FieldRole.IMMEDIATE,
            FieldRule.ANY_BITS,
        ),
        _field(
            "bits_high_u32",
            U32,
            "Arbitrary little-endian high 32 bits.",
            FieldRole.IMMEDIATE,
            FieldRule.ANY_BITS,
        ),
    ),
    semantics=None,
    behavior=(
        "Combines two naturally aligned 32-bit fields into one exact 64-bit pattern, "
        "avoiding an unaligned native 64-bit load."
    ),
    success=("destination_v8 receives both encoded halves exactly.",),
    assembly="%v<destination> = constant.i64 <bits>",
    pseudocode=(
        "low = load_le(32, &record[4]);\n"
        "high = load_le(32, &record[8]);\n"
        "values[destination_v8] = low | (u64(high) << 32);\n"
        "pc = pc + 12;"
    ),
)


def _pool_load(opcode: int, bit_width: int) -> Instruction:
    mnemonic = f"constant.pool.load.i{bit_width}"
    result = (
        "the selected pool cell's low 32 bits with its high cell half cleared"
        if bit_width == 32
        else "all 64 bits of the selected pool cell"
    )
    expression = (
        "bits_u32(constant_pool[constant_pool_ordinal_u16])"
        if bit_width == 32
        else "constant_pool[constant_pool_ordinal_u16]"
    )
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=CONSTANT_FAMILY,
        summary=f"Loads {result}.",
        fields=(
            _destination(),
            _field(
                "constant_pool_ordinal_u16",
                U16,
                "Direct module constant-pool ordinal.",
                FieldRole.IMMEDIATE,
                FieldRule.CONSTANT_POOL_ORDINAL,
            ),
        ),
        semantics=bit_width,
        behavior=f"Writes {result} to destination_v8.",
        success=(f"destination_v8 receives {result}.",),
        assembly=f"%v<destination> = {mnemonic} <pool_ordinal>",
        pseudocode=f"values[destination_v8] = {expression};\npc = pc + 4;",
    )


CONSTANT_INSTRUCTIONS = (
    CONSTANT_ZERO,
    CONSTANT_S16,
    CONSTANT_I32,
    CONSTANT_I64,
    _pool_load(0x1C, 32),
    _pool_load(0x1D, 64),
)
