# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core 0.0 constant instructions."""

from __future__ import annotations

from model.isa import InstructionFamily, InstructionFieldRole, InstructionSemantics
from model.isa.declarations import (
    core_instruction,
    instruction_field,
    value_register,
)
from model.isa.validation import ANY_BITS, CONSTANT_POOL_ORDINAL, ZERO
from model.schema import I16, U16, U32, RuleUse
from model.specification import CORE_0

FAMILY = InstructionFamily(
    entity_id="core.family.constant",
    since=CORE_0,
    summary="Inline and module-pool value constants.",
    dependencies=("core.contract.machine",),
    document_order=2,
    normative_text=(
        "Constant instructions define complete 64-bit value-register cells. "
        "An i32 suffix preserves or loads exactly 32 bits and clears the high "
        "cell half; an i64 suffix defines all 64 bits. constant.s16 explicitly "
        "sign-extends its encoded two's-complement immediate. The module "
        "constant pool is an eight-byte-aligned array of raw little-endian "
        "64-bit cells with at most 65,536 entries. After module verification, "
        "constant execution is infallible and never suspends."
    ),
)


def _destination():
    return value_register(
        "dst_v8",
        1,
        InstructionFieldRole.RESULT,
        "Destination value-register ordinal.",
    )


def _zero_u16():
    return instruction_field(
        "zero_padding_u16",
        2,
        U16.entity_id,
        InstructionFieldRole.PADDING,
        "Canonical zero padding.",
        (RuleUse(ZERO.entity_id),),
    )


CONSTANT_ZERO = core_instruction(
    entity_id="core.instruction.constant.zero",
    since=CORE_0,
    summary="Clears a complete value-register cell.",
    opcode=0x18,
    mnemonic="constant.zero",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(_destination(), _zero_u16()),
    state_effects=(),
    semantics=InstructionSemantics(
        description="Writes zero to all 64 bits of dst_v8.",
        verification=(
            "dst_v8 must be a valid value-register ordinal.",
            "zero_padding_u16 must equal zero.",
        ),
        preconditions=(),
        success=(
            "dst_v8 becomes the complete 64-bit zero pattern.",
            "The program counter advances by four bytes.",
        ),
        failures=(),
        ownership=(),
        assembly=("%v<dst> = constant.zero",),
        pseudocode="values[dst_v8] = 0;\npc = pc + 4;",
    ),
)

CONSTANT_S16 = core_instruction(
    entity_id="core.instruction.constant.s16",
    since=CORE_0,
    summary="Sign-extends a 16-bit immediate into a value cell.",
    opcode=0x19,
    mnemonic="constant.s16",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        _destination(),
        instruction_field(
            "immediate_i16",
            2,
            I16.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Signed little-endian two's-complement immediate.",
            (RuleUse(ANY_BITS.entity_id),),
        ),
    ),
    state_effects=(),
    semantics=InstructionSemantics(
        description=(
            "Sign-extends the encoded signed 16-bit immediate through all 64 "
            "bits of dst_v8."
        ),
        verification=(
            "dst_v8 must be a valid value-register ordinal; every immediate "
            "bit pattern is valid.",
        ),
        preconditions=(),
        success=(
            "dst_v8 receives the 64-bit sign extension of immediate_i16.",
            "The program counter advances by four bytes.",
        ),
        failures=(),
        ownership=(),
        assembly=("%v<dst> = constant.s16 <immediate>",),
        pseudocode=(
            "values[dst_v8] = sext_u64(load_le(16, &record[2]), 16);\npc = pc + 4;"
        ),
    ),
)

CONSTANT_I32 = core_instruction(
    entity_id="core.instruction.constant.i32",
    since=CORE_0,
    summary="Loads an inline 32-bit pattern into a value cell.",
    opcode=0x1A,
    mnemonic="constant.i32",
    byte_length=8,
    family_id=FAMILY.entity_id,
    fields=(
        _destination(),
        _zero_u16(),
        instruction_field(
            "bits_u32le",
            4,
            U32.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Arbitrary little-endian low 32 bits.",
            (RuleUse(ANY_BITS.entity_id),),
        ),
    ),
    state_effects=(),
    semantics=InstructionSemantics(
        description=(
            "Writes the encoded 32-bit pattern into the low half of dst_v8 "
            "and clears its high half."
        ),
        verification=(
            "dst_v8 must be a valid value-register ordinal.",
            "zero_padding_u16 must equal zero; every bits_u32le pattern is valid.",
        ),
        preconditions=(),
        success=(
            "dst_v8 receives bits_u32le zero-extended to 64 bits.",
            "The program counter advances by eight bytes.",
        ),
        failures=(),
        ownership=(),
        assembly=("%v<dst> = constant.i32 <bits>",),
        pseudocode=("values[dst_v8] = load_le(32, &record[4]);\npc = pc + 8;"),
    ),
)

CONSTANT_I64 = core_instruction(
    entity_id="core.instruction.constant.i64",
    since=CORE_0,
    summary="Loads an inline 64-bit pattern into a value cell.",
    opcode=0x1B,
    mnemonic="constant.i64",
    byte_length=12,
    family_id=FAMILY.entity_id,
    fields=(
        _destination(),
        _zero_u16(),
        instruction_field(
            "bits_low_u32le",
            4,
            U32.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Arbitrary little-endian low 32 bits.",
            (RuleUse(ANY_BITS.entity_id),),
        ),
        instruction_field(
            "bits_high_u32le",
            8,
            U32.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Arbitrary little-endian high 32 bits.",
            (RuleUse(ANY_BITS.entity_id),),
        ),
    ),
    state_effects=(),
    semantics=InstructionSemantics(
        description=(
            "Combines two naturally aligned little-endian 32-bit fields into "
            "one exact 64-bit destination pattern."
        ),
        verification=(
            "dst_v8 must be a valid value-register ordinal.",
            "zero_padding_u16 must equal zero; every immediate pattern is valid.",
        ),
        preconditions=(),
        success=(
            "dst_v8 receives bits_low_u32le in bits 0..31 and "
            "bits_high_u32le in bits 32..63.",
            "The program counter advances by twelve bytes.",
        ),
        failures=(),
        ownership=(),
        assembly=("%v<dst> = constant.i64 <bits>",),
        pseudocode=(
            "low = load_le(32, &record[4]);\n"
            "high = load_le(32, &record[8]);\n"
            "values[dst_v8] = low | (u64(high) << 32);\n"
            "pc = pc + 12;"
        ),
    ),
)


def _pool_load(
    *,
    entity_id: str,
    summary: str,
    opcode: int,
    mnemonic: str,
    result_effect: str,
    pseudocode: str,
):
    return core_instruction(
        entity_id=entity_id,
        since=CORE_0,
        summary=summary,
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=4,
        family_id=FAMILY.entity_id,
        fields=(
            _destination(),
            instruction_field(
                "pool_u16",
                2,
                U16.entity_id,
                InstructionFieldRole.IMMEDIATE,
                "Direct module constant-pool ordinal.",
                (RuleUse(CONSTANT_POOL_ORDINAL.entity_id),),
            ),
        ),
        state_effects=(),
        semantics=InstructionSemantics(
            description=result_effect,
            verification=(
                "dst_v8 must be a valid value-register ordinal.",
                "pool_u16 must be less than the module constant-pool count.",
            ),
            preconditions=(),
            success=(
                result_effect,
                "The program counter advances by four bytes.",
            ),
            failures=(),
            ownership=(),
            assembly=(f"%v<dst> = {mnemonic} <pool_ordinal>",),
            pseudocode=pseudocode,
        ),
    )


CONSTANT_POOL_LOAD_I32 = _pool_load(
    entity_id="core.instruction.constant.pool.load.i32",
    summary="Loads the low 32 bits of a module constant-pool cell.",
    opcode=0x1C,
    mnemonic="constant.pool.load.i32",
    result_effect=(
        "dst_v8 receives the selected pool cell's low 32 bits with its high "
        "32 bits cleared."
    ),
    pseudocode=("values[dst_v8] = bits_u32(constant_pool[pool_u16]);\npc = pc + 4;"),
)
CONSTANT_POOL_LOAD_I64 = _pool_load(
    entity_id="core.instruction.constant.pool.load.i64",
    summary="Loads a complete module constant-pool cell.",
    opcode=0x1D,
    mnemonic="constant.pool.load.i64",
    result_effect="dst_v8 receives all 64 bits of the selected pool cell.",
    pseudocode="values[dst_v8] = constant_pool[pool_u16];\npc = pc + 4;",
)

INSTRUCTIONS = (
    CONSTANT_ZERO,
    CONSTANT_S16,
    CONSTANT_I32,
    CONSTANT_I64,
    CONSTANT_POOL_LOAD_I32,
    CONSTANT_POOL_LOAD_I64,
)
ENTITIES = (FAMILY, *INSTRUCTIONS)
