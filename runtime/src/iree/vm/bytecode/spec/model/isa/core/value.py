# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core 0.0 value-register instructions."""

from __future__ import annotations

from model.isa import (
    InstructionFamily,
    InstructionFieldRole,
    InstructionSemantics,
)
from model.isa.declarations import (
    core_instruction,
    value_register,
    zero_padding,
)
from model.specification import CORE_0

FAMILY = InstructionFamily(
    entity_id="core.family.value",
    since=CORE_0,
    summary="Whole-cell value-register operations.",
    dependencies=("core.contract.machine",),
    document_order=11,
    normative_text=(
        "Value registers are complete 64-bit cells with no intrinsic scalar "
        "type. Value operations copy all 64 bits, never access refs, never "
        "fail after verification, and never suspend. A destination may alias "
        "any source because execution reads every required source before "
        "publishing the destination."
    ),
)

VALUE_COPY = core_instruction(
    entity_id="core.instruction.value.copy",
    since=CORE_0,
    summary="Copies one complete value-register cell.",
    opcode=0x10,
    mnemonic="value.copy",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        value_register(
            "dst_v8",
            1,
            InstructionFieldRole.RESULT,
            "Destination value-register ordinal.",
        ),
        value_register(
            "src_v8",
            2,
            InstructionFieldRole.OPERAND,
            "Source value-register ordinal.",
        ),
        zero_padding("zero_padding_u8", 3, 1),
    ),
    semantics=InstructionSemantics(
        description=(
            "Copies all 64 bits from src_v8 to dst_v8 without assigning a "
            "scalar interpretation to the bit pattern."
        ),
        verification=(
            "dst_v8 and src_v8 must be valid value-register ordinals.",
            "zero_padding_u8 must equal zero.",
        ),
        preconditions=(),
        success=(
            "dst_v8 receives the exact 64-bit pattern read from src_v8.",
            "The program counter advances by four bytes.",
        ),
        failures=(),
        ownership=(),
        assembly=(
            "%v<dst> = value.copy %v<src>",
            "%v5 = value.copy %v2",
        ),
        pseudocode=(
            "source_bits = values[src_v8];\nvalues[dst_v8] = source_bits;\npc = pc + 4;"
        ),
    ),
)

VALUE_SELECT = core_instruction(
    entity_id="core.instruction.value.select",
    since=CORE_0,
    summary="Selects one complete value-register cell.",
    opcode=0x11,
    mnemonic="value.select",
    byte_length=8,
    family_id=FAMILY.entity_id,
    fields=(
        value_register(
            "dst_v8",
            1,
            InstructionFieldRole.RESULT,
            "Destination value-register ordinal.",
        ),
        value_register(
            "condition_v8",
            2,
            InstructionFieldRole.OPERAND,
            "Complete 64-bit truth-condition value register.",
        ),
        value_register(
            "true_v8",
            3,
            InstructionFieldRole.OPERAND,
            "Source selected when condition_v8 is nonzero.",
        ),
        value_register(
            "false_v8",
            4,
            InstructionFieldRole.OPERAND,
            "Source selected when condition_v8 is zero.",
        ),
        zero_padding("zero_padding_u8", 5, 3),
    ),
    semantics=InstructionSemantics(
        description=(
            "Copies all 64 bits from true_v8 when the complete 64-bit value "
            "in condition_v8 is nonzero, otherwise from false_v8."
        ),
        verification=(
            "All four named fields must be valid value-register ordinals.",
            "Every zero_padding_u8 byte must equal zero.",
        ),
        preconditions=(),
        success=(
            "dst_v8 receives the exact 64-bit pattern from the selected source.",
            "The program counter advances by eight bytes.",
        ),
        failures=(),
        ownership=(),
        assembly=(
            "%v<dst> = value.select %v<condition>, %v<true>, %v<false>",
            "%v7 = value.select %v1, %v3, %v4",
        ),
        pseudocode=(
            "condition_bits = values[condition_v8];\n"
            "selected_bits = condition_bits != 0\n"
            "    ? values[true_v8]\n"
            "    : values[false_v8];\n"
            "values[dst_v8] = selected_bits;\n"
            "pc = pc + 8;"
        ),
    ),
)

INSTRUCTIONS = (VALUE_COPY, VALUE_SELECT)
ENTITIES = (FAMILY, *INSTRUCTIONS)
