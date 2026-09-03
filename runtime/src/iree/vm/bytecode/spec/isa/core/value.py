# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Whole-cell value-register instructions."""

from iree.vm.bytecode.spec.isa import (
    FieldRole,
    FieldRuleUse,
    Instruction,
    InstructionFamily,
    InstructionField,
)
from iree.vm.bytecode.spec.isa.core.rules import FieldRule
from iree.vm.bytecode.spec.schema import U8, Field
from iree.vm.bytecode.spec.version import CORE_0

VALUE_FAMILY = InstructionFamily(
    name="value",
    since=CORE_0,
    summary="Whole-cell value-register operations.",
    contract=(
        "Value registers are complete 64-bit cells with no intrinsic scalar type. "
        "Value operations copy all 64 bits, never access refs, never fail after "
        "verification, and never suspend. Every required source is read before a "
        "destination is published, so destinations may alias any source."
    ),
)


def _register(name: str, role: FieldRole, summary: str) -> InstructionField:
    return InstructionField(
        Field(name, U8, summary), role, FieldRuleUse(FieldRule.REGISTER_VALUE)
    )


def _padding(element_count: int = 1) -> InstructionField:
    return InstructionField(
        Field("zero_padding_u8", U8, "Canonical zero padding.", element_count),
        FieldRole.PADDING,
        FieldRuleUse(FieldRule.ZERO),
    )


VALUE_COPY = Instruction(
    opcode=0x10,
    mnemonic="value.copy",
    since=CORE_0,
    family=VALUE_FAMILY,
    summary="Copies one complete value-register cell.",
    fields=(
        _register(
            "destination_v8",
            FieldRole.RESULT,
            "Value-register ordinal receiving the copied bits.",
        ),
        _register(
            "source_v8",
            FieldRole.OPERAND,
            "Value-register ordinal providing the copied bits.",
        ),
        _padding(),
    ),
    semantics=None,
    behavior=(
        "Reads all 64 bits from source_v8 before replacing destination_v8 without "
        "assigning a scalar interpretation to the pattern."
    ),
    success=("destination_v8 receives the exact source pattern.",),
    assembly="%v<destination> = value.copy %v<source>",
    pseudocode=(
        "source_bits = values[source_v8];\n"
        "values[destination_v8] = source_bits;\n"
        "pc = pc + 4;"
    ),
)

VALUE_SELECT = Instruction(
    opcode=0x11,
    mnemonic="value.select",
    since=CORE_0,
    family=VALUE_FAMILY,
    summary="Selects one complete value-register cell.",
    fields=(
        _register(
            "destination_v8",
            FieldRole.RESULT,
            "Value-register ordinal receiving the selected bits.",
        ),
        _register(
            "condition_v8",
            FieldRole.OPERAND,
            "Complete 64-bit truth-condition value-register ordinal.",
        ),
        _register(
            "true_v8",
            FieldRole.OPERAND,
            "Source selected when the condition is nonzero.",
        ),
        _register(
            "false_v8",
            FieldRole.OPERAND,
            "Source selected when the condition is zero.",
        ),
        _padding(3),
    ),
    semantics=None,
    behavior=(
        "Reads the condition and selected candidate before replacing destination_v8, "
        "copying all 64 bits from true_v8 when the complete condition is nonzero and "
        "otherwise from false_v8."
    ),
    success=("destination_v8 receives the exact selected source pattern.",),
    assembly=("%v<destination> = value.select %v<condition>, %v<true>, %v<false>"),
    pseudocode=(
        "condition_bits = values[condition_v8];\n"
        "selected_bits = condition_bits != 0\n"
        "    ? values[true_v8] : values[false_v8];\n"
        "values[destination_v8] = selected_bits;\n"
        "pc = pc + 8;"
    ),
)

VALUE_INSTRUCTIONS = (VALUE_COPY, VALUE_SELECT)
