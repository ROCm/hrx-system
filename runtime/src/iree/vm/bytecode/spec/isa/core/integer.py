# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Wrapping scalar integer instructions."""

from __future__ import annotations

import enum
from typing import NamedTuple

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


class IntegerBinaryOperation(enum.Enum):
    ADD = "add"
    SUB = "sub"
    MUL = "mul"


class IntegerBinarySemantics(NamedTuple):
    operation: IntegerBinaryOperation
    bit_width: int


INTEGER_FAMILY = InstructionFamily(
    name="integer",
    since=CORE_0,
    summary="Wrapping scalar integer operations.",
    contract=(
        "Integer operations interpret the low 32 bits or complete 64 bits as "
        "exact two's-complement bit patterns; signedness belongs to an operation, "
        "not a register. Every i32 result clears the high cell half. Ordinary "
        "arithmetic wraps modulo its width. Implementations use unsigned host "
        "arithmetic so no architectural result inherits C signed-overflow behavior. "
        "All operands are read before any destination is written, permitting "
        "arbitrary source and destination aliasing."
    ),
)

_OPERATION_LANGUAGE = {
    IntegerBinaryOperation.ADD: ("Adds", "sum"),
    IntegerBinaryOperation.SUB: ("Subtracts", "difference"),
    IntegerBinaryOperation.MUL: ("Multiplies", "product"),
}


def _value_field(name: str, role: FieldRole, summary: str) -> InstructionField:
    return InstructionField(
        Field(name, U8, summary),
        role,
        FieldRuleUse(FieldRule.REGISTER_VALUE),
    )


def _binary(
    opcode: int, operation: IntegerBinaryOperation, bit_width: int
) -> Instruction:
    verb, result_noun = _OPERATION_LANGUAGE[operation]
    mnemonic = f"integer.{operation.value}.i{bit_width}"
    high_half = " and clears the high 32 cell bits" if bit_width == 32 else ""
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=INTEGER_FAMILY,
        summary=f"{verb} {bit_width}-bit patterns modulo 2^{bit_width}.",
        fields=(
            _value_field(
                "destination_v8",
                FieldRole.RESULT,
                "Value-register ordinal receiving the result.",
            ),
            _value_field("left_v8", FieldRole.OPERAND, "Left value-register ordinal."),
            _value_field(
                "right_v8", FieldRole.OPERAND, "Right value-register ordinal."
            ),
        ),
        semantics=IntegerBinarySemantics(operation, bit_width),
        behavior=(
            f"Reads both operands before computing their wrapping {bit_width}-bit "
            f"{result_noun}."
        ),
        success=(
            f"destination_v8 receives the low {bit_width} {result_noun} bits"
            f"{high_half}.",
        ),
        assembly=f"%v<destination> = {mnemonic} %v<left>, %v<right>",
    )


INTEGER_INSTRUCTIONS = tuple(
    _binary(opcode, operation, bit_width)
    for opcode, operation, bit_width in (
        (0x40, IntegerBinaryOperation.ADD, 32),
        (0x41, IntegerBinaryOperation.ADD, 64),
        (0x42, IntegerBinaryOperation.SUB, 32),
        (0x43, IntegerBinaryOperation.SUB, 64),
        (0x44, IntegerBinaryOperation.MUL, 32),
        (0x45, IntegerBinaryOperation.MUL, 64),
    )
)
