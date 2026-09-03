# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Closed declarations for physical VM instruction records."""

import enum
from typing import NamedTuple

from iree.vm.bytecode.spec.schema import Field, place_fields
from iree.vm.bytecode.spec.version import Version


class FieldRole(enum.Enum):
    RESULT = "result"
    OPERAND = "operand"
    IMMEDIATE = "immediate"
    CONSTRAINT_MEMBER = "constraint_member"
    PADDING = "padding"


class FieldRule(enum.Enum):
    ANY_BITS = "any_bits"
    ZERO = "zero"
    VALUE_REGISTER = "value_register"
    CONTROL_TARGET_S16 = "control_target_s16"
    CONTROL_TARGET_S32 = "control_target_s32"


class ControlFlow(enum.Enum):
    SEQUENTIAL = "sequential"
    BLOCK = "block"
    RETURN = "return"
    YIELD = "yield"
    BRANCH = "branch"
    CONDITIONAL_BRANCH = "conditional_branch"
    SWITCH = "switch"


class Suspension(enum.Enum):
    NEVER = "never"
    ALWAYS = "always"


class StateAccess(enum.Enum):
    READ = "read"
    WRITE = "write"


class StateResource(enum.Enum):
    INVOCATION_RESULTS = "invocation.results"


class IntegerBinaryOperation(enum.Enum):
    ADD = "add"
    SUB = "sub"
    MUL = "mul"


class BranchCondition(enum.Enum):
    ALWAYS = "always"
    NONZERO = "nonzero"
    ZERO = "zero"


class RecordRule(enum.Enum):
    RETURN_SIGNATURE = "return_signature"


class SwitchTargetsRule(NamedTuple):
    count_field: str
    base_field: str


InstructionRecordRule = RecordRule | SwitchTargetsRule


class InstructionField(NamedTuple):
    field: Field
    role: FieldRole
    rule: FieldRule


class StateEffect(NamedTuple):
    access: StateAccess
    resource: StateResource
    resource_fields: tuple[str, ...] = ()


class FailureCase(NamedTuple):
    status: str
    condition: str
    atomicity: str


class IntegerBinarySemantics(NamedTuple):
    operation: IntegerBinaryOperation
    bit_width: int


InstructionSemantics = IntegerBinarySemantics | BranchCondition | None


class InstructionFamily(NamedTuple):
    name: str
    since: Version
    summary: str
    contract: str


class Instruction(NamedTuple):
    """One complete immutable physical instruction declaration."""

    opcode: int
    mnemonic: str
    since: Version
    family: InstructionFamily
    summary: str
    fields: tuple[InstructionField, ...]
    semantics: InstructionSemantics
    behavior: str
    success: tuple[str, ...]
    assembly: str
    pseudocode: str | None = None
    rules: tuple[InstructionRecordRule, ...] = ()
    control_flow: ControlFlow = ControlFlow.SEQUENTIAL
    suspension: Suspension = Suspension.NEVER
    state_effects: tuple[StateEffect, ...] = ()
    preconditions: tuple[str, ...] = ()
    failures: tuple[FailureCase, ...] = ()
    ownership: tuple[str, ...] = ()

    @property
    def field_offsets(self) -> tuple[int, ...]:
        """Returns field offsets following the implicit one-byte opcode."""
        return place_fields(
            (instruction_field.field for instruction_field in self.fields),
            initial_offset=1,
        )

    @property
    def byte_length(self) -> int:
        """Returns the exact encoded record length."""

        return 1 + sum(field.field.byte_length for field in self.fields)
