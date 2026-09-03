# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Closed declarations for physical VM instruction records."""

import enum
from typing import NamedTuple

from iree.vm.bytecode.spec import schema
from iree.vm.bytecode.spec.version import Version

FieldRuleUse = schema.FieldRuleUse


class FieldRole(enum.Enum):
    RESULT = "result"
    OPERAND = "operand"
    IMMEDIATE = "immediate"
    CONSTRAINT_MEMBER = "constraint_member"
    PADDING = "padding"


class PackedSelectorComponent(NamedTuple):
    name: str
    bit_offset: int
    bit_length: int
    table: schema.NumericTable
    allowed_values: tuple[int, ...] = ()


class RecordRule(NamedTuple):
    kind: schema.RuleKind
    fields: tuple[str, ...] = ()
    values: tuple[int, ...] = ()
    data: tuple[PackedSelectorComponent, ...] | None = None
    summary: str = ""


class ControlFlow(enum.Enum):
    SEQUENTIAL = "sequential"
    BLOCK = "block"
    RETURN = "return"
    YIELD = "yield"
    BRANCH = "branch"
    CONDITIONAL_BRANCH = "conditional_branch"
    SWITCH = "switch"
    CALL = "call"
    FAIL = "fail"


class Suspension(enum.Enum):
    NEVER = "never"
    ALWAYS = "always"
    TARGET_DEPENDENT = "target_dependent"


class StateEffect(NamedTuple):
    access: enum.Enum
    resource: enum.Enum
    resource_fields: tuple[str, ...] = ()


class RuntimeRefPolicy(NamedTuple):
    type_contract: str
    null_policy: enum.Enum
    ownership: enum.Enum


class InstructionField(NamedTuple):
    field: schema.Field
    role: FieldRole
    rule: schema.FieldRuleUse
    ref_policy: RuntimeRefPolicy | None = None


class FailureCase(NamedTuple):
    status: str
    condition: str
    atomicity: str


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
    semantics: object | None
    behavior: str
    success: tuple[str, ...]
    assembly: str
    pseudocode: str | None = None
    rules: tuple[RecordRule, ...] = ()
    control_flow: ControlFlow = ControlFlow.SEQUENTIAL
    suspension: Suspension = Suspension.NEVER
    state_effects: tuple[StateEffect, ...] = ()
    preconditions: tuple[str, ...] = ()
    failures: tuple[FailureCase, ...] = ()
    ownership: tuple[str, ...] = ()

    @property
    def field_offsets(self) -> tuple[int, ...]:
        return schema.place_fields(
            (instruction_field.field for instruction_field in self.fields),
            initial_offset=1,
        )

    @property
    def byte_length(self) -> int:
        return 1 + sum(field.field.byte_length for field in self.fields)
