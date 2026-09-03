# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Closed declarations for physical VM instruction records."""

import enum
from typing import NamedTuple

from iree.vm.bytecode.spec.schema import Field, NumericTable, RuleKind, place_fields
from iree.vm.bytecode.spec.version import Version


class FieldRole(enum.Enum):
    RESULT = "result"
    OPERAND = "operand"
    IMMEDIATE = "immediate"
    RANGE_BASE = "range_base"
    RANGE_COUNT = "range_count"
    CONSTRAINT_MEMBER = "constraint_member"
    PADDING = "padding"


class PackedSelectorComponent(NamedTuple):
    name: str
    bit_offset: int
    bit_length: int
    table: NumericTable
    allowed_values: tuple[int, ...] = ()


class FieldRuleUse(NamedTuple):
    kind: RuleKind
    fields: tuple[str, ...] = ()
    values: tuple[int, ...] = ()
    data: NumericTable | tuple[PackedSelectorComponent, ...] | None = None


class RecordRule(NamedTuple):
    kind: RuleKind
    fields: tuple[str, ...] = ()
    values: tuple[int, ...] = ()
    names: tuple[str, ...] = ()
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
    field: Field
    role: FieldRole
    rule: FieldRuleUse
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
        return place_fields(
            (instruction_field.field for instruction_field in self.fields),
            initial_offset=1,
        )

    @property
    def byte_length(self) -> int:
        return 1 + sum(field.field.byte_length for field in self.fields)
