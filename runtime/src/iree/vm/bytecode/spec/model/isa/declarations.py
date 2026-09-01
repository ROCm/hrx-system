# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Concise constructors for version-zero instruction declarations.

The helpers factor representation invariants shared by many records. They do
not infer opcodes, field offsets, control flow, ownership, or behavior.
"""

from __future__ import annotations

from model.isa import (
    ControlFlow,
    Instruction,
    InstructionField,
    InstructionFieldRole,
    InstructionRangeGroup,
    InstructionSemantics,
    RuntimeRefPolicy,
    StateAccess,
    StateEffect,
    StateResource,
    Suspension,
)
from model.isa.validation import (
    REGISTER_FUNCTION,
    REGISTER_REF,
    REGISTER_VALUE,
    ZERO,
)
from model.schema import U8, RuleUse
from model.specification import CORE_0, HAL_0, Version


def instruction_field(
    name: str,
    offset: int,
    encoding_id: str,
    role: InstructionFieldRole,
    description: str,
    validation: tuple[RuleUse, ...],
    *,
    runtime_ref_policy: RuntimeRefPolicy | None = None,
    array_length: int = 1,
) -> InstructionField:
    """Declares one explicitly encoded instruction field."""

    return InstructionField(
        name=name,
        offset=offset,
        encoding_id=encoding_id,
        role=role,
        description=description,
        validation=validation,
        runtime_ref_policy=runtime_ref_policy,
        array_length=array_length,
    )


def core_instruction(
    *,
    entity_id: str,
    summary: str,
    opcode: int,
    mnemonic: str,
    byte_length: int,
    family_id: str,
    fields: tuple[InstructionField, ...],
    state_effects: tuple[StateEffect, ...],
    semantics: InstructionSemantics,
    since: Version = CORE_0,
    range_groups: tuple[InstructionRangeGroup, ...] = (),
    constraints: tuple[RuleUse, ...] = (),
    control_flow: ControlFlow = ControlFlow.SEQUENTIAL,
    suspension: Suspension = Suspension.NEVER,
) -> Instruction:
    """Declares one core-page physical instruction."""

    return Instruction(
        entity_id=entity_id,
        since=since,
        summary=summary,
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=byte_length,
        family_id=family_id,
        fields=fields,
        range_groups=range_groups,
        constraints=constraints,
        control_flow=control_flow,
        suspension=suspension,
        state_effects=state_effects,
        semantics=semantics,
    )


def hal_instruction(
    *,
    entity_id: str,
    summary: str,
    opcode: int,
    mnemonic: str,
    byte_length: int,
    family_id: str,
    fields: tuple[InstructionField, ...],
    state_effects: tuple[StateEffect, ...],
    semantics: InstructionSemantics,
    since: Version = HAL_0,
    range_groups: tuple[InstructionRangeGroup, ...] = (),
    constraints: tuple[RuleUse, ...] = (),
    control_flow: ControlFlow = ControlFlow.SEQUENTIAL,
    suspension: Suspension = Suspension.NEVER,
) -> Instruction:
    """Declares one HAL-page physical instruction."""

    return Instruction(
        entity_id=entity_id,
        since=since,
        summary=summary,
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=byte_length,
        family_id=family_id,
        fields=fields,
        range_groups=range_groups,
        constraints=constraints,
        control_flow=control_flow,
        suspension=suspension,
        state_effects=state_effects,
        semantics=semantics,
    )


def state_read(
    resource: StateResource,
    *resource_fields: str,
) -> StateEffect:
    """Declares a possible read from one architectural state resource."""

    return StateEffect(StateAccess.READ, resource, resource_fields)


def state_unknown() -> StateEffect:
    """Declares effects that may alias any architectural state resource."""

    return StateEffect(StateAccess.UNKNOWN, StateResource.ANY)


def state_write(
    resource: StateResource,
    *resource_fields: str,
) -> StateEffect:
    """Declares a possible write to one architectural state resource."""

    return StateEffect(StateAccess.WRITE, resource, resource_fields)


def state_allocate(
    resource: StateResource,
    *resource_fields: str,
) -> StateEffect:
    """Declares creation of one independently tracked resource lifetime."""

    return StateEffect(StateAccess.ALLOCATE, resource, resource_fields)


def state_release(
    resource: StateResource,
    *resource_fields: str,
) -> StateEffect:
    """Declares invalidation of one independently tracked resource lifetime."""

    return StateEffect(StateAccess.RELEASE, resource, resource_fields)


def state_synchronize(
    resource: StateResource,
    *resource_fields: str,
) -> StateEffect:
    """Declares an ordering or completion dependency on one state resource."""

    return StateEffect(StateAccess.SYNCHRONIZE, resource, resource_fields)


def value_register(
    name: str,
    offset: int,
    role: InstructionFieldRole,
    description: str,
) -> InstructionField:
    """Declares one encoded u8 value-register ordinal."""

    return InstructionField(
        name=name,
        offset=offset,
        encoding_id=U8.entity_id,
        role=role,
        description=description,
        validation=(RuleUse(REGISTER_VALUE.entity_id),),
    )


def ref_register(
    name: str,
    offset: int,
    role: InstructionFieldRole,
    description: str,
    runtime_ref_policy: RuntimeRefPolicy,
) -> InstructionField:
    """Declares one encoded u8 ref-register ordinal and runtime policy."""

    return InstructionField(
        name=name,
        offset=offset,
        encoding_id=U8.entity_id,
        role=role,
        description=description,
        validation=(RuleUse(REGISTER_REF.entity_id),),
        runtime_ref_policy=runtime_ref_policy,
    )


def function_register(
    name: str,
    offset: int,
    role: InstructionFieldRole,
    description: str,
) -> InstructionField:
    """Declares one encoded u8 function-register ordinal."""

    return InstructionField(
        name=name,
        offset=offset,
        encoding_id=U8.entity_id,
        role=role,
        description=description,
        validation=(RuleUse(REGISTER_FUNCTION.entity_id),),
    )


def zero_padding(
    name: str,
    offset: int,
    byte_length: int,
) -> InstructionField:
    """Declares a complete byte range of canonical zero padding."""

    return InstructionField(
        name=name,
        offset=offset,
        encoding_id=U8.entity_id,
        role=InstructionFieldRole.PADDING,
        description="Canonical zero padding.",
        validation=(RuleUse(ZERO.entity_id),),
        array_length=byte_length,
    )
