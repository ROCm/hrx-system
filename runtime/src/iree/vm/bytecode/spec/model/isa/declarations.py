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
        semantics=semantics,
    )


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
