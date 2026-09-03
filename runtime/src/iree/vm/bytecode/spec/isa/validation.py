# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Structural validation for physical VM instruction declarations."""

import re

from iree.vm.bytecode.spec import isa
from iree.vm.bytecode.spec.schema import I16, I32, U8, U16, U32

_MNEMONIC_PATTERN = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+")
_FIELD_RULE_ENCODINGS = {
    isa.FieldRule.VALUE_REGISTER: U8,
    isa.FieldRule.CONTROL_TARGET_S16: I16,
    isa.FieldRule.CONTROL_TARGET_S32: I32,
}


def validate_instruction(instruction: isa.Instruction) -> None:
    """Rejects an internally inconsistent instruction declaration."""

    if not 1 <= instruction.opcode <= 0xEF:
        raise ValueError(f"{instruction.mnemonic}: invalid core opcode")
    if not _MNEMONIC_PATTERN.fullmatch(instruction.mnemonic):
        raise ValueError(f"invalid instruction mnemonic {instruction.mnemonic!r}")
    if not instruction.summary.strip() or not instruction.behavior.strip():
        raise ValueError(f"{instruction.mnemonic}: missing normative behavior")
    if not instruction.success or not instruction.assembly.strip():
        raise ValueError(f"{instruction.mnemonic}: incomplete success contract")
    statements = (
        *instruction.success,
        *instruction.preconditions,
        *instruction.ownership,
    )
    if any(not statement.strip() for statement in statements) or any(
        not failure.status.strip()
        or not failure.condition.strip()
        or not failure.atomicity.strip()
        for failure in instruction.failures
    ):
        raise ValueError(f"{instruction.mnemonic}: empty normative statement")
    if instruction.byte_length % 4:
        raise ValueError(f"{instruction.mnemonic}: record is not four-byte framed")

    _ = instruction.field_offsets
    fields = {field.field.name: field for field in instruction.fields}
    for instruction_field in instruction.fields:
        field = instruction_field.field
        rule = instruction_field.rule
        if not isinstance(rule, isa.FieldRule) or not isinstance(
            instruction_field.role, isa.FieldRole
        ):
            raise ValueError(f"{instruction.mnemonic}: unknown field contract")
        expected_encoding = _FIELD_RULE_ENCODINGS.get(rule)
        if expected_encoding is not None and field.encoding != expected_encoding:
            raise ValueError(f"{instruction.mnemonic}: field rule has wrong encoding")
        if rule == isa.FieldRule.VALUE_REGISTER and instruction_field.role not in (
            isa.FieldRole.OPERAND,
            isa.FieldRole.RESULT,
        ):
            raise ValueError(f"{instruction.mnemonic}: invalid value-register field")
        if (
            instruction_field.role == isa.FieldRole.PADDING
            and rule != isa.FieldRule.ZERO
        ):
            raise ValueError(f"{instruction.mnemonic}: padding is not canonical zero")
        if (
            rule
            in (
                isa.FieldRule.CONTROL_TARGET_S16,
                isa.FieldRule.CONTROL_TARGET_S32,
            )
            and instruction_field.role != isa.FieldRole.IMMEDIATE
        ):
            raise ValueError(f"{instruction.mnemonic}: target is not an immediate")

    direct_targets = sum(
        field.rule
        in (isa.FieldRule.CONTROL_TARGET_S16, isa.FieldRule.CONTROL_TARGET_S32)
        for field in instruction.fields
    )
    expects_direct_target = instruction.control_flow in (
        isa.ControlFlow.BRANCH,
        isa.ControlFlow.CONDITIONAL_BRANCH,
        isa.ControlFlow.YIELD,
    )
    if direct_targets != int(expects_direct_target):
        raise ValueError(f"{instruction.mnemonic}: invalid direct-target count")
    if (instruction.suspension == isa.Suspension.ALWAYS) != (
        instruction.control_flow == isa.ControlFlow.YIELD
    ):
        raise ValueError(f"{instruction.mnemonic}: inconsistent suspension contract")

    return_rules = [
        rule for rule in instruction.rules if rule == isa.RecordRule.RETURN_SIGNATURE
    ]
    switch_rules = [
        rule for rule in instruction.rules if isinstance(rule, isa.SwitchTargetsRule)
    ]
    if len(return_rules) + len(switch_rules) != len(instruction.rules):
        raise ValueError(f"{instruction.mnemonic}: unknown record rule")
    if (len(return_rules) == 1) != (instruction.control_flow == isa.ControlFlow.RETURN):
        raise ValueError(f"{instruction.mnemonic}: inconsistent return-signature rule")
    if (len(switch_rules) == 1) != (instruction.control_flow == isa.ControlFlow.SWITCH):
        raise ValueError(f"{instruction.mnemonic}: inconsistent switch-target rule")
    if switch_rules:
        rule = switch_rules[0]
        if rule.count_field not in fields or rule.base_field not in fields:
            raise ValueError(f"{instruction.mnemonic}: switch rule names missing field")
        encodings = (
            fields[rule.count_field].field.encoding,
            fields[rule.base_field].field.encoding,
        )
        if encodings != (U16, U32):
            raise ValueError(f"{instruction.mnemonic}: invalid switch field encodings")
        roles = (fields[rule.count_field].role, fields[rule.base_field].role)
        if roles != (isa.FieldRole.CONSTRAINT_MEMBER,) * 2:
            raise ValueError(f"{instruction.mnemonic}: invalid switch field roles")

    if isinstance(instruction.semantics, isa.IntegerBinarySemantics):
        if (
            not isinstance(instruction.semantics.operation, isa.IntegerBinaryOperation)
            or instruction.control_flow != isa.ControlFlow.SEQUENTIAL
            or instruction.semantics.bit_width not in (32, 64)
        ):
            raise ValueError(f"{instruction.mnemonic}: invalid integer semantics")
    elif isinstance(instruction.semantics, isa.BranchCondition):
        expected_flow = (
            isa.ControlFlow.BRANCH
            if instruction.semantics == isa.BranchCondition.ALWAYS
            else isa.ControlFlow.CONDITIONAL_BRANCH
        )
        if instruction.control_flow != expected_flow:
            raise ValueError(f"{instruction.mnemonic}: mismatched branch semantics")
    elif instruction.semantics is not None or instruction.control_flow not in (
        isa.ControlFlow.BLOCK,
        isa.ControlFlow.RETURN,
        isa.ControlFlow.YIELD,
        isa.ControlFlow.SWITCH,
    ):
        raise ValueError(f"{instruction.mnemonic}: unknown instruction semantics")

    if len(set(instruction.state_effects)) != len(instruction.state_effects):
        raise ValueError(f"{instruction.mnemonic}: duplicate state effect")
    for effect in instruction.state_effects:
        if not isinstance(effect.access, isa.StateAccess) or not isinstance(
            effect.resource, isa.StateResource
        ):
            raise ValueError(f"{instruction.mnemonic}: unknown state effect")
        if any(field_name not in fields for field_name in effect.resource_fields):
            raise ValueError(
                f"{instruction.mnemonic}: state effect names missing field"
            )
