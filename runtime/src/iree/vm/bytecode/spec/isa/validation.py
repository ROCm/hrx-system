# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cross-declaration validation for physical VM instructions."""

from iree.vm.bytecode.spec import isa
from iree.vm.bytecode.spec.isa.core import rules
from iree.vm.bytecode.spec.schema import NumericTable, require


def _field_rule_parts(rule):
    if isinstance(rule, isa.FieldRuleUse):
        return rule.kind, rule.fields, rule.values, rule.data
    return rule, (), (), None


def _validate_field(instruction, instruction_field, fields, tables) -> None:
    field = instruction_field.field
    kind, related_fields, values, data = _field_rule_parts(instruction_field.rule)
    table = data if isinstance(data, NumericTable) else None
    policy = instruction_field.ref_policy
    valid = (
        kind in rules.FIELD_RULES
        and kind.accepts(related_fields, values, data=data)
        and all(name in fields for name in related_fields)
        and (kind.encoding is None or field.encoding == kind.encoding)
        and (table is None or table in tables and field.encoding == table.encoding)
        and (
            kind not in rules.DIRECT_TARGET_RULES
            or instruction_field.role == isa.FieldRole.IMMEDIATE
        )
        and (kind == rules.FieldRule.REGISTER_REF) == (policy is not None)
    )
    require(valid, f"{instruction.mnemonic}: malformed field rule")
    is_padding = instruction_field.role == isa.FieldRole.PADDING
    require(
        is_padding == (kind == rules.FieldRule.ZERO),
        f"{instruction.mnemonic}: padding is not canonical zero",
    )


def _validate_record_rule(instruction, rule, fields) -> None:
    valid = rule.kind in rules.RECORD_RULES
    valid &= rule.kind.accepts(rule.fields, rule.values, rule.names)
    require(valid, f"{instruction.mnemonic}: malformed record rule")
    require(
        all(name in fields for name in rule.fields),
        f"{instruction.mnemonic}: record rule names missing field",
    )


def validate_instruction(
    instruction: isa.Instruction, tables: tuple[NumericTable, ...] = ()
) -> None:
    valid = 1 <= instruction.opcode <= 0xEF
    valid &= not instruction.byte_length % 4
    require(valid, f"{instruction.mnemonic}: invalid instruction identity or layout")
    _ = instruction.field_offsets
    fields = {item.field.name: item for item in instruction.fields}
    for instruction_field in instruction.fields:
        _validate_field(instruction, instruction_field, fields, tables)

    target_count = sum(
        _field_rule_parts(item.rule)[0] in rules.DIRECT_TARGET_RULES
        for item in instruction.fields
    )
    for rule in instruction.rules:
        _validate_record_rule(instruction, rule, fields)
    expected = rules.CONTROL_CONTRACTS.get(
        instruction.control_flow, (0, isa.Suspension.NEVER, ())
    )
    control_rules = tuple(
        rule.kind for rule in instruction.rules if rule.kind in rules.CONTROL_RULES
    )
    require(
        target_count == expected[0]
        and instruction.suspension == expected[1]
        and len(control_rules) == int(bool(expected[2]))
        and all(kind in expected[2] for kind in control_rules),
        f"{instruction.mnemonic}: inconsistent control rule",
    )

    valid = all(
        (effect.access == rules.StateAccess.UNKNOWN)
        == (effect.resource == rules.StateResource.ANY)
        and all(name in fields for name in effect.resource_fields)
        for effect in instruction.state_effects
    )
    require(valid, f"{instruction.mnemonic}: invalid state effects")
