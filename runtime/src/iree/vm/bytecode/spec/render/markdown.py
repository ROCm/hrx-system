# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Renders the normative Markdown view of a VM specification."""

from __future__ import annotations

from iree.vm.bytecode.spec import module
from iree.vm.bytecode.spec.isa.core import rules as core_rules
from iree.vm.bytecode.spec.module import rules as module_rules
from iree.vm.bytecode.spec.specification import Specification


def _instruction_rule(rule) -> str:
    return {
        core_rules.FieldRule.ANY_BITS: "Any bit pattern.",
        core_rules.FieldRule.ZERO: "Must be zero.",
        core_rules.FieldRule.REGISTER_VALUE: "Must name an in-range value register.",
        core_rules.FieldRule.CONTROL_TARGET_S16: "Must resolve to an in-function `control.block` using widened signed arithmetic.",
        core_rules.FieldRule.CONTROL_TARGET_S32: "Must resolve to an in-function `control.block` using widened signed arithmetic.",
    }[rule]


def _module_rule(rule) -> str:
    if isinstance(rule, module.FieldRuleUse):
        if rule.kind == module_rules.FieldRule.EXACT_BYTES:
            return f"Must equal `{rule.data!r}` byte-for-byte."
        if rule.kind == module_rules.FieldRule.ALLOWED_RANGE:
            return (
                f"Must be in the inclusive range [{rule.values[0]}, {rule.values[1]}]."
            )
        if rule.kind == module_rules.FieldRule.SIGNATURE_DESCRIPTOR:
            return "Must match the scalar, ref, or function kind in `kind_u16`."
        raise ValueError(f"unsupported module field rule {rule!r}")
    if rule == module_rules.FieldRule.ANY_BITS:
        return "Any bit pattern."
    if rule == module_rules.FieldRule.ZERO:
        return "Must be zero."
    if rule == module_rules.FieldRule.CORE_MAJOR:
        return "Must equal the loader's Core major version."
    if rule == module_rules.FieldRule.CORE_REQUIRED_MINOR:
        return "Must not exceed the loader's supported Core minor version."
    raise ValueError(f"unsupported module field rule {rule!r}")


def _field_table_header() -> list[str]:
    return [
        "| Offset | Bytes | Field | Encoding | Role | Verification |",
        "| ---: | ---: | --- | --- | --- | --- |",
    ]


def _statements(lines: list[str], heading: str, statements: tuple[str, ...]) -> None:
    if not statements:
        return
    lines.extend([f"#### {heading}", ""])
    lines.extend(f"- {statement}" for statement in statements)
    lines.append("")


def render_specification(specification: Specification) -> str:
    """Renders every normative fact in the selected specification view."""

    version = specification.version
    lines = [
        "# IREE VM bytecode specification",
        "",
        f"Core version: **{version.major}.{version.minor}**",
        "",
        "This document is generated from the authoritative Python declaration model. ",
        "The current projection is a representative reconstruction slice, not the complete Core ISA.",
        "All multi-byte scalars are little-endian. Module records use their stated natural alignment; instruction records begin on four-byte boundaries and have four-byte-multiple lengths.",
        "",
        "## Module records",
        "",
    ]
    for ordinal, record in enumerate(specification.records):
        lines.extend(
            [
                f"### `{record.name}`",
                "",
                f"Ordinal **{ordinal}**; C overlay `{record.c_type}`; {record.byte_length} bytes; {record.alignment}-byte natural alignment; introduced in Core {record.since.major}.{record.since.minor}.",
                "",
                record.summary,
                "",
                record.contract,
                "",
                *_field_table_header(),
            ]
        )
        for wire_field, offset in zip(record.fields, record.field_offsets, strict=True):
            field = wire_field.field
            lines.append(
                f"| {offset} | {field.byte_length} | `{field.name}` | `{field.encoding.name}` | module field | {_module_rule(wire_field.rule)} {field.summary} |"
            )
        lines.append("")
    lines.extend(["## Core instruction set", ""])
    for family in specification.families:
        family_instructions = tuple(
            instruction
            for instruction in specification.instructions
            if instruction.family == family
        )
        lines.extend(
            [
                f"### {family.name}",
                "",
                family.summary,
                "",
                family.contract,
                "",
                "| Opcode | Mnemonic | Bytes | Control |",
                "| ---: | --- | ---: | --- |",
            ]
        )
        for instruction in family_instructions:
            lines.append(
                f"| `0x{instruction.opcode:02X}` | [`{instruction.mnemonic}`](#{instruction.mnemonic.replace('.', '-')}) | {instruction.byte_length} | `{instruction.control_flow.value}` |"
            )
        lines.append("")
        for instruction in family_instructions:
            lines.extend(
                [
                    f'<a id="{instruction.mnemonic.replace(".", "-")}"></a>',
                    f"#### `{instruction.mnemonic}`",
                    "",
                    f"Opcode `0x{instruction.opcode:02X}`; {instruction.byte_length} bytes; introduced in Core {instruction.since.major}.{instruction.since.minor}.",
                    "",
                    instruction.summary,
                    "",
                    instruction.behavior,
                    "",
                    *_field_table_header(),
                    "| 0 | 1 | `opcode` | `u8` | opcode | Must equal the opcode above. |",
                ]
            )
            for instruction_field, offset in zip(
                instruction.fields, instruction.field_offsets, strict=True
            ):
                field = instruction_field.field
                lines.append(
                    f"| {offset} | {field.byte_length} | `{field.name}` | `{field.encoding.name}` | `{instruction_field.role.value}` | {_instruction_rule(instruction_field.rule)} {field.summary} |"
                )
            lines.extend(
                ["", "Assembly:", "", "```loom", instruction.assembly, "```", ""]
            )
            _statements(lines, "Preconditions", instruction.preconditions)
            _statements(lines, "Successful execution", instruction.success)
            if instruction.state_effects:
                _statements(
                    lines,
                    "State effects",
                    tuple(
                        f"`{effect.access.value}` `{effect.resource.value}`"
                        + (
                            f" selected by {', '.join(effect.resource_fields)}"
                            if effect.resource_fields
                            else ""
                        )
                        + "."
                        for effect in instruction.state_effects
                    ),
                )
            if instruction.failures:
                lines.extend(
                    [
                        "#### Failures",
                        "",
                        "| Status | Condition | Atomicity |",
                        "| --- | --- | --- |",
                    ]
                )
                lines.extend(
                    f"| `{failure.status}` | {failure.condition} | {failure.atomicity} |"
                    for failure in instruction.failures
                )
                lines.append("")
            _statements(lines, "Ownership", instruction.ownership)
            if instruction.pseudocode:
                lines.extend(
                    [
                        "#### Reference pseudocode",
                        "",
                        "```c",
                        instruction.pseudocode,
                        "```",
                        "",
                    ]
                )
    return "\n".join(lines)
