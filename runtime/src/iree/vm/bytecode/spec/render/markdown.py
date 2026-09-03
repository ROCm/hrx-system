# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Renders the normative Markdown view of a VM specification."""

from __future__ import annotations

from iree.vm.bytecode.spec.isa.core import rules as core_rules
from iree.vm.bytecode.spec.module import rules as module_rules
from iree.vm.bytecode.spec.schema import NumericKind, UnknownNumericPolicy
from iree.vm.bytecode.spec.specification import Specification


def _instruction_rule(instruction_field) -> str:
    rule = instruction_field.rule
    text = rule.kind.summary
    if rule.kind == core_rules.FieldRule.SELECTOR:
        text = f"Must be an assigned `{rule.data.name}` selector."
    if not text:
        raise ValueError(f"unsupported instruction field rule {rule!r}")
    policy = instruction_field.ref_policy
    if policy:
        text += (
            f" Ref type `{policy.type_contract}`; null policy "
            f"`{policy.null_policy.value}`; ownership `{policy.ownership.value}`."
        )
    return text


def _module_rule(rule) -> str:
    kind = rule.kind
    text = kind.summary
    if text:
        return text
    if kind == module_rules.FieldRule.ALLOWED_BITS:
        return f"May set only bits in `0x{rule.values[0]:X}`."
    if kind == module_rules.FieldRule.ALLOWED_RANGE:
        return f"Must be in the inclusive range [{rule.values[0]}, {rule.values[1]}]."
    if kind == module_rules.FieldRule.EXACT_BYTES:
        return f"Must equal `{rule.data!r}` byte-for-byte."
    if kind == module_rules.FieldRule.MULTIPLE:
        return f"Must be an exact multiple of {rule.values[0]}."
    if kind == module_rules.FieldRule.BYTE_ALIGNMENT:
        return f"Must be a power-of-two byte alignment of at least {rule.values[0]}."
    if kind == module_rules.FieldRule.ORDINAL:
        return f"Must be an in-range `{rule.data.name.lower()}` ordinal."
    if kind == module_rules.FieldRule.ORDINAL_OR_NULL:
        return (
            f"Must be an in-range `{rule.data.name.lower()}` ordinal or canonical "
            f"null `0x{rule.values[0]:X}`."
        )
    if kind == module_rules.FieldRule.SIGNATURE_DESCRIPTOR:
        return f"Must match the scalar, ref, or function kind in `{rule.fields[0]}`."
    raise ValueError(f"unsupported module field rule {rule!r}")


def _field_table_header() -> list[str]:
    return [
        "| Offset | Bytes | Field | Encoding | Role | Verification |",
        "| ---: | ---: | --- | --- | --- | --- |",
    ]


def _render_record(lines: list[str], record, ordinal: int) -> None:
    lines.extend(
        [
            f"#### `{record.name}`",
            "",
            f"Record ordinal **{ordinal}**; C overlay `{record.c_type}`; "
            f"{record.byte_length} bytes; {record.alignment}-byte natural alignment; "
            f"introduced in Core {record.since.major}.{record.since.minor}.",
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
            f"| {offset} | {field.byte_length} | `{field.name}` | "
            f"`{field.encoding.name}` | module field | "
            f"{_module_rule(wire_field.rule)} {field.summary} |"
        )
    lines.append("")


def _render_constraints(lines: list[str], constraints) -> None:
    lines.extend(
        [
            "| Obligation | Inputs | Contract |",
            "| --- | --- | --- |",
        ]
    )
    for constraint in constraints:
        inputs = ", ".join(
            f"`{reference.record.name}"
            + (f".{reference.field_name}" if reference.field_name else "")
            + "`"
            for reference in constraint.inputs
        )
        lines.append(
            f"| `{constraint.name}` | {inputs or 'none'} | {constraint.contract} |"
        )
    lines.append("")


def _render_numeric_tables(lines, heading, tables) -> None:
    lines.extend([heading, ""])
    for table in tables:
        unknowns = (
            "preserved when nonzero"
            if table.unknown_policy == UnknownNumericPolicy.PRESERVE_NONZERO
            else "rejected"
        )
        lines.extend(
            [
                f"#### `{table.name}`",
                "",
                table.summary,
                "",
                f"Encoding `{table.encoding.name}`; kind `{table.kind.value}`; "
                f"unknown values are {unknowns}.",
                "",
                "| Value | Name | Meaning | Introduced |",
                "| ---: | --- | --- | --- |",
            ]
        )
        width = table.encoding.byte_length * 2
        for value in table.values:
            bit = "bit " if table.kind == NumericKind.FLAGS else ""
            lines.append(
                f"| {bit}`0x{value.value:0{width}X}` | `{value.name}` | "
                f"{value.summary} | Core {value.since.major}.{value.since.minor} |"
            )
        lines.append("")


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
        "This document is generated from the authoritative Python declaration model.",
        "It contains the module container and every Core declaration in this versioned view.",
        "All multi-byte scalars are little-endian. Module records use their stated natural alignment; instruction records begin on four-byte boundaries and have four-byte-multiple lengths.",
        "",
        "## Module container",
        "",
        specification.module_format.summary,
        "",
        specification.module_format.contract,
        "",
        f"The image base and every section payload are aligned to at least {specification.module_format.image_alignment} bytes. A directory row may require a greater power-of-two payload alignment.",
        "",
    ]
    _render_numeric_tables(
        lines,
        "### Module numeric domains",
        specification.module_format.numeric_tables,
    )
    record_ordinals = {
        record: ordinal
        for ordinal, record in enumerate(specification.module_format.records)
    }
    lines.extend(["### Image envelope", ""])
    for record in specification.module_format.envelope:
        _render_record(lines, record, record_ordinals[record])
    lines.extend(["### Container verification obligations", ""])
    _render_constraints(lines, specification.module_format.constraints)
    lines.extend(["## Module sections", ""])
    for section in specification.module_format.sections:
        lines.extend(
            [
                f"### `{section.name}`",
                "",
                f"Section type `0x{section.section_type:04X}`; required flags "
                f"`0x{section.required_flags:04X}`; introduced in Core "
                f"{section.since.major}.{section.since.minor}.",
                "",
                section.summary,
                "",
                section.contract,
                "",
            ]
        )
        for record in section.records:
            _render_record(lines, record, record_ordinals[record])
        lines.extend(["#### Structural verification obligations", ""])
        _render_constraints(lines, section.constraints)
    _render_numeric_tables(lines, "## Core selector domains", specification.selectors)
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
                    f"| {offset} | {field.byte_length} | `{field.name}` | `{field.encoding.name}` | `{instruction_field.role.value}` | {_instruction_rule(instruction_field)} {field.summary} |"
                )
            if instruction.rules:
                lines.append("")
            _statements(
                lines,
                "Record verification",
                tuple(rule.summary for rule in instruction.rules),
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
