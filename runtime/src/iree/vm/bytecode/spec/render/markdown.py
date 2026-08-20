# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Standalone normative Markdown projections."""

from __future__ import annotations

import enum
import posixpath
from collections.abc import Callable, Iterable
from typing import cast

from model.isa import Instruction, InstructionFamily
from model.module import (
    EnvelopeRecord,
    ModuleFormat,
    OrdinalDomain,
    Section,
    SectionRecord,
    ValidationObligation,
)
from model.schema import (
    EntityReference,
    FieldReference,
    NumericTable,
    RuleUse,
    ScalarEncoding,
    ValidationRule,
    WireRecord,
    selected_numeric_values,
    selected_record_layouts,
)
from model.specification import Entity, NormativeClause, Projection


def _anchor(entity: Entity) -> str:
    return f'<a id="{entity.normative_anchor}"></a>'


EntityLink = Callable[[str], str]


def _local_entity_link(entity_id: str) -> str:
    anchor = "spec-" + entity_id.replace(".", "-").replace("_", "-")
    return f"[`{entity_id}`](#{anchor})"


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _list(lines: list[str], values: Iterable[str]) -> None:
    items = tuple(values)
    if not items:
        lines.append("None.")
        lines.append("")
        return
    for value in items:
        lines.append(f"- {value}")
    lines.append("")


def _argument(value: object, entity_link: EntityLink) -> str:
    if isinstance(value, EntityReference):
        return entity_link(value.entity_id)
    if isinstance(value, FieldReference):
        return f"field `{value.field_name}`"
    if isinstance(value, bytes):
        return "`" + " ".join(f"{byte:02X}" for byte in value) + "`"
    if isinstance(value, enum.Enum):
        return f"`{value.value}`"
    if isinstance(value, tuple):
        return (
            "[" + ", ".join(_argument(element, entity_link) for element in value) + "]"
        )
    if isinstance(value, str):
        return f"`{value}`"
    if isinstance(value, int):
        return f"`{value}`"
    return f"`{value!r}`"


def _rule_use(rule_use: RuleUse, entity_link: EntityLink) -> str:
    result = entity_link(rule_use.rule_id)
    if rule_use.arguments:
        result += (
            "("
            + ", ".join(_argument(value, entity_link) for value in rule_use.arguments)
            + ")"
        )
    return result


def _render_preamble(lines: list[str], title: str, projection: Projection) -> None:
    lines.extend(
        (
            "<!-- GENERATED FILE: DO NOT EDIT. -->",
            "<!-- Rebuild with: python dev.py docs build -->",
            "",
            f"# {title}",
            "",
            "This document is generated from the declarative Python specification. "
            "The Python declarations are the editable authority; this Markdown is "
            "their complete normative human projection.",
            "",
            f"Projection: `{projection.version_label()}`.",
            "",
            "An entity's introduction version is the first specification "
            "revision that names and interprets it. Its minimum-consumer "
            "version is the oldest revision that can safely accept the "
            "encoding, including by preserving or skipping it without "
            "interpretation.",
            "",
        )
    )


def render_specification_index_markdown(
    module_projection: Projection,
    core_projection: Projection,
    hal_projection: Projection,
) -> str:
    """Renders the published VM bytecode specification landing page."""

    lines = [
        "<!-- GENERATED FILE: DO NOT EDIT. -->",
        "<!-- Rebuild with: python dev.py docs build -->",
        "",
        "# IREE VM Bytecode Specification",
        "",
        "This reference is generated from the declarative Python specification "
        "that defines the VM module container and instruction encodings. The "
        "Python declarations are the editable authority; these pages are their "
        "complete normative human projection.",
        "",
        "- [Module container](module-format.md) defines the mmap-compatible image "
        "format and its verification obligations.",
        "- [Core ISA](isa/core/index.md) defines the required machine, encoding, "
        "verification, and instruction contracts.",
        "- [HAL ISA](isa/hal/index.md) defines the optional hardware abstraction "
        "instruction page.",
        "",
        "## Published Projection",
        "",
        "| Surface | Version projection |",
        "| --- | --- |",
        f"| Module container | `{module_projection.version_label()}` |",
        f"| Core ISA | `{core_projection.version_label()}` |",
        f"| HAL ISA | `{hal_projection.version_label()}` |",
        "",
        "Each entity records both its introduction version and its minimum "
        "consumer version so tools can project older views without maintaining "
        "parallel specifications.",
        "",
    ]
    return "\n".join(lines)


def _render_encodings(lines: list[str], entities: Iterable[Entity]) -> None:
    encodings = [entity for entity in entities if isinstance(entity, ScalarEncoding)]
    if not encodings:
        return
    lines.extend(("## Scalar Encodings", ""))
    for encoding in encodings:
        lines.extend(
            (
                _anchor(encoding),
                f"### `{encoding.entity_id}`",
                "",
                encoding.summary,
                "",
                f"C type: `{encoding.c_type}`. Wire bytes: {encoding.byte_length}. "
                f"Natural alignment: {encoding.alignment}.",
                "",
            )
        )


def _render_validation_rules(lines: list[str], entities: Iterable[Entity]) -> None:
    rules = [entity for entity in entities if isinstance(entity, ValidationRule)]
    if not rules:
        return
    lines.extend(("## Validation Rule Catalog", ""))
    for rule in rules:
        parameters = ", ".join(
            f"`{parameter.name}: {parameter.shape.kind.value}`"
            for parameter in rule.parameters
        )
        lines.extend(
            (
                _anchor(rule),
                f"### `{rule.entity_id}`",
                "",
                rule.summary,
                "",
                f"Scope: `{rule.scope.value}`. Parameters: {parameters or 'none'}.",
                "",
                rule.normative_text,
                "",
            )
        )


def _render_numeric_tables(
    lines: list[str],
    projection: Projection,
    entities: Iterable[Entity],
    entity_link: EntityLink,
) -> None:
    tables = [entity for entity in entities if isinstance(entity, NumericTable)]
    if not tables:
        return
    values_by_table = selected_numeric_values(projection)
    lines.extend(("## Numeric Tables", ""))
    for table in tables:
        lines.extend(
            (
                _anchor(table),
                f"### `{table.entity_id}`",
                "",
                table.summary,
                "",
                f"Kind: `{table.table_kind.value}`. Encoding: "
                f"{entity_link(table.encoding_id)}. Unknown values: "
                f"`{table.unknown_value_policy.value}`.",
                "",
                "| Value | Name | Introduced | Minimum consumer | Meaning |",
                "| ---: | --- | --- | --- | --- |",
            )
        )
        for value in values_by_table[table.entity_id]:
            minimum_consumer = value.minimum_consumer_version
            if minimum_consumer is None:
                raise AssertionError(f"{value.entity_id}: missing consumer version")
            lines.append(
                f"| {_anchor(value)}`0x{value.value:X}` | `{_cell(value.name)}` | "
                f"`{value.since.major}.{value.since.minor}` | "
                f"`{minimum_consumer.major}.{minimum_consumer.minor}` | "
                f"{_cell(value.summary)} |"
            )
        lines.append("")


def render_module_markdown(projection: Projection) -> str:
    """Renders a complete standalone module-container specification."""

    lines: list[str] = []
    _render_preamble(lines, "IREE VM Module Container", projection)
    entities = projection.entities
    entity_map = projection.entity_map()
    formats = [entity for entity in entities if isinstance(entity, ModuleFormat)]
    if len(formats) != 1:
        raise ValueError("module projection must contain exactly one format")
    module_format = formats[0]
    lines.extend(
        (
            _anchor(module_format),
            "## Image Contract",
            "",
            module_format.summary,
            "",
            module_format.normative_text,
            "",
            f"The image base is aligned to {module_format.image_alignment} bytes. "
            "Every section begins at an image-relative offset aligned to "
            f"{module_format.section_alignment} bytes. Natural field alignment is "
            "part of the "
            "wire format so a verified little-endian image may be mmap-overlaid on "
            "alignment-sensitive targets.",
            "",
        )
    )

    clauses = [entity for entity in entities if isinstance(entity, NormativeClause)]
    if clauses:
        lines.extend(("## Module Contracts", ""))
        for clause in clauses:
            lines.extend(
                (
                    _anchor(clause),
                    f"### `{clause.entity_id}`",
                    "",
                    clause.summary,
                    "",
                    clause.normative_text,
                    "",
                )
            )

    ordinal_domains = [
        entity for entity in entities if isinstance(entity, OrdinalDomain)
    ]
    if ordinal_domains:
        lines.extend(("## Ordinal Domains", ""))
        for domain in ordinal_domains:
            base = (
                _local_entity_link(domain.base_domain_id)
                if domain.base_domain_id is not None
                else "none"
            )
            lines.extend(
                (
                    _anchor(domain),
                    f"### `{domain.entity_id}`",
                    "",
                    domain.summary,
                    "",
                    f"Maximum count: {domain.maximum_count}. Base domain: {base}. "
                    f"Requires nonempty values: `{domain.require_nonempty_value}`.",
                    "",
                )
            )

    _render_numeric_tables(
        lines,
        projection,
        entities,
        _local_entity_link,
    )
    _render_encodings(lines, entities)

    selected_layouts = selected_record_layouts(projection)
    records = [entity for entity in entities if isinstance(entity, WireRecord)]
    lines.extend(("## Fixed Wire Records", ""))
    for record in records:
        layout = selected_layouts[record.entity_id]
        lines.extend(
            (
                _anchor(record),
                f"### `{record.c_type}`",
                "",
                record.summary,
                "",
                _anchor(layout),
                f"Selected layout: {_local_entity_link(layout.entity_id)}; "
                f"{layout.byte_length} bytes, alignment {layout.alignment}.",
                "",
                "| Offset | Bytes | Field | Meaning | Verification |",
                "| ---: | ---: | --- | --- | --- |",
            )
        )
        for field in layout.fields:
            encoding = cast(ScalarEncoding, entity_map[field.encoding_id])
            field_bytes = encoding.byte_length * field.array_length
            checks = "; ".join(
                _rule_use(use, _local_entity_link) for use in field.validation
            )
            lines.append(
                f"| {field.offset} | {field_bytes} | `{field.name}` | "
                f"{_cell(field.description)} | {_cell(checks)} |"
            )
        lines.append("")

    envelopes = sorted(
        (entity for entity in entities if isinstance(entity, EnvelopeRecord)),
        key=lambda value: value.document_order,
    )
    lines.extend(("## Image Envelope", ""))
    for envelope in envelopes:
        lines.append(
            f"- {_anchor(envelope)}{_local_entity_link(envelope.record_id)} — "
            f"{envelope.summary}"
        )
    lines.append("")

    sections = sorted(
        (entity for entity in entities if isinstance(entity, Section)),
        key=lambda value: value.section_type,
    )
    section_members = [
        entity for entity in entities if isinstance(entity, SectionRecord)
    ]
    lines.extend(("## Sections", ""))
    for section in sections:
        lines.extend(
            (
                _anchor(section),
                f"### `0x{section.section_type:04X}` — `{section.entity_id}`",
                "",
                section.summary,
                "",
                f"Required flags: `0x{section.required_flags:04X}`.",
                "",
                section.grammar,
                "",
                section.normative_text,
                "",
                "Record grammar:",
                "",
            )
        )
        members = sorted(
            (
                value
                for value in section_members
                if value.section_id == section.entity_id
            ),
            key=lambda value: value.document_order,
        )
        for member in members:
            lines.append(
                f"- {_anchor(member)}{_local_entity_link(member.record_id)} — "
                f"{member.summary}"
            )
        lines.append("")

    obligations = [
        entity for entity in entities if isinstance(entity, ValidationObligation)
    ]
    lines.extend(("## Loader Verification Obligations", ""))
    for obligation in obligations:
        inputs = ", ".join(
            _local_entity_link(value.record_id)
            + (f".`{value.field_name}`" if value.field_name is not None else "")
            for value in obligation.inputs
        )
        lines.extend(
            (
                _anchor(obligation),
                f"### `{obligation.entity_id}`",
                "",
                obligation.summary,
                "",
                f"Scope: `{obligation.scope.value}`. Kind: `{obligation.kind}`. "
                f"Inputs: {inputs or 'none'}.",
                "",
                obligation.normative_text,
                "",
            )
        )

    _render_validation_rules(lines, entities)
    return "\n".join(lines).rstrip() + "\n"


def _render_range_groups(lines: list[str], instruction: Instruction) -> None:
    if not instruction.range_groups:
        return
    lines.extend(("Counted ranges:", ""))
    for group in instruction.range_groups:
        lines.append(f"- `{group.name}` uses count field `{group.count_field}`:")
        for member in group.members:
            ref_policy = ""
            if member.runtime_ref_policy is not None:
                policy = member.runtime_ref_policy
                ref_policy = (
                    f", `{policy.type_contract}` / `{policy.null_policy.value}` / "
                    f"`{policy.ownership.value}`"
                )
            lines.append(
                f"  - `{member.base_field}`: `{member.storage.value}`, "
                f"{member.element_byte_length}-byte elements aligned to "
                f"{member.element_alignment}{ref_policy}."
            )
    lines.append("")


def _render_instruction(
    lines: list[str],
    instruction: Instruction,
    projection: Projection,
    entity_link: EntityLink,
    heading_level: int = 3,
) -> None:
    heading = "#" * heading_level
    detail_heading = "#" * (heading_level + 1)
    entity_map = projection.entity_map()
    domain = next(
        value
        for value in projection.domains
        if value.domain == instruction.since.domain
    )
    lines.extend(
        (
            _anchor(instruction),
            f"{heading} `{instruction.mnemonic}`",
            "",
            instruction.summary,
            "",
            f"Identity: page `0x{domain.page_id:02X}`, opcode "
            f"`0x{instruction.opcode:02X}`, {instruction.byte_length} bytes, "
            f"control `{instruction.control_flow.value}`, suspension "
            f"`{instruction.suspension.value}`, introduced "
            f"`{instruction.since.major}.{instruction.since.minor}`, minimum "
            f"consumer `{instruction.minimum_consumer_version.major}."
            f"{instruction.minimum_consumer_version.minor}`.",
            "",
            instruction.semantics.description,
            "",
            f"{detail_heading} Encoding",
            "",
            "| Offset | Bytes | Field | Role | Meaning | Verification | Ref policy |",
            "| ---: | ---: | --- | --- | --- | --- | --- |",
        )
    )
    if domain.page_id != 0:
        lines.append(
            f"| 0 | 1 | `page_u8` | header | Page selector `0x{domain.page_id:02X}`. "
            "| exact page | — |"
        )
        opcode_offset = 1
    else:
        opcode_offset = 0
    lines.append(
        f"| {opcode_offset} | 1 | `opcode_u8` | header | Opcode "
        f"`0x{instruction.opcode:02X}`. | exact opcode | — |"
    )
    for field in instruction.fields:
        encoding = cast(ScalarEncoding, entity_map[field.encoding_id])
        field_bytes = encoding.byte_length * field.array_length
        checks = "; ".join(_rule_use(use, entity_link) for use in field.validation)
        if field.runtime_ref_policy is None:
            ref_policy = "—"
        else:
            policy = field.runtime_ref_policy
            ref_policy = (
                f"`{policy.type_contract}` / `{policy.null_policy.value}` / "
                f"`{policy.ownership.value}`"
            )
        lines.append(
            f"| {field.offset} | {field_bytes} | `{field.name}` | "
            f"`{field.role.value}` | {_cell(field.description)} | "
            f"{_cell(checks)} | {_cell(ref_policy)} |"
        )
    lines.append("")
    _render_range_groups(lines, instruction)
    if instruction.constraints:
        lines.extend(("Record constraints:", ""))
        for constraint in instruction.constraints:
            lines.append(f"- {_rule_use(constraint, entity_link)}")
        lines.append("")

    lines.extend((f"{detail_heading} Verification", ""))
    _list(lines, instruction.semantics.verification)
    lines.extend((f"{detail_heading} Dynamic Preconditions", ""))
    _list(lines, instruction.semantics.preconditions)
    lines.extend((f"{detail_heading} Success", ""))
    _list(lines, instruction.semantics.success)
    lines.extend((f"{detail_heading} Failure And Atomicity", ""))
    if instruction.semantics.failures:
        lines.extend(
            (
                "| Status | Condition | Mutation boundary |",
                "| --- | --- | --- |",
            )
        )
        for failure in instruction.semantics.failures:
            lines.append(
                f"| `{failure.status}` | {_cell(failure.condition)} | "
                f"{_cell(failure.atomicity)} |"
            )
        lines.append("")
    else:
        lines.extend(("No dynamic failure remains after successful verification.", ""))
    lines.extend((f"{detail_heading} Ownership And Lifetime", ""))
    _list(lines, instruction.semantics.ownership)
    lines.extend((f"{detail_heading} Assembly", "", "```text"))
    lines.extend(instruction.semantics.assembly)
    lines.extend(("```", "", f"{detail_heading} C-like Pseudocode", "", "```c"))
    lines.extend(instruction.semantics.pseudocode.splitlines())
    lines.extend(("```", ""))


def isa_family_path(family: InstructionFamily) -> str:
    """Returns the published documentation path for one ISA family."""

    marker = ".family."
    _, separator, family_name = family.entity_id.partition(marker)
    if not separator:
        raise ValueError(f"{family.entity_id}: missing family marker")
    return f"isa/{family.since.domain}/{family_name}.md"


def _isa_entity_paths(projection: Projection) -> dict[str, str]:
    families = {
        entity.entity_id: entity
        for entity in projection.entities
        if isinstance(entity, InstructionFamily)
    }
    paths = {
        entity.entity_id: f"isa/{entity.since.domain}/index.md"
        for entity in projection.entities
    }
    for family in families.values():
        paths[family.entity_id] = isa_family_path(family)
    for entity in projection.entities:
        if isinstance(entity, Instruction):
            paths[entity.entity_id] = isa_family_path(families[entity.family_id])
    return paths


def _isa_entity_link(
    entity_paths: dict[str, str],
    source_path: str,
) -> EntityLink:
    def link(entity_id: str) -> str:
        target_path = entity_paths.get(entity_id)
        if target_path is None:
            raise KeyError(f"no document path for {entity_id}")
        anchor = "spec-" + entity_id.replace(".", "-").replace("_", "-")
        if target_path == source_path:
            location = f"#{anchor}"
        else:
            relative_path = posixpath.relpath(
                target_path,
                posixpath.dirname(source_path) or ".",
            )
            location = f"{relative_path}#{anchor}"
        return f"[`{entity_id}`]({location})"

    return link


def _render_instruction_summary(
    lines: list[str],
    instructions: Iterable[Instruction],
    entity_link: EntityLink,
) -> None:
    lines.extend(
        (
            "| Opcode | Instruction | Bytes | Control | Suspension | Meaning |",
            "| ---: | --- | ---: | --- | --- | --- |",
        )
    )
    for instruction in sorted(instructions, key=lambda value: value.opcode):
        lines.append(
            f"| `0x{instruction.opcode:02X}` | "
            f"{entity_link(instruction.entity_id)} | {instruction.byte_length} | "
            f"`{instruction.control_flow.value}` | "
            f"`{instruction.suspension.value}` | {_cell(instruction.summary)} |"
        )
    lines.append("")


def render_isa_index_markdown(
    projection: Projection,
    domain_name: str,
) -> str:
    """Renders one instruction page's contracts, selectors, and family index."""

    lines: list[str] = []
    domain_label = "Core" if domain_name == "core" else domain_name.upper()
    _render_preamble(
        lines,
        f"IREE VM {domain_label} Instruction Set Architecture",
        projection,
    )
    entities = tuple(
        entity for entity in projection.entities if entity.since.domain == domain_name
    )
    if not entities:
        raise ValueError(f"ISA projection has no {domain_name!r} entities")
    source_path = f"isa/{domain_name}/index.md"
    entity_link = _isa_entity_link(_isa_entity_paths(projection), source_path)
    if domain_name != "core":
        lines.extend(
            (
                "This page extends the [Core ISA](../core/index.md) and inherits its "
                "machine, encoding, and verification contracts.",
                "",
            )
        )
    clauses = [entity for entity in entities if isinstance(entity, NormativeClause)]
    if clauses:
        lines.extend(("## Architectural Contracts", ""))
        for clause in clauses:
            lines.extend(
                (
                    _anchor(clause),
                    f"### `{clause.entity_id}`",
                    "",
                    clause.summary,
                    "",
                    clause.normative_text,
                    "",
                )
            )

    families = sorted(
        (entity for entity in entities if isinstance(entity, InstructionFamily)),
        key=lambda value: value.document_order,
    )
    instructions = [entity for entity in entities if isinstance(entity, Instruction)]
    lines.extend(("## Instruction Families", ""))
    for family in families:
        lines.extend(
            (
                f"### {entity_link(family.entity_id)}",
                "",
                family.summary,
                "",
            )
        )
        family_instructions = sorted(
            (value for value in instructions if value.family_id == family.entity_id),
            key=lambda value: value.opcode,
        )
        _render_instruction_summary(lines, family_instructions, entity_link)

    _render_numeric_tables(lines, projection, entities, entity_link)
    _render_encodings(lines, entities)
    _render_validation_rules(lines, entities)
    return "\n".join(lines).rstrip() + "\n"


def render_isa_family_markdown(
    projection: Projection,
    family_id: str,
) -> str:
    """Renders complete normative semantics for one instruction family."""

    family = projection.require_entity(family_id)
    if not isinstance(family, InstructionFamily):
        raise ValueError(f"{family_id}: not an instruction family")
    source_path = isa_family_path(family)
    entity_link = _isa_entity_link(_isa_entity_paths(projection), source_path)
    instructions = sorted(
        (
            entity
            for entity in projection.entities
            if isinstance(entity, Instruction) and entity.family_id == family_id
        ),
        key=lambda value: value.opcode,
    )
    if not instructions:
        raise ValueError(f"{family_id}: family has no instructions")

    lines: list[str] = []
    domain_label = (
        "Core" if family.since.domain == "core" else family.since.domain.upper()
    )
    family_name = family.entity_id.partition(".family.")[2].replace("_", " ")
    _render_preamble(
        lines,
        f"IREE VM {domain_label} {family_name.title()} Instruction Family",
        projection,
    )
    lines.extend(
        (
            f"[Back to the {family.since.domain.upper()} ISA index](index.md).",
            "",
            _anchor(family),
            "## Family Contract",
            "",
            f"Family identity: `{family.entity_id}`.",
            "",
            family.normative_text,
            "",
            "## Instruction Summary",
            "",
        )
    )
    _render_instruction_summary(lines, instructions, entity_link)
    lines.extend(("## Instruction Semantics", ""))
    for instruction in instructions:
        _render_instruction(
            lines,
            instruction,
            projection,
            entity_link,
            heading_level=3,
        )
    return "\n".join(lines).rstrip() + "\n"
