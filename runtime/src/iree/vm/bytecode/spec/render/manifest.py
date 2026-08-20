# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Completeness and version-release manifests."""

from __future__ import annotations

import collections
import json

from model.isa import Instruction, InstructionFamily
from model.module import Section, ValidationObligation
from model.schema import NumericTable, NumericValue, ValidationRule, WireRecord
from model.specification import Projection, Specification, compare_projections


def _json(payload: object) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _type_counts(projection: Projection) -> dict[str, int]:
    counts = collections.Counter(
        type(entity).__name__ for entity in projection.entities
    )
    return dict(sorted(counts.items()))


def render_completeness_manifest(
    module_projection: Projection,
    isa_projection: Projection,
) -> str:
    """Proves every projected fact has stable identity and required coverage."""

    all_entities = (*module_projection.entities, *isa_projection.entities)
    anchors: dict[str, str] = {}
    for entity in all_entities:
        prior = anchors.get(entity.normative_anchor)
        if prior is not None and prior != entity.entity_id:
            raise ValueError(
                f"normative anchor {entity.normative_anchor} collides for "
                f"{prior} and {entity.entity_id}"
            )
        anchors[entity.normative_anchor] = entity.entity_id

    instructions = [
        entity for entity in isa_projection.entities if isinstance(entity, Instruction)
    ]
    instruction_rows = []
    for instruction in instructions:
        semantics = instruction.semantics
        instruction_rows.append(
            {
                "entity_id": instruction.entity_id,
                "anchor": instruction.normative_anchor,
                "mnemonic": instruction.mnemonic,
                "family_id": instruction.family_id,
                "field_count": len(instruction.fields),
                "range_group_count": len(instruction.range_groups),
                "constraint_count": len(instruction.constraints),
                "coverage": {
                    "summary": bool(instruction.summary.strip()),
                    "encoding": instruction.byte_length > 0
                    and bool(instruction.fields),
                    "field_meanings": all(
                        bool(field.description.strip()) for field in instruction.fields
                    ),
                    "verification": bool(semantics.verification),
                    "dynamic_preconditions": semantics.preconditions is not None,
                    "success": bool(semantics.success)
                    or instruction.control_flow.value == "fail",
                    "failure_atomicity": all(
                        bool(failure.atomicity.strip())
                        for failure in semantics.failures
                    ),
                    "ownership": semantics.ownership is not None,
                    "control_and_suspension": True,
                    "assembly": bool(semantics.assembly),
                    "pseudocode": bool(semantics.pseudocode.strip()),
                },
            }
        )
    for row in instruction_rows:
        missing = [name for name, present in row["coverage"].items() if not present]
        if missing:
            raise ValueError(f"{row['entity_id']}: incomplete semantics {missing}")

    records = [
        entity
        for entity in module_projection.entities
        if isinstance(entity, WireRecord)
    ]
    rules = [
        entity
        for entity in (*module_projection.entities, *isa_projection.entities)
        if isinstance(entity, ValidationRule)
    ]
    numeric_tables = [
        entity
        for entity in (*module_projection.entities, *isa_projection.entities)
        if isinstance(entity, NumericTable)
    ]
    numeric_values = [
        entity
        for entity in (*module_projection.entities, *isa_projection.entities)
        if isinstance(entity, NumericValue)
    ]
    payload = {
        "schema": "iree.vm.bytecode.completeness",
        "schema_version": 1,
        "module": {
            "projection": module_projection.version_label(),
            "entity_type_counts": _type_counts(module_projection),
            "wire_record_count": len(records),
            "section_count": sum(
                isinstance(entity, Section) for entity in module_projection.entities
            ),
            "loader_obligation_count": sum(
                isinstance(entity, ValidationObligation)
                for entity in module_projection.entities
            ),
        },
        "isa": {
            "projection": isa_projection.version_label(),
            "entity_type_counts": _type_counts(isa_projection),
            "family_count": sum(
                isinstance(entity, InstructionFamily)
                for entity in isa_projection.entities
            ),
            "instruction_count": len(instructions),
            "instruction_field_count": sum(
                len(instruction.fields) for instruction in instructions
            ),
            "instructions": instruction_rows,
        },
        "shared": {
            "validation_rule_count": len({rule.entity_id for rule in rules}),
            "numeric_table_count": len({table.entity_id for table in numeric_tables}),
            "numeric_value_count": len({value.entity_id for value in numeric_values}),
            "normative_anchor_count": len(anchors),
        },
    }
    return _json(payload)


def _release_projection(
    specification: Specification,
    domain_name: str,
    minor: int,
) -> Projection:
    entity_ids = tuple(
        entity.entity_id
        for entity in specification.entities
        if entity.since.domain == domain_name and entity.since.minor <= minor
    )
    if not entity_ids:
        return Projection(specification.name, (), (), ())
    versions = specification.derive_projection_versions(entity_ids)
    return specification.project(versions)


def render_release_manifest(specification: Specification) -> str:
    """Renders exact additive stable-identity diffs for every domain minor."""

    releases = []
    entity_map = specification.entity_map()
    for domain in specification.domains:
        for minor in range(domain.latest_minor + 1):
            before = (
                Projection(specification.name, (), (), ())
                if minor == 0
                else _release_projection(specification, domain.domain, minor - 1)
            )
            after = _release_projection(specification, domain.domain, minor)
            diff = compare_projections(before, after)
            added = tuple(
                entity_id
                for entity_id in diff.added
                if entity_map[entity_id].since.domain == domain.domain
            )
            removed = tuple(
                entity_id
                for entity_id in diff.removed
                if entity_map[entity_id].since.domain == domain.domain
            )
            changed = tuple(
                entity_id
                for entity_id in diff.changed
                if entity_map[entity_id].since.domain == domain.domain
            )
            if removed or changed:
                raise ValueError(
                    f"{domain.domain} {domain.major}.{minor} is not additive"
                )
            releases.append(
                {
                    "domain": domain.domain,
                    "page_id": domain.page_id,
                    "major": domain.major,
                    "minor": minor,
                    "added": list(added),
                    "removed": list(removed),
                    "changed": list(changed),
                }
            )
    return _json(
        {
            "schema": "iree.vm.bytecode.release-diffs",
            "schema_version": 1,
            "specification": specification.name,
            "releases": releases,
        }
    )
