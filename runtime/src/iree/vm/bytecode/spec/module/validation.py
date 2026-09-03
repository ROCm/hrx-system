# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Structural validation for VM module declarations."""

from iree.vm.bytecode.spec import module, schema
from iree.vm.bytecode.spec.module import ModuleFormat, rules
from iree.vm.bytecode.spec.version import Version


def validate_wire_record(record: module.WireRecord) -> None:
    valid = schema.is_name(record.name) and bool(record.fields)
    schema.require(valid, f"{record.name}: invalid wire record")
    _ = record.field_offsets
    valid = not record.byte_length % record.alignment
    schema.require(valid, f"{record.name}: invalid length")
    fields = {item.field.name: item for item in record.fields}
    for wire_field in record.fields:
        field, rule = wire_field
        if not isinstance(rule, module.FieldRuleUse):
            rule = module.FieldRuleUse(rule)
        kind, related_fields, values, data = rule
        valid = (
            kind in rules.FIELD_RULES
            and kind.accepts(related_fields, values, data=data)
            and all(name in fields for name in related_fields)
            and (kind.encoding is None or field.encoding == kind.encoding)
        )
        valid &= kind != rules.FieldRule.EXACT_BYTES or len(data) == field.byte_length
        schema.require(valid, f"{record.name}.{field.name}: invalid field rule")


def _validate_constraint(constraint, records, version) -> None:
    valid = schema.is_name(constraint.name)
    valid &= constraint.since.is_available_in(version)
    valid &= all(reference.record in records for reference in constraint.inputs)
    valid &= all(
        reference.field_name is None
        or reference.field_name in (item.field.name for item in reference.record.fields)
        for reference in constraint.inputs
    )
    schema.require(valid, f"{constraint.name}: invalid structural constraint")


def validate_module_format(module_format: ModuleFormat, version: Version) -> None:
    valid = module_format.since.is_available_in(version)
    valid &= schema.is_power_of_two(module_format.image_alignment)
    valid &= schema.is_power_of_two(module_format.minimum_section_alignment)
    valid &= module_format.image_alignment >= module_format.minimum_section_alignment
    valid &= bool(module_format.envelope)
    sections = module_format.sections
    numeric_tables = module_format.numeric_tables
    records = module_format.records
    valid &= len(records) <= 0xFF
    valid &= len({record.name for record in records}) == len(records)
    valid &= len({record.c_type for record in records}) == len(records)
    valid &= len({section.name for section in sections}) == len(sections)
    valid &= len({table.name for table in numeric_tables}) == len(numeric_tables)
    schema.require(valid, "invalid module format")
    for record in records:
        validate_wire_record(record)
        valid = record.since.is_available_in(version)
        schema.require(valid, f"{record.name}: unavailable")
    for table in numeric_tables:
        schema.validate_numeric_table(table, version)
    section_types = [section.section_type for section in sections]
    valid = section_types == sorted(set(section_types))
    schema.require(valid, "sections are not ordered")
    for section in sections:
        valid = schema.is_name(section.name) and 1 <= section.section_type <= 0xFFFF
        valid &= 0 <= section.required_flags <= 0xFFFF
        valid &= section.since.authority != "core" or not section.section_type >> 8
        valid &= section.since.is_available_in(version)
        valid &= bool(section.records and section.constraints)
        schema.require(valid, f"{section.name}: invalid module section")
        for constraint in section.constraints:
            _validate_constraint(constraint, records, version)
    for constraint in module_format.constraints:
        _validate_constraint(constraint, records, version)


def project_module_format(module_format, version):
    sections = tuple(
        section._replace(
            records=version.select(section.records),
            constraints=version.select(section.constraints),
        )
        for section in version.select(module_format.sections)
    )
    return module_format._replace(
        numeric_tables=tuple(
            table._replace(values=version.select(table.values))
            for table in version.select(module_format.numeric_tables)
        ),
        envelope=version.select(module_format.envelope),
        sections=sections,
        constraints=version.select(module_format.constraints),
    )
