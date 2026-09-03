# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Structural validation for VM module declarations."""

from iree.vm.bytecode.spec import module
from iree.vm.bytecode.spec.module import rules
from iree.vm.bytecode.spec.schema import is_name, is_power_of_two, require
from iree.vm.bytecode.spec.version import Version


def validate_wire_record(record: module.WireRecord) -> None:
    valid = is_name(record.name) and bool(record.fields)
    require(valid, f"{record.name}: invalid wire record")
    _ = record.field_offsets
    require(not record.byte_length % record.alignment, f"{record.name}: invalid length")
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
        require(valid, f"{record.name}.{field.name}: invalid field rule")


def _validate_constraint(constraint, records, version) -> None:
    valid = is_name(constraint.name)
    valid &= constraint.since.is_valid() and constraint.since.is_available_in(version)
    valid &= all(reference.record in records for reference in constraint.inputs)
    valid &= all(
        reference.field_name is None
        or reference.field_name in (item.field.name for item in reference.record.fields)
        for reference in constraint.inputs
    )
    require(valid, f"{constraint.name}: invalid structural constraint")


def validate_module_format(
    module_format: module.ModuleFormat, version: Version
) -> None:
    valid = module_format.since.is_valid() and module_format.since.is_available_in(
        version
    )
    valid &= is_power_of_two(module_format.image_alignment)
    valid &= is_power_of_two(module_format.minimum_section_alignment)
    valid &= module_format.image_alignment >= module_format.minimum_section_alignment
    valid &= bool(module_format.envelope)
    require(valid, "invalid module format")
    records = (
        *module_format.envelope,
        *(record for section in module_format.sections for record in section.records),
    )
    require(len(set(records)) == len(records), "module records have multiple owners")
    for record in records:
        validate_wire_record(record)
        require(record.since.is_available_in(version), f"{record.name}: unavailable")
    section_types = [section.section_type for section in module_format.sections]
    require(section_types == sorted(set(section_types)), "sections are not ordered")
    for section in module_format.sections:
        valid = is_name(section.name) and 1 <= section.section_type <= 0xFFFF
        valid &= 0 <= section.required_flags <= 0xFFFF
        valid &= section.since.authority != "core" or not section.section_type >> 8
        valid &= section.since.is_valid() and section.since.is_available_in(version)
        valid &= bool(section.records and section.constraints)
        require(valid, f"{section.name}: invalid module section")
        for constraint in section.constraints:
            _validate_constraint(constraint, section.records, version)
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
        envelope=version.select(module_format.envelope),
        sections=sections,
        constraints=version.select(module_format.constraints),
    )
