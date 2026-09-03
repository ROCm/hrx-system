# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Structural validation for VM module wire-record declarations."""

import re

from iree.vm.bytecode.spec import module
from iree.vm.bytecode.spec.schema import U16

_RECORD_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_C_TYPE_PATTERN = re.compile(r"iree_vm_bytecode_[a-z0-9_]+_t")


def validate_wire_record(record: module.WireRecord) -> None:
    """Rejects an internally inconsistent module record declaration."""

    if not _RECORD_NAME_PATTERN.fullmatch(record.name):
        raise ValueError(f"invalid wire record name {record.name!r}")
    if not _C_TYPE_PATTERN.fullmatch(record.c_type):
        raise ValueError(f"{record.name}: invalid C type {record.c_type!r}")
    if not record.summary.strip() or not record.contract.strip():
        raise ValueError(f"{record.name}: missing normative contract")
    if not record.fields:
        raise ValueError(f"{record.name}: empty wire record")
    _ = record.field_offsets
    if record.byte_length % record.alignment:
        raise ValueError(f"{record.name}: length violates natural alignment")

    fields = {field.field.name: field for field in record.fields}
    for wire_field in record.fields:
        field = wire_field.field
        rule = wire_field.rule
        if isinstance(rule, module.ExactBytesRule):
            if len(rule.expected) != field.byte_length:
                raise ValueError(
                    f"{record.name}.{field.name}: exact byte length differs"
                )
        elif isinstance(rule, module.AllowedRangeRule):
            maximum_value = (1 << (field.byte_length * 8)) - 1
            if (
                field.element_count != 1
                or not field.encoding.name.startswith("u")
                or not 0 <= rule.minimum <= rule.maximum <= maximum_value
            ):
                raise ValueError(f"{record.name}.{field.name}: invalid allowed range")
        elif rule in (
            module.FieldRule.CORE_MAJOR,
            module.FieldRule.CORE_REQUIRED_MINOR,
        ):
            if field.encoding != U16 or field.element_count != 1:
                raise ValueError(
                    f"{record.name}.{field.name}: core version must be u16"
                )
        elif not isinstance(rule, module.FieldRule):
            raise ValueError(f"{record.name}.{field.name}: unknown field rule")

    if len(record.rules) > 1:
        raise ValueError(f"{record.name}: duplicate record rules")
    if record.rules:
        if record.rules[0] != module.RecordRule.SIGNATURE_DESCRIPTOR:
            raise ValueError(f"{record.name}: unknown record rule")
        if "kind_u16" not in fields or "type_ordinal_u16" not in fields:
            raise ValueError(f"{record.name}: signature rule names missing field")
        if (
            fields["kind_u16"].field.encoding,
            fields["type_ordinal_u16"].field.encoding,
        ) != (U16, U16):
            raise ValueError(f"{record.name}: signature fields must be u16")
