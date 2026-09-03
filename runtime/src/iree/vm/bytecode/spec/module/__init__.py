# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Closed declarations for VM module wire records."""

import enum
from typing import NamedTuple

from iree.vm.bytecode.spec.schema import Field, RuleKind, place_fields
from iree.vm.bytecode.spec.version import Version


class FieldRuleUse(NamedTuple):
    kind: RuleKind
    fields: tuple[str, ...] = ()
    values: tuple[int, ...] = ()
    data: bytes | enum.Enum | None = None


class WireField(NamedTuple):
    field: Field
    rule: RuleKind | FieldRuleUse


class WireRecord(NamedTuple):
    """One naturally aligned fixed module wire record."""

    name: str
    c_type: str
    since: Version
    summary: str
    contract: str
    fields: tuple[WireField, ...]

    @property
    def field_offsets(self) -> tuple[int, ...]:
        return place_fields(wire_field.field for wire_field in self.fields)

    @property
    def byte_length(self) -> int:
        return sum(field.field.byte_length for field in self.fields)

    @property
    def alignment(self) -> int:
        return max((field.field.encoding.alignment for field in self.fields), default=1)


class RecordFieldReference(NamedTuple):
    record: WireRecord
    field_name: str | None = None


class StructuralConstraint(NamedTuple):
    name: str
    since: Version
    inputs: tuple[RecordFieldReference, ...]
    contract: str


class Section(NamedTuple):
    name: str
    section_type: int
    required_flags: int
    since: Version
    summary: str
    contract: str
    records: tuple[WireRecord, ...]
    constraints: tuple[StructuralConstraint, ...]


class ModuleFormat(NamedTuple):
    since: Version
    image_alignment: int
    minimum_section_alignment: int
    summary: str
    contract: str
    envelope: tuple[WireRecord, ...]
    sections: tuple[Section, ...]
    constraints: tuple[StructuralConstraint, ...]
