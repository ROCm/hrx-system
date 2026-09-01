# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Declarative module-container structure and loader obligations."""

from __future__ import annotations

import dataclasses
import re

from model.schema import ValidationScope, WireRecord, WireRecordLayout
from model.specification import Entity, Specification

_KIND_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_FIELD_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class OrdinalDomain(Entity):
    """One stable numbering domain referenced by fixed-width ordinals."""

    maximum_count: int
    base_domain_id: str | None = None
    require_nonempty_value: bool = False

    def referenced_entity_ids(self) -> tuple[str, ...]:
        return (self.base_domain_id,) if self.base_domain_id is not None else ()

    def validate(self, specification: Specification) -> None:
        if not 1 <= self.maximum_count <= 65536:
            raise ValueError(f"{self.entity_id}: invalid maximum ordinal count")
        if self.base_domain_id is not None:
            base_domain = specification.entity_map().get(self.base_domain_id)
            if not isinstance(base_domain, OrdinalDomain):
                raise ValueError(f"{self.entity_id}: invalid base ordinal domain")
            if self.maximum_count > base_domain.maximum_count:
                raise ValueError(
                    f"{self.entity_id}: derived ordinal domain exceeds its base"
                )
        elif self.require_nonempty_value:
            raise ValueError(
                f"{self.entity_id}: nonempty constraint has no base domain"
            )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ModuleFormat(Entity):
    """Top-level image packing and alignment contract."""

    image_alignment: int
    minimum_section_alignment: int
    normative_text: str

    def __post_init__(self) -> None:
        super(ModuleFormat, self).__post_init__()
        if not self.normative_text.strip():
            raise ValueError(f"{self.entity_id}: missing normative text")

    def validate(self, specification: Specification) -> None:
        formats = [
            entity
            for entity in specification.entities
            if isinstance(entity, ModuleFormat)
        ]
        if len(formats) != 1:
            raise ValueError(
                f"{self.entity_id}: specification has {len(formats)} module formats"
            )
        if not _is_power_of_two(self.image_alignment):
            raise ValueError(f"{self.entity_id}: invalid image alignment")
        if not _is_power_of_two(self.minimum_section_alignment):
            raise ValueError(f"{self.entity_id}: invalid minimum section alignment")
        if self.image_alignment < self.minimum_section_alignment:
            raise ValueError(
                f"{self.entity_id}: image alignment does not preserve sections"
            )

        records = {
            entity.entity_id
            for entity in specification.entities
            if isinstance(entity, WireRecord)
        }
        envelope_members = [
            entity
            for entity in specification.entities
            if isinstance(entity, EnvelopeRecord) and entity.format_id == self.entity_id
        ]
        if not envelope_members:
            raise ValueError(f"{self.entity_id}: no envelope records")
        section_members = [
            entity
            for entity in specification.entities
            if isinstance(entity, SectionRecord)
        ]
        owned_records = [
            member.record_id for member in (*envelope_members, *section_members)
        ]
        duplicate_records = sorted(
            record_id
            for record_id in set(owned_records)
            if owned_records.count(record_id) != 1
        )
        if duplicate_records:
            raise ValueError(
                f"{self.entity_id}: multiply owned records {duplicate_records}"
            )
        missing_records = records - set(owned_records)
        if missing_records:
            raise ValueError(
                f"{self.entity_id}: unowned records {sorted(missing_records)}"
            )
        extra_records = set(owned_records) - records
        if extra_records:
            raise ValueError(
                f"{self.entity_id}: unknown owned records {sorted(extra_records)}"
            )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EnvelopeRecord(Entity):
    """Ordered use of one fixed record in the image envelope."""

    format_id: str
    record_id: str
    document_order: int

    def referenced_entity_ids(self) -> tuple[str, ...]:
        return (self.format_id, self.record_id)

    def validate(self, specification: Specification) -> None:
        entities_by_id = specification.entity_map()
        if not isinstance(entities_by_id.get(self.format_id), ModuleFormat):
            raise ValueError(f"{self.entity_id}: unknown module format")
        if not isinstance(entities_by_id.get(self.record_id), WireRecord):
            raise ValueError(f"{self.entity_id}: envelope member is not a record")
        module_format = entities_by_id[self.format_id]
        record = entities_by_id[self.record_id]
        if module_format.since.domain != record.since.domain:
            raise ValueError(
                f"{self.entity_id}: envelope record belongs to another domain"
            )
        if self.document_order < 0:
            raise ValueError(f"{self.entity_id}: negative document order")
        peers = [
            entity
            for entity in specification.entities
            if isinstance(entity, EnvelopeRecord)
            and entity.format_id == self.format_id
            and entity.document_order == self.document_order
        ]
        if len(peers) != 1:
            raise ValueError(f"{self.entity_id}: duplicate envelope document order")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Section(Entity):
    """One version-owned module section identity and payload grammar."""

    section_type: int
    required_flags: int
    grammar: str
    normative_text: str

    def __post_init__(self) -> None:
        super(Section, self).__post_init__()
        if not self.normative_text.strip():
            raise ValueError(f"{self.entity_id}: missing normative text")

    def validate(self, specification: Specification) -> None:
        domain = specification.domain_map()[self.since.domain]
        if not 1 <= self.section_type <= 0xFFFF:
            raise ValueError(
                f"{self.entity_id}: invalid section type {self.section_type:#x}"
            )
        if self.section_type >> 8 != domain.page_id:
            raise ValueError(
                f"{self.entity_id}: section type is outside its page domain"
            )
        if not 0 <= self.required_flags <= 0xFFFF:
            raise ValueError(f"{self.entity_id}: section flags exceed u16")
        if not self.grammar.strip():
            raise ValueError(f"{self.entity_id}: missing section grammar")
        for other in specification.entities:
            if (
                isinstance(other, Section)
                and other is not self
                and other.section_type == self.section_type
            ):
                raise ValueError(
                    f"{self.entity_id}: section type collides with {other.entity_id}"
                )
        members = [
            entity
            for entity in specification.entities
            if isinstance(entity, SectionRecord) and entity.section_id == self.entity_id
        ]
        if not members:
            raise ValueError(f"{self.entity_id}: section has no record grammar")
        if not any(member.since == self.since for member in members):
            raise ValueError(
                f"{self.entity_id}: no record grammar exists at introduction"
            )
        obligations = [
            entity
            for entity in specification.entities
            if isinstance(entity, ValidationObligation)
            and entity.scope == ValidationScope.SECTION
            and entity.section_id == self.entity_id
        ]
        if not obligations:
            raise ValueError(f"{self.entity_id}: section has no validation rules")
        if not any(obligation.since == self.since for obligation in obligations):
            raise ValueError(
                f"{self.entity_id}: no validation rule exists at introduction"
            )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SectionRecord(Entity):
    """Ordered use of one fixed record in a section grammar."""

    section_id: str
    record_id: str
    document_order: int

    def referenced_entity_ids(self) -> tuple[str, ...]:
        return (self.section_id, self.record_id)

    def validate(self, specification: Specification) -> None:
        entities_by_id = specification.entity_map()
        section = entities_by_id.get(self.section_id)
        record = entities_by_id.get(self.record_id)
        if not isinstance(section, Section):
            raise ValueError(f"{self.entity_id}: unknown owning section")
        if not isinstance(record, WireRecord):
            raise ValueError(f"{self.entity_id}: section member is not a record")
        if section.since.domain != record.since.domain:
            raise ValueError(f"{self.entity_id}: section and record domains differ")
        if self.document_order < 0:
            raise ValueError(f"{self.entity_id}: negative document order")
        peers = [
            entity
            for entity in specification.entities
            if isinstance(entity, SectionRecord)
            and entity.section_id == self.section_id
            and entity.document_order == self.document_order
        ]
        if len(peers) != 1:
            raise ValueError(f"{self.entity_id}: duplicate section document order")


@dataclasses.dataclass(frozen=True, slots=True)
class RecordFieldReference:
    """One record or record-field input to a loader obligation."""

    record_id: str
    field_name: str | None = None

    def __post_init__(self) -> None:
        if self.field_name is not None and not _FIELD_PATTERN.fullmatch(
            self.field_name
        ):
            raise ValueError(f"invalid record field reference {self.field_name!r}")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ValidationObligation(Entity):
    """One named loader obligation without generated implementation control flow."""

    scope: ValidationScope
    kind: str
    inputs: tuple[RecordFieldReference, ...]
    normative_text: str
    section_id: str | None = None

    def __post_init__(self) -> None:
        super(ValidationObligation, self).__post_init__()
        if self.scope not in {
            ValidationScope.IMAGE,
            ValidationScope.DIRECTORY,
            ValidationScope.SECTION,
            ValidationScope.CROSS_SECTION,
        }:
            raise ValueError(f"{self.entity_id}: invalid loader obligation scope")
        if not _KIND_PATTERN.fullmatch(self.kind):
            raise ValueError(f"{self.entity_id}: invalid obligation kind {self.kind!r}")
        if not self.normative_text.strip():
            raise ValueError(f"{self.entity_id}: missing normative obligation")
        if (self.scope == ValidationScope.SECTION) != (self.section_id is not None):
            raise ValueError(f"{self.entity_id}: section scope and section ID disagree")

    def referenced_entity_ids(self) -> tuple[str, ...]:
        references = [item.record_id for item in self.inputs]
        if self.section_id is not None:
            references.append(self.section_id)
        return tuple(sorted(set(references)))

    def validate(self, specification: Specification) -> None:
        entities_by_id = specification.entity_map()
        if self.section_id is not None and not isinstance(
            entities_by_id.get(self.section_id), Section
        ):
            raise ValueError(f"{self.entity_id}: unknown obligation section")
        for item in self.inputs:
            record = entities_by_id.get(item.record_id)
            if not isinstance(record, WireRecord):
                raise ValueError(
                    f"{self.entity_id}: input {item.record_id!r} is not a record"
                )
            if item.field_name is None:
                continue
            layouts = [
                entity
                for entity in specification.entities
                if isinstance(entity, WireRecordLayout)
                and entity.record_id == item.record_id
                and entity.since.domain == self.since.domain
                and entity.since.major == self.since.major
                and entity.since.minor <= self.since.minor
            ]
            if not layouts:
                raise ValueError(
                    f"{self.entity_id}: no available input layout for {item.record_id}"
                )
            layout = max(layouts, key=lambda value: value.since.minor)
            if item.field_name not in {field.name for field in layout.fields}:
                raise ValueError(
                    f"{self.entity_id}: unknown input field "
                    f"{item.record_id}.{item.field_name}"
                )
