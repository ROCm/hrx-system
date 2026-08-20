# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Canonical scalar encodings, field rules, and wire-layout entities.

The module format and instruction sets both use these scalar and fixed-record
concepts. This module owns scalar identities and context-free field rules;
concrete records and contextual validation live with their specifications.
"""

from __future__ import annotations

import dataclasses
import enum
import re

from model.specification import CORE_0, Entity, Projection, Specification

_C_TYPE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_ ]*")
_FIELD_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_NUMERIC_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*")


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ScalarEncoding(Entity):
    """One fixed-width scalar wire encoding."""

    c_type: str
    byte_length: int
    alignment: int

    def validate(self, specification: Specification) -> None:
        del specification
        if not _C_TYPE_PATTERN.fullmatch(self.c_type):
            raise ValueError(f"{self.entity_id}: invalid C type {self.c_type!r}")
        if not _is_power_of_two(self.byte_length):
            raise ValueError(f"{self.entity_id}: invalid scalar byte length")
        if not _is_power_of_two(self.alignment):
            raise ValueError(f"{self.entity_id}: invalid scalar alignment")
        if self.alignment > self.byte_length:
            raise ValueError(f"{self.entity_id}: over-aligned scalar encoding")


U8 = ScalarEncoding(
    entity_id="core.encoding.u8",
    since=CORE_0,
    summary="Little-endian unsigned eight-bit integer.",
    c_type="uint8_t",
    byte_length=1,
    alignment=1,
)
I16 = ScalarEncoding(
    entity_id="core.encoding.i16",
    since=CORE_0,
    summary="Little-endian two's-complement signed 16-bit integer.",
    c_type="int16_t",
    byte_length=2,
    alignment=2,
)
U16 = ScalarEncoding(
    entity_id="core.encoding.u16",
    since=CORE_0,
    summary="Little-endian unsigned 16-bit integer.",
    c_type="uint16_t",
    byte_length=2,
    alignment=2,
)
I32 = ScalarEncoding(
    entity_id="core.encoding.i32",
    since=CORE_0,
    summary="Little-endian two's-complement signed 32-bit integer.",
    c_type="int32_t",
    byte_length=4,
    alignment=4,
)
U32 = ScalarEncoding(
    entity_id="core.encoding.u32",
    since=CORE_0,
    summary="Little-endian unsigned 32-bit integer.",
    c_type="uint32_t",
    byte_length=4,
    alignment=4,
)
U64 = ScalarEncoding(
    entity_id="core.encoding.u64",
    since=CORE_0,
    summary="Little-endian unsigned 64-bit integer.",
    c_type="uint64_t",
    byte_length=8,
    alignment=8,
)

SCALAR_ENCODINGS = (U8, I16, U16, I32, U32, U64)


class NumericTableKind(enum.Enum):
    """Generation and validation policy for a numeric domain."""

    ENUM = "enum"
    FLAGS = "flags"
    SELECTOR = "selector"


class UnknownNumericValuePolicy(enum.Enum):
    """How a consumer handles values absent from the selected version."""

    REJECT = "reject"
    PRESERVE_NONZERO = "preserve_nonzero"


class ValidationScope(enum.Enum):
    """Architectural scope at which a validation rule is evaluated."""

    FIELD = "field"
    RECORD = "record"
    DIRECTORY = "directory"
    SECTION = "section"
    IMAGE = "image"
    CROSS_SECTION = "cross_section"


class ArgumentKind(enum.Enum):
    """Self-describing value shapes accepted by validation rules."""

    INTEGER = "integer"
    TEXT = "text"
    BYTES = "bytes"
    ENTITY = "entity"
    FIELD = "field"
    SEQUENCE = "sequence"
    TUPLE = "tuple"


@dataclasses.dataclass(frozen=True, slots=True)
class ArgumentShape:
    """Recursive shape of one declarative validation-rule argument."""

    kind: ArgumentKind
    elements: tuple[ArgumentShape, ...] = ()
    minimum_count: int = 0

    def __post_init__(self) -> None:
        if self.minimum_count < 0:
            raise ValueError("negative validation argument minimum count")
        if self.kind == ArgumentKind.SEQUENCE:
            if len(self.elements) != 1:
                raise ValueError("sequence argument requires one element shape")
        elif self.kind == ArgumentKind.TUPLE:
            if not self.elements:
                raise ValueError("tuple argument requires element shapes")
            if self.minimum_count:
                raise ValueError("tuple argument cannot have a minimum count")
        elif self.elements or self.minimum_count:
            raise ValueError(
                f"{self.kind.value} argument cannot have nested shape data"
            )

    @staticmethod
    def sequence(
        element: ArgumentShape,
        *,
        minimum_count: int = 0,
    ) -> ArgumentShape:
        return ArgumentShape(
            ArgumentKind.SEQUENCE,
            (element,),
            minimum_count,
        )

    @staticmethod
    def tuple_of(*elements: ArgumentShape) -> ArgumentShape:
        return ArgumentShape(ArgumentKind.TUPLE, tuple(elements))


INTEGER_ARGUMENT = ArgumentShape(ArgumentKind.INTEGER)
TEXT_ARGUMENT = ArgumentShape(ArgumentKind.TEXT)
BYTES_ARGUMENT = ArgumentShape(ArgumentKind.BYTES)
ENTITY_ARGUMENT = ArgumentShape(ArgumentKind.ENTITY)
FIELD_ARGUMENT = ArgumentShape(ArgumentKind.FIELD)


@dataclasses.dataclass(frozen=True, slots=True)
class EntityReference:
    """Marks a validation argument as a specification dependency."""

    entity_id: str


@dataclasses.dataclass(frozen=True, slots=True)
class FieldReference:
    """Marks a validation argument as a field in the owning record."""

    field_name: str

    def __post_init__(self) -> None:
        if not _FIELD_PATTERN.fullmatch(self.field_name):
            raise ValueError(f"invalid field reference {self.field_name!r}")


@dataclasses.dataclass(frozen=True, slots=True)
class RuleParameter:
    """One named parameter in a validation-rule declaration."""

    name: str
    shape: ArgumentShape

    def __post_init__(self) -> None:
        if not _FIELD_PATTERN.fullmatch(self.name):
            raise ValueError(f"invalid rule parameter {self.name!r}")


@dataclasses.dataclass(frozen=True, slots=True)
class RuleUse:
    """One application of a declared validation rule."""

    rule_id: str
    arguments: tuple[object, ...] = ()

    def referenced_entity_ids(self) -> tuple[str, ...]:
        references = [self.rule_id]

        def collect(value: object) -> None:
            if isinstance(value, EntityReference):
                references.append(value.entity_id)
            elif isinstance(value, tuple):
                for element in value:
                    collect(element)

        for argument in self.arguments:
            collect(argument)
        return tuple(sorted(set(references)))


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ValidationRule(Entity):
    """One versioned, self-describing structural validation contract."""

    scope: ValidationScope
    parameters: tuple[RuleParameter, ...]
    normative_text: str

    def __post_init__(self) -> None:
        super(ValidationRule, self).__post_init__()
        parameter_names = [parameter.name for parameter in self.parameters]
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError(f"{self.entity_id}: duplicate rule parameter")
        if not self.normative_text.strip():
            raise ValueError(f"{self.entity_id}: missing normative rule text")


_NONEMPTY_INTEGER_SEQUENCE = ArgumentShape.sequence(
    INTEGER_ARGUMENT,
    minimum_count=1,
)

ANY_BITS = ValidationRule(
    entity_id="core.validation.field.any_bits",
    since=CORE_0,
    summary="Accepts every bit pattern representable by the field.",
    scope=ValidationScope.FIELD,
    parameters=(),
    normative_text=(
        "Structural verification imposes no value restriction beyond the "
        "field's encoded width. Contextual rules may still constrain it."
    ),
)
ZERO = ValidationRule(
    entity_id="core.validation.field.zero",
    since=CORE_0,
    summary="Requires every encoded field bit to be zero.",
    scope=ValidationScope.FIELD,
    parameters=(),
    normative_text="The field value must equal zero.",
)
ALLOWED_BITS = ValidationRule(
    entity_id="core.validation.field.allowed_bits",
    since=CORE_0,
    summary="Restricts a field to a declared mask of assigned bits.",
    scope=ValidationScope.FIELD,
    parameters=(RuleParameter("allowed_mask", INTEGER_ARGUMENT),),
    normative_text=(
        "Every bit outside allowed_mask must be zero. Assigned bits may be "
        "combined unless another rule states otherwise."
    ),
)
ALLOWED_BITS_EXACTLY_ONE = ValidationRule(
    entity_id="core.validation.field.allowed_bits_exactly_one",
    since=CORE_0,
    summary="Restricts bits and requires one selected bit from a submask.",
    scope=ValidationScope.FIELD,
    parameters=(
        RuleParameter("allowed_mask", INTEGER_ARGUMENT),
        RuleParameter("exactly_one_mask", INTEGER_ARGUMENT),
    ),
    normative_text=(
        "Every bit outside allowed_mask must be zero and exactly one bit in "
        "exactly_one_mask must be set. exactly_one_mask must be a nonzero "
        "subset of allowed_mask."
    ),
)
ALLOWED_RANGE = ValidationRule(
    entity_id="core.validation.field.allowed_range",
    since=CORE_0,
    summary="Restricts an unsigned field to an inclusive range.",
    scope=ValidationScope.FIELD,
    parameters=(
        RuleParameter("minimum", INTEGER_ARGUMENT),
        RuleParameter("maximum", INTEGER_ARGUMENT),
    ),
    normative_text="The decoded unsigned value must be in [minimum, maximum].",
)
ALLOWED_VALUES = ValidationRule(
    entity_id="core.validation.field.allowed_values",
    since=CORE_0,
    summary="Restricts a field to an explicit set of values.",
    scope=ValidationScope.FIELD,
    parameters=(RuleParameter("values", _NONEMPTY_INTEGER_SEQUENCE),),
    normative_text=(
        "The decoded unsigned value must equal one member of the nonempty, "
        "duplicate-free values sequence."
    ),
)
EXACT_BYTES = ValidationRule(
    entity_id="core.validation.field.exact_bytes",
    since=CORE_0,
    summary="Requires one exact byte sequence.",
    scope=ValidationScope.FIELD,
    parameters=(RuleParameter("expected", BYTES_ARGUMENT),),
    normative_text=(
        "The field bytes must exactly equal expected, including embedded zero "
        "bytes. The expected sequence length must equal the field length."
    ),
)
MULTIPLE = ValidationRule(
    entity_id="core.validation.field.multiple",
    since=CORE_0,
    summary="Requires an unsigned field to be a positive-unit multiple.",
    scope=ValidationScope.FIELD,
    parameters=(RuleParameter("unit", INTEGER_ARGUMENT),),
    normative_text="unit must be positive and the decoded value modulo unit is zero.",
)

BASIC_FIELD_RULES = (
    ANY_BITS,
    ZERO,
    ALLOWED_BITS,
    ALLOWED_BITS_EXACTLY_ONE,
    ALLOWED_RANGE,
    ALLOWED_VALUES,
    EXACT_BYTES,
    MULTIPLE,
)


def _validate_argument(
    owner: str,
    value: object,
    shape: ArgumentShape,
    specification: Specification,
    field_names: frozenset[str],
) -> None:
    if shape.kind == ArgumentKind.INTEGER:
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif shape.kind == ArgumentKind.TEXT:
        valid = isinstance(value, str)
    elif shape.kind == ArgumentKind.BYTES:
        valid = isinstance(value, bytes)
    elif shape.kind == ArgumentKind.ENTITY:
        valid = isinstance(value, EntityReference)
        if valid and value.entity_id not in specification.entity_map():
            raise ValueError(f"{owner}: unknown entity argument {value.entity_id!r}")
    elif shape.kind == ArgumentKind.FIELD:
        valid = isinstance(value, FieldReference)
        if valid and value.field_name not in field_names:
            raise ValueError(f"{owner}: unknown field argument {value.field_name!r}")
    elif shape.kind == ArgumentKind.SEQUENCE:
        valid = isinstance(value, tuple) and len(value) >= shape.minimum_count
        if valid:
            for element in value:
                _validate_argument(
                    owner,
                    element,
                    shape.elements[0],
                    specification,
                    field_names,
                )
    elif shape.kind == ArgumentKind.TUPLE:
        valid = isinstance(value, tuple) and len(value) == len(shape.elements)
        if valid:
            for element, element_shape in zip(
                value,
                shape.elements,
                strict=True,
            ):
                _validate_argument(
                    owner,
                    element,
                    element_shape,
                    specification,
                    field_names,
                )
    else:
        raise AssertionError(f"unhandled argument kind {shape.kind}")
    if not valid:
        raise ValueError(f"{owner}: argument does not match {shape.kind.value} shape")


def validate_rule_use(
    owner: str,
    rule_use: RuleUse,
    expected_scope: ValidationScope,
    specification: Specification,
    *,
    field_names: frozenset[str] = frozenset(),
) -> None:
    """Validates one rule application without implementing the rule itself."""

    rule = specification.entity_map().get(rule_use.rule_id)
    if not isinstance(rule, ValidationRule):
        raise ValueError(f"{owner}: {rule_use.rule_id!r} is not a validation rule")
    if rule.scope != expected_scope:
        raise ValueError(
            f"{owner}: {rule.entity_id} has {rule.scope.value} scope, expected "
            f"{expected_scope.value}"
        )
    if len(rule_use.arguments) != len(rule.parameters):
        raise ValueError(
            f"{owner}: {rule.entity_id} expects {len(rule.parameters)} "
            f"arguments, got {len(rule_use.arguments)}"
        )
    for argument, parameter in zip(
        rule_use.arguments,
        rule.parameters,
        strict=True,
    ):
        _validate_argument(
            f"{owner}.{parameter.name}",
            argument,
            parameter.shape,
            specification,
            field_names,
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class NumericTable(Entity):
    """One explicitly encoded enum, flag, or instruction selector domain."""

    encoding_id: str
    table_kind: NumericTableKind
    unknown_value_policy: UnknownNumericValuePolicy

    def referenced_entity_ids(self) -> tuple[str, ...]:
        return (self.encoding_id,)

    def validate(self, specification: Specification) -> None:
        entities_by_id = specification.entity_map()
        encoding = entities_by_id.get(self.encoding_id)
        if not isinstance(encoding, ScalarEncoding):
            raise ValueError(f"{self.entity_id}: numeric table encoding is not scalar")
        values = [
            entity
            for entity in specification.entities
            if isinstance(entity, NumericValue) and entity.table_id == self.entity_id
        ]
        if not values:
            raise ValueError(f"{self.entity_id}: numeric table has no values")
        if not any(value.since == self.since for value in values):
            raise ValueError(f"{self.entity_id}: no values exist at table introduction")
        names: set[str] = set()
        encoded_values: set[int] = set()
        maximum_value = (1 << (encoding.byte_length * 8)) - 1
        for value in values:
            if value.since.domain != self.since.domain:
                raise ValueError(
                    f"{value.entity_id}: numeric value and table domains differ"
                )
            if value.since.major != self.since.major:
                raise ValueError(
                    f"{value.entity_id}: numeric value and table majors differ"
                )
            if value.since.minor < self.since.minor:
                raise ValueError(f"{value.entity_id}: numeric value predates its table")
            if (
                self.unknown_value_policy == UnknownNumericValuePolicy.PRESERVE_NONZERO
                and value.minimum_consumer_version != self.since
            ):
                raise ValueError(
                    f"{value.entity_id}: an open numeric value must remain "
                    "compatible with the table introduction"
                )
            if value.name in names:
                raise ValueError(
                    f"{self.entity_id}: duplicate numeric name {value.name}"
                )
            if value.value in encoded_values:
                raise ValueError(
                    f"{self.entity_id}: duplicate encoded value {value.value}"
                )
            if not 0 <= value.value <= maximum_value:
                raise ValueError(
                    f"{value.entity_id}: value does not fit table encoding"
                )
            if (
                self.table_kind == NumericTableKind.FLAGS
                and value.value != 0
                and value.value & (value.value - 1)
            ):
                raise ValueError(f"{value.entity_id}: flag is not one bit")
            names.add(value.name)
            encoded_values.add(value.value)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class NumericValue(Entity):
    """One explicit append-only value in a numeric table."""

    table_id: str
    name: str
    value: int

    def __post_init__(self) -> None:
        super(NumericValue, self).__post_init__()
        if not _NUMERIC_NAME_PATTERN.fullmatch(self.name):
            raise ValueError(f"{self.entity_id}: invalid numeric name {self.name!r}")

    def referenced_entity_ids(self) -> tuple[str, ...]:
        return (self.table_id,)

    def validate(self, specification: Specification) -> None:
        table = specification.entity_map().get(self.table_id)
        if not isinstance(table, NumericTable):
            raise ValueError(
                f"{self.entity_id}: table {self.table_id!r} is not numeric"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class WireField:
    """One complete named byte range in a fixed record layout."""

    name: str
    offset: int
    encoding_id: str
    description: str
    validation: tuple[RuleUse, ...]
    array_length: int = 1

    def __post_init__(self) -> None:
        if not _FIELD_PATTERN.fullmatch(self.name):
            raise ValueError(f"invalid wire field name {self.name!r}")
        if self.offset < 0:
            raise ValueError(f"{self.name}: negative wire offset")
        if self.array_length <= 0:
            raise ValueError(f"{self.name}: non-positive array length")
        if not self.description.strip():
            raise ValueError(f"{self.name}: missing field description")
        if not self.validation:
            raise ValueError(f"{self.name}: missing field validation")

    def referenced_entity_ids(self) -> tuple[str, ...]:
        references = [self.encoding_id]
        for rule_use in self.validation:
            references.extend(rule_use.referenced_entity_ids())
        return tuple(sorted(set(references)))


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class WireRecord(Entity):
    """Stable identity and C spelling for one fixed wire record."""

    c_type: str

    def validate(self, specification: Specification) -> None:
        if not _C_TYPE_PATTERN.fullmatch(self.c_type):
            raise ValueError(f"{self.entity_id}: invalid C type {self.c_type!r}")
        layouts = sorted(
            (
                entity
                for entity in specification.entities
                if isinstance(entity, WireRecordLayout)
                and entity.record_id == self.entity_id
            ),
            key=lambda layout: (layout.since.major, layout.since.minor),
        )
        if not layouts:
            raise ValueError(f"{self.entity_id}: no wire layout")
        first_layout = layouts[0]
        if first_layout.since != self.since:
            raise ValueError(
                f"{self.entity_id}: initial layout version does not match record"
            )
        previous_layout: WireRecordLayout | None = None
        seen_versions = set()
        for layout in layouts:
            if layout.since.domain != self.since.domain:
                raise ValueError(
                    f"{layout.entity_id}: layout and record domains differ"
                )
            if layout.since.major != self.since.major:
                raise ValueError(f"{layout.entity_id}: layout and record majors differ")
            version_key = (layout.since.major, layout.since.minor)
            if version_key in seen_versions:
                raise ValueError(
                    f"{self.entity_id}: multiple layouts at version "
                    f"{layout.since.major}.{layout.since.minor}"
                )
            seen_versions.add(version_key)
            expected_previous = (
                previous_layout.entity_id if previous_layout is not None else None
            )
            if layout.previous_layout_id != expected_previous:
                raise ValueError(
                    f"{layout.entity_id}: previous layout is "
                    f"{layout.previous_layout_id!r}, expected "
                    f"{expected_previous!r}"
                )
            previous_layout = layout


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class WireRecordLayout(Entity):
    """One version-selected complete layout for a fixed wire record."""

    record_id: str
    byte_length: int
    alignment: int
    fields: tuple[WireField, ...]
    previous_layout_id: str | None = None
    scalar_alias: bool = False

    def referenced_entity_ids(self) -> tuple[str, ...]:
        references = [self.record_id]
        if self.previous_layout_id is not None:
            references.append(self.previous_layout_id)
        for field in self.fields:
            references.extend(field.referenced_entity_ids())
        return tuple(sorted(references))

    def validate(self, specification: Specification) -> None:
        entities_by_id = specification.entity_map()
        record = entities_by_id.get(self.record_id)
        if not isinstance(record, WireRecord):
            raise ValueError(
                f"{self.entity_id}: record {self.record_id!r} is not a wire record"
            )
        if self.previous_layout_id is not None:
            previous = entities_by_id.get(self.previous_layout_id)
            if not isinstance(previous, WireRecordLayout):
                raise ValueError(
                    f"{self.entity_id}: previous layout is not a wire layout"
                )
            if previous.record_id != self.record_id:
                raise ValueError(
                    f"{self.entity_id}: previous layout belongs to another record"
                )
            if (
                previous.since.domain != self.since.domain
                or previous.since.major != self.since.major
                or previous.since.minor >= self.since.minor
            ):
                raise ValueError(f"{self.entity_id}: previous layout is not older")
        if self.byte_length <= 0:
            raise ValueError(f"{self.entity_id}: non-positive byte length")
        if not _is_power_of_two(self.alignment):
            raise ValueError(f"{self.entity_id}: invalid record alignment")
        if self.byte_length % self.alignment:
            raise ValueError(
                f"{self.entity_id}: byte length does not preserve alignment"
            )
        if not self.fields:
            raise ValueError(f"{self.entity_id}: no fields")
        if self.scalar_alias and len(self.fields) != 1:
            raise ValueError(f"{self.entity_id}: scalar alias does not have one field")

        occupied: list[str | None] = [None] * self.byte_length
        field_names: set[str] = set()
        maximum_alignment = 1
        for field in self.fields:
            if field.name in field_names:
                raise ValueError(f"{self.entity_id}: duplicate field {field.name}")
            field_names.add(field.name)
            encoding = entities_by_id.get(field.encoding_id)
            if not isinstance(encoding, ScalarEncoding):
                raise ValueError(
                    f"{self.entity_id}.{field.name}: encoding "
                    f"{field.encoding_id!r} is not scalar"
                )
            if field.offset % encoding.alignment:
                raise ValueError(f"{self.entity_id}.{field.name}: naturally misaligned")
            field_byte_length = encoding.byte_length * field.array_length
            field_end = field.offset + field_byte_length
            if field_end > self.byte_length:
                raise ValueError(f"{self.entity_id}.{field.name}: outside record")
            for byte_offset in range(field.offset, field_end):
                previous_owner = occupied[byte_offset]
                if previous_owner is not None:
                    raise ValueError(
                        f"{self.entity_id}.{field.name}: overlaps "
                        f"{previous_owner} at byte {byte_offset}"
                    )
                occupied[byte_offset] = field.name
            maximum_alignment = max(maximum_alignment, encoding.alignment)

        uncovered = [
            byte_offset for byte_offset, owner in enumerate(occupied) if owner is None
        ]
        if uncovered:
            raise ValueError(f"{self.entity_id}: uncovered bytes {uncovered}")
        frozen_field_names = frozenset(field_names)
        for field in self.fields:
            for rule_use in field.validation:
                validate_rule_use(
                    f"{self.entity_id}.{field.name}",
                    rule_use,
                    ValidationScope.FIELD,
                    specification,
                    field_names=frozen_field_names,
                )
        if maximum_alignment != self.alignment:
            raise ValueError(
                f"{self.entity_id}: declared alignment {self.alignment} does "
                f"not match natural alignment {maximum_alignment}"
            )
        if self.scalar_alias:
            field = self.fields[0]
            encoding = entities_by_id[field.encoding_id]
            if (
                field.offset != 0
                or field.array_length != 1
                or not isinstance(encoding, ScalarEncoding)
                or encoding.byte_length != self.byte_length
            ):
                raise ValueError(f"{self.entity_id}: invalid scalar alias layout")


def selected_record_layouts(
    projection: Projection,
) -> dict[str, WireRecordLayout]:
    """Selects the newest available layout for every projected wire record."""

    records = {
        entity.entity_id: entity
        for entity in projection.entities
        if isinstance(entity, WireRecord)
    }
    selected: dict[str, WireRecordLayout] = {}
    for entity in projection.entities:
        if not isinstance(entity, WireRecordLayout):
            continue
        if entity.record_id not in records:
            raise ValueError(
                f"{entity.entity_id}: projected without record {entity.record_id}"
            )
        current = selected.get(entity.record_id)
        if current is None or current.since < entity.since:
            selected[entity.record_id] = entity
    missing = set(records) - set(selected)
    if missing:
        raise ValueError(f"projected records without layouts: {sorted(missing)}")
    return selected


def selected_numeric_values(
    projection: Projection,
) -> dict[str, tuple[NumericValue, ...]]:
    """Returns projected numeric values grouped and encoded-value sorted."""

    tables = {
        entity.entity_id: entity
        for entity in projection.entities
        if isinstance(entity, NumericTable)
    }
    grouped: dict[str, list[NumericValue]] = {table_id: [] for table_id in tables}
    for entity in projection.entities:
        if not isinstance(entity, NumericValue):
            continue
        values = grouped.get(entity.table_id)
        if values is None:
            raise ValueError(
                f"{entity.entity_id}: projected without table {entity.table_id}"
            )
        values.append(entity)
    missing = [table_id for table_id, values in grouped.items() if not values]
    if missing:
        raise ValueError(f"projected numeric tables without values: {missing}")
    return {
        table_id: tuple(sorted(values, key=lambda value: value.value))
        for table_id, values in grouped.items()
    }
