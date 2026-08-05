# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict immutable model of the canonical SPIR-V core JSON grammar."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import cast

from build_tools.spirv.registry.registry_source import (
    RegistryOrigin,
    RegistrySourceError,
)

_MAGIC_NUMBER_PATTERN = re.compile(r"0x[0-9A-Fa-f]{8}")
_BIT_VALUE_PATTERN = re.compile(r"0x[0-9A-Fa-f]+")
_VERSION_PATTERN = re.compile(r"([1-9][0-9]*)\.(0|[1-9][0-9]*)")
_SPIRV_MAGIC_NUMBER = 0x07230203
_MAX_WORD_VALUE = 0xFFFFFFFF
_MAX_OPCODE = 0xFFFF


@dataclass(frozen=True, order=True, slots=True)
class SpirvVersion:
    """One SPIR-V core major/minor version."""

    # Major version encoded in the SPIR-V binary version word.
    major: int
    # Minor version encoded in the SPIR-V binary version word.
    minor: int

    def __post_init__(self) -> None:
        if type(self.major) is not int or not 1 <= self.major <= 0xFF:
            raise ValueError("SPIR-V major version must be a nonzero 8-bit integer")
        if type(self.minor) is not int or not 0 <= self.minor <= 0xFF:
            raise ValueError("SPIR-V minor version must be an 8-bit integer")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


class SpirvOperandCategory(Enum):
    """Canonical operand-kind categories."""

    BIT_ENUM = "BitEnum"
    VALUE_ENUM = "ValueEnum"
    ID = "Id"
    LITERAL = "Literal"
    COMPOSITE = "Composite"


class SpirvOperandQuantifier(Enum):
    """Cardinality modifiers accepted on instruction operands and parameters."""

    OPTIONAL = "?"
    VARIADIC = "*"


@dataclass(frozen=True, slots=True)
class SpirvInstructionPrintingClass:
    """One instruction grouping used by canonical assembly printing."""

    # Stable class tag referenced by instruction definitions.
    tag: str
    # Human-readable section heading, when this class is printed.
    heading: str | None
    # Canonical source definition of this printing class.
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class SpirvInstructionOperand:
    """One typed operand in an instruction signature."""

    # Operand-kind name defining the encoded operand.
    kind: str
    # Specification display name for the operand, when present.
    name: str | None
    # Optional or variadic cardinality, when the operand is not singular.
    quantifier: SpirvOperandQuantifier | None
    # Canonical source definition of this operand occurrence.
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class SpirvEnumerantParameter:
    """One typed parameter carried by an operand enumerant."""

    # Operand-kind name defining the encoded parameter.
    kind: str
    # Specification display name for the parameter, when present.
    name: str | None
    # Optional or variadic cardinality, when the parameter is not singular.
    quantifier: SpirvOperandQuantifier | None
    # Canonical source definition of this parameter occurrence.
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class SpirvEnumerant:
    """One canonical operand enumerant and its availability requirements."""

    # Primary specification name of the enumerant.
    name: str
    # Equivalent historical or extension spellings of the enumerant.
    aliases: tuple[str, ...]
    # Numeric word or mask value encoded in SPIR-V.
    value: int
    # First core version, or None for an extension-only/unversioned enumerant.
    version: SpirvVersion | None
    # Last core version containing the enumerant, when it was later removed.
    last_version: SpirvVersion | None
    # True when the definition is provisional rather than ratified.
    provisional: bool
    # Canonical primary capability names. These are alternative guards for
    # ordinary enumerants and capabilities implied by Capability enumerants.
    capabilities: tuple[str, ...]
    # Alternative extensions through which the enumerant is available.
    extensions: tuple[str, ...]
    # Encoded parameters selected by this enumerant.
    parameters: tuple[SpirvEnumerantParameter, ...]
    # Canonical source definition of this enumerant.
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class SpirvOperandKind:
    """One operand kind with category-specific structure and direct indexes."""

    # Primary grammar name referenced by instructions and parameters.
    name: str
    # Structural category controlling this kind's remaining fields.
    category: SpirvOperandCategory
    # Literal or ID documentation supplied by the grammar.
    doc: str | None
    # Ordered component kind names for a composite kind.
    bases: tuple[str, ...]
    # Ordered definitions for a value or bit enumerant kind.
    enumerants: tuple[SpirvEnumerant, ...]
    # Canonical source definition of this operand kind.
    origin: RegistryOrigin
    # Direct lookup by every primary and alias enumerant name.
    enumerants_by_name: Mapping[str, SpirvEnumerant] = field(
        repr=False, compare=False, hash=False
    )
    # Direct lookup by encoded enumerant value.
    enumerants_by_value: Mapping[int, SpirvEnumerant] = field(
        repr=False, compare=False, hash=False
    )


@dataclass(frozen=True, slots=True)
class SpirvInstruction:
    """One canonical SPIR-V instruction definition."""

    # Primary specification name of the instruction.
    opname: str
    # Equivalent historical or extension spellings of the instruction.
    aliases: tuple[str, ...]
    # Printing-class tag grouping the instruction in specification output.
    printing_class: str
    # Numeric opcode encoded in the low half of the first instruction word.
    opcode: int
    # First core version, or None for an extension-only instruction.
    version: SpirvVersion | None
    # Last core version containing the instruction, when it was later removed.
    last_version: SpirvVersion | None
    # True when the definition is provisional rather than ratified.
    provisional: bool
    # Canonical primary names of alternative enabling capabilities.
    capabilities: tuple[str, ...]
    # Alternative extensions through which the instruction is available.
    extensions: tuple[str, ...]
    # Ordered encoded operand signature.
    operands: tuple[SpirvInstructionOperand, ...]
    # Canonical source definition of this instruction.
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class SpirvCoreGrammar:
    """The complete core grammar with immutable direct lookup indexes."""

    # Root identity of the canonical source document.
    source_origin: RegistryOrigin
    # Copyright and source-integrity notice retained from the document.
    copyright: tuple[str, ...]
    # Binary module magic number declared by the grammar.
    magic_number: int
    # Latest core version represented by the grammar.
    version: SpirvVersion
    # Grammar schema/content revision within that core version.
    revision: int
    # Printing classes in canonical source order.
    printing_classes: tuple[SpirvInstructionPrintingClass, ...]
    # Instructions in canonical opcode order.
    instructions: tuple[SpirvInstruction, ...]
    # Operand kinds in canonical source order.
    operand_kinds: tuple[SpirvOperandKind, ...]
    # Every extension string referenced by an instruction or enumerant.
    referenced_extensions: frozenset[str]
    # Direct lookup by printing-class tag.
    printing_classes_by_tag: Mapping[str, SpirvInstructionPrintingClass] = field(
        repr=False, compare=False, hash=False
    )
    # Direct lookup by every primary and alias instruction name.
    instructions_by_name: Mapping[str, SpirvInstruction] = field(
        repr=False, compare=False, hash=False
    )
    # Direct lookup by encoded instruction opcode.
    instructions_by_opcode: Mapping[int, SpirvInstruction] = field(
        repr=False, compare=False, hash=False
    )
    # Direct lookup by operand-kind name.
    operand_kinds_by_name: Mapping[str, SpirvOperandKind] = field(
        repr=False, compare=False, hash=False
    )
    # Direct lookup by every primary and alias capability name.
    capabilities_by_name: Mapping[str, SpirvEnumerant] = field(
        repr=False, compare=False, hash=False
    )
    # Direct implications keyed by every capability primary/alias name.
    capability_direct_implications: Mapping[str, tuple[str, ...]] = field(
        repr=False, compare=False, hash=False
    )
    # Transitive implications keyed by every capability primary/alias name.
    capability_implication_closures: Mapping[str, frozenset[str]] = field(
        repr=False, compare=False, hash=False
    )


class _DuplicateJsonKeyError(Exception):
    def __init__(self, key: str) -> None:
        self.key = key


class _InvalidJsonConstantError(Exception):
    def __init__(self, value: str) -> None:
        self.value = value


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise _InvalidJsonConstantError(value)


def _expect_object(value: object, origin: RegistryOrigin) -> dict[str, object]:
    if type(value) is not dict:
        raise RegistrySourceError(origin, "expected a JSON object")
    return cast(dict[str, object], value)


def _expect_fields(
    value: dict[str, object],
    origin: RegistryOrigin,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            parts.append(f"unknown fields: {', '.join(unknown)}")
        raise RegistrySourceError(origin, "; ".join(parts))


def _expect_list(value: object, origin: RegistryOrigin) -> list[object]:
    if type(value) is not list:
        raise RegistrySourceError(origin, "expected a JSON array")
    return cast(list[object], value)


def _expect_string(
    value: object,
    origin: RegistryOrigin,
    *,
    allow_empty: bool = False,
) -> str:
    if type(value) is not str:
        raise RegistrySourceError(origin, "expected a string")
    if not value and not allow_empty:
        raise RegistrySourceError(origin, "expected a non-empty string")
    return value


def _expect_integer(
    value: object,
    origin: RegistryOrigin,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int:
        raise RegistrySourceError(origin, "expected an integer")
    if not minimum <= value <= maximum:
        raise RegistrySourceError(
            origin,
            f"integer {value} is outside [{minimum}, {maximum}]",
        )
    return value


def _named_origin(origin: RegistryOrigin, name: str) -> RegistryOrigin:
    return RegistryOrigin(origin.source, f"{origin.locator}({name})")


def _expect_optional_string(
    value: dict[str, object], key: str, origin: RegistryOrigin
) -> str | None:
    if key not in value:
        return None
    return _expect_string(value[key], origin.child(key))


def _expect_string_tuple(
    value: dict[str, object],
    key: str,
    origin: RegistryOrigin,
    *,
    required: bool = False,
    unique: bool = True,
) -> tuple[str, ...]:
    if key not in value:
        if required:
            raise RegistrySourceError(origin, f"missing fields: {key}")
        return ()
    values = _expect_list(value[key], origin.child(key))
    if not values:
        raise RegistrySourceError(origin.child(key), "expected a non-empty array")
    result = tuple(
        _expect_string(item, origin.child(key).child(f"[{index}]"))
        for index, item in enumerate(values)
    )
    seen_names: set[str] = set()
    duplicate_names: set[str] = set()
    if unique:
        for name in result:
            if name in seen_names:
                duplicate_names.add(name)
            else:
                seen_names.add(name)
    if duplicate_names:
        raise RegistrySourceError(
            origin.child(key),
            f"duplicate names: {', '.join(sorted(duplicate_names))}",
        )
    return result


def _parse_version(
    value: object,
    origin: RegistryOrigin,
    *,
    allow_none: bool,
) -> SpirvVersion | None:
    text = _expect_string(value, origin)
    if text == "None":
        if allow_none:
            return None
        raise RegistrySourceError(origin, "lastVersion cannot be 'None'")
    match = _VERSION_PATTERN.fullmatch(text)
    if match is None:
        raise RegistrySourceError(origin, f"invalid SPIR-V version {text!r}")
    try:
        return SpirvVersion(int(match.group(1)), int(match.group(2)))
    except ValueError as exception:
        raise RegistrySourceError(origin, str(exception)) from exception


def _parse_optional_version(
    value: dict[str, object],
    key: str,
    origin: RegistryOrigin,
    *,
    allow_none: bool,
) -> SpirvVersion | None:
    if key not in value:
        return None
    return _parse_version(value[key], origin.child(key), allow_none=allow_none)


def _parse_provisional(value: dict[str, object], origin: RegistryOrigin) -> bool:
    if "provisional" not in value:
        return False
    provisional = value["provisional"]
    if provisional is not True:
        raise RegistrySourceError(
            origin.child("provisional"),
            "provisional must be the boolean true when present",
        )
    return True


def _validate_entity_versions(
    version: SpirvVersion | None,
    last_version: SpirvVersion | None,
    provisional: bool,
    grammar_version: SpirvVersion,
    origin: RegistryOrigin,
) -> None:
    if version is not None and version > grammar_version:
        raise RegistrySourceError(
            origin.child("version"),
            f"version {version} exceeds grammar version {grammar_version}",
        )
    if last_version is not None:
        if version is None:
            raise RegistrySourceError(
                origin.child("lastVersion"),
                "lastVersion requires a core introduction version",
            )
        if last_version < version:
            raise RegistrySourceError(
                origin.child("lastVersion"),
                f"lastVersion {last_version} precedes version {version}",
            )
        if last_version > grammar_version:
            raise RegistrySourceError(
                origin.child("lastVersion"),
                f"lastVersion {last_version} exceeds grammar version {grammar_version}",
            )
    if provisional and version is not None:
        raise RegistrySourceError(
            origin.child("provisional"),
            "provisional definitions must have version 'None'",
        )


def _parse_quantifier(
    value: dict[str, object], origin: RegistryOrigin
) -> SpirvOperandQuantifier | None:
    if "quantifier" not in value:
        return None
    text = _expect_string(value["quantifier"], origin.child("quantifier"))
    try:
        return SpirvOperandQuantifier(text)
    except ValueError as exception:
        raise RegistrySourceError(
            origin.child("quantifier"),
            f"unsupported operand quantifier {text!r}",
        ) from exception


def _parse_printing_class(
    value: object, origin: RegistryOrigin
) -> SpirvInstructionPrintingClass:
    fields = _expect_object(value, origin)
    _expect_fields(
        fields,
        origin,
        required=frozenset(("tag",)),
        optional=frozenset(("heading",)),
    )
    tag = _expect_string(fields["tag"], origin.child("tag"))
    definition_origin = _named_origin(origin, tag)
    return SpirvInstructionPrintingClass(
        tag=tag,
        heading=_expect_optional_string(fields, "heading", definition_origin),
        origin=definition_origin,
    )


def _parse_enumerant_parameter(
    value: object, origin: RegistryOrigin
) -> SpirvEnumerantParameter:
    fields = _expect_object(value, origin)
    _expect_fields(
        fields,
        origin,
        required=frozenset(("kind",)),
        optional=frozenset(("name", "quantifier")),
    )
    return SpirvEnumerantParameter(
        kind=_expect_string(fields["kind"], origin.child("kind")),
        name=_expect_optional_string(fields, "name", origin),
        quantifier=_parse_quantifier(fields, origin),
        origin=origin,
    )


def _parse_enumerant(
    value: object,
    category: SpirvOperandCategory,
    grammar_version: SpirvVersion,
    origin: RegistryOrigin,
) -> SpirvEnumerant:
    fields = _expect_object(value, origin)
    _expect_fields(
        fields,
        origin,
        required=frozenset(("enumerant", "value")),
        optional=frozenset(
            (
                "aliases",
                "capabilities",
                "extensions",
                "parameters",
                "version",
                "lastVersion",
                "provisional",
            )
        ),
    )
    name = _expect_string(fields["enumerant"], origin.child("enumerant"))
    definition_origin = _named_origin(origin, name)

    if category is SpirvOperandCategory.BIT_ENUM:
        encoded_value = _expect_string(
            fields["value"], definition_origin.child("value")
        )
        if _BIT_VALUE_PATTERN.fullmatch(encoded_value) is None:
            raise RegistrySourceError(
                definition_origin.child("value"),
                "bit-enum values must be hexadecimal strings",
            )
        enumerant_value = int(encoded_value, 16)
        if enumerant_value > _MAX_WORD_VALUE:
            raise RegistrySourceError(
                definition_origin.child("value"),
                "bit-enum value exceeds one SPIR-V word",
            )
    else:
        enumerant_value = _expect_integer(
            fields["value"],
            definition_origin.child("value"),
            minimum=0,
            maximum=_MAX_WORD_VALUE,
        )

    if "version" not in fields and not (
        category is SpirvOperandCategory.BIT_ENUM and enumerant_value == 0
    ):
        raise RegistrySourceError(definition_origin, "missing fields: version")

    version = _parse_optional_version(
        fields, "version", definition_origin, allow_none=True
    )
    last_version = _parse_optional_version(
        fields, "lastVersion", definition_origin, allow_none=False
    )
    provisional = _parse_provisional(fields, definition_origin)
    _validate_entity_versions(
        version, last_version, provisional, grammar_version, definition_origin
    )

    parameters = ()
    if "parameters" in fields:
        parameter_values = _expect_list(
            fields["parameters"], definition_origin.child("parameters")
        )
        if not parameter_values:
            raise RegistrySourceError(
                definition_origin.child("parameters"),
                "expected a non-empty array",
            )
        parameters = tuple(
            _parse_enumerant_parameter(
                parameter,
                definition_origin.child("parameters").child(f"[{index}]"),
            )
            for index, parameter in enumerate(parameter_values)
        )

    return SpirvEnumerant(
        name=name,
        aliases=_expect_string_tuple(fields, "aliases", definition_origin),
        value=enumerant_value,
        version=version,
        last_version=last_version,
        provisional=provisional,
        capabilities=_expect_string_tuple(fields, "capabilities", definition_origin),
        extensions=_expect_string_tuple(fields, "extensions", definition_origin),
        parameters=parameters,
        origin=definition_origin,
    )


def _build_enumerant_indexes(
    kind_name: str,
    enumerants: Sequence[SpirvEnumerant],
) -> tuple[Mapping[str, SpirvEnumerant], Mapping[int, SpirvEnumerant]]:
    by_name: dict[str, SpirvEnumerant] = {}
    by_value: dict[int, SpirvEnumerant] = {}
    previous_value: int | None = None
    for enumerant in enumerants:
        for name in (enumerant.name, *enumerant.aliases):
            existing = by_name.get(name)
            if existing is not None:
                raise RegistrySourceError(
                    enumerant.origin,
                    f"operand kind {kind_name!r} enumerant name {name!r} "
                    f"collides with {existing.origin}",
                )
            by_name[name] = enumerant
        existing_value = by_value.get(enumerant.value)
        if existing_value is not None:
            raise RegistrySourceError(
                enumerant.origin,
                f"operand kind {kind_name!r} value {enumerant.value} "
                f"collides with {existing_value.origin}",
            )
        if previous_value is not None and enumerant.value < previous_value:
            raise RegistrySourceError(
                enumerant.origin.child("value"),
                f"operand kind {kind_name!r} enumerants are not in increasing "
                "value order",
            )
        by_value[enumerant.value] = enumerant
        previous_value = enumerant.value
    return MappingProxyType(by_name), MappingProxyType(by_value)


def _parse_operand_kind(
    value: object,
    grammar_version: SpirvVersion,
    origin: RegistryOrigin,
) -> SpirvOperandKind:
    fields = _expect_object(value, origin)
    _expect_fields(
        fields,
        origin,
        required=frozenset(("category", "kind")),
        optional=frozenset(("bases", "doc", "enumerants")),
    )
    category_text = _expect_string(fields["category"], origin.child("category"))
    try:
        category = SpirvOperandCategory(category_text)
    except ValueError as exception:
        raise RegistrySourceError(
            origin.child("category"),
            f"unsupported operand category {category_text!r}",
        ) from exception
    kind_name = _expect_string(fields["kind"], origin.child("kind"))
    definition_origin = _named_origin(origin, kind_name)

    if category in (SpirvOperandCategory.BIT_ENUM, SpirvOperandCategory.VALUE_ENUM):
        expected_field = "enumerants"
    elif category in (SpirvOperandCategory.ID, SpirvOperandCategory.LITERAL):
        expected_field = "doc"
    else:
        expected_field = "bases"
    category_fields = {key for key in ("bases", "doc", "enumerants") if key in fields}
    if category_fields != {expected_field}:
        raise RegistrySourceError(
            definition_origin,
            f"operand category {category.value!r} requires only {expected_field!r}; "
            f"found {', '.join(sorted(category_fields)) or 'none'}",
        )

    doc = None
    bases: tuple[str, ...] = ()
    enumerants: tuple[SpirvEnumerant, ...] = ()
    if expected_field == "doc":
        doc = _expect_string(fields["doc"], definition_origin.child("doc"))
    elif expected_field == "bases":
        bases = _expect_string_tuple(
            fields,
            "bases",
            definition_origin,
            required=True,
            unique=False,
        )
    else:
        enumerant_values = _expect_list(
            fields["enumerants"], definition_origin.child("enumerants")
        )
        if not enumerant_values:
            raise RegistrySourceError(
                definition_origin.child("enumerants"),
                "expected a non-empty array",
            )
        enumerants = tuple(
            _parse_enumerant(
                enumerant,
                category,
                grammar_version,
                definition_origin.child("enumerants").child(f"[{index}]"),
            )
            for index, enumerant in enumerate(enumerant_values)
        )
    by_name, by_value = _build_enumerant_indexes(kind_name, enumerants)
    return SpirvOperandKind(
        name=kind_name,
        category=category,
        doc=doc,
        bases=bases,
        enumerants=enumerants,
        origin=definition_origin,
        enumerants_by_name=by_name,
        enumerants_by_value=by_value,
    )


def _canonicalize_capability_references(
    operand_kinds: Sequence[SpirvOperandKind],
    capabilities_by_name: Mapping[str, SpirvEnumerant],
) -> tuple[SpirvOperandKind, ...]:
    result = []
    for operand_kind in operand_kinds:
        enumerants = []
        for enumerant in operand_kind.enumerants:
            canonical_names = []
            for name in enumerant.capabilities:
                capability = capabilities_by_name.get(name)
                if capability is None:
                    raise RegistrySourceError(
                        enumerant.origin.child("capabilities"),
                        f"unknown capability {name!r}",
                    )
                if capability.name in canonical_names:
                    raise RegistrySourceError(
                        enumerant.origin.child("capabilities"),
                        f"capability {capability.name!r} is repeated through aliases",
                    )
                canonical_names.append(capability.name)
            enumerants.append(replace(enumerant, capabilities=tuple(canonical_names)))
        by_name, by_value = _build_enumerant_indexes(operand_kind.name, enumerants)
        result.append(
            replace(
                operand_kind,
                enumerants=tuple(enumerants),
                enumerants_by_name=by_name,
                enumerants_by_value=by_value,
            )
        )
    return tuple(result)


def _parse_instruction_operand(
    value: object, origin: RegistryOrigin
) -> SpirvInstructionOperand:
    fields = _expect_object(value, origin)
    _expect_fields(
        fields,
        origin,
        required=frozenset(("kind",)),
        optional=frozenset(("name", "quantifier")),
    )
    return SpirvInstructionOperand(
        kind=_expect_string(fields["kind"], origin.child("kind")),
        name=_expect_optional_string(fields, "name", origin),
        quantifier=_parse_quantifier(fields, origin),
        origin=origin,
    )


def _parse_instruction(
    value: object,
    grammar_version: SpirvVersion,
    capabilities_by_name: Mapping[str, SpirvEnumerant],
    origin: RegistryOrigin,
) -> SpirvInstruction:
    fields = _expect_object(value, origin)
    _expect_fields(
        fields,
        origin,
        required=frozenset(("class", "opcode", "opname", "version")),
        optional=frozenset(
            (
                "aliases",
                "capabilities",
                "extensions",
                "lastVersion",
                "operands",
                "provisional",
            )
        ),
    )
    opname = _expect_string(fields["opname"], origin.child("opname"))
    definition_origin = _named_origin(origin, opname)
    version = _parse_version(
        fields["version"], definition_origin.child("version"), allow_none=True
    )
    last_version = _parse_optional_version(
        fields, "lastVersion", definition_origin, allow_none=False
    )
    provisional = _parse_provisional(fields, definition_origin)
    _validate_entity_versions(
        version, last_version, provisional, grammar_version, definition_origin
    )

    capabilities = []
    for name in _expect_string_tuple(fields, "capabilities", definition_origin):
        capability = capabilities_by_name.get(name)
        if capability is None:
            raise RegistrySourceError(
                definition_origin.child("capabilities"),
                f"unknown capability {name!r}",
            )
        if capability.name in capabilities:
            raise RegistrySourceError(
                definition_origin.child("capabilities"),
                f"capability {capability.name!r} is repeated through aliases",
            )
        capabilities.append(capability.name)
    extensions = _expect_string_tuple(fields, "extensions", definition_origin)
    if version is None and not capabilities and not extensions:
        raise RegistrySourceError(
            definition_origin,
            "an instruction outside all core versions requires a capability or "
            "extension guard",
        )

    operands = ()
    if "operands" in fields:
        operand_values = _expect_list(
            fields["operands"], definition_origin.child("operands")
        )
        if not operand_values:
            raise RegistrySourceError(
                definition_origin.child("operands"),
                "expected a non-empty array",
            )
        operands = tuple(
            _parse_instruction_operand(
                operand,
                definition_origin.child("operands").child(f"[{index}]"),
            )
            for index, operand in enumerate(operand_values)
        )

    return SpirvInstruction(
        opname=opname,
        aliases=_expect_string_tuple(fields, "aliases", definition_origin),
        printing_class=_expect_string(
            fields["class"], definition_origin.child("class")
        ),
        opcode=_expect_integer(
            fields["opcode"],
            definition_origin.child("opcode"),
            minimum=0,
            maximum=_MAX_OPCODE,
        ),
        version=version,
        last_version=last_version,
        provisional=provisional,
        capabilities=tuple(capabilities),
        extensions=extensions,
        operands=operands,
        origin=definition_origin,
    )


def _build_capability_implication_closures(
    capability_kind: SpirvOperandKind,
) -> tuple[Mapping[str, tuple[str, ...]], Mapping[str, frozenset[str]]]:
    direct_by_primary_name = {
        capability.name: capability.capabilities
        for capability in capability_kind.enumerants
    }
    closures_by_primary_name: dict[str, frozenset[str]] = {}
    active_names: list[str] = []

    def visit(name: str) -> frozenset[str]:
        existing = closures_by_primary_name.get(name)
        if existing is not None:
            return existing
        if name in active_names:
            cycle = active_names[active_names.index(name) :] + [name]
            capability = capability_kind.enumerants_by_name[name]
            raise RegistrySourceError(
                capability.origin.child("capabilities"),
                f"capability dependency cycle: {' -> '.join(cycle)}",
            )
        active_names.append(name)
        closure: set[str] = set()
        for dependency in direct_by_primary_name[name]:
            closure.add(dependency)
            closure.update(visit(dependency))
        active_names.pop()
        result = frozenset(closure)
        closures_by_primary_name[name] = result
        return result

    for capability in capability_kind.enumerants:
        visit(capability.name)

    direct_by_name: dict[str, tuple[str, ...]] = {}
    closures_by_name: dict[str, frozenset[str]] = {}
    for capability in capability_kind.enumerants:
        for name in (capability.name, *capability.aliases):
            direct_by_name[name] = direct_by_primary_name[capability.name]
            closures_by_name[name] = closures_by_primary_name[capability.name]
    return MappingProxyType(direct_by_name), MappingProxyType(closures_by_name)


def _parse_document(document: object, source: str) -> SpirvCoreGrammar:
    root_origin = RegistryOrigin(source, "$")
    fields = _expect_object(document, root_origin)
    _expect_fields(
        fields,
        root_origin,
        required=frozenset(
            (
                "copyright",
                "instruction_printing_class",
                "instructions",
                "magic_number",
                "major_version",
                "minor_version",
                "operand_kinds",
                "revision",
            )
        ),
    )

    copyright_values = _expect_list(fields["copyright"], root_origin.child("copyright"))
    copyright_lines = tuple(
        _expect_string(
            value,
            root_origin.child("copyright").child(f"[{index}]"),
            allow_empty=True,
        )
        for index, value in enumerate(copyright_values)
    )

    magic_number_text = _expect_string(
        fields["magic_number"], root_origin.child("magic_number")
    )
    if _MAGIC_NUMBER_PATTERN.fullmatch(magic_number_text) is None:
        raise RegistrySourceError(
            root_origin.child("magic_number"),
            "magic_number must be an eight-digit hexadecimal string",
        )
    magic_number = int(magic_number_text, 16)
    if magic_number != _SPIRV_MAGIC_NUMBER:
        raise RegistrySourceError(
            root_origin.child("magic_number"),
            f"expected SPIR-V magic number 0x{_SPIRV_MAGIC_NUMBER:08X}",
        )
    grammar_version = SpirvVersion(
        _expect_integer(
            fields["major_version"],
            root_origin.child("major_version"),
            minimum=0,
            maximum=0xFF,
        ),
        _expect_integer(
            fields["minor_version"],
            root_origin.child("minor_version"),
            minimum=0,
            maximum=0xFF,
        ),
    )
    revision = _expect_integer(
        fields["revision"],
        root_origin.child("revision"),
        minimum=0,
        maximum=_MAX_WORD_VALUE,
    )

    printing_class_values = _expect_list(
        fields["instruction_printing_class"],
        root_origin.child("instruction_printing_class"),
    )
    if not printing_class_values:
        raise RegistrySourceError(
            root_origin.child("instruction_printing_class"),
            "expected a non-empty array",
        )
    printing_classes = tuple(
        _parse_printing_class(
            value,
            root_origin.child("instruction_printing_class").child(f"[{index}]"),
        )
        for index, value in enumerate(printing_class_values)
    )
    printing_classes_by_tag: dict[str, SpirvInstructionPrintingClass] = {}
    for printing_class in printing_classes:
        existing = printing_classes_by_tag.get(printing_class.tag)
        if existing is not None:
            raise RegistrySourceError(
                printing_class.origin,
                f"printing class tag {printing_class.tag!r} collides with "
                f"{existing.origin}",
            )
        printing_classes_by_tag[printing_class.tag] = printing_class

    operand_kind_values = _expect_list(
        fields["operand_kinds"], root_origin.child("operand_kinds")
    )
    if not operand_kind_values:
        raise RegistrySourceError(
            root_origin.child("operand_kinds"), "expected a non-empty array"
        )
    raw_operand_kinds = tuple(
        _parse_operand_kind(
            value,
            grammar_version,
            root_origin.child("operand_kinds").child(f"[{index}]"),
        )
        for index, value in enumerate(operand_kind_values)
    )
    raw_operand_kinds_by_name: dict[str, SpirvOperandKind] = {}
    for operand_kind in raw_operand_kinds:
        existing = raw_operand_kinds_by_name.get(operand_kind.name)
        if existing is not None:
            raise RegistrySourceError(
                operand_kind.origin,
                f"operand kind {operand_kind.name!r} collides with {existing.origin}",
            )
        raw_operand_kinds_by_name[operand_kind.name] = operand_kind

    raw_capability_kind = raw_operand_kinds_by_name.get("Capability")
    if raw_capability_kind is None:
        raise RegistrySourceError(
            root_origin.child("operand_kinds"),
            "missing required Capability operand kind",
        )
    if raw_capability_kind.category is not SpirvOperandCategory.VALUE_ENUM:
        raise RegistrySourceError(
            raw_capability_kind.origin,
            "Capability operand kind must be a ValueEnum",
        )
    for capability in raw_capability_kind.enumerants:
        if capability.version is None and not capability.extensions:
            raise RegistrySourceError(
                capability.origin,
                "a capability outside all core versions requires an extension guard",
            )
    raw_capabilities_by_name = raw_capability_kind.enumerants_by_name
    operand_kinds = _canonicalize_capability_references(
        raw_operand_kinds, raw_capabilities_by_name
    )
    operand_kinds_by_name = {kind.name: kind for kind in operand_kinds}
    capability_kind = operand_kinds_by_name["Capability"]
    capabilities_by_name = capability_kind.enumerants_by_name

    for operand_kind in operand_kinds:
        for base in operand_kind.bases:
            if base not in operand_kinds_by_name:
                raise RegistrySourceError(
                    operand_kind.origin.child("bases"),
                    f"unknown operand kind {base!r}",
                )
        for enumerant in operand_kind.enumerants:
            for parameter in enumerant.parameters:
                if parameter.kind not in operand_kinds_by_name:
                    raise RegistrySourceError(
                        parameter.origin.child("kind"),
                        f"unknown operand kind {parameter.kind!r}",
                    )

    instruction_values = _expect_list(
        fields["instructions"], root_origin.child("instructions")
    )
    if not instruction_values:
        raise RegistrySourceError(
            root_origin.child("instructions"), "expected a non-empty array"
        )
    instructions = tuple(
        _parse_instruction(
            value,
            grammar_version,
            capabilities_by_name,
            root_origin.child("instructions").child(f"[{index}]"),
        )
        for index, value in enumerate(instruction_values)
    )
    instructions_by_name: dict[str, SpirvInstruction] = {}
    instructions_by_opcode: dict[int, SpirvInstruction] = {}
    previous_opcode: int | None = None
    for instruction in instructions:
        if instruction.printing_class not in printing_classes_by_tag:
            raise RegistrySourceError(
                instruction.origin.child("class"),
                f"unknown instruction printing class {instruction.printing_class!r}",
            )
        for operand in instruction.operands:
            if operand.kind not in operand_kinds_by_name:
                raise RegistrySourceError(
                    operand.origin.child("kind"),
                    f"unknown operand kind {operand.kind!r}",
                )
        for name in (instruction.opname, *instruction.aliases):
            existing = instructions_by_name.get(name)
            if existing is not None:
                raise RegistrySourceError(
                    instruction.origin,
                    f"instruction name {name!r} collides with {existing.origin}",
                )
            instructions_by_name[name] = instruction
        existing_opcode = instructions_by_opcode.get(instruction.opcode)
        if existing_opcode is not None:
            raise RegistrySourceError(
                instruction.origin.child("opcode"),
                f"opcode {instruction.opcode} collides with {existing_opcode.origin}",
            )
        if previous_opcode is not None and instruction.opcode < previous_opcode:
            raise RegistrySourceError(
                instruction.origin.child("opcode"),
                "instructions are not in increasing opcode order",
            )
        instructions_by_opcode[instruction.opcode] = instruction
        previous_opcode = instruction.opcode

    direct_implications, implication_closures = _build_capability_implication_closures(
        capability_kind
    )
    referenced_extensions = frozenset(
        extension
        for operand_kind in operand_kinds
        for enumerant in operand_kind.enumerants
        for extension in enumerant.extensions
    ) | frozenset(
        extension
        for instruction in instructions
        for extension in instruction.extensions
    )

    return SpirvCoreGrammar(
        source_origin=root_origin,
        copyright=copyright_lines,
        magic_number=magic_number,
        version=grammar_version,
        revision=revision,
        printing_classes=printing_classes,
        instructions=instructions,
        operand_kinds=operand_kinds,
        referenced_extensions=referenced_extensions,
        printing_classes_by_tag=MappingProxyType(printing_classes_by_tag),
        instructions_by_name=MappingProxyType(instructions_by_name),
        instructions_by_opcode=MappingProxyType(instructions_by_opcode),
        operand_kinds_by_name=MappingProxyType(operand_kinds_by_name),
        capabilities_by_name=capabilities_by_name,
        capability_direct_implications=direct_implications,
        capability_implication_closures=implication_closures,
    )


def parse_spirv_core_grammar(
    contents: str | bytes | bytearray,
    *,
    source: str = "<SPIR-V core grammar>",
) -> SpirvCoreGrammar:
    """Parses and validates one complete SPIR-V core grammar document."""

    if not isinstance(contents, (str, bytes, bytearray)):
        raise TypeError("SPIR-V core grammar contents must be text or bytes")
    if not isinstance(source, str) or not source:
        raise TypeError("SPIR-V core grammar source must be a non-empty string")
    try:
        document = json.loads(
            contents,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exception:
        origin = RegistryOrigin(
            source,
            f"line {exception.lineno}, column {exception.colno}",
        )
        raise RegistrySourceError(origin, exception.msg) from exception
    except UnicodeDecodeError as exception:
        raise RegistrySourceError(
            RegistryOrigin(source, f"byte {exception.start}"),
            "grammar is not valid Unicode text",
        ) from exception
    except _DuplicateJsonKeyError as exception:
        raise RegistrySourceError(
            RegistryOrigin(source, "$"),
            f"duplicate JSON object field {exception.key!r}",
        ) from exception
    except _InvalidJsonConstantError as exception:
        raise RegistrySourceError(
            RegistryOrigin(source, "$"),
            f"non-standard JSON constant {exception.value!r}",
        ) from exception
    return _parse_document(document, source)


def load_spirv_core_grammar(
    path: str | os.PathLike[str],
) -> SpirvCoreGrammar:
    """Loads and validates one complete SPIR-V core grammar file."""

    source_path = Path(path)
    return parse_spirv_core_grammar(
        source_path.read_bytes(),
        source=os.fspath(source_path),
    )
