# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict importer for Vulkan SPIR-V availability metadata.

This importer owns the Vulkan-registry facts a SPIR-V compiler consumes. It
retains core versions, extensions, availability blocks, the referenced
structure/member and property-constant closure, and the ordered alternatives
that enable SPIR-V extensions and capabilities. Unrelated Vulkan API schemas
are classified at the document boundary but are not materialized.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

from build_tools.spirv.registry.dependency_expression import (
    DependencyExpression,
    DependencyExpressionParseError,
    dependency_expression_names,
    parse_dependency_expression,
)
from build_tools.spirv.registry.registry_source import (
    RegistryOrigin,
    RegistrySourceError,
)
from build_tools.spirv.registry.vulkan_spirv_availability_model import (
    VulkanApi,
    VulkanAvailabilityBlock,
    VulkanAvailabilityBlockKind,
    VulkanConstant,
    VulkanCoreFeature,
    VulkanExtension,
    VulkanExtensionType,
    VulkanFeatureReference,
    VulkanSpirvAvailability,
    VulkanSpirvAvailabilityRegistry,
    VulkanSpirvCoreEnable,
    VulkanSpirvEnable,
    VulkanSpirvExtensionEnable,
    VulkanSpirvFeatureEnable,
    VulkanSpirvPropertyEnable,
    VulkanStructure,
    VulkanStructureMember,
    VulkanTypeReference,
    VulkanVersion,
)

_VERSION_PATTERN = re.compile(r"([1-9][0-9]*)\.(0|[1-9][0-9]*)")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TOP_LEVEL_SECTIONS = frozenset(
    (
        "commands",
        "comment",
        "enums",
        "extensions",
        "feature",
        "formats",
        "platforms",
        "spirvcapabilities",
        "spirvextensions",
        "sync",
        "tags",
        "types",
        "videocodecs",
    )
)
_SINGLETON_TOP_LEVEL_SECTIONS = frozenset(
    (
        "commands",
        "extensions",
        "formats",
        "platforms",
        "spirvcapabilities",
        "spirvextensions",
        "sync",
        "tags",
        "types",
        "videocodecs",
    )
)
_REQUIRED_TOP_LEVEL_SECTIONS = frozenset(
    ("extensions", "spirvcapabilities", "spirvextensions", "types")
)
_BLOCK_ENUM_ATTRIBUTES = frozenset(
    (
        "alias",
        "api",
        "bitpos",
        "comment",
        "deprecated",
        "dir",
        "extends",
        "extnumber",
        "name",
        "offset",
        "protect",
        "value",
    )
)
_AVAILABILITY_BLOCK_TAGS = frozenset(kind.value for kind in VulkanAvailabilityBlockKind)


@dataclass(frozen=True, slots=True)
class _RawType:
    name: str
    category: str | None
    alias: str | None
    element: ET.Element
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class _RawFeatureReference:
    structure_name: str
    member_names: tuple[str, ...]
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class _RawAvailabilityBlock:
    kind: VulkanAvailabilityBlockKind
    apis: frozenset[VulkanApi]
    dependencies: DependencyExpression | None
    feature_references: tuple[_RawFeatureReference, ...]
    type_references: tuple[VulkanTypeReference, ...]
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class _RawCoreFeature:
    name: str
    version: VulkanVersion
    apis: frozenset[VulkanApi]
    internal: bool
    dependencies: DependencyExpression | None
    blocks: tuple[_RawAvailabilityBlock, ...]
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class _RawExtension:
    name: str
    number: int
    extension_type: VulkanExtensionType | None
    supported_apis: frozenset[VulkanApi]
    dependencies: DependencyExpression | None
    promoted_to: str | None
    deprecated_by: str | None
    obsoleted_by: str | None
    platform: str | None
    provisional: bool
    blocks: tuple[_RawAvailabilityBlock, ...]
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class _RawSpirvFeatureEnable:
    structure_name: str
    member_name: str
    profile_member_alias: str | None
    requirements: DependencyExpression
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class _RawSpirvPropertyEnable:
    structure_name: str
    member_name: str
    constant_name: str
    requirements: DependencyExpression
    origin: RegistryOrigin


_RawSpirvEnable: TypeAlias = (
    VulkanSpirvCoreEnable
    | VulkanSpirvExtensionEnable
    | _RawSpirvFeatureEnable
    | _RawSpirvPropertyEnable
)


@dataclass(frozen=True, slots=True)
class _RawSpirvAvailability:
    name: str
    enables: tuple[_RawSpirvEnable, ...]
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class _EnumRow:
    element: ET.Element
    origin: RegistryOrigin


def _named_origin(origin: RegistryOrigin, name: str) -> RegistryOrigin:
    return RegistryOrigin(origin.source, f"{origin.locator}({name})")


def _expect_attributes(
    element: ET.Element,
    origin: RegistryOrigin,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    fields = frozenset(element.attrib)
    missing = sorted(required - fields)
    unknown = sorted(fields - required - optional)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing attributes: {', '.join(missing)}")
        if unknown:
            parts.append(f"unknown attributes: {', '.join(unknown)}")
        raise RegistrySourceError(origin, "; ".join(parts))


def _expect_leaf(element: ET.Element, origin: RegistryOrigin) -> None:
    if len(element):
        raise RegistrySourceError(origin, "expected an element with no children")
    if element.text is not None and element.text.strip():
        raise RegistrySourceError(origin, "expected an element with no text")


def _expect_identifier(value: str, origin: RegistryOrigin) -> str:
    if _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise RegistrySourceError(origin, f"invalid identifier {value!r}")
    return value


def _required_identifier(
    element: ET.Element, attribute: str, origin: RegistryOrigin
) -> str:
    value = element.attrib[attribute]
    if _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise RegistrySourceError(
            origin.child(f"@{attribute}"), f"invalid identifier {value!r}"
        )
    return value


def _optional_identifier(
    element: ET.Element, attribute: str, origin: RegistryOrigin
) -> str | None:
    value = element.get(attribute)
    if value is None:
        return None
    if _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise RegistrySourceError(
            origin.child(f"@{attribute}"), f"invalid identifier {value!r}"
        )
    return value


def _optional_text(
    element: ET.Element,
    attribute: str,
    origin: RegistryOrigin,
    *,
    allow_empty: bool = False,
) -> str | None:
    value = element.get(attribute)
    if value is None:
        return None
    if not value and not allow_empty:
        raise RegistrySourceError(
            origin.child(f"@{attribute}"), "expected a non-empty string"
        )
    return value


def _parse_positive_integer(value: str, origin: RegistryOrigin) -> int:
    if not value.isascii() or not value.isdecimal():
        raise RegistrySourceError(origin, "expected a decimal integer")
    result = int(value)
    if result <= 0:
        raise RegistrySourceError(origin, "expected a positive integer")
    return result


def _parse_nonnegative_integer(value: str, origin: RegistryOrigin) -> int:
    if not value.isascii() or not value.isdecimal():
        raise RegistrySourceError(origin, "expected a decimal integer")
    return int(value)


def _parse_version(value: str, origin: RegistryOrigin) -> VulkanVersion:
    match = _VERSION_PATTERN.fullmatch(value)
    if match is None:
        raise RegistrySourceError(origin, "expected a major.minor Vulkan version")
    return VulkanVersion(int(match.group(1)), int(match.group(2)))


def _parse_api_set(
    value: str,
    origin: RegistryOrigin,
    *,
    allow_disabled: bool = False,
) -> frozenset[VulkanApi]:
    if allow_disabled and value == "disabled":
        return frozenset()
    names = value.split(",")
    if not names or any(not name for name in names):
        raise RegistrySourceError(origin, "expected a comma-separated API set")
    if len(names) != len(set(names)):
        raise RegistrySourceError(origin, "API names must not be repeated")
    result = set()
    for name in names:
        try:
            result.add(VulkanApi(name))
        except ValueError as exception:
            raise RegistrySourceError(
                origin, f"unsupported Vulkan API {name!r}"
            ) from exception
    return frozenset(result)


def _parse_dependency(
    value: str,
    origin: RegistryOrigin,
    cache: dict[str, DependencyExpression],
) -> DependencyExpression:
    existing = cache.get(value)
    if existing is not None:
        return existing
    try:
        result = parse_dependency_expression(value, source=str(origin))
    except DependencyExpressionParseError as exception:
        raise RegistrySourceError(
            origin,
            f"invalid dependency expression at column {exception.column}: "
            f"{exception.reason}",
        ) from exception
    cache[value] = result
    return result


def _parse_optional_dependency(
    value: str | None,
    origin: RegistryOrigin,
    cache: dict[str, DependencyExpression],
) -> DependencyExpression | None:
    if value is None:
        return None
    return _parse_dependency(value, origin, cache)


def _parse_member_names(value: str, origin: RegistryOrigin) -> tuple[str, ...]:
    names = tuple(value.split(","))
    if not names:
        raise RegistrySourceError(origin, "expected at least one feature member")
    for name in names:
        _expect_identifier(name, origin)
    if len(names) != len(set(names)):
        raise RegistrySourceError(origin, "feature members must not be repeated")
    return names


def _index_top_level_sections(
    root: ET.Element, root_origin: RegistryOrigin
) -> Mapping[str, tuple[ET.Element, ...]]:
    if root.tag != "registry":
        raise RegistrySourceError(root_origin, "expected a registry root element")
    _expect_attributes(root, root_origin, required=frozenset())

    sections: dict[str, list[ET.Element]] = {}
    for index, element in enumerate(root):
        if element.tag not in _TOP_LEVEL_SECTIONS:
            raise RegistrySourceError(
                root_origin.child(f"[{index}]"),
                f"unknown top-level section {element.tag!r}",
            )
        sections.setdefault(element.tag, []).append(element)

    for name in _SINGLETON_TOP_LEVEL_SECTIONS:
        count = len(sections.get(name, ()))
        if count > 1:
            raise RegistrySourceError(
                root_origin,
                f"top-level section {name!r} appears {count} times",
            )
    missing = sorted(
        name for name in _REQUIRED_TOP_LEVEL_SECTIONS if name not in sections
    )
    if missing:
        raise RegistrySourceError(
            root_origin,
            f"missing top-level sections: {', '.join(missing)}",
        )
    return MappingProxyType(
        {name: tuple(elements) for name, elements in sections.items()}
    )


def _extract_type_name(element: ET.Element, origin: RegistryOrigin) -> str | None:
    attribute_name = element.get("name")
    if attribute_name is not None:
        if not attribute_name:
            raise RegistrySourceError(origin.child("@name"), "type name is empty")
        return attribute_name

    name_element = element.find("name")
    if name_element is None:
        name_element = element.find("./proto/name")
    if name_element is not None:
        if name_element.attrib or len(name_element):
            raise RegistrySourceError(
                origin.child("name"), "expected a plain type name element"
            )
        if name_element.text is None:
            raise RegistrySourceError(origin.child("name"), "type name is empty")
        child_name = name_element.text.strip()
        _expect_identifier(child_name, origin.child("name"))
        return child_name
    return None


def _scan_types(
    container: ET.Element, origin: RegistryOrigin
) -> tuple[tuple[_RawType, ...], Mapping[str, tuple[_RawType, ...]]]:
    _expect_attributes(
        container,
        origin,
        required=frozenset(("comment",)),
    )
    raw_types = []
    by_name_lists: dict[str, list[_RawType]] = {}
    type_index = 0
    for child in container:
        if child.tag == "comment":
            _expect_attributes(child, origin.child("comment"), required=frozenset())
            if len(child):
                raise RegistrySourceError(
                    origin.child("comment"), "type comments must not have children"
                )
            continue
        if child.tag != "type":
            raise RegistrySourceError(
                origin.child(f"[{type_index}]"),
                f"unknown types child {child.tag!r}",
            )
        provisional_origin = origin.child(f"type[{type_index}]")
        type_index += 1
        name = _extract_type_name(child, provisional_origin)
        if name is None:
            continue
        type_origin = _named_origin(provisional_origin, name)
        category = child.get("category")
        if category is not None and not category:
            raise RegistrySourceError(
                type_origin.child("@category"), "type category is empty"
            )
        alias = _optional_identifier(child, "alias", type_origin)
        raw_type = _RawType(name, category, alias, child, type_origin)
        by_name_lists.setdefault(name, []).append(raw_type)
        raw_types.append(raw_type)
    return tuple(raw_types), MappingProxyType(
        {name: tuple(values) for name, values in by_name_lists.items()}
    )


def _scan_platform_names(
    sections: Mapping[str, tuple[ET.Element, ...]], root_origin: RegistryOrigin
) -> frozenset[str]:
    platform_sections = sections.get("platforms", ())
    if not platform_sections:
        return frozenset()
    container = platform_sections[0]
    origin = root_origin.child("platforms")
    _expect_attributes(
        container,
        origin,
        required=frozenset(("comment",)),
    )
    names = set()
    for index, element in enumerate(container):
        element_origin = origin.child(f"platform[{index}]")
        if element.tag != "platform":
            raise RegistrySourceError(
                element_origin, f"unknown platforms child {element.tag!r}"
            )
        _expect_attributes(
            element,
            element_origin,
            required=frozenset(("comment", "name", "protect")),
        )
        _expect_leaf(element, element_origin)
        name = _required_identifier(element, "name", element_origin)
        if name in names:
            raise RegistrySourceError(
                element_origin, f"platform name {name!r} is repeated"
            )
        names.add(name)
    return frozenset(names)


def _validate_ignored_block_reference(
    element: ET.Element, origin: RegistryOrigin
) -> None:
    _expect_attributes(
        element,
        origin,
        required=frozenset(("name",)),
        optional=frozenset(("comment", "supersededby")),
    )
    _expect_leaf(element, origin)
    _required_identifier(element, "name", origin)
    _optional_identifier(element, "supersededby", origin)


def _parse_block(
    element: ET.Element,
    origin: RegistryOrigin,
    raw_types_by_name: Mapping[str, tuple[_RawType, ...]],
    dependency_cache: dict[str, DependencyExpression],
) -> _RawAvailabilityBlock:
    try:
        kind = VulkanAvailabilityBlockKind(element.tag)
    except ValueError as exception:
        raise RegistrySourceError(
            origin, f"unsupported availability block {element.tag!r}"
        ) from exception
    _expect_attributes(
        element,
        origin,
        required=frozenset(),
        optional=frozenset(
            ("api", "comment", "depends", "explanationlink", "reasonlink")
        ),
    )
    api_text = element.get("api")
    apis = (
        _parse_api_set(api_text, origin.child("@api"))
        if api_text is not None
        else frozenset()
    )
    dependencies = _parse_optional_dependency(
        element.get("depends"), origin.child("@depends"), dependency_cache
    )
    feature_references = []
    type_references = []
    child_counts: dict[str, int] = {}
    for child in element:
        child_index = child_counts.get(child.tag, 0)
        child_counts[child.tag] = child_index + 1
        child_locator = f"{child.tag}[{child_index}]"
        if child.tag == "enum":
            # General enum values are outside this compiler-facing model. Keep
            # their block-reference schema strict while deferring value/alias
            # interpretation until a SPIR-V property predicate names one.
            name = child.get("name")
            if (
                name is None
                or not child.attrib.keys() <= _BLOCK_ENUM_ATTRIBUTES
                or _IDENTIFIER_PATTERN.fullmatch(name) is None
                or len(child)
                or (child.text is not None and child.text.strip())
            ):
                child_origin = origin.child(child_locator)
                if name is None:
                    raise RegistrySourceError(
                        child_origin, "enum reference is missing its name"
                    )
                unknown_attributes = sorted(
                    child.attrib.keys() - _BLOCK_ENUM_ATTRIBUTES
                )
                if unknown_attributes:
                    raise RegistrySourceError(
                        child_origin,
                        "unknown enum attributes: " + ", ".join(unknown_attributes),
                    )
                _expect_identifier(name, child_origin.child("@name"))
                _expect_leaf(child, child_origin)
            continue
        if child.tag == "comment":
            if child.attrib or len(child):
                child_origin = origin.child(child_locator)
                _expect_attributes(child, child_origin, required=frozenset())
                if len(child):
                    raise RegistrySourceError(
                        child_origin, "availability comments must not have children"
                    )
            continue
        child_origin = origin.child(child_locator)
        if child.tag == "feature":
            _expect_attributes(
                child,
                child_origin,
                required=frozenset(("name", "struct")),
            )
            _expect_leaf(child, child_origin)
            structure_name = _required_identifier(child, "struct", child_origin)
            member_names = _parse_member_names(
                child.attrib["name"], child_origin.child("@name")
            )
            feature_references.append(
                _RawFeatureReference(
                    structure_name,
                    member_names,
                    _named_origin(child_origin, structure_name),
                )
            )
        elif child.tag == "type":
            _validate_ignored_block_reference(child, child_origin)
            name = child.attrib["name"]
            if name not in raw_types_by_name:
                raise RegistrySourceError(
                    child_origin.child("@name"), f"unknown Vulkan type {name!r}"
                )
            superseded_by = child.get("supersededby")
            if superseded_by is not None and superseded_by not in raw_types_by_name:
                raise RegistrySourceError(
                    child_origin.child("@supersededby"),
                    f"unknown Vulkan type {superseded_by!r}",
                )
            type_references.append(
                VulkanTypeReference(
                    name, superseded_by, _named_origin(child_origin, name)
                )
            )
        elif child.tag == "command":
            _validate_ignored_block_reference(child, child_origin)
        else:
            raise RegistrySourceError(
                child_origin, f"unknown availability child {child.tag!r}"
            )
    return _RawAvailabilityBlock(
        kind,
        apis,
        dependencies,
        tuple(feature_references),
        tuple(type_references),
        origin,
    )


def _parse_core_features(
    elements: Sequence[ET.Element],
    root_origin: RegistryOrigin,
    raw_types_by_name: Mapping[str, tuple[_RawType, ...]],
    dependency_cache: dict[str, DependencyExpression],
) -> tuple[_RawCoreFeature, ...]:
    result = []
    names: dict[str, RegistryOrigin] = {}
    for index, element in enumerate(elements):
        origin = root_origin.child(f"feature[{index}]")
        _expect_attributes(
            element,
            origin,
            required=frozenset(("api", "comment", "name", "number")),
            optional=frozenset(("apitype", "depends")),
        )
        name = _required_identifier(element, "name", origin)
        origin = _named_origin(origin, name)
        existing = names.get(name)
        if existing is not None:
            raise RegistrySourceError(
                origin, f"core feature name {name!r} collides with {existing}"
            )
        names[name] = origin
        api_type = element.get("apitype")
        if api_type not in (None, "internal"):
            raise RegistrySourceError(
                origin.child("@apitype"), f"unsupported API type {api_type!r}"
            )
        blocks = []
        block_counts: dict[str, int] = {}
        for child in element:
            if child.tag not in _AVAILABILITY_BLOCK_TAGS:
                raise RegistrySourceError(
                    origin, f"unknown core feature child {child.tag!r}"
                )
            block_index = block_counts.get(child.tag, 0)
            block_counts[child.tag] = block_index + 1
            blocks.append(
                _parse_block(
                    child,
                    origin.child(f"{child.tag}[{block_index}]"),
                    raw_types_by_name,
                    dependency_cache,
                )
            )
        result.append(
            _RawCoreFeature(
                name,
                _parse_version(element.attrib["number"], origin.child("@number")),
                _parse_api_set(element.attrib["api"], origin.child("@api")),
                api_type == "internal",
                _parse_optional_dependency(
                    element.get("depends"),
                    origin.child("@depends"),
                    dependency_cache,
                ),
                tuple(blocks),
                origin,
            )
        )
    return tuple(result)


def _parse_extensions(
    container: ET.Element,
    root_origin: RegistryOrigin,
    raw_types_by_name: Mapping[str, tuple[_RawType, ...]],
    platform_names: frozenset[str],
    dependency_cache: dict[str, DependencyExpression],
) -> tuple[_RawExtension, ...]:
    origin = root_origin.child("extensions")
    _expect_attributes(
        container,
        origin,
        required=frozenset(("comment",)),
    )
    result = []
    names: dict[str, RegistryOrigin] = {}
    numbers: dict[int, RegistryOrigin] = {}
    for index, element in enumerate(container):
        extension_origin = origin.child(f"extension[{index}]")
        if element.tag != "extension":
            raise RegistrySourceError(
                extension_origin, f"unknown extensions child {element.tag!r}"
            )
        _expect_attributes(
            element,
            extension_origin,
            required=frozenset(("name", "number", "supported")),
            optional=frozenset(
                (
                    "author",
                    "comment",
                    "contact",
                    "depends",
                    "deprecatedby",
                    "nofeatures",
                    "obsoletedby",
                    "platform",
                    "promotedto",
                    "provisional",
                    "ratified",
                    "sortorder",
                    "specialuse",
                    "type",
                )
            ),
        )
        name = _required_identifier(element, "name", extension_origin)
        extension_origin = _named_origin(extension_origin, name)
        existing_name = names.get(name)
        if existing_name is not None:
            raise RegistrySourceError(
                extension_origin,
                f"extension name {name!r} collides with {existing_name}",
            )
        names[name] = extension_origin
        number = _parse_positive_integer(
            element.attrib["number"], extension_origin.child("@number")
        )
        existing_number = numbers.get(number)
        if existing_number is not None:
            raise RegistrySourceError(
                extension_origin.child("@number"),
                f"extension number {number} collides with {existing_number}",
            )
        numbers[number] = extension_origin

        supported_apis = _parse_api_set(
            element.attrib["supported"],
            extension_origin.child("@supported"),
            allow_disabled=True,
        )
        if VulkanApi.VULKAN_BASE in supported_apis:
            raise RegistrySourceError(
                extension_origin.child("@supported"),
                "extensions cannot target the internal vulkanbase API",
            )
        extension_type_text = element.get("type")
        extension_type = None
        if extension_type_text is not None:
            try:
                extension_type = VulkanExtensionType(extension_type_text)
            except ValueError as exception:
                raise RegistrySourceError(
                    extension_origin.child("@type"),
                    f"unsupported extension type {extension_type_text!r}",
                ) from exception
        if supported_apis and extension_type is None:
            raise RegistrySourceError(
                extension_origin,
                "a supported extension requires an instance or device type",
            )
        for attribute in ("author", "comment", "contact", "specialuse"):
            _optional_text(element, attribute, extension_origin)
        no_features = element.get("nofeatures")
        if no_features not in (None, "true"):
            raise RegistrySourceError(
                extension_origin.child("@nofeatures"),
                "nofeatures must be 'true' when present",
            )
        provisional_text = element.get("provisional")
        if provisional_text not in (None, "true"):
            raise RegistrySourceError(
                extension_origin.child("@provisional"),
                "provisional must be 'true' when present",
            )
        ratified = element.get("ratified")
        if ratified is not None:
            ratified_apis = _parse_api_set(
                ratified, extension_origin.child("@ratified")
            )
            if VulkanApi.VULKAN_BASE in ratified_apis:
                raise RegistrySourceError(
                    extension_origin.child("@ratified"),
                    "extensions cannot be ratified for vulkanbase",
                )
        sort_order = element.get("sortorder")
        if sort_order is not None:
            _parse_positive_integer(sort_order, extension_origin.child("@sortorder"))

        promoted_to = _optional_identifier(element, "promotedto", extension_origin)
        deprecated_by = _optional_text(
            element, "deprecatedby", extension_origin, allow_empty=True
        )
        if deprecated_by:
            _expect_identifier(deprecated_by, extension_origin.child("@deprecatedby"))
        obsoleted_by = _optional_identifier(element, "obsoletedby", extension_origin)
        platform = _optional_identifier(element, "platform", extension_origin)
        if platform is not None and platform not in platform_names:
            raise RegistrySourceError(
                extension_origin.child("@platform"),
                f"unknown Vulkan platform {platform!r}",
            )

        blocks = []
        block_counts: dict[str, int] = {}
        for child in element:
            if child.tag not in _AVAILABILITY_BLOCK_TAGS:
                raise RegistrySourceError(
                    extension_origin, f"unknown extension child {child.tag!r}"
                )
            block_index = block_counts.get(child.tag, 0)
            block_counts[child.tag] = block_index + 1
            blocks.append(
                _parse_block(
                    child,
                    extension_origin.child(f"{child.tag}[{block_index}]"),
                    raw_types_by_name,
                    dependency_cache,
                )
            )
        result.append(
            _RawExtension(
                name,
                number,
                extension_type,
                supported_apis,
                _parse_optional_dependency(
                    element.get("depends"),
                    extension_origin.child("@depends"),
                    dependency_cache,
                ),
                promoted_to,
                deprecated_by,
                obsoleted_by,
                platform,
                provisional_text == "true",
                tuple(blocks),
                extension_origin,
            )
        )
    return tuple(result)


def _parse_spirv_enable(
    element: ET.Element,
    origin: RegistryOrigin,
    dependency_cache: dict[str, DependencyExpression],
) -> _RawSpirvEnable:
    if element.tag != "enable":
        raise RegistrySourceError(
            origin, f"expected an enable element, got {element.tag!r}"
        )
    _expect_leaf(element, origin)
    fields = frozenset(element.attrib)
    if fields == frozenset(("version",)):
        return VulkanSpirvCoreEnable(
            _required_identifier(element, "version", origin), origin
        )
    if fields == frozenset(("extension",)):
        return VulkanSpirvExtensionEnable(
            _required_identifier(element, "extension", origin), origin
        )
    if fields in (
        frozenset(("feature", "requires", "struct")),
        frozenset(("alias", "feature", "requires", "struct")),
    ):
        return _RawSpirvFeatureEnable(
            _required_identifier(element, "struct", origin),
            _required_identifier(element, "feature", origin),
            _optional_identifier(element, "alias", origin),
            _parse_dependency(
                element.attrib["requires"],
                origin.child("@requires"),
                dependency_cache,
            ),
            origin,
        )
    if fields == frozenset(("member", "property", "requires", "value")):
        return _RawSpirvPropertyEnable(
            _required_identifier(element, "property", origin),
            _required_identifier(element, "member", origin),
            _required_identifier(element, "value", origin),
            _parse_dependency(
                element.attrib["requires"],
                origin.child("@requires"),
                dependency_cache,
            ),
            origin,
        )
    raise RegistrySourceError(
        origin,
        "unsupported SPIR-V enable attributes: "
        + (", ".join(sorted(fields)) if fields else "none"),
    )


def _parse_spirv_availabilities(
    container: ET.Element,
    root_origin: RegistryOrigin,
    *,
    container_name: str,
    entry_name: str,
    dependency_cache: dict[str, DependencyExpression],
) -> tuple[_RawSpirvAvailability, ...]:
    origin = root_origin.child(container_name)
    _expect_attributes(
        container,
        origin,
        required=frozenset(("comment",)),
    )
    result = []
    names: dict[str, RegistryOrigin] = {}
    for index, element in enumerate(container):
        entry_origin = origin.child(f"{entry_name}[{index}]")
        if element.tag != entry_name:
            raise RegistrySourceError(
                entry_origin, f"unknown {container_name} child {element.tag!r}"
            )
        _expect_attributes(
            element,
            entry_origin,
            required=frozenset(("name",)),
        )
        name = _required_identifier(element, "name", entry_origin)
        entry_origin = _named_origin(entry_origin, name)
        existing = names.get(name)
        if existing is not None:
            raise RegistrySourceError(
                entry_origin, f"SPIR-V name {name!r} collides with {existing}"
            )
        names[name] = entry_origin
        enables = tuple(
            _parse_spirv_enable(
                enable,
                entry_origin.child(f"enable[{enable_index}]"),
                dependency_cache,
            )
            for enable_index, enable in enumerate(element)
        )
        if not enables:
            raise RegistrySourceError(
                entry_origin, "SPIR-V availability requires at least one alternative"
            )
        result.append(_RawSpirvAvailability(name, enables, entry_origin))
    return tuple(result)


def _index_structure_types(
    raw_types: Sequence[_RawType],
) -> tuple[
    Mapping[str, tuple[_RawType, ...]],
    Mapping[str, tuple[_RawType, ...]],
]:
    types_by_name: dict[str, list[_RawType]] = {}
    aliases_by_target: dict[str, list[_RawType]] = {}
    for raw_type in raw_types:
        if raw_type.category != "struct":
            continue
        types_by_name.setdefault(raw_type.name, []).append(raw_type)
        if raw_type.alias is not None:
            aliases_by_target.setdefault(raw_type.alias, []).append(raw_type)
    return (
        MappingProxyType(
            {name: tuple(raw_types) for name, raw_types in types_by_name.items()}
        ),
        MappingProxyType(
            {name: tuple(raw_types) for name, raw_types in aliases_by_target.items()}
        ),
    )


def _record_dependency_structure_requests(
    expression: DependencyExpression | None,
    origin: RegistryOrigin,
    requests: dict[str, RegistryOrigin],
) -> None:
    if expression is None:
        return
    for atom in dependency_expression_names(expression):
        if "::" not in atom:
            continue
        parts = atom.split("::")
        if len(parts) != 2:
            raise RegistrySourceError(
                origin, f"invalid structure/member dependency atom {atom!r}"
            )
        structure_name, member_name = parts
        _expect_identifier(structure_name, origin)
        _expect_identifier(member_name, origin)
        requests.setdefault(structure_name, origin)


def _collect_structure_requests(
    core_features: Sequence[_RawCoreFeature],
    extensions: Sequence[_RawExtension],
    spirv_extensions: Sequence[_RawSpirvAvailability],
    spirv_capabilities: Sequence[_RawSpirvAvailability],
) -> Mapping[str, RegistryOrigin]:
    requests: dict[str, RegistryOrigin] = {}
    for owner in (*core_features, *extensions):
        _record_dependency_structure_requests(
            owner.dependencies, owner.origin.child("@depends"), requests
        )
        for block in owner.blocks:
            _record_dependency_structure_requests(
                block.dependencies, block.origin.child("@depends"), requests
            )
            for feature in block.feature_references:
                requests.setdefault(feature.structure_name, feature.origin)
    for availability in (*spirv_extensions, *spirv_capabilities):
        for enable in availability.enables:
            if isinstance(enable, (_RawSpirvFeatureEnable, _RawSpirvPropertyEnable)):
                requests.setdefault(enable.structure_name, enable.origin)
    return MappingProxyType(requests)


def _simple_child_identifier(
    element: ET.Element, tag: str, origin: RegistryOrigin
) -> str:
    children = [child for child in element if child.tag == tag]
    if len(children) != 1:
        raise RegistrySourceError(
            origin, f"expected exactly one {tag!r} child, found {len(children)}"
        )
    child = children[0]
    if child.attrib or len(child):
        raise RegistrySourceError(
            origin.child(tag), f"expected a plain {tag!r} element"
        )
    if child.text is None:
        raise RegistrySourceError(origin.child(tag), f"{tag} is empty")
    return _expect_identifier(child.text.strip(), origin.child(tag))


def _parse_structure_member(
    element: ET.Element, origin: RegistryOrigin
) -> VulkanStructureMember:
    _expect_attributes(
        element,
        origin,
        required=frozenset(),
        optional=frozenset(
            (
                "featurelink",
                "len",
                "limittype",
                "noautovalidity",
                "optional",
                "values",
            )
        ),
    )
    allowed_children = frozenset(("comment", "enum", "name", "type"))
    for child in element:
        if child.tag not in allowed_children:
            raise RegistrySourceError(
                origin, f"unknown structure member child {child.tag!r}"
            )
        if child.tag == "name":
            if len(child):
                raise RegistrySourceError(
                    origin.child(child.tag), "member names must not have children"
                )
            continue
        if child.attrib or len(child):
            raise RegistrySourceError(
                origin.child(child.tag),
                f"expected a plain {child.tag!r} element",
            )
    _simple_child_identifier(element, "type", origin)
    name_children = [child for child in element if child.tag == "name"]
    if len(name_children) != 1:
        raise RegistrySourceError(
            origin,
            f"expected exactly one 'name' child, found {len(name_children)}",
        )
    name_element = name_children[0]
    _expect_attributes(
        name_element,
        origin.child("name"),
        required=frozenset(),
        optional=frozenset(("alias",)),
    )
    if len(name_element) or name_element.text is None:
        raise RegistrySourceError(origin.child("name"), "expected a plain member name")
    name = _expect_identifier(name_element.text.strip(), origin.child("name"))
    member_origin = _named_origin(origin, name)
    availability_alias = name_element.get("alias")
    if availability_alias is not None:
        if "::" in availability_alias:
            parts = availability_alias.split("::")
            if len(parts) != 2:
                raise RegistrySourceError(
                    member_origin.child("name@alias"),
                    f"invalid member alias {availability_alias!r}",
                )
            for part in parts:
                _expect_identifier(part, member_origin.child("name@alias"))
        else:
            _expect_identifier(availability_alias, member_origin.child("name@alias"))
    values = element.get("values")
    if values is not None:
        _expect_identifier(values, member_origin.child("@values"))
        if name != "sType":
            raise RegistrySourceError(
                member_origin.child("@values"),
                "only an sType member may carry a structure identity",
            )
    for attribute in ("noautovalidity", "optional"):
        value = element.get(attribute)
        if value not in (None, "true"):
            raise RegistrySourceError(
                member_origin.child(f"@{attribute}"),
                f"{attribute} must be 'true' when present",
            )
    return VulkanStructureMember(name, availability_alias, member_origin)


def _parse_structure(raw_type: _RawType, aliases: tuple[str, ...]) -> VulkanStructure:
    element = raw_type.element
    origin = raw_type.origin
    _expect_attributes(
        element,
        origin,
        required=frozenset(("category", "name")),
        optional=frozenset(
            (
                "allowduplicate",
                "comment",
                "requiredlimittype",
                "returnedonly",
                "structextends",
            )
        ),
    )
    if element.attrib["category"] != "struct":
        raise RegistrySourceError(origin, "availability type must be a structure")
    if element.attrib["name"] != raw_type.name:
        raise RegistrySourceError(origin, "structure name changed during type scan")
    members = []
    members_by_name: dict[str, VulkanStructureMember] = {}
    structure_type = None
    member_index = 0
    for child in element:
        if child.tag == "comment":
            _expect_attributes(child, origin.child("comment"), required=frozenset())
            if len(child):
                raise RegistrySourceError(
                    origin.child("comment"), "structure comments must not have children"
                )
            continue
        if child.tag != "member":
            raise RegistrySourceError(origin, f"unknown structure child {child.tag!r}")
        member = _parse_structure_member(child, origin.child(f"member[{member_index}]"))
        member_index += 1
        existing = members_by_name.get(member.name)
        if existing is not None:
            raise RegistrySourceError(
                member.origin,
                f"member name {member.name!r} collides with {existing.origin}",
            )
        members.append(member)
        members_by_name[member.name] = member
        if member.name == "sType":
            structure_type = child.get("values")
    if not members:
        raise RegistrySourceError(origin, "availability structure has no members")
    structure_type_member = members_by_name.get("sType")
    if structure_type_member is not None:
        if structure_type is None:
            raise RegistrySourceError(
                structure_type_member.origin,
                "sType member has no symbolic structure identity",
            )
    elif raw_type.name != "VkPhysicalDeviceFeatures":
        raise RegistrySourceError(
            origin,
            "availability structure has no sType member",
        )
    return VulkanStructure(
        raw_type.name,
        aliases,
        tuple(members),
        structure_type,
        origin,
        MappingProxyType(members_by_name),
    )


def _build_structure_model(
    raw_types: Sequence[_RawType],
    requests: Mapping[str, RegistryOrigin],
) -> tuple[
    tuple[VulkanStructure, ...],
    Mapping[str, VulkanStructure],
    Mapping[str, VulkanStructureMember],
]:
    structure_types_by_name, structure_aliases_by_target = _index_structure_types(
        raw_types
    )
    resolved_types: dict[str, _RawType] = {}
    active_aliases: list[str] = []

    def resolve_type(name: str, request_origin: RegistryOrigin) -> _RawType:
        existing = resolved_types.get(name)
        if existing is not None:
            return existing
        definitions = structure_types_by_name.get(name, ())
        if not definitions:
            raise RegistrySourceError(
                request_origin, f"unknown Vulkan structure {name!r}"
            )
        if len(definitions) != 1:
            raise RegistrySourceError(
                request_origin,
                f"Vulkan structure {name!r} has {len(definitions)} definitions",
            )
        raw_type = definitions[0]
        if name in active_aliases:
            cycle = active_aliases[active_aliases.index(name) :] + [name]
            raise RegistrySourceError(
                raw_type.origin.child("@alias"),
                "type alias cycle: " + " -> ".join(cycle),
            )
        if raw_type.alias is None:
            resolved_types[name] = raw_type
            return raw_type
        active_aliases.append(name)
        canonical = resolve_type(raw_type.alias, raw_type.origin.child("@alias"))
        active_aliases.pop()
        resolved_types[name] = canonical
        return canonical

    canonical_names = set()
    pending_names = []
    for name, request_origin in requests.items():
        canonical = resolve_type(name, request_origin)
        if canonical.name not in canonical_names:
            canonical_names.add(canonical.name)
            pending_names.append(canonical.name)

    parsed_structures_by_name: dict[str, VulkanStructure] = {}
    while pending_names:
        canonical_name = pending_names.pop()
        raw_type = resolved_types[canonical_name]
        structure = _parse_structure(raw_type, ())
        parsed_structures_by_name[canonical_name] = structure
        for member in structure.members:
            availability_alias = member.availability_alias
            if availability_alias is None or "::" not in availability_alias:
                continue
            target_name, _, _ = availability_alias.partition("::")
            target = resolve_type(target_name, member.origin.child("name@alias"))
            if target.name not in canonical_names:
                canonical_names.add(target.name)
                pending_names.append(target.name)

    selected_alias_names = set()
    pending_target_names = list(parsed_structures_by_name)
    while pending_target_names:
        target_name = pending_target_names.pop()
        for raw_alias in structure_aliases_by_target.get(target_name, ()):
            resolve_type(raw_alias.name, raw_alias.origin)
            if raw_alias.name not in selected_alias_names:
                selected_alias_names.add(raw_alias.name)
                pending_target_names.append(raw_alias.name)

    aliases_by_canonical: dict[str, list[str]] = {name: [] for name in canonical_names}
    for raw_type in raw_types:
        if raw_type.name not in selected_alias_names:
            continue
        canonical = resolve_type(raw_type.name, raw_type.origin)
        _expect_attributes(
            raw_type.element,
            raw_type.origin,
            required=frozenset(("alias", "category", "name")),
        )
        _expect_leaf(raw_type.element, raw_type.origin)
        aliases_by_canonical[canonical.name].append(raw_type.name)

    structures = tuple(
        replace(
            parsed_structures_by_name[raw_type.name],
            aliases=tuple(aliases_by_canonical[raw_type.name]),
        )
        for raw_type in raw_types
        if raw_type.name in canonical_names
    )
    structures_by_name: dict[str, VulkanStructure] = {}
    members_by_name: dict[str, VulkanStructureMember] = {}
    for structure in structures:
        for name in (structure.name, *structure.aliases):
            existing = structures_by_name.get(name)
            if existing is not None:
                raise RegistrySourceError(
                    structure.origin,
                    f"structure name {name!r} collides with {existing.origin}",
                )
            structures_by_name[name] = structure
            for member in structure.members:
                members_by_name[f"{name}::{member.name}"] = member
    return (
        structures,
        MappingProxyType(structures_by_name),
        MappingProxyType(members_by_name),
    )


def _scan_enum_elements(
    root: ET.Element,
) -> Mapping[str, tuple[ET.Element, ...]]:
    rows_by_name: dict[str, list[ET.Element]] = {}
    for element in root.iter("enum"):
        name = element.get("name")
        if name is not None:
            rows_by_name.setdefault(name, []).append(element)
    return MappingProxyType(
        {name: tuple(elements) for name, elements in rows_by_name.items()}
    )


def _constant_definition(
    name: str,
    elements_by_name: Mapping[str, tuple[ET.Element, ...]],
    request_origin: RegistryOrigin,
) -> _EnumRow:
    definitions = tuple(
        element
        for element in elements_by_name.get(name, ())
        if any(
            attribute in element.attrib
            for attribute in ("alias", "bitpos", "offset", "value")
        )
    )
    if not definitions:
        raise RegistrySourceError(
            request_origin, f"Vulkan constant {name!r} has no definition"
        )
    if len(definitions) != 1:
        raise RegistrySourceError(
            request_origin,
            f"Vulkan constant {name!r} has {len(definitions)} definitions",
        )
    return _EnumRow(
        definitions[0],
        _named_origin(RegistryOrigin(request_origin.source, "$.enum"), name),
    )


def _validate_property_constant_definition(row: _EnumRow) -> None:
    element = row.element
    _expect_attributes(
        element,
        row.origin,
        required=frozenset(("name",)),
        optional=frozenset(("alias", "bitpos", "comment", "extends", "type", "value")),
    )
    _expect_leaf(element, row.origin)
    payloads = [name for name in ("alias", "bitpos", "value") if name in element.attrib]
    if len(payloads) != 1:
        raise RegistrySourceError(
            row.origin,
            "a property constant requires exactly one alias, bitpos, or value payload",
        )
    _required_identifier(element, "name", row.origin)
    _optional_identifier(element, "alias", row.origin)
    _optional_identifier(element, "extends", row.origin)


def _parse_property_constant_value(row: _EnumRow) -> int:
    element = row.element
    bit_position = element.get("bitpos")
    if bit_position is not None:
        position = _parse_nonnegative_integer(bit_position, row.origin.child("@bitpos"))
        if position >= 64:
            raise RegistrySourceError(
                row.origin.child("@bitpos"),
                "property constant bit positions must fit in 64 bits",
            )
        return 1 << position
    value = element.get("value")
    if value is None:
        raise RegistrySourceError(row.origin, "canonical property constant is an alias")
    try:
        return int(value, 0)
    except ValueError as exception:
        raise RegistrySourceError(
            row.origin.child("@value"),
            "property constant value must be an integer literal",
        ) from exception


def _build_constant_model(
    elements_by_name: Mapping[str, tuple[ET.Element, ...]],
    requests: Mapping[str, RegistryOrigin],
) -> tuple[tuple[VulkanConstant, ...], Mapping[str, VulkanConstant]]:
    definitions: dict[str, _EnumRow] = {}
    canonical_by_name: dict[str, str] = {}
    active: list[str] = []

    def resolve(name: str, request_origin: RegistryOrigin) -> str:
        existing = canonical_by_name.get(name)
        if existing is not None:
            return existing
        if name in active:
            cycle = active[active.index(name) :] + [name]
            raise RegistrySourceError(
                request_origin,
                "property constant alias cycle: " + " -> ".join(cycle),
            )
        row = _constant_definition(name, elements_by_name, request_origin)
        _validate_property_constant_definition(row)
        definitions[name] = row
        alias = row.element.get("alias")
        if alias is None:
            canonical_by_name[name] = name
            return name
        active.append(name)
        canonical = resolve(alias, row.origin.child("@alias"))
        active.pop()
        canonical_by_name[name] = canonical
        return canonical

    for name, origin in requests.items():
        resolve(name, origin)

    aliases_by_canonical: dict[str, list[str]] = {}
    for name, canonical_name in canonical_by_name.items():
        if name != canonical_name:
            aliases_by_canonical.setdefault(canonical_name, []).append(name)

    canonical_names = set(canonical_by_name.values())
    constants = []
    seen = set()
    for name in elements_by_name:
        if name not in canonical_names or name in seen:
            continue
        row = definitions[name]
        seen.add(name)
        constants.append(
            VulkanConstant(
                name,
                tuple(aliases_by_canonical.get(name, ())),
                _parse_property_constant_value(row),
                row.origin,
            )
        )
    if seen != canonical_names:
        missing = sorted(canonical_names - seen)
        raise RegistrySourceError(
            next(iter(requests.values())),
            f"property constant definitions were not retained: {', '.join(missing)}",
        )

    constants_by_name: dict[str, VulkanConstant] = {}
    for constant in constants:
        for name in (constant.name, *constant.aliases):
            existing = constants_by_name.get(name)
            if existing is not None:
                raise RegistrySourceError(
                    constant.origin,
                    f"constant name {name!r} collides with {existing.origin}",
                )
            constants_by_name[name] = constant
    return tuple(constants), MappingProxyType(constants_by_name)


def _validate_dependency_references(
    expression: DependencyExpression | None,
    origin: RegistryOrigin,
    core_feature_names: frozenset[str],
    extension_names: frozenset[str],
    structure_members_by_name: Mapping[str, VulkanStructureMember],
) -> None:
    if expression is None:
        return
    for name in dependency_expression_names(expression):
        if "::" in name:
            if name not in structure_members_by_name:
                raise RegistrySourceError(
                    origin, f"unknown Vulkan feature predicate {name!r}"
                )
        elif name not in core_feature_names and name not in extension_names:
            raise RegistrySourceError(origin, f"unknown Vulkan dependency {name!r}")


def _finalize_block(
    block: _RawAvailabilityBlock,
    structures_by_name: Mapping[str, VulkanStructure],
) -> VulkanAvailabilityBlock:
    feature_references = []
    for reference in block.feature_references:
        structure = structures_by_name[reference.structure_name]
        for member_name in reference.member_names:
            if member_name not in structure.members_by_name:
                raise RegistrySourceError(
                    reference.origin,
                    f"unknown member {reference.structure_name}::{member_name}",
                )
        feature_references.append(
            VulkanFeatureReference(
                reference.structure_name,
                structure.name,
                reference.member_names,
                reference.origin,
            )
        )
    return VulkanAvailabilityBlock(
        block.kind,
        block.apis,
        block.dependencies,
        tuple(feature_references),
        block.type_references,
        block.origin,
    )


def _finalize_core_features(
    raw_features: Sequence[_RawCoreFeature],
    structures_by_name: Mapping[str, VulkanStructure],
) -> tuple[tuple[VulkanCoreFeature, ...], Mapping[str, VulkanCoreFeature]]:
    features = tuple(
        VulkanCoreFeature(
            feature.name,
            feature.version,
            feature.apis,
            feature.internal,
            feature.dependencies,
            tuple(
                _finalize_block(block, structures_by_name) for block in feature.blocks
            ),
            feature.origin,
        )
        for feature in raw_features
    )
    return features, MappingProxyType({feature.name: feature for feature in features})


def _finalize_extensions(
    raw_extensions: Sequence[_RawExtension],
    structures_by_name: Mapping[str, VulkanStructure],
) -> tuple[
    tuple[VulkanExtension, ...],
    Mapping[str, VulkanExtension],
    Mapping[int, VulkanExtension],
]:
    extensions = tuple(
        VulkanExtension(
            extension.name,
            extension.number,
            extension.extension_type,
            extension.supported_apis,
            extension.dependencies,
            extension.promoted_to,
            extension.deprecated_by,
            extension.obsoleted_by,
            extension.platform,
            extension.provisional,
            tuple(
                _finalize_block(block, structures_by_name) for block in extension.blocks
            ),
            extension.origin,
        )
        for extension in raw_extensions
    )
    return (
        extensions,
        MappingProxyType({extension.name: extension for extension in extensions}),
        MappingProxyType({extension.number: extension for extension in extensions}),
    )


def _finalize_spirv_availabilities(
    raw_availabilities: Sequence[_RawSpirvAvailability],
    core_features_by_name: Mapping[str, VulkanCoreFeature],
    extensions_by_name: Mapping[str, VulkanExtension],
    structures_by_name: Mapping[str, VulkanStructure],
    constants_by_name: Mapping[str, VulkanConstant],
) -> tuple[
    tuple[VulkanSpirvAvailability, ...],
    Mapping[str, VulkanSpirvAvailability],
]:
    result = []
    for availability in raw_availabilities:
        enables = []
        for enable in availability.enables:
            if isinstance(enable, VulkanSpirvCoreEnable):
                if enable.core_feature_name not in core_features_by_name:
                    raise RegistrySourceError(
                        enable.origin,
                        f"unknown Vulkan core feature {enable.core_feature_name!r}",
                    )
                enables.append(enable)
            elif isinstance(enable, VulkanSpirvExtensionEnable):
                if enable.extension_name not in extensions_by_name:
                    raise RegistrySourceError(
                        enable.origin,
                        f"unknown Vulkan extension {enable.extension_name!r}",
                    )
                enables.append(enable)
            elif isinstance(enable, _RawSpirvFeatureEnable):
                structure = structures_by_name[enable.structure_name]
                if enable.member_name not in structure.members_by_name:
                    raise RegistrySourceError(
                        enable.origin,
                        f"unknown member {enable.structure_name}::{enable.member_name}",
                    )
                enables.append(
                    VulkanSpirvFeatureEnable(
                        enable.structure_name,
                        structure.name,
                        enable.member_name,
                        enable.profile_member_alias,
                        enable.requirements,
                        enable.origin,
                    )
                )
            elif isinstance(enable, _RawSpirvPropertyEnable):
                structure = structures_by_name[enable.structure_name]
                if enable.member_name not in structure.members_by_name:
                    raise RegistrySourceError(
                        enable.origin,
                        f"unknown member {enable.structure_name}::{enable.member_name}",
                    )
                constant = constants_by_name[enable.constant_name]
                enables.append(
                    VulkanSpirvPropertyEnable(
                        enable.structure_name,
                        structure.name,
                        enable.member_name,
                        enable.constant_name,
                        constant.name,
                        constant.value,
                        enable.requirements,
                        enable.origin,
                    )
                )
            else:
                raise TypeError("expected a raw SPIR-V availability alternative")
        result.append(
            VulkanSpirvAvailability(
                availability.name, tuple(enables), availability.origin
            )
        )
    availabilities = tuple(result)
    return availabilities, MappingProxyType(
        {availability.name: availability for availability in availabilities}
    )


def _validate_raw_references(
    core_features: Sequence[_RawCoreFeature],
    extensions: Sequence[_RawExtension],
    spirv_extensions: Sequence[_RawSpirvAvailability],
    spirv_capabilities: Sequence[_RawSpirvAvailability],
    structures: Sequence[VulkanStructure],
    structure_members_by_name: Mapping[str, VulkanStructureMember],
) -> None:
    core_feature_names = frozenset(feature.name for feature in core_features)
    extension_names = frozenset(extension.name for extension in extensions)
    overlap = core_feature_names & extension_names
    if overlap:
        name = min(overlap)
        feature = next(feature for feature in core_features if feature.name == name)
        raise RegistrySourceError(
            feature.origin,
            f"core feature and extension namespaces collide at {name!r}",
        )

    availability_names = core_feature_names | extension_names
    for structure in structures:
        for member in structure.members:
            alias = member.availability_alias
            if alias is None:
                continue
            if "::" in alias:
                if alias not in structure_members_by_name:
                    raise RegistrySourceError(
                        member.origin.child("name@alias"),
                        f"unknown Vulkan member alias {alias!r}",
                    )
            elif alias not in availability_names:
                raise RegistrySourceError(
                    member.origin.child("name@alias"),
                    f"unknown Vulkan availability alias {alias!r}",
                )

    for feature in core_features:
        _validate_dependency_references(
            feature.dependencies,
            feature.origin.child("@depends"),
            core_feature_names,
            extension_names,
            structure_members_by_name,
        )
        for block in feature.blocks:
            if block.apis and not block.apis <= feature.apis:
                raise RegistrySourceError(
                    block.origin.child("@api"),
                    "block APIs are not a subset of the owning core feature",
                )
            _validate_dependency_references(
                block.dependencies,
                block.origin.child("@depends"),
                core_feature_names,
                extension_names,
                structure_members_by_name,
            )

    for extension in extensions:
        _validate_dependency_references(
            extension.dependencies,
            extension.origin.child("@depends"),
            core_feature_names,
            extension_names,
            structure_members_by_name,
        )
        for relation_name, relation in (
            ("promotedto", extension.promoted_to),
            ("deprecatedby", extension.deprecated_by),
            ("obsoletedby", extension.obsoleted_by),
        ):
            if relation and relation not in availability_names:
                raise RegistrySourceError(
                    extension.origin.child(f"@{relation_name}"),
                    f"unknown Vulkan version or extension {relation!r}",
                )
        for block in extension.blocks:
            if block.apis and not block.apis <= extension.supported_apis:
                raise RegistrySourceError(
                    block.origin.child("@api"),
                    "block APIs are not a subset of the owning extension",
                )
            _validate_dependency_references(
                block.dependencies,
                block.origin.child("@depends"),
                core_feature_names,
                extension_names,
                structure_members_by_name,
            )

    for availability in (*spirv_extensions, *spirv_capabilities):
        for enable in availability.enables:
            if isinstance(enable, (_RawSpirvFeatureEnable, _RawSpirvPropertyEnable)):
                _validate_dependency_references(
                    enable.requirements,
                    enable.origin.child("@requires"),
                    core_feature_names,
                    extension_names,
                    structure_members_by_name,
                )


def _parse_document(root: ET.Element, source: str) -> VulkanSpirvAvailabilityRegistry:
    root_origin = RegistryOrigin(source, "$")
    dependency_cache: dict[str, DependencyExpression] = {}
    sections = _index_top_level_sections(root, root_origin)
    raw_types, raw_types_by_name = _scan_types(
        sections["types"][0], root_origin.child("types")
    )
    platform_names = _scan_platform_names(sections, root_origin)
    raw_core_features = _parse_core_features(
        sections.get("feature", ()),
        root_origin,
        raw_types_by_name,
        dependency_cache,
    )
    if not raw_core_features:
        raise RegistrySourceError(root_origin, "registry has no core features")
    raw_extensions = _parse_extensions(
        sections["extensions"][0],
        root_origin,
        raw_types_by_name,
        platform_names,
        dependency_cache,
    )
    raw_spirv_extensions = _parse_spirv_availabilities(
        sections["spirvextensions"][0],
        root_origin,
        container_name="spirvextensions",
        entry_name="spirvextension",
        dependency_cache=dependency_cache,
    )
    raw_spirv_capabilities = _parse_spirv_availabilities(
        sections["spirvcapabilities"][0],
        root_origin,
        container_name="spirvcapabilities",
        entry_name="spirvcapability",
        dependency_cache=dependency_cache,
    )

    structure_requests = _collect_structure_requests(
        raw_core_features,
        raw_extensions,
        raw_spirv_extensions,
        raw_spirv_capabilities,
    )
    structures, structures_by_name, structure_members_by_name = _build_structure_model(
        raw_types, structure_requests
    )
    _validate_raw_references(
        raw_core_features,
        raw_extensions,
        raw_spirv_extensions,
        raw_spirv_capabilities,
        structures,
        structure_members_by_name,
    )

    constant_requests: dict[str, RegistryOrigin] = {}
    for availability in (*raw_spirv_extensions, *raw_spirv_capabilities):
        for enable in availability.enables:
            if isinstance(enable, _RawSpirvPropertyEnable):
                constant_requests.setdefault(enable.constant_name, enable.origin)
    constants, constants_by_name = _build_constant_model(
        _scan_enum_elements(root), MappingProxyType(constant_requests)
    )

    core_features, core_features_by_name = _finalize_core_features(
        raw_core_features, structures_by_name
    )
    extensions, extensions_by_name, extensions_by_number = _finalize_extensions(
        raw_extensions, structures_by_name
    )
    spirv_extensions, spirv_extensions_by_name = _finalize_spirv_availabilities(
        raw_spirv_extensions,
        core_features_by_name,
        extensions_by_name,
        structures_by_name,
        constants_by_name,
    )
    spirv_capabilities, spirv_capabilities_by_name = _finalize_spirv_availabilities(
        raw_spirv_capabilities,
        core_features_by_name,
        extensions_by_name,
        structures_by_name,
        constants_by_name,
    )
    return VulkanSpirvAvailabilityRegistry(
        root_origin,
        core_features,
        extensions,
        structures,
        constants,
        spirv_extensions,
        spirv_capabilities,
        core_features_by_name,
        extensions_by_name,
        extensions_by_number,
        structures_by_name,
        structure_members_by_name,
        constants_by_name,
        spirv_extensions_by_name,
        spirv_capabilities_by_name,
    )


def parse_vulkan_spirv_availability(
    contents: str | bytes | bytearray,
    *,
    source: str = "<Vulkan registry>",
) -> VulkanSpirvAvailabilityRegistry:
    """Parses one Vulkan registry's complete SPIR-V availability surface."""

    if not isinstance(contents, (str, bytes, bytearray)):
        raise TypeError("Vulkan registry contents must be text or bytes")
    if not isinstance(source, str) or not source:
        raise TypeError("Vulkan registry source must be a non-empty string")
    if isinstance(contents, bytearray):
        contents = bytes(contents)
    doctype_marker = "<!DOCTYPE" if isinstance(contents, str) else b"<!DOCTYPE"
    if doctype_marker in contents:
        raise RegistrySourceError(
            RegistryOrigin(source, "$"),
            "Vulkan registry must not contain a document type declaration",
        )
    try:
        root = ET.fromstring(contents)
    except ET.ParseError as exception:
        line, column = exception.position
        raise RegistrySourceError(
            RegistryOrigin(source, f"line {line}, column {column + 1}"),
            str(exception),
        ) from exception
    return _parse_document(root, source)


def load_vulkan_spirv_availability(
    path: str | os.PathLike[str],
) -> VulkanSpirvAvailabilityRegistry:
    """Loads and validates one complete Vulkan registry XML file."""

    source_path = Path(path)
    return parse_vulkan_spirv_availability(
        source_path.read_bytes(),
        source=os.fspath(source_path),
    )


__all__ = [
    "VulkanApi",
    "VulkanAvailabilityBlock",
    "VulkanAvailabilityBlockKind",
    "VulkanConstant",
    "VulkanCoreFeature",
    "VulkanExtension",
    "VulkanExtensionType",
    "VulkanFeatureReference",
    "VulkanSpirvAvailability",
    "VulkanSpirvAvailabilityRegistry",
    "VulkanSpirvCoreEnable",
    "VulkanSpirvEnable",
    "VulkanSpirvExtensionEnable",
    "VulkanSpirvFeatureEnable",
    "VulkanSpirvPropertyEnable",
    "VulkanStructure",
    "VulkanStructureMember",
    "VulkanTypeReference",
    "VulkanVersion",
    "load_vulkan_spirv_availability",
    "parse_vulkan_spirv_availability",
]
