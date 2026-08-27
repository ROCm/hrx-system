# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cross-validates the Loom SPIR-V feature catalog against Khronos sources."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from build_tools.spirv.registry.registry_source import RegistryOrigin
from build_tools.spirv.registry.spirv_core_grammar import (
    SpirvCoreGrammar,
    SpirvEnumerant,
    SpirvInstruction,
    SpirvVersion,
)
from build_tools.spirv.registry.vulkan_spirv_availability_model import (
    VulkanSpirvAvailabilityRegistry,
)

from loom.target.arch.spirv.features import (
    ADDRESSING_MODEL_UNSPECIFIED,
    MEMORY_MODEL_UNSPECIFIED,
    FeatureAtom,
)


@dataclass(frozen=True, slots=True)
class _CatalogField:
    name: str
    c_prefix: str
    operand_kind_name: str | None
    unspecified_value: str | None = None


@dataclass(frozen=True, slots=True)
class _IsaAssignment:
    value: int
    origin: RegistryOrigin


@dataclass(frozen=True, slots=True)
class _AtomClosure:
    atom_keys: frozenset[str]
    spirv_version: SpirvVersion
    extensions: frozenset[str]
    capabilities: frozenset[str]


type _GrammarEntity = SpirvEnumerant | SpirvInstruction

_CATALOG_FIELDS = (
    _CatalogField("capabilities", "LOOM_SPIRV_CAPABILITY_", "Capability"),
    _CatalogField("opcodes", "LOOM_SPIRV_OP_", None),
    _CatalogField("storage_classes", "LOOM_SPIRV_STORAGE_CLASS_", "StorageClass"),
    _CatalogField("decorations", "LOOM_SPIRV_DECORATION_", "Decoration"),
    _CatalogField(
        "addressing_model",
        "LOOM_SPIRV_ADDRESSING_MODEL_",
        "AddressingModel",
        ADDRESSING_MODEL_UNSPECIFIED,
    ),
    _CatalogField(
        "memory_model",
        "LOOM_SPIRV_MEMORY_MODEL_",
        "MemoryModel",
        MEMORY_MODEL_UNSPECIFIED,
    ),
)

_ISA_DECLARATION_PATTERN = re.compile(r"^\s*(?P<symbol>LOOM_SPIRV_[A-Z0-9_]+)\b")
_ISA_ASSIGNMENT_PATTERN = re.compile(
    r"^\s*(?P<symbol>LOOM_SPIRV_[A-Z0-9_]+)\s*=\s*"
    r"(?P<value>-?(?:0x[0-9A-Fa-f]+|[0-9]+))\s*,\s*(?://.*)?$"
)


def validate_feature_catalog_sources(
    *,
    atoms: tuple[FeatureAtom, ...],
    isa_header: str,
    isa_source: str,
    grammar: SpirvCoreGrammar,
    vulkan_registry: VulkanSpirvAvailabilityRegistry,
) -> None:
    """Validates a structurally valid feature catalog against canonical sources."""

    references = _collect_catalog_references(atoms)
    entities = _resolve_catalog_entities(references, grammar)
    demanded_symbols = frozenset(symbol for field_references in references.values() for symbol in field_references)
    isa_assignments = _parse_demanded_isa_assignments(
        isa_header,
        source=isa_source,
        demanded_symbols=demanded_symbols,
    )
    _validate_wire_references(
        references=references,
        entities=entities,
        assignments=isa_assignments,
        isa_source=isa_source,
        grammar=grammar,
    )

    atoms_by_key = {atom.key: atom for atom in atoms}
    closures_by_key = _build_atom_closures(atoms, atoms_by_key, entities, grammar)
    capability_providers = _build_capability_providers(atoms, entities)
    for atom in atoms:
        closure = closures_by_key[atom.key]
        _validate_atom_extensions(atom, vulkan_registry)
        for field in _CATALOG_FIELDS:
            for symbol in _field_symbols(atom, field):
                entity = entities[field.name][symbol]
                if field.name == "capabilities":
                    _validate_capability(
                        atom,
                        symbol,
                        entity,
                        closure,
                        capability_providers,
                        grammar,
                        vulkan_registry,
                    )
                else:
                    _validate_ordinary_entity(atom, field, symbol, entity, closure)


def _collect_catalog_references(
    atoms: tuple[FeatureAtom, ...],
) -> dict[str, dict[str, tuple[str, ...]]]:
    mutable_references: dict[str, dict[str, list[str]]] = {field.name: defaultdict(list) for field in _CATALOG_FIELDS}
    for atom in atoms:
        for field in _CATALOG_FIELDS:
            for symbol in _field_symbols(atom, field):
                mutable_references[field.name][symbol].append(atom.key)
    return {field_name: {symbol: tuple(atom_keys) for symbol, atom_keys in field_references.items()} for field_name, field_references in mutable_references.items()}


def _field_symbols(atom: FeatureAtom, field: _CatalogField) -> tuple[str, ...]:
    value = getattr(atom, field.name)
    values = (value,) if isinstance(value, str) else value
    if field.unspecified_value is None:
        return values
    return tuple(item for item in values if item != field.unspecified_value)


def _resolve_catalog_entities(
    references: Mapping[str, Mapping[str, tuple[str, ...]]],
    grammar: SpirvCoreGrammar,
) -> dict[str, dict[str, _GrammarEntity]]:
    result: dict[str, dict[str, _GrammarEntity]] = {}
    for field in _CATALOG_FIELDS:
        demanded_symbols = references[field.name].keys()
        resolved: dict[str, _GrammarEntity] = {}
        for entity in _field_entities(field, grammar):
            for spelling in _entity_spellings(entity):
                symbol = _c_symbol(field, spelling)
                if symbol not in demanded_symbols:
                    continue
                existing = resolved.get(symbol)
                if existing is not None and existing is not entity:
                    raise ValueError(f"SPIR-V catalog {field.name} symbol {symbol!r} resolves to both {existing.origin} and {entity.origin}")
                resolved[symbol] = entity
        result[field.name] = resolved
    return result


def _field_entities(
    field: _CatalogField,
    grammar: SpirvCoreGrammar,
) -> Iterable[_GrammarEntity]:
    if field.operand_kind_name is None:
        return grammar.instructions
    operand_kind = grammar.operand_kinds_by_name.get(field.operand_kind_name)
    if operand_kind is None:
        raise ValueError(f"{grammar.source_origin}: SPIR-V grammar has no {field.operand_kind_name!r} operand kind")
    return operand_kind.enumerants


def _entity_spellings(entity: _GrammarEntity) -> tuple[str, ...]:
    if isinstance(entity, SpirvInstruction):
        return (entity.opname, *entity.aliases)
    return (entity.name, *entity.aliases)


def _c_symbol(field: _CatalogField, spelling: str) -> str:
    if field.operand_kind_name is None:
        spelling = spelling.removeprefix("Op")
    return field.c_prefix + _upper_snake_case(spelling)


def _upper_snake_case(value: str) -> str:
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value).upper()


def _parse_demanded_isa_assignments(
    contents: str,
    *,
    source: str,
    demanded_symbols: frozenset[str],
) -> dict[str, _IsaAssignment]:
    if not isinstance(contents, str):
        raise TypeError("SPIR-V ISA header contents must be text")
    if not isinstance(source, str) or not source:
        raise TypeError("SPIR-V ISA header source must be a non-empty string")
    assignments: dict[str, _IsaAssignment] = {}
    for line_number, line in enumerate(contents.splitlines(), start=1):
        declaration_match = _ISA_DECLARATION_PATTERN.match(line)
        if declaration_match is None:
            continue
        symbol = declaration_match.group("symbol")
        if symbol not in demanded_symbols:
            continue
        origin = RegistryOrigin(source, f"line {line_number}")
        assignment_match = _ISA_ASSIGNMENT_PATTERN.match(line)
        if assignment_match is None:
            raise ValueError(f"{origin}: demanded SPIR-V ISA symbol {symbol!r} must have a literal integer enum assignment")
        if symbol in assignments:
            raise ValueError(f"{origin}: demanded SPIR-V ISA symbol {symbol!r} duplicates {assignments[symbol].origin}")
        assignments[symbol] = _IsaAssignment(
            value=int(assignment_match.group("value"), 0),
            origin=origin,
        )
    return assignments


def _validate_wire_references(
    *,
    references: Mapping[str, Mapping[str, tuple[str, ...]]],
    entities: Mapping[str, Mapping[str, _GrammarEntity]],
    assignments: Mapping[str, _IsaAssignment],
    isa_source: str,
    grammar: SpirvCoreGrammar,
) -> None:
    for field in _CATALOG_FIELDS:
        for symbol, atom_keys in references[field.name].items():
            atom_text = ", ".join(repr(key) for key in atom_keys)
            entity = entities[field.name].get(symbol)
            if entity is None:
                raise ValueError(f"SPIR-V feature atom(s) {atom_text} {field.name} symbol {symbol!r} has no primary or alias spelling in {grammar.source_origin}")
            assignment = assignments.get(symbol)
            if assignment is None:
                raise ValueError(f"SPIR-V feature atom(s) {atom_text} {field.name} symbol {symbol!r} has no literal assignment in {isa_source}")
            expected_value = _entity_value(entity)
            if assignment.value != expected_value:
                raise ValueError(
                    f"SPIR-V feature atom(s) {atom_text} {field.name} symbol "
                    f"{symbol!r} has value {assignment.value} at "
                    f"{assignment.origin}, but {_entity_label(entity)} at "
                    f"{entity.origin} encodes {expected_value}"
                )


def _entity_value(entity: _GrammarEntity) -> int:
    if isinstance(entity, SpirvInstruction):
        return entity.opcode
    return entity.value


def _entity_label(entity: _GrammarEntity) -> str:
    if isinstance(entity, SpirvInstruction):
        return f"instruction {entity.opname!r}"
    return f"enumerant {entity.name!r}"


def _build_atom_closures(
    atoms: tuple[FeatureAtom, ...],
    atoms_by_key: Mapping[str, FeatureAtom],
    entities: Mapping[str, Mapping[str, _GrammarEntity]],
    grammar: SpirvCoreGrammar,
) -> dict[str, _AtomClosure]:
    versions_by_key = {atom.key: _decode_spirv_version(atom, grammar) for atom in atoms}
    ordered_closures: dict[str, tuple[FeatureAtom, ...]] = {}

    def build_ordered_closure(atom: FeatureAtom) -> tuple[FeatureAtom, ...]:
        existing = ordered_closures.get(atom.key)
        if existing is not None:
            return existing
        ordered: list[FeatureAtom] = []
        seen: set[str] = set()
        for required_key in atom.required:
            for required_atom in build_ordered_closure(atoms_by_key[required_key]):
                if required_atom.key not in seen:
                    seen.add(required_atom.key)
                    ordered.append(required_atom)
        if atom.key not in seen:
            ordered.append(atom)
        result = tuple(ordered)
        ordered_closures[atom.key] = result
        return result

    result: dict[str, _AtomClosure] = {}
    for atom in atoms:
        closure_atoms = build_ordered_closure(atom)
        result[atom.key] = _AtomClosure(
            atom_keys=frozenset(item.key for item in closure_atoms),
            spirv_version=max(versions_by_key[item.key] for item in closure_atoms),
            extensions=frozenset(extension for item in closure_atoms for extension in item.extensions),
            capabilities=frozenset(_capability_entity(entities, symbol).name for item in closure_atoms for symbol in item.capabilities),
        )
    return result


def _decode_spirv_version(atom: FeatureAtom, grammar: SpirvCoreGrammar) -> SpirvVersion:
    value = atom.minimum_spirv_version
    if type(value) is not int or not 0 <= value <= 0xFFFFFFFF or value & 0xFF0000FF or ((value >> 16) & 0xFF) == 0:
        raise ValueError(f"SPIR-V feature atom {atom.key!r} has malformed minimum version {value!r}")
    version = SpirvVersion(
        major=(value >> 16) & 0xFF,
        minor=(value >> 8) & 0xFF,
    )
    if version > grammar.version:
        raise ValueError(f"SPIR-V feature atom {atom.key!r} requires version {version}, beyond {grammar.source_origin} version {grammar.version}")
    return version


def _capability_entity(
    entities: Mapping[str, Mapping[str, _GrammarEntity]],
    symbol: str,
) -> SpirvEnumerant:
    return cast(SpirvEnumerant, entities["capabilities"][symbol])


def _build_capability_providers(
    atoms: tuple[FeatureAtom, ...],
    entities: Mapping[str, Mapping[str, _GrammarEntity]],
) -> dict[str, frozenset[str]]:
    mutable_providers: dict[str, set[str]] = defaultdict(set)
    for atom in atoms:
        for symbol in atom.capabilities:
            mutable_providers[_capability_entity(entities, symbol).name].add(atom.key)
    return {capability_name: frozenset(atom_keys) for capability_name, atom_keys in mutable_providers.items()}


def _validate_atom_extensions(
    atom: FeatureAtom,
    vulkan_registry: VulkanSpirvAvailabilityRegistry,
) -> None:
    for extension in atom.extensions:
        if extension not in vulkan_registry.spirv_extensions_by_name:
            raise ValueError(f"SPIR-V feature atom {atom.key!r} extension {extension!r} has no Vulkan enable mapping in {vulkan_registry.source_origin}")


def _validate_capability(
    atom: FeatureAtom,
    symbol: str,
    entity: _GrammarEntity,
    closure: _AtomClosure,
    capability_providers: Mapping[str, frozenset[str]],
    grammar: SpirvCoreGrammar,
    vulkan_registry: VulkanSpirvAvailabilityRegistry,
) -> None:
    entity = cast(SpirvEnumerant, entity)
    if not _has_version_or_extension_route(entity, closure):
        _raise_missing_route(atom, "capabilities", symbol, entity, closure)

    availability_names = (entity.name, *entity.aliases)
    if not any(name in vulkan_registry.spirv_capabilities_by_name for name in availability_names):
        raise ValueError(
            f"SPIR-V feature atom {atom.key!r} capabilities symbol {symbol!r} resolves to {entity.name!r} at {entity.origin}, which has no Vulkan enable mapping in {vulkan_registry.source_origin}"
        )

    for implied_name in grammar.capability_implication_closures[entity.name]:
        providers = capability_providers.get(implied_name)
        if providers is None or not closure.atom_keys.isdisjoint(providers):
            continue
        provider_text = ", ".join(repr(key) for key in sorted(providers))
        raise ValueError(
            f"SPIR-V feature atom {atom.key!r} capability {entity.name!r} at "
            f"{entity.origin} implies modeled capability {implied_name!r}, "
            f"provided by atom(s) {provider_text}, outside its declared "
            "dependency closure"
        )


def _validate_ordinary_entity(
    atom: FeatureAtom,
    field: _CatalogField,
    symbol: str,
    entity: _GrammarEntity,
    closure: _AtomClosure,
) -> None:
    if entity.capabilities and closure.capabilities.isdisjoint(entity.capabilities):
        guards = ", ".join(repr(name) for name in entity.capabilities)
        raise ValueError(
            f"SPIR-V feature atom {atom.key!r} {field.name} symbol {symbol!r} "
            f"resolves to {_entity_label(entity)} at {entity.origin}, but its "
            f"dependency closure supplies none of the capability guards {guards}"
        )
    if _has_version_or_extension_route(entity, closure):
        return
    capability_only_route = entity.version is None and not entity.extensions and bool(entity.capabilities) and not closure.capabilities.isdisjoint(entity.capabilities)
    if not capability_only_route:
        _raise_missing_route(atom, field.name, symbol, entity, closure)


def _has_version_or_extension_route(
    entity: _GrammarEntity,
    closure: _AtomClosure,
) -> bool:
    core_route = entity.version is not None and entity.version <= closure.spirv_version and (entity.last_version is None or closure.spirv_version <= entity.last_version)
    extension_route = not closure.extensions.isdisjoint(entity.extensions)
    return core_route or extension_route


def _raise_missing_route(
    atom: FeatureAtom,
    field_name: str,
    symbol: str,
    entity: _GrammarEntity,
    closure: _AtomClosure,
) -> None:
    core_range = "none"
    if entity.version is not None:
        core_range = str(entity.version)
        if entity.last_version is not None:
            core_range += f" through {entity.last_version}"
    extensions = ", ".join(repr(name) for name in entity.extensions) or "none"
    raise ValueError(
        f"SPIR-V feature atom {atom.key!r} {field_name} symbol {symbol!r} "
        f"resolves to {_entity_label(entity)} at {entity.origin}, available from "
        f"core version {core_range} or extension(s) {extensions}; its dependency "
        f"closure supplies SPIR-V {closure.spirv_version} and extensions "
        f"{sorted(closure.extensions)!r}"
    )


__all__ = ["validate_feature_catalog_sources"]
