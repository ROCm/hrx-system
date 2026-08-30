# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates the Loom language reference from canonical declarations."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loom.dsl import (
    AttrDef,
    Dialect,
    Effect,
    EncodingFamilyDef,
    EnumDef,
    Op,
    OperandOwnershipEffect,
    ParameterizedAttrDef,
    ResultOwnershipEffect,
    TypeDef,
    type_constraint_name,
)
from loom.gen.ops.model import DialectGeneration, GenerationModel, load_generation_model

_GENERATED_NOTICE = "<!-- Generated from the canonical Loom declaration model. -->"
_SLUG_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")


@dataclass(frozen=True)
class ReferenceSection:
    """One user-facing group in the dialect reference."""

    key: str
    title: str
    doc: str


@dataclass(frozen=True)
class DialectReferenceSpec:
    """Publication placement for one canonical dialect."""

    name: str
    section: str | None

    @property
    def is_published(self) -> bool:
        return self.section is not None


REFERENCE_SECTIONS = (
    ReferenceSection(
        "program",
        "Program IR",
        "Target-independent program structure, computation, data, and launch semantics.",
    ),
    ReferenceSection(
        "testing",
        "Testing and compiler control",
        "Executable checks, benchmark cases, pipeline selection, and diagnostics.",
    ),
    ReferenceSection(
        "target",
        "Target and low-level IR",
        "Target witnesses and representations used after target-independent lowering.",
    ),
)

# This list is intentionally exhaustive. Adding a canonical dialect requires an
# explicit decision about its user-facing role instead of silently publishing it
# according to declaration or import order.
DIALECT_REFERENCE_SPECS = (
    DialectReferenceSpec("scalar", "program"),
    DialectReferenceSpec("func", "program"),
    DialectReferenceSpec("template", "program"),
    DialectReferenceSpec("encoding", "program"),
    DialectReferenceSpec("pool", "program"),
    DialectReferenceSpec("global", "program"),
    DialectReferenceSpec("scf", "program"),
    DialectReferenceSpec("cfg", "program"),
    DialectReferenceSpec("command", "program"),
    DialectReferenceSpec("buffer", "program"),
    DialectReferenceSpec("view", "program"),
    DialectReferenceSpec("vector", "program"),
    DialectReferenceSpec("index", "program"),
    DialectReferenceSpec("kernel", "program"),
    DialectReferenceSpec("target", "program"),
    DialectReferenceSpec("config", "program"),
    DialectReferenceSpec("check", "testing"),
    DialectReferenceSpec("pass", "testing"),
    DialectReferenceSpec("sanitizer", "testing"),
    DialectReferenceSpec("low", "target"),
    DialectReferenceSpec("llvmir", "target"),
    DialectReferenceSpec("amdgpu", "target"),
    DialectReferenceSpec("spirv", "target"),
    DialectReferenceSpec("x86", "target"),
    DialectReferenceSpec("wasm", "target"),
    DialectReferenceSpec("test", None),
)


def _slug(value: str) -> str:
    slug = value.lower().replace(".", "-")
    if not _SLUG_PATTERN.fullmatch(slug):
        raise ValueError(f"reference name {value!r} does not have a stable ASCII path spelling")
    return slug


def _escape_table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    materialized_rows = [tuple(row) for row in rows]
    if any(len(row) != len(headers) for row in materialized_rows):
        raise ValueError("Markdown table row does not match its header")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_escape_table_cell(cell) for cell in row) + " |" for row in materialized_rows)
    return lines


def _summary(doc: str) -> str:
    if not doc:
        return "—"
    paragraph = doc.strip().split("\n\n", maxsplit=1)[0].replace("\n", " ")
    sentence_end = paragraph.find(". ")
    return paragraph if sentence_end < 0 else paragraph[: sentence_end + 1]


def _code_list(values: Iterable[Any]) -> str:
    rendered = [f"`{value}`" for value in values]
    return ", ".join(rendered) if rendered else "—"


def _enum_name(value: Any) -> str:
    return value.name.lower().replace("_", " ")


def _phase_name(op: Op, dialect: Dialect) -> str:
    phase = op.phase or dialect.default_phase
    return _enum_name(phase) if phase is not None else "—"


def _attr_type_name(attr: AttrDef) -> str:
    if attr.enum_def is not None:
        return f"{attr.attr_type} `{attr.enum_def.name}`"
    if attr.parameterized_attr is not None:
        return f"{attr.attr_type} `#{attr.parameterized_attr.name}`"
    if attr.symbol_ref is not None:
        interfaces = _code_list(attr.symbol_ref.interfaces)
        return f"symbol ({interfaces})"
    return attr.attr_type


def _attr_cardinality(attr: AttrDef) -> str:
    if attr.optional:
        return "optional"
    if attr.default is not None:
        return f"default `{attr.default!r}`"
    return "required"


def _value_cardinality(field: Any) -> str:
    if getattr(field, "variadic", False):
        return "zero or more" if getattr(field, "optional", False) else "variadic"
    return "optional" if getattr(field, "optional", False) else "required"


def _field_rows(op: Op) -> list[tuple[str, str, str, str, str]]:
    rows = [
        (
            "Operand",
            f"`{operand.name}`",
            f"`{type_constraint_name(operand.type_constraint)}`",
            _value_cardinality(operand),
            operand.doc or "—",
        )
        for operand in op.operands
    ]
    for result in op.results:
        result_kind = "Tied result" if hasattr(result, "tied_to") else "Result"
        details = result.doc or "—"
        tied_to = getattr(result, "tied_to", None)
        if tied_to is not None:
            details = f"{details} Tied to `{tied_to}`."
        elif getattr(result, "allocates", False):
            details = f"{details} Produces a fresh allocation."
        rows.append(
            (
                result_kind,
                f"`{result.name}`",
                f"`{type_constraint_name(result.type_constraint)}`",
                _value_cardinality(result),
                details,
            )
        )
    rows.extend(
        (
            "Attribute",
            f"`{attr.name}`",
            _attr_type_name(attr),
            _attr_cardinality(attr),
            attr.doc or "—",
        )
        for attr in op.attrs
    )
    rows.extend(
        (
            "Successor",
            f"`{successor.name}`",
            "block",
            _value_cardinality(successor),
            successor.doc or "—",
        )
        for successor in op.successors
    )
    for region in op.regions:
        properties = []
        if region.single_block:
            properties.append("single block")
        if region.terminator:
            properties.append(f"terminator `{region.terminator}`")
        description = region.doc or "—"
        if properties:
            description = f"{description} ({', '.join(properties)}.)"
        rows.append(
            (
                "Region",
                f"`{region.name}`",
                "region",
                _value_cardinality(region),
                description,
            )
        )
    return rows


def _interface_name(interface: Any) -> str:
    return type(interface).__name__.removesuffix("Interface")


def _effect_name(effect: Effect) -> str:
    return f"{effect.kind.value} {effect.operand}"


def _ownership_effect_name(
    effect: OperandOwnershipEffect | ResultOwnershipEffect,
) -> str:
    if isinstance(effect, OperandOwnershipEffect):
        return f"{effect.kind.value} {effect.operand} ({effect.carrier.value})"
    source = f" from {effect.source}" if effect.source is not None else ""
    return f"{effect.kind.value} {effect.result}{source}"


def _operation_path(dialect_name: str, op: Op) -> str:
    prefix = f"{dialect_name}."
    if not op.name.startswith(prefix):
        raise ValueError(f"operation {op.name!r} is not in dialect namespace {dialect_name!r}")
    return f"dialects/{dialect_name}/ops/{_slug(op.name[len(prefix) :])}.md"


def _render_operation(dialect: Dialect, op: Op) -> str:
    metadata: list[tuple[str, str]] = [("Semantic phase", _phase_name(op, dialect))]
    category = op.category or dialect.default_category
    if category is not None:
        metadata.append(("Category", f"`{category.key}`"))
    if op.traits:
        metadata.append(("Traits", _code_list(op.traits)))
    if op.contracts:
        metadata.append(("Target contracts", _code_list(c.key for c in op.contracts)))
    if op.interfaces:
        metadata.append(("Interfaces", _code_list(_interface_name(i) for i in op.interfaces)))
    if op.effects:
        metadata.append(("Memory effects", _code_list(_effect_name(e) for e in op.effects)))
    if op.ownership_effects:
        metadata.append(
            (
                "Ownership effects",
                _code_list(_ownership_effect_name(e) for e in op.ownership_effects),
            )
        )

    lines = [
        _GENERATED_NOTICE,
        "",
        f"# `{op.name}`",
        "",
        f"[← `{dialect.name}` dialect](../index.md)",
        "",
        op.doc.strip(),
        "",
        "## Operation contract",
        "",
        *_table(("Property", "Value"), metadata),
        "",
    ]

    if op.symbol_def is not None:
        symbol = op.symbol_def
        flags = []
        if symbol.is_declaration:
            flags.append("declaration")
        if symbol.is_test_only:
            flags.append("test only")
        lines.extend(
            [
                "## Symbol contract",
                "",
                *_table(
                    ("Property", "Value"),
                    (
                        ("Kind", f"`{symbol.name}`"),
                        ("Defining field", f"`{symbol.field}`"),
                        ("Interfaces", _code_list(symbol.interfaces)),
                        ("Flags", _code_list(flags)),
                    ),
                ),
                "",
            ]
        )

    field_rows = _field_rows(op)
    lines.extend(["## Signature", ""])
    if field_rows:
        lines.extend(
            _table(
                ("Kind", "Name", "Type", "Cardinality", "Description"),
                field_rows,
            )
        )
    else:
        lines.append("This operation has no operands, results, attributes, or regions.")
    lines.append("")

    if op.constraints:
        lines.extend(
            [
                "## Verification constraints",
                "",
                *[f"- `{constraint!r}`" for constraint in op.constraints],
                "",
            ]
        )

    lines.extend(["## Examples", ""])
    if op.examples:
        for example in op.examples:
            lines.extend(["```loom", example.rstrip(), "```", ""])
    else:
        lines.extend(
            [
                "!!! note",
                "    A canonical source example has not been added for this operation yet.",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _group_operations(
    generation: DialectGeneration,
) -> list[tuple[str, str, list[Op]]]:
    dialect = generation.dialect
    grouped: dict[str | None, list[Op]] = defaultdict(list)
    for op in generation.ops:
        category = op.category or dialect.default_category
        grouped[category.key if category is not None else None].append(op)

    groups = []
    for category in dialect.categories:
        ops = sorted(grouped.pop(category.key, ()), key=lambda op: op.name)
        if ops:
            groups.append((category.key.replace("_", " ").title(), category.doc, ops))
    uncategorized = sorted(grouped.pop(None, ()), key=lambda op: op.name)
    if uncategorized:
        groups.insert(0, ("Operations", "", uncategorized))
    if grouped:
        unknown = ", ".join(sorted(grouped))
        raise ValueError(f"dialect {dialect.name!r} has undeclared categories: {unknown}")
    return groups


def _type_path(type_def: TypeDef) -> str:
    return f"types/{_slug(type_def.name)}.md"


def _parameterized_attr_path(attr: ParameterizedAttrDef) -> str:
    return f"attributes/families/{_slug(attr.name)}.md"


def _enum_path(owner: str, enum_def: EnumDef) -> str:
    return f"attributes/enums/{_slug(owner)}-{_slug(enum_def.name)}.md"


def _encoding_path(encoding: EncodingFamilyDef) -> str:
    return f"encodings/{_slug(encoding.name)}.md"


def _render_dialect(generation: DialectGeneration) -> str:
    dialect = generation.dialect
    dialect_id = f"`0x{dialect.dialect_id:02x}`" if dialect.dialect_id else "unassigned"
    lines = [
        _GENERATED_NOTICE,
        "",
        f"# `{dialect.name}` dialect",
        "",
        "[← Dialect reference](../index.md)",
        "",
        dialect.doc.strip(),
        "",
        *_table(
            ("Property", "Value"),
            (
                ("Bytecode dialect ID", dialect_id),
                ("Registered by default", "yes" if dialect.register_by_default else "no"),
                ("Operations", len(generation.ops)),
            ),
        ),
        "",
    ]

    for title, doc, ops in _group_operations(generation):
        lines.extend([f"## {title}", ""])
        if doc:
            lines.extend([doc, ""])
        lines.extend(
            _table(
                ("Operation", "Summary"),
                (
                    (
                        f"[`{op.name}`](ops/{Path(_operation_path(dialect.name, op)).name})",
                        _summary(op.doc),
                    )
                    for op in ops
                ),
            )
        )
        lines.append("")

    if generation.parameterized_attrs:
        lines.extend(
            [
                "## Parameterized attributes",
                "",
                *_table(
                    ("Attribute", "Summary"),
                    (
                        (
                            f"[`#{attr.name}`](../../{_parameterized_attr_path(attr)})",
                            _summary(attr.doc),
                        )
                        for attr in generation.parameterized_attrs
                    ),
                ),
                "",
            ]
        )
    if generation.encoding_families:
        lines.extend(
            [
                "## Encoding families",
                "",
                *_table(
                    ("Encoding", "Summary"),
                    (
                        (
                            f"[`#{encoding.name}`](../../{_encoding_path(encoding)})",
                            _summary(encoding.doc),
                        )
                        for encoding in generation.encoding_families
                    ),
                ),
                "",
            ]
        )
    if generation.types:
        lines.extend(
            [
                "## Types",
                "",
                *_table(
                    ("Type", "Summary"),
                    (
                        (
                            f"[`{type_def.name}`](../../{_type_path(type_def)})",
                            _summary(type_def.doc),
                        )
                        for type_def in generation.types
                    ),
                ),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_dialect_index(
    generations: dict[str, DialectGeneration],
    specs: Sequence[DialectReferenceSpec],
) -> str:
    lines = [
        _GENERATED_NOTICE,
        "",
        "# Dialect reference",
        "",
        "This reference is generated from the same declarations used by Loom's parser, printer, verifier, builders, and bytecode implementation.",
        "",
        "The programming guide introduces these operations in executable programs. Use this reference when you need the complete contract for a specific op.",
        "",
    ]
    for section in REFERENCE_SECTIONS:
        section_specs = sorted(
            (spec for spec in specs if spec.section == section.key),
            key=lambda spec: spec.name,
        )
        lines.extend([f"## {section.title}", "", section.doc, ""])
        lines.extend(
            _table(
                ("Dialect", "Summary", "Operations"),
                (
                    (
                        f"[`{spec.name}`]({spec.name}/index.md)",
                        _summary(generations[spec.name].dialect.doc),
                        len(generations[spec.name].ops),
                    )
                    for spec in section_specs
                ),
            )
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _type_parameter_rows(type_def: TypeDef) -> list[tuple[str, str, str, str]]:
    rows = []
    for parameter in type_def.params:
        if isinstance(parameter, AttrDef):
            rows.append(
                (
                    f"`{parameter.name}`",
                    _attr_type_name(parameter),
                    _attr_cardinality(parameter),
                    parameter.doc or "—",
                )
            )
            continue
        kind = type(parameter).__name__.removesuffix("Param").lower()
        constraint = getattr(parameter, "constraint", None)
        if constraint is not None:
            kind = f"{kind} `{type_constraint_name(constraint)}`"
        cardinality = "optional" if getattr(parameter, "optional", False) else "required"
        rows.append((f"`{parameter.name}`", kind, cardinality, parameter.doc or "—"))
    return rows


def _render_type(type_def: TypeDef) -> str:
    lines = [
        _GENERATED_NOTICE,
        "",
        f"# `{type_def.name}` type",
        "",
        "[← Type reference](index.md)",
        "",
        type_def.doc.strip(),
        "",
        *_table(
            ("Property", "Value"),
            (
                ("Representation", f"`{type_def.ir_kind}`"),
                ("Semantic role", _enum_name(type_def.semantic)),
                ("Target contracts", _code_list(c.key for c in type_def.contracts)),
            ),
        ),
        "",
    ]
    parameter_rows = _type_parameter_rows(type_def)
    if parameter_rows:
        lines.extend(
            [
                "## Parameters",
                "",
                *_table(("Name", "Kind", "Cardinality", "Description"), parameter_rows),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_type_index(types: Sequence[TypeDef]) -> str:
    return (
        "\n".join(
            [
                _GENERATED_NOTICE,
                "",
                "# Type reference",
                "",
                "Structured Loom types declared by the canonical language model.",
                "",
                *_table(
                    ("Type", "Summary"),
                    (
                        (
                            f"[`{type_def.name}`]({Path(_type_path(type_def)).name})",
                            _summary(type_def.doc),
                        )
                        for type_def in types
                    ),
                ),
            ]
        ).rstrip()
        + "\n"
    )


def _parameter_rows(parameters: Sequence[AttrDef]) -> list[tuple[str, str, str, str]]:
    return [
        (
            f"`{parameter.name}`",
            _attr_type_name(parameter),
            _attr_cardinality(parameter),
            parameter.doc or "—",
        )
        for parameter in parameters
    ]


def _render_parameterized_attr(attr: ParameterizedAttrDef) -> str:
    lines = [
        _GENERATED_NOTICE,
        "",
        f"# `#{attr.name}` attribute",
        "",
        "[← Attribute reference](../index.md)",
        "",
        attr.doc.strip(),
        "",
    ]
    if attr.target_condition is not None:
        lines.extend(
            [
                "This attribute can be used as a static target-applicability requirement.",
                "",
            ]
        )
    if attr.primary_parameter is not None:
        lines.extend(
            [
                f"`{attr.primary_parameter.name}` is the compact positional parameter.",
                "",
            ]
        )
    parameter_rows = _parameter_rows(attr.parameters)
    if parameter_rows:
        lines.extend(
            [
                "## Parameters",
                "",
                *_table(("Name", "Kind", "Cardinality", "Description"), parameter_rows),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_enum(owner: str, enum_def: EnumDef, uses: Sequence[str]) -> str:
    return (
        "\n".join(
            [
                _GENERATED_NOTICE,
                "",
                f"# `{owner}.{enum_def.name}` enum",
                "",
                "[← Attribute reference](../index.md)",
                "",
                enum_def.doc.strip() or "No enum-level description has been added.",
                "",
                "## Cases",
                "",
                *_table(
                    ("Keyword", "Value", "Description"),
                    ((f"`{case.keyword}`", case.value, case.doc or "—") for case in enum_def.cases),
                ),
                "",
                "## Used by",
                "",
                *[f"- `{use}`" for use in uses],
            ]
        ).rstrip()
        + "\n"
    )


def _render_attribute_index(
    attrs: Sequence[ParameterizedAttrDef],
    enums: Sequence[tuple[str, EnumDef, Sequence[str]]],
) -> str:
    lines = [
        _GENERATED_NOTICE,
        "",
        "# Attribute reference",
        "",
        "Parameterized attributes and enum vocabularies used by canonical Loom IR.",
        "",
    ]
    if attrs:
        lines.extend(
            [
                "## Parameterized attributes",
                "",
                *_table(
                    ("Attribute", "Summary"),
                    (
                        (
                            f"[`#{attr.name}`](families/{Path(_parameterized_attr_path(attr)).name})",
                            _summary(attr.doc),
                        )
                        for attr in attrs
                    ),
                ),
                "",
            ]
        )
    if enums:
        lines.extend(
            [
                "## Enum vocabularies",
                "",
                *_table(
                    ("Enum", "Summary"),
                    (
                        (
                            f"[`{owner}.{enum_def.name}`](enums/{Path(_enum_path(owner, enum_def)).name})",
                            _summary(enum_def.doc),
                        )
                        for owner, enum_def, _ in enums
                    ),
                ),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_encoding(encoding: EncodingFamilyDef) -> str:
    lines = [
        _GENERATED_NOTICE,
        "",
        f"# `#{encoding.name}` encoding",
        "",
        "[← Encoding reference](index.md)",
        "",
        encoding.doc.strip(),
        "",
        *_table(
            ("Property", "Value"),
            (
                ("Role", _enum_name(encoding.role)),
                (
                    "Implicit shaped attachment",
                    "yes" if encoding.implicit_shaped_attachment else "no",
                ),
            ),
        ),
        "",
    ]
    parameter_rows = _parameter_rows(encoding.parameters)
    if parameter_rows:
        lines.extend(
            [
                "## Static parameters",
                "",
                *_table(("Name", "Kind", "Cardinality", "Description"), parameter_rows),
                "",
            ]
        )
    if encoding.dynamic_parameters:
        lines.extend(
            [
                "## Dynamic parameters",
                "",
                *_table(
                    ("Name", "Type", "Description"),
                    (
                        (
                            f"`{parameter.name}`",
                            f"`{type_constraint_name(parameter.type_constraint)}`",
                            parameter.doc or "—",
                        )
                        for parameter in encoding.dynamic_parameters
                    ),
                ),
                "",
            ]
        )
    if encoding.aliases:
        lines.extend(
            [
                "## Canonical aliases",
                "",
                *_table(
                    ("Alias", "Fixed parameters", "Default parameters"),
                    (
                        (
                            f"`#{alias.name}`",
                            _code_list(f"{name}={value!r}" for name, value in alias.fixed_parameters),
                            _code_list(f"{name}={value!r}" for name, value in alias.default_parameters),
                        )
                        for alias in encoding.aliases
                    ),
                ),
                "",
            ]
        )
    if encoding.fixed_record is not None:
        record = encoding.fixed_record
        lines.extend(
            [
                "## Fixed storage geometry",
                "",
                *_table(
                    ("Logical elements", "Storage bytes", "Required alignment"),
                    (
                        (
                            record.logical_element_count,
                            record.storage_byte_count,
                            record.required_alignment,
                        ),
                    ),
                ),
                "",
            ]
        )
    if encoding.required_auxiliary_keys:
        lines.extend(
            [
                "## Required auxiliary values",
                "",
                *_table(
                    ("Key", "Description"),
                    ((f"`{case.keyword}`", case.doc or "—") for case in encoding.required_auxiliary_keys),
                ),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_encoding_index(encodings: Sequence[EncodingFamilyDef]) -> str:
    return (
        "\n".join(
            [
                _GENERATED_NOTICE,
                "",
                "# Encoding reference",
                "",
                "Encoding families describe logical layout, storage schema, physical storage, and numeric transforms without baking those choices into operation ABIs.",
                "",
                *_table(
                    ("Encoding", "Role", "Summary"),
                    (
                        (
                            f"[`#{encoding.name}`]({Path(_encoding_path(encoding)).name})",
                            _enum_name(encoding.role),
                            _summary(encoding.doc),
                        )
                        for encoding in encodings
                    ),
                ),
            ]
        ).rstrip()
        + "\n"
    )


def _validate_specs(
    model: GenerationModel,
) -> tuple[dict[str, DialectGeneration], dict[str, DialectReferenceSpec]]:
    generations: dict[str, DialectGeneration] = {}
    for generation in model.dialects:
        name = generation.dialect.name
        if name in generations:
            raise ValueError(f"duplicate dialect generation for {name!r}")
        generations[name] = generation

    specs: dict[str, DialectReferenceSpec] = {}
    valid_sections = {section.key for section in REFERENCE_SECTIONS}
    for spec in DIALECT_REFERENCE_SPECS:
        if spec.name in specs:
            raise ValueError(f"duplicate dialect reference placement for {spec.name!r}")
        if spec.section is not None and spec.section not in valid_sections:
            raise ValueError(f"dialect {spec.name!r} uses unknown reference section {spec.section!r}")
        specs[spec.name] = spec

    missing_specs = sorted(generations.keys() - specs.keys())
    stale_specs = sorted(specs.keys() - generations.keys())
    if missing_specs or stale_specs:
        raise ValueError(f"dialect reference placement is not exhaustive; missing={missing_specs}, stale={stale_specs}")
    return generations, specs


def _public_types(model: GenerationModel, specs: dict[str, DialectReferenceSpec]) -> list[TypeDef]:
    types = list(model.types)
    for generation in model.dialects:
        if specs[generation.dialect.name].is_published:
            types.extend(generation.types)
    by_name: dict[str, TypeDef] = {}
    for type_def in types:
        if type_def.name in by_name:
            raise ValueError(f"duplicate public type declaration {type_def.name!r}")
        by_name[type_def.name] = type_def
    return [by_name[name] for name in sorted(by_name)]


def _public_parameterized_attrs(model: GenerationModel, specs: dict[str, DialectReferenceSpec]) -> list[ParameterizedAttrDef]:
    attrs = [attr for generation in model.dialects if specs[generation.dialect.name].is_published for attr in generation.parameterized_attrs]
    names = [attr.name for attr in attrs]
    if len(names) != len(set(names)):
        raise ValueError("duplicate public parameterized attribute declaration")
    return sorted(attrs, key=lambda attr: attr.name)


def _public_encodings(model: GenerationModel, specs: dict[str, DialectReferenceSpec]) -> list[EncodingFamilyDef]:
    encodings = [encoding for generation in model.dialects if specs[generation.dialect.name].is_published for encoding in generation.encoding_families]
    names = [encoding.name for encoding in encodings]
    if len(names) != len(set(names)):
        raise ValueError("duplicate public encoding family declaration")
    return sorted(encodings, key=lambda encoding: encoding.name)


def _collect_public_enums(
    model: GenerationModel,
    specs: dict[str, DialectReferenceSpec],
    types: Sequence[TypeDef],
) -> list[tuple[str, EnumDef, Sequence[str]]]:
    definitions: dict[tuple[str, str], EnumDef] = {}
    uses: dict[tuple[str, str], set[str]] = defaultdict(set)

    def collect(owner: str, attr: AttrDef, use: str) -> None:
        enum_def = attr.enum_def
        if enum_def is None:
            return
        key = (owner, enum_def.name)
        previous = definitions.setdefault(key, enum_def)
        if previous != enum_def:
            raise ValueError(f"enum name {owner}.{enum_def.name!r} has multiple public definitions")
        uses[key].add(use)

    for generation in model.dialects:
        if not specs[generation.dialect.name].is_published:
            continue
        owner = generation.dialect.name
        for op in generation.ops:
            for attr in op.attrs:
                collect(owner, attr, f"{op.name}.{attr.name}")
        for family in generation.parameterized_attrs:
            for parameter in family.parameters:
                collect(owner, parameter, f"#{family.name}.{parameter.name}")
        for encoding in generation.encoding_families:
            for parameter in encoding.parameters:
                collect(owner, parameter, f"#{encoding.name}.{parameter.name}")
            if encoding.auxiliary_key_enum is not None:
                auxiliary = encoding.auxiliary_key_enum
                key = (owner, auxiliary.name)
                previous = definitions.setdefault(key, auxiliary)
                if previous != auxiliary:
                    raise ValueError(f"enum name {owner}.{auxiliary.name!r} has multiple public definitions")
                uses[key].add(f"#{encoding.name} auxiliary values")
    for type_def in types:
        owner = type_def.name.split(".", maxsplit=1)[0]
        for parameter in type_def.params:
            if isinstance(parameter, AttrDef):
                collect(owner, parameter, f"{type_def.name}.{parameter.name}")

    return [(owner, definitions[(owner, name)], sorted(uses[(owner, name)])) for owner, name in sorted(definitions)]


def _coverage(
    model: GenerationModel,
    specs: dict[str, DialectReferenceSpec],
    types: Sequence[TypeDef],
    attrs: Sequence[ParameterizedAttrDef],
    enums: Sequence[tuple[str, EnumDef, Sequence[str]]],
    encodings: Sequence[EncodingFamilyDef],
) -> dict[str, Any]:
    dialect_coverage = {}
    total_ops = 0
    documented_ops = 0
    ops_with_examples = 0
    field_count = 0
    documented_field_count = 0
    for generation in model.dialects:
        name = generation.dialect.name
        fields = [
            field
            for op in generation.ops
            for field in (
                *op.operands,
                *op.results,
                *op.attrs,
                *op.successors,
                *op.regions,
            )
        ]
        operation_count = len(generation.ops)
        dialect_documented_ops = sum(bool(op.doc.strip()) for op in generation.ops)
        dialect_ops_with_examples = sum(bool(op.examples) for op in generation.ops)
        dialect_documented_fields = sum(bool(field.doc.strip()) for field in fields)
        total_ops += operation_count
        documented_ops += dialect_documented_ops
        ops_with_examples += dialect_ops_with_examples
        field_count += len(fields)
        documented_field_count += dialect_documented_fields
        dialect_coverage[name] = {
            "documented_field_count": dialect_documented_fields,
            "documented_operation_count": dialect_documented_ops,
            "field_count": len(fields),
            "operation_count": operation_count,
            "operations_with_examples": dialect_ops_with_examples,
            "published": specs[name].is_published,
            "section": specs[name].section,
        }
    return {
        "schema_version": 1,
        "dialects": dialect_coverage,
        "totals": {
            "documented_field_count": documented_field_count,
            "documented_operation_count": documented_ops,
            "encoding_family_count": len(encodings),
            "enum_count": len(enums),
            "field_count": field_count,
            "operation_count": total_ops,
            "operations_with_examples": ops_with_examples,
            "parameterized_attribute_count": len(attrs),
            "type_count": len(types),
        },
    }


def generate_reference_files(model: GenerationModel | None = None) -> dict[str, str]:
    """Returns the deterministic generated reference file set."""

    model = model or load_generation_model()
    generations, specs = _validate_specs(model)
    types = _public_types(model, specs)
    attrs = _public_parameterized_attrs(model, specs)
    encodings = _public_encodings(model, specs)
    enums = _collect_public_enums(model, specs, types)

    files: dict[str, str] = {
        "dialects/index.md": _render_dialect_index(generations, DIALECT_REFERENCE_SPECS),
        "types/index.md": _render_type_index(types),
        "attributes/index.md": _render_attribute_index(attrs, enums),
        "encodings/index.md": _render_encoding_index(encodings),
    }
    for spec in DIALECT_REFERENCE_SPECS:
        if not spec.is_published:
            continue
        generation = generations[spec.name]
        files[f"dialects/{spec.name}/index.md"] = _render_dialect(generation)
        for op in generation.ops:
            path = _operation_path(spec.name, op)
            if path in files:
                raise ValueError(f"duplicate generated reference path {path!r}")
            files[path] = _render_operation(generation.dialect, op)
    for type_def in types:
        files[_type_path(type_def)] = _render_type(type_def)
    for attr in attrs:
        files[_parameterized_attr_path(attr)] = _render_parameterized_attr(attr)
    for owner, enum_def, uses in enums:
        files[_enum_path(owner, enum_def)] = _render_enum(owner, enum_def, uses)
    for encoding in encodings:
        files[_encoding_path(encoding)] = _render_encoding(encoding)

    coverage = _coverage(model, specs, types, attrs, enums, encodings)
    files["coverage.json"] = json.dumps(coverage, indent=2, sort_keys=True) + "\n"
    return dict(sorted(files.items()))


def write_reference_files(output_dir: Path, files: dict[str, str]) -> None:
    """Writes a generated reference file set beneath ``output_dir``."""

    for relative_path, contents in files.items():
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(contents, encoding="utf-8")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Loom language reference from canonical declarations.")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory receiving generated reference files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    files = generate_reference_files()
    write_reference_files(args.output, files)
    print(f"Generated {len(files)} Loom reference files under {args.output}")


if __name__ == "__main__":
    main()
