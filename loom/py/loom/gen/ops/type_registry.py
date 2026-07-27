# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: Type declarations -> C type registry tables."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from loom.dsl import ContractFamily, TypeDef, TypeSemantic
from loom.gen.assembly.tokens import KEYWORD_MAP
from loom.gen.support.generated_file import line_comment_header

COPYRIGHT = """\
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

GENERATED_HEADER = COPYRIGHT + "\n" + "\n".join(line_comment_header("//", generator="loom.gen.ops.c_tables")) + "\n// clang-format off"

_IR_KIND_MAP: dict[str, str] = {
    "tile": "LOOM_TYPE_TILE",
    "tensor": "LOOM_TYPE_TENSOR",
    "vector": "LOOM_TYPE_VECTOR",
    "view": "LOOM_TYPE_VIEW",
    "buffer": "LOOM_TYPE_BUFFER",
    "storage": "LOOM_TYPE_STORAGE",
    "pool": "LOOM_TYPE_POOL",
    "group": "LOOM_TYPE_GROUP",
    "dialect": "LOOM_TYPE_DIALECT",
}

_C_SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")


def _contract_family_mask(contracts: Sequence[ContractFamily]) -> str:
    """Returns a stable C bitmask expression for contract families."""

    unique_contracts = set(contracts)
    if len(unique_contracts) != len(contracts):
        duplicate_names = sorted(family.name for family in contracts if contracts.count(family) > 1)
        raise ValueError(f"duplicate contract families in semantic metadata: {duplicate_names}")
    ordered_contracts = [family for family in ContractFamily if family in unique_contracts]
    if not ordered_contracts:
        return "0"
    return " | ".join(f"LOOM_CONTRACT_{family.name}" for family in ordered_contracts)


def _type_semantic_c_name(semantic: TypeSemantic) -> str:
    """Returns the C enum for a type semantic role."""

    return semantic.c_name


def _emit_type_semantics(lines: list[str], type_def: TypeDef) -> None:
    """Appends a sparse initializer for one type semantic metadata row."""

    semantic = _type_semantic_c_name(type_def.semantic)
    contract_families = _contract_family_mask(type_def.contracts)
    if semantic == "LOOM_TYPE_SEMANTIC_ORDINARY" and contract_families == "0":
        return
    lines.append("    .semantics = {")
    if semantic != "LOOM_TYPE_SEMANTIC_ORDINARY":
        lines.append(f"        .semantic = {semantic},")
    if contract_families != "0":
        lines.append(f"        .contract_families = {contract_families},")
    lines.append("    },")


def _type_c_ident(name: str) -> str:
    """Convert a type name to a C identifier fragment."""
    return name.replace(".", "_")


def _type_fact_domain_symbol(type_def: Any) -> str | None:
    """Returns the validated C fact-domain symbol for a TypeDef, if any."""
    fact_domain = getattr(type_def, "fact_domain", None)
    if fact_domain is None:
        return None
    if not isinstance(fact_domain, str) or not _C_SYMBOL_RE.fullmatch(fact_domain):
        raise ValueError(f"TypeDef {type_def.name!r}: fact_domain must be a C symbol name, got {fact_domain!r}")
    return fact_domain


def _translate_type_format_elements(
    type_def: Any,
) -> list[str]:
    """Translate a TypeDef's format elements to C initializer strings."""
    from loom.assembly import (
        Attr,
        Clause,
        EncodingOf,
        Glue,
        Keyword,
        OptionalGroup,
        ScalarOf,
        ShapeOf,
        TypeOf,
    )

    param_names = [p.name for p in type_def.params]

    def param_index(field: str) -> int:
        return param_names.index(field)

    def translate(element: Any) -> list[str]:
        match element:
            case ShapeOf(field=field):
                return [f"{{LOOM_TYPE_FMT_SHAPE, {param_index(field)}, 0}}"]
            case ScalarOf(field=field):
                return [f"{{LOOM_TYPE_FMT_SCALAR, {param_index(field)}, 0}}"]
            case EncodingOf(field=field):
                return [f"{{LOOM_TYPE_FMT_ENCODING, {param_index(field)}, 0}}"]
            case TypeOf(field=field):
                return [f"{{LOOM_TYPE_FMT_TYPE, {param_index(field)}, 0}}"]
            case Attr(field=field):
                return [f"{{LOOM_TYPE_FMT_ATTR, {param_index(field)}, 0}}"]
            case Glue():
                return ["{LOOM_TYPE_FMT_GLUE, 0, 0}"]
            case Keyword(text=text):
                kw_name = KEYWORD_MAP.get(text)
                if kw_name is None:
                    raise ValueError(f"unknown keyword in type format: {text!r}")
                return [f"{{LOOM_TYPE_FMT_KEYWORD, 0, {kw_name}}}"]
            case Clause(name=name, elements=elements):
                kw_name = KEYWORD_MAP.get(name)
                if kw_name is None:
                    raise ValueError(f"unknown clause name in type format: {name!r}")
                result = [
                    f"{{LOOM_TYPE_FMT_KEYWORD, 0, {kw_name}}}",
                    "{LOOM_TYPE_FMT_GLUE, 0, 0}",
                    f"{{LOOM_TYPE_FMT_KEYWORD, 0, {KEYWORD_MAP['(']}}}",
                ]
                for e in elements:
                    result.extend(translate(e))
                result.append(f"{{LOOM_TYPE_FMT_KEYWORD, 0, {KEYWORD_MAP[')']}}}")
                return result
            case OptionalGroup(elements=elements, anchor=anchor):
                anchor_idx = param_index(anchor)
                inner = []
                for e in elements:
                    inner.extend(translate(e))
                skip_count = len(inner)
                data = f"({skip_count} << 8) | {anchor_idx}"
                result = [f"{{LOOM_TYPE_FMT_OPTIONAL, {anchor_idx}, {data}}}"]
                result.extend(inner)
                return result
            case _:
                raise ValueError(f"unsupported type format element: {element!r}")

    elements: list[str] = []
    for element in type_def.format:
        elements.extend(translate(element))
    return elements


def _type_builtin_name_rows(all_types: Sequence[Any]) -> dict[str, str]:
    """Returns sparse loom_type_kind_t name rows for built-in registry types."""

    rows: dict[str, str] = {}
    for type_def in all_types:
        if type_def.ir_kind == "dialect":
            continue
        kind = _IR_KIND_MAP[type_def.ir_kind]
        previous = rows.get(kind)
        if previous is not None:
            raise ValueError(f"Type kind {type_def.ir_kind!r} has duplicate registry names {previous!r} and {type_def.name!r}")
        rows[kind] = type_def.name
    return rows


def _emit_tables_header() -> str:
    guard = "LOOM_OPS_TYPE_REGISTRY_TABLES_H_"
    lines = [
        GENERATED_HEADER,
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        '#include "loom/ops/type_registry.h"',
        "",
        "extern const iree_string_view_t",
        "    loom_type_registry_builtin_names[LOOM_TYPE_COUNT_];",
        "",
        "extern const loom_type_registry_entry_t",
        "    loom_type_registry_entries_storage[];",
        "extern const iree_host_size_t loom_type_registry_entry_count;",
        "",
        f"#endif  // {guard}",
        "",
    ]
    return "\n".join(lines)


def generate_type_registry(
    all_types: list[Any],
) -> tuple[str, str, str]:
    """Generate type_registry.h, type_registry_tables.h, and type_registry_tables.c."""
    header = [GENERATED_HEADER]
    header.append("#ifndef LOOM_OPS_TYPE_REGISTRY_H_")
    header.append("#define LOOM_OPS_TYPE_REGISTRY_H_")
    header.append("")
    header.append('#include "iree/base/api.h"')
    header.append('#include "loom/ir/types.h"')
    header.append('#include "loom/ops/op_defs.h"')
    header.append("")
    header.append("#ifdef __cplusplus")
    header.append('extern "C" {')
    header.append("#endif")
    header.append("")
    header.append("typedef struct loom_value_fact_domain_t loom_value_fact_domain_t;")
    header.append("")
    header.append("// Format element kinds for type interiors (inside <...>).")
    header.append("// These are separate from op format elements because type")
    header.append("// interiors have different semantics (shape dims, element")
    header.append("// types, encodings) than op bodies (operand refs, attr values).")
    header.append("typedef enum loom_type_format_kind_e {")
    header.append("  LOOM_TYPE_FMT_SHAPE = 0,      // Dimension list: 4x[%M]x...")
    header.append("  LOOM_TYPE_FMT_SCALAR = 1,      // Element type keyword: f32, i8.")
    header.append("  LOOM_TYPE_FMT_ENCODING = 2,    // Encoding ref: #q8_0 or %enc.")
    header.append("  LOOM_TYPE_FMT_TYPE = 3,         // Recursive type: vm.ref<T>.")
    header.append("  LOOM_TYPE_FMT_ATTR = 4,         // Bare identifier: group<workgroup>.")
    header.append("  LOOM_TYPE_FMT_KEYWORD = 5,      // Literal punctuation/word.")
    header.append("  LOOM_TYPE_FMT_OPTIONAL = 6,     // Conditional elements.")
    header.append("  LOOM_TYPE_FMT_GLUE = 7,         // Suppress space.")
    header.append("} loom_type_format_kind_t;")
    header.append("")
    header.append("// A 4-byte format element for type interiors. Same layout")
    header.append("// as loom_format_element_t for consistent handling.")
    header.append("typedef struct loom_type_format_element_t {")
    header.append("  // Format opcode, encoded as loom_type_format_kind_t.")
    header.append("  uint8_t kind;")
    header.append("")
    header.append("  // Parameter index consumed by this element.")
    header.append("  uint8_t field_index;")
    header.append("")
    header.append("  // Kind-specific payload such as a keyword ID or skip count.")
    header.append("  uint16_t data;")
    header.append("} loom_type_format_element_t;")
    header.append("")
    header.append("// Descriptor for a registered type. Contains the name,")
    header.append("// the IR type kind to construct, parameter count, and")
    header.append("// format elements describing the type interior syntax.")
    header.append("typedef struct loom_type_descriptor_t {")
    header.append('  // B-string name: [length]"tile", [length]"hal.buffer".')
    header.append("  const uint8_t* name;")
    header.append("")
    header.append("  // What IR type kind to construct when parsing.")
    header.append("  loom_type_kind_t ir_kind;")
    header.append("")
    header.append("  // Number of declared parameters.")
    header.append("  uint8_t param_count;")
    header.append("")
    header.append("  // Optional type-owned value fact domain. NULL means the type only has generic")
    header.append("  // scalar facts or uses the domain-free extension behavior.")
    header.append("  const loom_value_fact_domain_t* fact_domain;")
    header.append("")
    header.append("  // Semantic role and target-contract families for this type.")
    header.append("  loom_type_semantics_t semantics;")
    header.append("")
    header.append("  // Format element array for the type interior (inside <...>).")
    header.append("  // NULL for opaque types (no angle brackets).")
    header.append("  const loom_type_format_element_t* format_elements;")
    header.append("")
    header.append("  // Number of entries in |format_elements|.")
    header.append("  uint8_t format_element_count;")
    header.append("} loom_type_descriptor_t;")
    header.append("")
    header.append("// Entry in the sorted type registry.")
    header.append("typedef struct loom_type_registry_entry_t {")
    header.append("  iree_string_view_t name;")
    header.append("  const loom_type_descriptor_t* descriptor;")
    header.append("} loom_type_registry_entry_t;")
    header.append("")
    header.append("// Returns the number of entries in the global type registry.")
    header.append("iree_host_size_t loom_type_registry_count(void);")
    header.append("")
    header.append("// Returns the sorted registry array (for iteration/testing).")
    header.append("const loom_type_registry_entry_t* loom_type_registry_entries(void);")
    header.append("")
    header.append('// Looks up a type descriptor by name (e.g., "tile", "hal.buffer").')
    header.append("// Returns the descriptor on success, NULL if not found.")
    header.append("const loom_type_descriptor_t* loom_type_registry_lookup(")
    header.append("    iree_string_view_t name);")
    header.append("")
    header.append("// Resolves the type-owned value fact domain for |type|, or NULL if the")
    header.append("// registered type has no extension fact domain.")
    header.append("const loom_value_fact_domain_t* loom_type_registry_resolve_fact_domain(")
    header.append("    void* user_data, const loom_fact_context_t* context,")
    header.append("    const loom_module_t* module, loom_type_t type);")
    header.append("")
    header.append("// Installs the generated type-registry fact-domain resolver on |context|.")
    header.append("void loom_type_registry_configure_fact_context(")
    header.append("    loom_fact_context_t* context);")
    header.append("")
    header.append("#ifdef __cplusplus")
    header.append("}")
    header.append("#endif")
    header.append("")
    header.append("#endif  // LOOM_OPS_TYPE_REGISTRY_H_")
    header.append("")

    tables_header = _emit_tables_header()

    source = [GENERATED_HEADER]
    source.append('#include "loom/ops/type_registry_tables.h"')
    source.append("")

    for type_def in all_types:
        ident = _type_c_ident(type_def.name)
        name_len = len(type_def.name)
        source.append(f'static const uint8_t loom_type_{ident}_name[] = "\\x{name_len:02x}" "{type_def.name}";')

    source.append("")

    fact_domain_symbols = sorted({fact_domain for type_def in all_types if (fact_domain := _type_fact_domain_symbol(type_def)) is not None})
    if fact_domain_symbols:
        source.extend(f"extern const loom_value_fact_domain_t {fact_domain};" for fact_domain in fact_domain_symbols)
        source.append("")

    for type_def in all_types:
        if not type_def.format:
            continue
        ident = _type_c_ident(type_def.name)
        elements = _translate_type_format_elements(type_def)
        source.append(f"static const loom_type_format_element_t loom_type_{ident}_format[] = {{")
        source.extend(f"    {element}," for element in elements)
        source.append("};")

    source.append("")

    for type_def in all_types:
        ident = _type_c_ident(type_def.name)
        ir_kind = _IR_KIND_MAP[type_def.ir_kind]
        param_count = len(type_def.params)
        fact_domain = _type_fact_domain_symbol(type_def)
        if type_def.format:
            elements = _translate_type_format_elements(type_def)
            format_ref = f"loom_type_{ident}_format"
            format_count = len(elements)
        else:
            format_ref = "NULL"
            format_count = 0
        source.append(f"static const loom_type_descriptor_t loom_type_{ident}_descriptor = {{")
        source.append(f"    .name = loom_type_{ident}_name,")
        source.append(f"    .ir_kind = {ir_kind},")
        if param_count != 0:
            source.append(f"    .param_count = {param_count},")
        if fact_domain:
            source.append(f"    .fact_domain = &{fact_domain},")
        _emit_type_semantics(source, type_def)
        if format_ref != "NULL":
            source.append(f"    .format_elements = {format_ref},")
            source.append(f"    .format_element_count = {format_count},")
        source.append("};")
        source.append("")

    sorted_types = sorted(all_types, key=lambda type_def: type_def.name)
    source.append("const loom_type_registry_entry_t")
    source.append("    loom_type_registry_entries_storage[] = {")
    for type_def in sorted_types:
        ident = _type_c_ident(type_def.name)
        source.append(f'    {{IREE_SVL("{type_def.name}"), &loom_type_{ident}_descriptor}},')
    source.append("};")
    source.append("")
    count = len(sorted_types)
    source.append("const iree_host_size_t loom_type_registry_entry_count =")
    source.append(f"    {count};")
    source.append("")
    source.append("const iree_string_view_t")
    source.append("    loom_type_registry_builtin_names[LOOM_TYPE_COUNT_] = {")
    for kind, name in sorted(_type_builtin_name_rows(all_types).items()):
        source.append(f'    [{kind}] = IREE_SVL("{name}"),')
    source.append("};")
    source.append("")

    return "\n".join(header), tables_header, "\n".join(source)
