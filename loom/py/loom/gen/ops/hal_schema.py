# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""HAL semantic schema rows shared by verification and target projections."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from loom.dialect.hal import ALL_HAL_TYPES
from loom.dialect.hal.methods import (
    ALL_HAL_METHODS,
    Method,
    Optional,
    Range,
    Record,
    Shape,
    Value,
)
from loom.gen.support.files import write_text_file
from loom.gen.support.generated_file import GeneratedFileSet
from loom.ir import ScalarTypeKind, scalar_type_name

_HEADER = """// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.hal_schema.
// clang-format off
"""


def children(shape: Shape) -> tuple[Shape, ...]:
    match shape:
        case Record(fields=fields):
            return tuple(field.value for field in fields)
        case Range(element=element) | Optional(value=element):
            return (element,)
        case Value():
            return ()


def nodes(methods: tuple[Method, ...] = ALL_HAL_METHODS) -> tuple[list[Shape], dict[Shape, int]]:
    """Checks declarations once before emitting trusted C metadata."""
    ordered: list[Shape] = []
    ordinals: dict[Shape, int] = {}
    records: dict[str, Record] = {}
    types = {definition.name for definition in ALL_HAL_TYPES}
    types.update(scalar_type_name(kind) for kind in ScalarTypeKind if kind.name not in ("INDEX", "OFFSET"))

    def identifier(name: str) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"invalid HAL identifier: {name!r}")

    def add(shape: Shape) -> None:
        if shape in ordinals:
            return
        match shape:
            case Value(type=name):
                if name not in types:
                    raise ValueError(f"unsupported HAL value type: {name!r}")
            case Record(name=name, fields=fields):
                identifier(name)
                if name in records and records[name] != shape:
                    raise ValueError(f"conflicting HAL record name: {name!r}")
                records[name] = shape
                names = [field.name for field in fields]
                if len(set(names)) != len(names):
                    raise ValueError(f"duplicate field in HAL record: {name!r}")
                for field_name in names:
                    identifier(field_name)
            case Optional(value=Optional()):
                raise ValueError("nested HAL optionals have ambiguous source presence")
        for child in children(shape):
            add(child)
        ordinals[shape] = len(ordered)
        ordered.append(shape)

    if not methods or [method.name for method in methods] != sorted({method.name for method in methods}):
        raise ValueError("HAL methods must be nonempty and strictly name-sorted")
    for method in methods:
        for component in method.name.split("."):
            identifier(component)
        if type(method.version) is not int or not 0 <= method.version <= 0xFFFFFFFF:
            raise ValueError("HAL method version must fit u32")
        add(method.arguments)
    if len(ordered) > 65535 or sum(len(children(shape)) for shape in ordered) > 65535:
        raise ValueError("HAL schema exceeds the u16 node/edge domain")
    return ordered, ordinals


def source_tables() -> str:
    ordered, ordinals = nodes()
    edges: list[int] = []
    rows: list[str] = []
    for shape in ordered:
        child_shapes = children(shape)
        child_base = len(edges)
        edges.extend(ordinals[child] for child in child_shapes)
        kind = {
            Value: "VALUE",
            Record: "RECORD",
            Range: "RANGE",
            Optional: "OPTIONAL",
        }[type(shape)]
        scalar = "NONE"
        reference = ""
        if isinstance(shape, Value):
            if shape.type.startswith("hal."):
                reference = shape.type
            else:
                scalar = shape.type.upper()
        rows.append(f'  {{LOOM_HAL_SCHEMA_{kind}, LOOM_SCALAR_TYPE_{scalar}, {len(child_shapes)}, {child_base}, IREE_SVL("{reference}")}},')
    if len(edges) > 65535:
        raise ValueError("HAL schema exceeds the u16 edge domain")
    method_rows = [f'  {{IREE_SVL("{method.name}"), {method.version}, {ordinals[method.arguments]}}},' for method in ALL_HAL_METHODS]
    return (
        _HEADER
        + '\n#include "loom/ops/hal/schema.h"\n\n'
        + "const uint16_t loom_hal_schema_children[] = {\n  "
        + ", ".join(str(edge) for edge in edges)
        + "\n};\n\nconst loom_hal_schema_node_t loom_hal_schema_nodes[] = {\n"
        + "\n".join(rows)
        + "\n};\n\nconst loom_hal_method_t loom_hal_methods[] = {\n"
        + "\n".join(method_rows)
        + "\n};\n"
    )


def native_model(methods: tuple[Method, ...] = ALL_HAL_METHODS):
    """Projects semantic grouping into fixed VM records, before compilation."""
    ordered, _ = nodes(methods)
    layouts: list[tuple[tuple[str, int], ...]] = []
    layout_ordinals: dict[tuple[tuple[str, int], ...], int] = {}
    references = {definition.name: i for i, definition in enumerate(ALL_HAL_TYPES)}

    def fields(shape: Shape) -> tuple[tuple[str, int], ...]:
        match shape:
            case Value(type=name):
                if name in references:
                    return (("IREE_VM_NATIVE_CALL_FIELD_REF", references[name]),)
                # Match the VM's ordinary scalar carriers (including i1).
                carrier = {"i1": "I32", "f8E4M3": "F8E4M3FN"}.get(name, name.upper())
                return ((f"IREE_VM_SCALAR_TYPE_{carrier}", 0),)
            case Record(fields=members):
                return tuple(field for member in members for field in fields(member.value))
            case Range(element=element) | Optional(value=element):
                return (("IREE_VM_NATIVE_CALL_FIELD_RANGE", layout(element)),)

    def layout(shape: Shape) -> int:
        row = fields(shape)
        if row not in layout_ordinals:
            layout_ordinals[row] = len(layouts)
            layouts.append(row)
        return layout_ordinals[row]

    roots = [layout(method.arguments) for method in methods]
    projection = []
    for shape in ordered:
        row = fields(shape)
        width = sum(2 if kind == "IREE_VM_NATIVE_CALL_FIELD_RANGE" else 1 for kind, _ in row)
        child = row[0][1] if isinstance(shape, (Range, Optional)) else 0
        if width > 65535:
            raise ValueError("HAL fixed record width exceeds the u16 domain")
        projection.append((width, child))
    depths: list[int] = []
    for row in layouts:
        depths.append(max((depths[child] + 1 for kind, child in row if kind == "IREE_VM_NATIVE_CALL_FIELD_RANGE"), default=0))
    return layouts, roots, depths, projection


def native_header() -> str:
    layouts, roots, depths, projection = native_model()
    ordered, ordinals = nodes()
    record_layouts = {method.arguments: root for method, root in zip(ALL_HAL_METHODS, roots, strict=True)}
    for shape, (_, child_layout) in zip(ordered, projection, strict=True):
        if isinstance(shape, (Range, Optional)) and isinstance(children(shape)[0], Record):
            record_layouts[children(shape)[0]] = child_layout

    def has_optional(shape: Shape) -> bool:
        return isinstance(shape, Optional) or any(has_optional(child) for child in children(shape))

    edges: list[int] = []
    node_rows: list[str] = []
    for shape, (width, _) in zip(ordered, projection, strict=True):
        child_shapes = children(shape)
        base = len(edges)
        edges.extend(ordinals[child] for child in child_shapes)
        kind = {Value: "VALUE", Record: "RECORD", Range: "RANGE", Optional: "OPTIONAL"}[type(shape)]
        node_rows.append(f"  {{IREE_HAL_VM_NATIVE_{kind}, {int(has_optional(shape))}, {len(child_shapes)}, {base}, {width}}},")
    constants = [f"  IREE_HAL_NATIVE_METHOD_COUNT = {len(ALL_HAL_METHODS)},"]
    for index, definition in enumerate(ALL_HAL_TYPES):
        constants.append(f"  IREE_HAL_NATIVE_REF_{definition.name.removeprefix('hal.').upper()} = {index},")
    for index, method in enumerate(ALL_HAL_METHODS):
        prefix = "IREE_HAL_NATIVE_" + method.name.upper().replace(".", "_")
        constants.extend((f"  {prefix} = {index},", f"  {prefix}_VERSION = {method.version},", f"  {prefix}_ROOT = {ordinals[method.arguments]},"))

    def field_offsets(shape: Record, prefix: str, base: int = 0) -> None:
        for member in shape.fields:
            name = prefix + "_" + member.name.upper()
            if isinstance(member.value, Record):
                field_offsets(member.value, name, base)
            else:
                constants.append(f"  {name} = {base},")
            base += projection[ordinals[member.value]][0]

    for shape in ordered:
        if isinstance(shape, Record):
            prefix = "IREE_HAL_NATIVE_" + shape.name.upper()
            constants.append(f"  {prefix}_WORD_COUNT = {projection[ordinals[shape]][0]},")
            if shape in record_layouts:
                constants.append(f"  {prefix}_LAYOUT = {record_layouts[shape]},")
            field_offsets(shape, prefix)
    fields = [field for row in layouts for field in row]
    field_rows = [f"  {{{kind}, {ordinal}}}," for kind, ordinal in fields]
    layout_rows = []
    base = 0
    for row, depth in zip(layouts, depths, strict=True):
        width = sum(2 if kind == "IREE_VM_NATIVE_CALL_FIELD_RANGE" else 1 for kind, _ in row)
        layout_rows.append(f"  {{{base}, {width}, {len(row)}, {depth}}},")
        base += len(row)
    schema_rows = [f"  {{kIreeHalNativeLayouts, kIreeHalNativeFields, {len(layouts)}, {len(fields)}, {root}}}," for root in roots]
    return (
        _HEADER
        + "\n#ifndef IREE_MODULE_HAL_NATIVE_SCHEMA_H_\n#define IREE_MODULE_HAL_NATIVE_SCHEMA_H_\n"
        + '\n#include "iree/module/hal/native_abi.h"\n\n'
        + "enum {\n"
        + "\n".join(constants)
        + "\n};\n\n"
        + "static const iree_string_view_t kIreeHalNativeMethodNames[] = {\n"
        + "\n".join(f'  IREE_SVL("{method.name}"),' for method in ALL_HAL_METHODS)
        + "\n};\n\n"
        + "static const uint16_t kIreeHalNativeChildren[] = {\n  "
        + ", ".join(str(edge) for edge in edges)
        + "\n};\n\nstatic const iree_hal_vm_native_node_t kIreeHalNativeNodes[] = {\n"
        + "\n".join(node_rows)
        + "\n};\n\n"
        + "static const iree_vm_native_call_field_t kIreeHalNativeFields[] = {\n"
        + "\n".join(field_rows)
        + "\n};\n\nstatic const iree_vm_native_call_layout_t kIreeHalNativeLayouts[] = {\n"
        + "\n".join(layout_rows)
        + "\n};\n\nstatic const iree_vm_native_call_schema_t kIreeHalNativeSchemas[] = {\n"
        + "\n".join(schema_rows)
        + "\n};\n\n#endif  // IREE_MODULE_HAL_NATIVE_SCHEMA_H_\n"
    )


def vm_projection() -> str:
    layouts, _, _, projection = native_model()
    rows = [f"  {width}," for width, _ in projection]
    names = [f'  IREE_SVL("{definition.name}"),' for definition in ALL_HAL_TYPES]
    scalars = [
        "LOOM_SCALAR_TYPE_" + kind.removeprefix("IREE_VM_SCALAR_TYPE_").removesuffix("FN") if kind.startswith("IREE_VM_SCALAR_TYPE_") else "LOOM_SCALAR_TYPE_NONE" for row in layouts for kind, _ in row
    ]

    def token_words(shape: Shape) -> int:
        match shape:
            case Optional(value=value):
                return 2 + token_words(value)
            case Range():
                return 2
            case Value():
                return 1
            case Record():
                return 0

    maximum_token_words = max(token_words(shape) for shape in nodes()[0])
    return (
        _HEADER
        + f"\nenum {{ LOOM_VM_HAL_MAXIMUM_TOKEN_WORDS = {maximum_token_words} }};\n"
        + "\nstatic const uint16_t kWordCounts[] = {\n"
        + "\n".join(rows)
        + "\n};\n\nstatic const iree_string_view_t kReferenceNames[] = {\n"
        + "\n".join(names)
        + "\n};\n\nstatic const loom_scalar_type_t kFieldScalarTypes[] = {\n  "
        + ", ".join(scalars)
        + "\n};\n"
    )


DESCRIPTION = "HAL native declarations"
REGENERATE_COMMAND = "python3 loom/py/loom/gen/run.py checked_in_artifacts --in-place"


def checked_in_file_set() -> GeneratedFileSet:
    return GeneratedFileSet.from_mapping(
        {
            "runtime/src/iree/module/hal/native_schema.h": native_header(),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--vm-projection", type=Path)
    args = parser.parse_args()
    if not args.source and not args.vm_projection:
        parser.error("at least one output is required")
    if args.source:
        write_text_file(args.source, source_tables())
    if args.vm_projection:
        write_text_file(args.vm_projection, vm_projection())


if __name__ == "__main__":
    main()
