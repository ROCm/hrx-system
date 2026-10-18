# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates typed C++ declarations and native scalar bindings together."""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Sequence
from pathlib import Path

from loom.dialect.scalar import FastMathFlags, TrigonometryMathFlags
from loom.dsl import ATTR_TYPE_FLAGS, FLOAT, INTEGER, Op
from loom.gen import bootstrap
from loom.gen.ops.model import load_dialect_generation
from loom.gen.support.files import write_text_file
from loom.gen.support.generated_file import GeneratedFileSet, line_comment_header, maintain_generated_file_set

_GENERATOR = "loom.gen.cxx.intrinsics"
DESCRIPTION = "Loom C++ scalar declarations"
REGENERATE_COMMAND = "python3 loom/py/loom/gen/run.py cxx_intrinsics --in-place"

# Integer bit counts preserve the input width and have no source permissions.
_INTEGER_PROJECTIONS = frozenset(("scalar.ctlzi", "scalar.cttzi", "scalar.ctpopi"))


def validate_scalar_projection(op: Op) -> None:
    """Checks the complete contract represented by the homogeneous projection."""
    if op.regions or op.successors or len(op.results) != 1 or not op.operands:
        raise ValueError(f"{op.name}: requires fixed operands and one scalar result")
    fields = (*op.operands, *op.results)
    category = op.results[0].type_constraint
    if category not in (FLOAT, INTEGER) or any(field.variadic or field.type_constraint != category for field in fields) or any(operand.optional for operand in op.operands):
        raise ValueError(f"{op.name}: requires fixed values in one scalar category")
    if (category == INTEGER and op.attrs) or any(
        attr.attr_type != ATTR_TYPE_FLAGS or (attr.name, attr.enum_def) not in (("fastmath", FastMathFlags), ("math", TrigonometryMathFlags)) for attr in op.attrs
    ):
        raise ValueError(f"{op.name}: requires an explicit attribute projection")
    if not any(trait.name == "Pure" for trait in op.traits):
        raise ValueError(f"{op.name}: requires a pure scalar operation")
    names = {field.name for field in fields}
    if not op.constraints or any(constraint.name != "SameType" or set(constraint.args) != names for constraint in op.constraints):
        raise ValueError(f"{op.name}: requires one homogeneous type relationship")


def scalar_projections() -> tuple[Op, ...]:
    """Selects homogeneous floating operations and integer bit counts, without
    inventing wrappers for regions, shaped types, or semantic attributes.
    """
    selected = []
    for op in load_dialect_generation("scalar").ops:
        if op.name in _INTEGER_PROJECTIONS:
            validate_scalar_projection(op)
            selected.append(op)
            continue
        if len(op.results) != 1 or not op.operands:
            continue
        if any(field.type_constraint != FLOAT for field in (*op.operands, *op.results)):
            continue
        if any(attr.name not in ("fastmath", "math") for attr in op.attrs):
            continue
        names = {field.name for field in (*op.operands, *op.results)}
        if not any(constraint.name == "SameType" and set(constraint.args) == names for constraint in op.constraints):
            continue
        validate_scalar_projection(op)
        selected.append(op)
    return tuple(sorted(selected, key=lambda op: op.name))


def generate_header(ops: Sequence[Op]) -> str:
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator=_GENERATOR, regenerate=REGENERATE_COMMAND),
        "// clang-format off",
        "#ifndef LOOMCXX_SCALAR_H_",
        "#define LOOMCXX_SCALAR_H_",
        "",
        "// Fixed scalar projections of the canonical Loom operation contracts.",
        "// The approximate namespace explicitly permits AFN and no other flags.",
        "// Its preserved trigonometry retains source-defined zero signs even",
        "// when the calling scope permits NSZ. Nonzero results remain approximate.",
        "namespace loom::scalar {",
    ]
    for namespace in (None, "approximate", "approximate::preserved"):
        if namespace:
            lines.extend(["", f"namespace {namespace} {{"])
        for op in ops:
            if namespace and not op.attrs:
                continue
            if namespace == "approximate::preserved" and op.attrs[0].enum_def != TrigonometryMathFlags:
                continue
            name = op.name.removeprefix("scalar.")
            flags = ', "afn"' if namespace else ""
            if namespace == "approximate::preserved":
                flags += ', "preserve"'
            lines.append("")
            lines.extend("// " + line for line in textwrap.wrap(op.doc, 76))
            if op.results[0].type_constraint == INTEGER:
                parameter = "Integer"
                constraint = "__is_integral(Integer) && !__is_same(Integer, bool)"
            else:
                parameter = "Float"
                constraint = "__is_floating_point(Float)"
            arguments = ", ".join(f"{parameter} {operand.name}" for operand in op.operands)
            lines.append(f"template <class {parameter}> requires ({constraint})")
            lines.append(f'[[loom::op("{op.name}"{flags})]] {parameter} {name}({arguments});')
        if namespace:
            lines.extend(["", f"}}  // namespace {namespace}"])
    lines.extend(["", "}  // namespace loom::scalar", "", "#endif  // LOOMCXX_SCALAR_H_", ""])
    return "\n".join(lines)


def generate_source(ops: Sequence[Op]) -> str:
    lines = [
        *line_comment_header("//", generator=_GENERATOR),
        '#include "loom/import/cxx/binding/scalar_bindings.h"',
        '#include "loom/ops/scalar/ops.h"',
        "",
    ]
    for op in ops:
        name = op.name.replace(".", "_")
        arguments = ", ".join(f"operands[{index}]" for index in range(len(op.operands)))
        flags = "flags, " if op.attrs else ""
        lines.extend(
            [
                f"static iree_status_t {name}(loom_builder_t* builder, uint8_t flags,",
                "    const loom_value_id_t* operands, loom_type_t result_type,",
                "    loom_location_id_t location, loom_op_t** out_op) {",
            ]
        )
        if not op.attrs:
            lines.append("  (void)flags;")
        lines.extend(
            [
                f"  return loom_{name}_build(builder, {flags}{arguments}, result_type, location, out_op);",
                "}",
                "",
            ]
        )
    lines.append("static const loom_cxx_scalar_binding_t bindings[] = {")
    for op in ops:
        category = "INTEGER" if op.results[0].type_constraint == INTEGER else "FLOAT"
        preserve_flag = next((case.value for attr in op.attrs for case in attr.enum_def.cases if case.keyword == "preserve"), 0)
        lines.append(f'    {{"{op.name}", LOOM_CXX_SCALAR_CATEGORY_{category}, {len(op.operands)}, {str(bool(op.attrs)).lower()}, {preserve_flag}, {op.name.replace(".", "_")}}},')
    lines.extend(
        [
            "};",
            "",
            "const loom_cxx_scalar_binding_t* loom_cxx_scalar_binding_find(iree_string_view_t name) {",
            "  iree_host_size_t first = 0, count = IREE_ARRAYSIZE(bindings);",
            "  while (count) {",
            "    iree_host_size_t step = count / 2, position = first + step;",
            "    int order = iree_string_view_compare(name, iree_make_cstring_view(bindings[position].name));",
            "    if (!order) return &bindings[position];",
            "    if (order < 0) { count = step; }",
            "    else { first = position + 1; count -= step + 1; }",
            "  }",
            "  return NULL;",
            "}",
            "",
            "bool loom_cxx_fastmath_flag_parse(iree_string_view_t name, uint8_t* out_flag) {",
        ]
    )
    lines.extend(f'  if (iree_string_view_equal(name, IREE_SV("{case.keyword}"))) {{ *out_flag = {case.value}; return true; }}' for case in FastMathFlags.cases)
    lines.extend(["  return false;", "}", ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--header", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--in-place", action="store_true")
    args = parser.parse_args(argv)
    if args.check or args.in_place:
        if args.source or args.header:
            parser.error("maintenance modes do not accept output paths")
        result = maintain_generated_file_set(
            bootstrap.find_repo_root(), checked_in_file_set(), mode="update" if args.in_place else "check", description=DESCRIPTION, regenerate_command=REGENERATE_COMMAND
        )
        return 0 if result.ok else 1
    if not args.source and not args.header:
        parser.error("select an output path or a maintenance mode")
    ops = scalar_projections()
    if args.source:
        write_text_file(args.source, generate_source(ops))
    if args.header:
        write_text_file(args.header, generate_header(ops))
    return 0


def checked_in_file_set() -> GeneratedFileSet:
    return GeneratedFileSet.from_mapping(
        {
            "loom/src/loom/import/cxx/include/loomcxx/scalar.h": generate_header(scalar_projections()),
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
