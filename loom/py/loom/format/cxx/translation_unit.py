# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C++ translation-unit declarations required by a rendered kernel."""

from dataclasses import dataclass

from loom.format.cxx.types import SCALAR_CPP_TYPES
from loom.ir import Operation


@dataclass(frozen=True, slots=True)
class TargetInfo:
    """Source target record required by the rendered kernel."""

    operation: Operation
    operation_name: str
    kind: str
    subgroup_size: int | None
    features: tuple[tuple[str, bool], ...]


def render_preamble(
    *,
    operation_names: set[str],
    scalar_names: set[str],
    vector_aliases: dict[tuple[str, tuple[int, ...], bool], str],
    record_declarations: dict[str, tuple[tuple[str, str], ...]],
    table_lookup_declarations: set[tuple[str, str, str, str]],
    targets: tuple[tuple[str, TargetInfo], ...],
    uses_numeric_conversion: bool,
) -> list[str]:
    """Renders includes, helper declarations, and source target metadata."""

    headers = {"loomcxx/kernel.h"}
    if targets:
        headers.add("loomcxx/target/amdgpu.h")
    if any(name.startswith("view.atomic.") for name in operation_names):
        headers.add("loomcxx/atomic.h")
    if "scalar.siluf" in operation_names:
        headers.add("loomcxx/scalar.h")
    if "index.assume" in operation_names:
        headers.add("loomcxx/predicate.h")
    if "encoding.define" in operation_names:
        headers.add("loomcxx/encoding.h")
    if uses_numeric_conversion:
        headers.add("loomcxx/numeric.h")
    if operation_names & {
        "kernel.subgroup.reduce",
        "vector.absf",
        "vector.absi",
        "vector.minimumf",
        "vector.maximumf",
        "vector.minnumf",
        "vector.maxnumf",
        "vector.dot2f",
        "vector.decode",
        "vector.fragment",
        "vector.load",
        "vector.mma",
        "vector.reduce",
        "vector.slice",
        "vector.store",
        "vector.table.lookup",
    }:
        headers.add("loomcxx/vector.h")
    if scalar_names & {"f8E4M3", "f8E5M2"}:
        headers.add("loomcxx/numeric.h")
    system_headers = {"stdfloat"} if scalar_names & {"f16", "bf16"} else set()
    preamble = [*(f"#include <{header}>" for header in sorted(headers)), ""]
    if system_headers:
        preamble.extend(
            [*(f"#include <{header}>" for header in sorted(system_headers)), ""]
        )

    for (element, dims, signed), alias in sorted(
        vector_aliases.items(), key=lambda item: item[1]
    ):
        base = SCALAR_CPP_TYPES[element]
        if element.startswith("i") and not signed:
            base = {
                "i8": "unsigned char",
                "i16": "unsigned short",
                "i32": "unsigned",
                "i64": "unsigned long long",
                "i1": "bool",
            }[element]
        preamble.append(
            f"using {alias} = {base} __attribute__((ext_vector_type({dims[0]})));"
        )
    if vector_aliases:
        preamble.append("")

    for record_name, fields in record_declarations.items():
        preamble.append(f"struct {record_name} {{")
        preamble.extend(
            f"  {field_type} {field_name};" for field_name, field_type in fields
        )
        preamble.append("};")
        preamble.append("")

    for function, result, table, indices in sorted(table_lookup_declarations):
        preamble.append('[[loom::op("vector.table.lookup")]]')
        preamble.append(f"{result} {function}({table} table, {indices} indices);")
        preamble.append("")

    for target_name, target in targets:
        target_fields = [f'.kind = "{target.kind}"']
        if target.subgroup_size is not None:
            target_fields.append(f".subgroup_size = {target.subgroup_size}")
        if target.features:
            target_fields.append(".features = {")
            target_fields.extend(
                f"  .{name} = loom::target::feature::"
                f"{'enabled' if enabled else 'disabled'},"
                for name, enabled in target.features
            )
            target_fields.append("}")
        if len(target_fields) == 1:
            preamble.append(
                f"constexpr loom::amdgpu::target {target_name}{{{target_fields[0]}}};"
            )
        else:
            preamble.append(f"constexpr loom::amdgpu::target {target_name}{{")
            for field in target_fields:
                suffix = "" if field.endswith(("{", ",")) else ","
                preamble.append(f"  {field}{suffix}")
            preamble.append("};")
        preamble.append("")
    return preamble
