# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C++ translation-unit declarations required by rendered callables."""

import json
from dataclasses import dataclass

from loom.format.cxx.errors import CxxRenderError
from loom.format.cxx.types import SCALAR_CPP_TYPES
from loom.ir import Operation, Utf8Attr


@dataclass(frozen=True, slots=True)
class TargetInfo:
    """Source target record required by a rendered callable."""

    operation: Operation
    operation_name: str
    kind: str | None
    subgroup_size: int | None
    features: tuple[tuple[str, bool], ...]


def _render_generic_target(
    target_name: str, target_type_name: str | None, target: TargetInfo
) -> list[str]:
    if target_type_name is None:
        raise CxxRenderError(
            target.operation, "generic target has no planned C++ type name"
        )
    field_names = [
        "kind",
        *sorted(
            name
            for name in target.operation.attributes
            if name not in {"kind", "symbol"}
        ),
    ]
    fields: list[tuple[str, str, str]] = []
    for field_name in field_names:
        value = target.operation.attributes[field_name]
        if isinstance(value, Utf8Attr):
            value = value.value
        if isinstance(value, str):
            fields.append((field_name, "const char*", json.dumps(value)))
        elif type(value) is int:
            fields.append((field_name, "long long", str(value)))
        elif type(value) is bool:
            fields.append((field_name, "bool", "true" if value else "false"))
        else:
            raise CxxRenderError(
                target.operation,
                f"target field {field_name!r} has no C++ record projection",
            )

    lines = [f'struct [[loom::target("target.generic")]] {target_type_name} {{']
    lines.extend(
        f"  {field_type} {field_name};" for field_name, field_type, _ in fields
    )
    lines.append("};")
    initializers = [f".{field_name} = {value}" for field_name, _, value in fields]
    if len(initializers) == 1:
        lines.append(
            f"constexpr {target_type_name} {target_name}{{{initializers[0]}}};"
        )
    else:
        lines.append(f"constexpr {target_type_name} {target_name}{{")
        lines.extend(f"  {initializer}," for initializer in initializers)
        lines.append("};")
    lines.append("")
    return lines


def _render_target_declaration(target_name: str, target: TargetInfo) -> list[str]:
    symbol = target.operation.attributes.get("symbol")
    if not isinstance(symbol, str):
        raise CxxRenderError(target.operation, "target declaration has no symbol name")
    return [
        f"[[loom::symbol({json.dumps(symbol)})]]",
        f"extern const loom::target::declaration {target_name};",
        "",
    ]


def _render_amdgpu_target(target_name: str, target: TargetInfo) -> list[str]:
    if target.kind is None:
        raise CxxRenderError(target.operation, "AMDGPU target has no selector")
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
        return [
            f"constexpr loom::amdgpu::target {target_name}{{{target_fields[0]}}};",
            "",
        ]

    lines = [f"constexpr loom::amdgpu::target {target_name}{{"]
    for field in target_fields:
        suffix = "" if field.endswith(("{", ",")) else ","
        lines.append(f"  {field}{suffix}")
    lines.extend(("};", ""))
    return lines


def _render_aie2p_target(target_name: str, target: TargetInfo) -> list[str]:
    if target.kind is None:
        raise CxxRenderError(target.operation, "AIE2P target has no selector")
    target_fields = [f".kind = {json.dumps(target.kind)}"]
    device_profile = target.operation.attributes.get("device_profile")
    if isinstance(device_profile, Utf8Attr):
        device_profile = device_profile.value
    if device_profile is not None:
        if not isinstance(device_profile, str):
            raise CxxRenderError(
                target.operation, "AIE2P device_profile is not a string"
            )
        target_fields.append(f".device_profile = {json.dumps(device_profile)}")
    if len(target_fields) == 1:
        return [
            f"constexpr loom::aie2p::target {target_name}{{{target_fields[0]}}};",
            "",
        ]
    lines = [f"constexpr loom::aie2p::target {target_name}{{"]
    lines.extend(f"  {field}," for field in target_fields)
    lines.extend(("};", ""))
    return lines


def _render_targets(
    targets: tuple[tuple[str, str | None, TargetInfo], ...],
) -> list[str]:
    lines: list[str] = []
    for target_name, target_type_name, target in targets:
        if target.operation_name == "target.decl":
            lines.extend(_render_target_declaration(target_name, target))
        elif target.operation_name == "target.generic":
            lines.extend(_render_generic_target(target_name, target_type_name, target))
        elif target.operation_name == "aie2p.target":
            lines.extend(_render_aie2p_target(target_name, target))
        else:
            lines.extend(_render_amdgpu_target(target_name, target))
    return lines


def render_preamble(
    *,
    operation_names: set[str],
    scalar_names: set[str],
    has_kernels: bool,
    uses_pipelines: bool,
    uses_buffers: bool,
    uses_views: bool,
    uses_encodings: bool,
    vector_aliases: dict[tuple[str, tuple[int, ...], bool], str],
    record_declarations: dict[str, tuple[tuple[str, str], ...]],
    table_lookup_declarations: set[tuple[str, str, str, str]],
    targets: tuple[tuple[str, str | None, TargetInfo], ...],
    uses_scalar_functions: bool,
    uses_numeric_conversion: bool,
    uses_predicate_functions: bool,
) -> list[str]:
    """Renders includes, helper declarations, and source target metadata."""

    headers: set[str] = set()
    uses_kernel_header = has_kernels or any(
        name.startswith("kernel.") for name in operation_names
    )
    if uses_kernel_header:
        headers.add("loomcxx/kernel.h")
    if uses_pipelines:
        headers.add("loomcxx/pipeline.h")
    if any(name.startswith("command.") for name in operation_names):
        headers.add("loomcxx/command.h")
    target_operations = {target.operation_name for _, _, target in targets}
    if "amdgpu.target" in target_operations:
        headers.add("loomcxx/target/amdgpu.h")
    if "aie2p.target" in target_operations:
        headers.add("loomcxx/target/aie2p.h")
    if target_operations & {"target.decl", "target.generic"}:
        headers.add("loomcxx/target.h")
    if any(name.startswith("view.atomic.") for name in operation_names):
        headers.add("loomcxx/atomic.h")
    if uses_scalar_functions:
        headers.add("loomcxx/scalar.h")
    if "index.assume" in operation_names or uses_predicate_functions:
        headers.add("loomcxx/predicate.h")
    if "encoding.define" in operation_names:
        headers.add("loomcxx/encoding.h")
    if uses_numeric_conversion:
        headers.add("loomcxx/numeric.h")
    uses_vector_header = bool(
        operation_names
        & {
            "kernel.subgroup.reduce",
            "vector.absf",
            "vector.absi",
            "vector.bitcast",
            "vector.concat",
            "vector.cmpf",
            "vector.cmpi",
            "vector.minimumf",
            "vector.maximumf",
            "vector.minsi",
            "vector.maxsi",
            "vector.minui",
            "vector.maxui",
            "vector.minnumf",
            "vector.maxnumf",
            "vector.dot2f",
            "vector.dotf",
            "vector.decode",
            "vector.fragment",
            "vector.fragment.load",
            "vector.fragment.store",
            "vector.interleave",
            "vector.iota",
            "vector.load",
            "vector.mma",
            "vector.reduce",
            "vector.select",
            "vector.slice",
            "vector.store",
            "vector.table.lookup",
            "vector.transpose",
            "vector.extract",
        }
    )
    if uses_vector_header:
        headers.add("loomcxx/vector.h")
    elif not uses_kernel_header and not uses_pipelines:
        if uses_views:
            headers.add("loomcxx/view.h")
        elif uses_buffers:
            headers.add("loomcxx/buffer.h")
        elif scalar_names & {"index", "offset"}:
            headers.add("loomcxx/index.h")
    if uses_encodings and headers.isdisjoint(
        {
            "loomcxx/command.h",
            "loomcxx/kernel.h",
            "loomcxx/pipeline.h",
            "loomcxx/vector.h",
            "loomcxx/view.h",
        }
    ):
        headers.add("loomcxx/encoding_type.h")
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
        if len(dims) == 1 and element != "i1":
            definition = f"{base} __attribute__((ext_vector_type({dims[0]})))"
        else:
            dimensions = ", ".join(str(dimension) for dimension in dims)
            definition = f"loom::type::vector<loom::type::shape<{dimensions}>, {base}>"
        preamble.append(f"using {alias} = {definition};")
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

    preamble.extend(_render_targets(targets))
    return preamble
