# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Preserves memory observations and their operands through public C++ import."""

import subprocess
import sys

from loom.fields import compute_layout, resolve_fields
from loom.format.cxx import load_module, render_module
from loom.format.cxx.registry import source_ops
from loom.format.cxx.types import SCALAR_CPP_TYPES
from loom.ir import Module, ShapedType, StaticDim


def _program() -> str:
    functions = [
        "#include <loomcxx/numeric.h>\n#include <loomcxx/vector.h>\n"
        "#include <stdfloat>\nusing Access = loom::memory_access;"
    ]
    for element, source_type in SCALAR_CPP_TYPES.items():
        # View elements are non-Boolean storage scalars. Index and offset are
        # SSA domains rather than addressable C++ element types.
        if element in {"i1", "index", "offset"}:
            continue
        for shape in ("", "4", "2, 4", "loom::type::dynamic, 4"):
            dynamic = "dynamic" in shape
            prefix = "vector" if shape else "view"
            payload = (
                f"loom::type::vector<loom::type::shape<{shape}>, {source_type}>"
                if shape
                else source_type
            )
            rows = "loom::type::dynamic" if dynamic else "4"
            view = f"loom::type::view<loom::type::shape<{rows}, 16>, {source_type}>"
            name = f"observe_{len(functions)}"
            parameters = ", loom::type::index rows" if dynamic else ""
            indices = "loom::type::indices{row, column}" if shape else "row, column"
            dimensions = ", {rows}" if dynamic else ""
            lines = [
                f"void {name}({view} input, {view} output, "
                f"loom::type::index row, loom::type::index column{parameters}) {{",
                f"  using Payload = {payload};",
            ]
            for index, flags in enumerate(
                (
                    "Access::none",
                    "Access::root_in_bounds",
                    "Access::volatile_",
                    "Access::volatile_ | Access::root_in_bounds",
                )
            ):
                selector = f"Payload, {flags}" if shape else flags
                lines.extend(
                    (
                        f"  auto value{index} = loom::{prefix}::load<{selector}>(input, {indices}{dimensions});",
                        f"  loom::{prefix}::store<{flags}>(value{index}, output, {indices});",
                    )
                )
            # Unused observations and repeated writes must not disappear.
            selector = "Payload, Access::volatile_" if shape else "Access::volatile_"
            lines.extend(
                (
                    f"  loom::{prefix}::load<{selector}>(input, {indices}{dimensions});",
                    f"  loom::{prefix}::load<{selector}>(input, {indices}{dimensions});",
                    f"  loom::{prefix}::store<Access::volatile_>(value3, output, {indices});",
                    f"  loom::{prefix}::store<Access::volatile_>(value3, output, {indices});",
                    "}",
                )
            )
            functions.append("\n".join(lines))
    return "\n".join(functions)


def _observations(module: Module) -> dict:
    layouts = {operation.name: compute_layout(operation) for operation in source_ops()}
    functions = {}
    for function in module.body.ops:
        assert function.name == "func.def"
        body = function.regions[0].blocks[0]
        identities = {value: module.values[value].name for value in body.arg_ids}
        observations = []
        for operation in body.ops:
            if operation.name == "func.return":
                continue
            if operation.name in {"view.refine", "vector.refine"}:
                identities[operation.results[0]] = identities[operation.operands[0]]
                continue
            assert operation.name in {
                "view.load",
                "view.store",
                "vector.load",
                "vector.store",
            }, operation.name
            resolved = resolve_fields(layouts[operation.name], operation, module)
            access = identities[resolved.value_id("view")]
            indices = tuple(
                identities[value] for value in resolved.value_ids("indices")
            )
            if operation.results:
                payload = operation.results[0]
                identity = ("read", len(observations))
                identities[payload] = identity
            else:
                payload = resolved.value_id("value")
                identity = identities[payload]
            value_type = module.values[payload].type
            footprint = (
                (
                    repr(value_type.element_type),
                    tuple(
                        dimension.size
                        if isinstance(dimension, StaticDim)
                        else identities[dimension.value_id]
                        for dimension in value_type.dims
                    ),
                )
                if isinstance(value_type, ShapedType)
                else repr(value_type)
            )
            observations.append(
                (
                    operation.name,
                    operation.attributes.get("memory_flags"),
                    access,
                    indices,
                    identity,
                    footprint,
                )
            )
        functions[function.attributes["callee"]] = observations
    return functions


def _import(importer: str, source: str) -> Module:
    result = subprocess.run(
        [importer, "--to=bc", "-"],
        input=source.encode(),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert result.stderr == b"", result.stderr.decode()
    return load_module(result.stdout)


def main(importer: str) -> None:
    original = _import(importer, _program())
    imported = _import(importer, render_module(original))
    expected = _observations(original)
    # Four ordinary/qualified load-store pairs, then two discarded reads and
    # two repeated stores. Both the initial importer and reimport must retain it.
    flags = [None, "root_in_bounds", "volatile", "volatile|root_in_bounds"]
    sequence = [flag for flag in flags for _ in range(2)] + ["volatile"] * 4
    assert all(
        [access[1] for access in trace] == sequence for trace in expected.values()
    )
    assert _observations(imported) == expected


if __name__ == "__main__":
    main(sys.argv[1])
