# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Resolves final, repository-owned CMake compile interfaces in dependency order.

The input is written by iree_finalize_compile_usage after native target creation.
Link dependencies remain native. Only context-independent compile requirements
cross ordinary edges; native providers remain compile dependencies at boundaries.
This is configure-local output, never a cache across configurations or edits.
"""

from __future__ import annotations

import argparse
import graphlib
import json
import re
from pathlib import Path

# These expressions depend on the consumer's configuration, toolchain, or build
# versus install context. Expressions that look up targets, directories, sources,
# or other properties must keep their original native evaluation context.
_PORTABLE_EXPRESSIONS = {
    "0",
    "1",
    "BOOL",
    "IF",
    "AND",
    "OR",
    "NOT",
    "STREQUAL",
    "EQUAL",
    "VERSION_LESS",
    "VERSION_LESS_EQUAL",
    "VERSION_GREATER",
    "VERSION_GREATER_EQUAL",
    "VERSION_EQUAL",
    "CONFIG",
    "COMPILE_LANGUAGE",
    "COMPILE_LANG_AND_ID",
    "C_COMPILER_ID",
    "CXX_COMPILER_ID",
    "C_COMPILER_VERSION",
    "CXX_COMPILER_VERSION",
    "PLATFORM_ID",
    "BUILD_INTERFACE",
    "BUILD_LOCAL_INTERFACE",
    "INSTALL_INTERFACE",
    "INSTALL_PREFIX",
    "SEMICOLON",
    "COMMA",
    "ANGLE-R",
}


def _items(value: str) -> list[str]:
    """Splits lists without splitting quoted expressions or escaped semicolons."""
    result = []
    start = depth = 0
    for index, character in enumerate(value):
        if value[index : index + 2] == "$<":
            depth += 1
        elif character == ">" and depth:
            depth -= 1
        elif character == ";" and not depth and (not index or value[index - 1] != "\\"):
            result.append(value[start:index])
            start = index + 1
    result.append(value[start:])
    return [item for item in result if item]


def _ordinary(row: dict) -> bool:
    if "$<" in row["links"] or "::@" in row["links"]:
        return False
    if (
        row["usage"]["INTERFACE_SOURCES"]
        or row["usage"]["INTERFACE_PRECOMPILE_HEADERS"]
    ):
        return False
    for value in row["usage"].values():
        for expression in re.findall(r"\$<([^:$<>]+)", value):
            if expression not in _PORTABLE_EXPRESSIONS:
                return False
    return True


def _quote(value: str) -> str:
    delimiter = "="
    while f"]{delimiter}]" in value:
        delimiter += "="
    # Bracket arguments discard an immediately following newline.
    return f"[{delimiter}[\n{value}]{delimiter}]"


def resolve(rows: list[dict]) -> str:
    graph = {row["target"]: row for row in rows if _ordinary(row)}
    predecessors = {
        name: tuple(
            dict.fromkeys(
                provider for provider in row["providers"].values() if provider in graph
            )
        )
        for name, row in graph.items()
    }
    order = graphlib.TopologicalSorter(predecessors)
    try:
        order.prepare()
    except graphlib.CycleError:
        # Native CMake supports static-library cycles. Resolve only the acyclic
        # portion; cycles and their dependents retain the original native graph.
        pass
    resolved = {}
    output = ["# Generated compile interfaces; native link ownership is unchanged.\n"]
    while order.is_active():
        ready = order.get_ready()
        for name in ready:
            row = graph[name]
            values = {
                prop: dict.fromkeys(_items(value))
                for prop, value in row["usage"].items()
            }
            opaque = {}
            links = []
            for dependency in _items(row["links"]):
                provider = row["providers"].get(dependency)
                if provider in resolved:
                    child, inherited = resolved[provider]
                    for prop in values:
                        values[prop].update(child[prop])
                    opaque.update(inherited)
                    links.append(f"$<LINK_ONLY:{dependency}>")
                else:
                    links.append(dependency)
                    if provider is not None:
                        opaque[provider] = None
                        for prop in values:
                            values[prop][f"$<TARGET_PROPERTY:{provider},{prop}>"] = None
            if opaque:
                links.append("$<COMPILE_ONLY:" + ";".join(opaque) + ">")
            resolved[name] = values, opaque
            for prop, entries in values.items():
                if entries:
                    output.append(
                        f"set_property(TARGET {name} PROPERTY {prop} {_quote(';'.join(entries))})\n"
                    )
            output.append(
                f"set_property(TARGET {name} PROPERTY INTERFACE_LINK_LIBRARIES {_quote(';'.join(links))})\n"
            )
        order.done(*ready)
    return "".join(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(
        resolve(json.loads(args.input.read_text(encoding="utf-8"))), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
