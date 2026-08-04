#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import sys
from pathlib import Path

SUMMARY_VERSION = 1


class SummaryError(ValueError):
    pass


@dataclasses.dataclass(frozen=True, order=True)
class SourcePoint:
    file: str
    line: int
    column: int


@dataclasses.dataclass(frozen=True)
class Function:
    id: str
    name: str
    point: SourcePoint


@dataclasses.dataclass(frozen=True)
class Edge:
    caller: str
    callee: str
    caller_name: str
    callee_name: str
    point: SourcePoint
    indirect: bool
    dispatcher: str | None


def _required_string(value: object, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value:
        raise SummaryError(f"{path}: {field} must be a non-empty string")
    return value


def _required_unsigned(value: object, field: str, path: Path) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SummaryError(f"{path}: {field} must be a non-negative integer")
    return value


def _source_point(record: dict[str, object], path: Path) -> SourcePoint:
    return SourcePoint(
        file=_required_string(record.get("file"), "file", path),
        line=_required_unsigned(record.get("line"), "line", path),
        column=_required_unsigned(record.get("column"), "column", path),
    )


def _records(value: object, field: str, path: Path) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise SummaryError(f"{path}: {field} must be an array")
    records: list[dict[str, object]] = []
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            raise SummaryError(f"{path}: {field}[{index}] must be an object")
        records.append(record)
    return records


def load_summary(path: Path) -> tuple[list[Function], list[Edge]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SummaryError(f"{path}: failed to read recursion summary: {exc}") from exc
    if not isinstance(document, dict):
        raise SummaryError(f"{path}: recursion summary must be an object")
    if document.get("version") != SUMMARY_VERSION:
        raise SummaryError(
            f"{path}: unsupported recursion summary version "
            f"{document.get('version')!r}; expected {SUMMARY_VERSION}"
        )
    _required_string(document.get("translation_unit"), "translation_unit", path)

    functions = [
        Function(
            id=_required_string(record.get("id"), "functions[].id", path),
            name=_required_string(record.get("name"), "functions[].name", path),
            point=_source_point(record, path),
        )
        for record in _records(document.get("functions"), "functions", path)
    ]
    edges: list[Edge] = []
    for record in _records(document.get("edges"), "edges", path):
        indirect = record.get("indirect")
        if not isinstance(indirect, bool):
            raise SummaryError(f"{path}: edges[].indirect must be a boolean")
        dispatcher = record.get("dispatcher")
        if dispatcher is not None and not isinstance(dispatcher, str):
            raise SummaryError(
                f"{path}: edges[].dispatcher must be a string when present"
            )
        edges.append(
            Edge(
                caller=_required_string(record.get("caller"), "edges[].caller", path),
                callee=_required_string(record.get("callee"), "edges[].callee", path),
                caller_name=_required_string(
                    record.get("caller_name"), "edges[].caller_name", path
                ),
                callee_name=_required_string(
                    record.get("callee_name"), "edges[].callee_name", path
                ),
                point=_source_point(record, path),
                indirect=indirect,
                dispatcher=dispatcher,
            )
        )
    return functions, edges


def strongly_connected_components(edges: list[Edge]) -> list[list[str]]:
    nodes = sorted({edge.caller for edge in edges} | {edge.callee for edge in edges})
    outgoing: dict[str, list[str]] = {node: [] for node in nodes}
    incoming: dict[str, list[str]] = {node: [] for node in nodes}
    for edge in edges:
        outgoing[edge.caller].append(edge.callee)
        incoming[edge.callee].append(edge.caller)
    for adjacency in (outgoing, incoming):
        for targets in adjacency.values():
            targets.sort()

    visited: set[str] = set()
    finish_order: list[str] = []
    for root in nodes:
        if root in visited:
            continue
        visited.add(root)
        frames: list[tuple[str, int]] = [(root, 0)]
        while frames:
            node, next_edge = frames[-1]
            if next_edge < len(outgoing[node]):
                target = outgoing[node][next_edge]
                frames[-1] = (node, next_edge + 1)
                if target not in visited:
                    visited.add(target)
                    frames.append((target, 0))
                continue
            finish_order.append(node)
            frames.pop()

    assigned: set[str] = set()
    components: list[list[str]] = []
    for root in reversed(finish_order):
        if root in assigned:
            continue
        assigned.add(root)
        worklist = [root]
        component: list[str] = []
        while worklist:
            node = worklist.pop()
            component.append(node)
            for predecessor in incoming[node]:
                if predecessor not in assigned:
                    assigned.add(predecessor)
                    worklist.append(predecessor)
        components.append(sorted(component))
    return components


def representative_cycle(
    start: str, component: list[str], outgoing: dict[str, list[Edge]]
) -> list[Edge]:
    members = set(component)
    for first in outgoing.get(start, []):
        if first.callee not in members:
            continue
        if first.callee == start:
            return [first]
        predecessor: dict[str, Edge | None] = {first.callee: None}
        queue: collections.deque[str] = collections.deque([first.callee])
        while queue and start not in predecessor:
            node = queue.popleft()
            for edge in outgoing.get(node, []):
                if edge.callee not in members:
                    continue
                if edge.callee not in predecessor:
                    predecessor[edge.callee] = edge
                    queue.append(edge.callee)
        if start not in predecessor:
            continue
        reverse_path: list[Edge] = []
        current = start
        while current != first.callee:
            edge = predecessor[current]
            if edge is None:
                raise AssertionError("cycle predecessor chain ended early")
            reverse_path.append(edge)
            current = edge.caller
        return [first, *reversed(reverse_path)]
    raise AssertionError("recursive SCC has no representative cycle")


def format_diagnostics(functions: list[Function], edges: list[Edge]) -> str:
    definitions: dict[str, Function] = {}
    for function in functions:
        current = definitions.get(function.id)
        if current and current.name != function.name:
            raise SummaryError(
                f"function identity collision for {function.id!r}: "
                f"{current.name!r} and {function.name!r}"
            )
        if current is None or function.point < current.point:
            definitions[function.id] = function

    outgoing: dict[str, list[Edge]] = collections.defaultdict(list)
    self_recursive: set[str] = set()
    for edge in edges:
        outgoing[edge.caller].append(edge)
        if edge.caller == edge.callee:
            self_recursive.add(edge.caller)
    for node_edges in outgoing.values():
        node_edges.sort(
            key=lambda edge: (
                edge.callee,
                edge.point,
                edge.indirect,
                edge.dispatcher or "",
            )
        )

    recursive_components = [
        component
        for component in strongly_connected_components(edges)
        if len(component) > 1 or (bool(component) and component[0] in self_recursive)
    ]
    recursive_components.sort(
        key=lambda component: min(
            (definitions[node].point for node in component if node in definitions),
            default=SourcePoint("", 0, 0),
        )
    )

    lines: list[str] = []
    for component in recursive_components:
        representative = min(
            (definitions[node] for node in component if node in definitions),
            key=lambda function: function.point,
            default=None,
        )
        cycle = representative_cycle(
            representative.id if representative else component[0], component, outgoing
        )
        point = representative.point if representative else cycle[0].point
        suffix = "" if len(component) == 1 else "s"
        definition_files = {
            definitions[node].point.file for node in component if node in definitions
        }
        cycle_scope = "cross-translation-unit " if len(definition_files) > 1 else ""
        lines.append(
            f"{point.file}:{point.line}:{point.column}: error: potentially "
            f"unbounded native recursion in a {cycle_scope}call cycle "
            f"containing {len(component)} function{suffix}; use an explicit "
            "worklist or establish a mechanically checked fixed bound "
            "[iree-unbounded-recursion]"
        )
        for edge in cycle:
            if edge.indirect and edge.dispatcher:
                message = (
                    f"call from {edge.caller_name} through callback dispatcher "
                    f"{edge.dispatcher} to {edge.callee_name} participates in "
                    "this cycle"
                )
            else:
                call_kind = "indirect" if edge.indirect else "direct"
                message = (
                    f"{call_kind} call from {edge.caller_name} to "
                    f"{edge.callee_name} participates in this cycle"
                )
            lines.append(
                f"{edge.point.file}:{edge.point.line}:{edge.point.column}: "
                f"note: {message}"
            )
    return "\n".join(lines) + ("\n" if lines else "")


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-diagnostics", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("summaries", nargs="*", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_arguments(argv)
    try:
        functions: list[Function] = []
        edges: list[Edge] = []
        for path in args.summaries:
            summary_functions, summary_edges = load_summary(path)
            functions.extend(summary_functions)
            edges.extend(summary_edges)
        diagnostics = format_diagnostics(functions, edges)
    except SummaryError as exc:
        diagnostics = f"error: {exc}\n"
        args.output.write_text(diagnostics, encoding="utf-8")
        print(diagnostics, end="", file=sys.stderr)
        return 2

    report = diagnostics or (
        f"No recursive call cycles found across {len(args.summaries)} "
        "translation units.\n"
    )
    args.output.write_text(report, encoding="utf-8")
    if not args.quiet:
        print(report, end="")
    if diagnostics and not args.allow_diagnostics:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
