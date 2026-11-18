# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command order and storage observations from a compiler diagnostic report."""

from __future__ import annotations

from loom.reporting.compile_report import CompileReportDocument, CompileReportError

_KINDS = (
    "fill",
    "copy",
    "dispatch_direct",
    "dispatch_indirect_static",
    "dispatch_indirect_dynamic",
    "barrier",
    "timestamp",
)
_COUNTS = (
    "command_count",
    "barrier_before_count",
    "fixed_buffer_count",
    "rebindable_binding_count",
    "executable_count",
    "entry_count",
    "binding_count",
    "parameter_root_count",
    "parameter_count",
)


def build_command_show(document: CompileReportDocument) -> dict[str, object] | None:
    """Keeps root order and distinguishes absent details from empty programs."""
    if "command_programs" not in document.report:
        return None
    path = f"{document.source}.command_programs"
    values = document.report["command_programs"]
    if not isinstance(values, list):
        raise CompileReportError(f"{path}: expected array")
    plans = []
    for index, value in enumerate(values):
        root_path = f"{path}[{index}]"
        source = _object(value, root_path)
        plan = {"root": _string(source.get("root"), f"{root_path}.root")}
        for field in _COUNTS:
            plan[field] = _count(source.get(field), f"{root_path}.{field}")
        counts = _object(source.get("command_counts"), f"{root_path}.command_counts")
        plan["command_counts"] = {
            kind: _count(counts.get(kind), f"{root_path}.command_counts.{kind}")
            for kind in _KINDS
        }
        if sum(plan["command_counts"].values()) != plan["command_count"]:
            raise CompileReportError(
                f"{root_path}.command_counts: total differs from command_count"
            )
        for name in ("transient", "launch_counts"):
            if name not in source:
                raise CompileReportError(
                    f"{root_path}.{name}: missing storage observation"
                )
            value = source[name]
            plan[name] = (
                None if value is None else _storage(value, f"{root_path}.{name}")
            )
        for name, count_field, parse in (
            ("commands", "command_count", _command),
            ("bindings", "binding_count", _binding),
            ("parameter_roots", "parameter_root_count", _storage),
            ("parameters", "parameter_count", _parameter),
        ):
            if name not in source:
                continue
            rows = source[name]
            count = plan[count_field]
            if not isinstance(rows, list) or len(rows) != count:
                raise CompileReportError(f"{root_path}.{name}: expected {count} rows")
            plan[name] = [
                parse(row, f"{root_path}.{name}[{i}]") for i, row in enumerate(rows)
            ]
        plans.append(plan)
    return {"count": len(plans), "rows": plans}


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CompileReportError(f"{path}: expected object")
    return value


def _count(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CompileReportError(f"{path}: expected nonnegative integer")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CompileReportError(f"{path}: expected nonempty string")
    return value


def _binding(value: object, path: str) -> dict[str, object]:
    source = _object(value, path)
    role = source.get("role")
    if role not in ("fixed", "rebindable"):
        raise CompileReportError(f"{path}.role: expected fixed or rebindable")
    return {
        "role": role,
        "root_index": _count(source.get("root_index"), f"{path}.root_index"),
    }


def _buffer(value: object, path: str) -> dict[str, object]:
    source = _object(value, path)
    result = _binding(source, path)
    result["byte_offset"] = _count(source.get("byte_offset"), f"{path}.byte_offset")
    if "byte_length" not in source:
        raise CompileReportError(f"{path}.byte_length: missing range length")
    length = source["byte_length"]
    result["byte_length"] = (
        None if length is None else _count(length, f"{path}.byte_length")
    )
    return result


def _storage(value: object, path: str) -> dict[str, object]:
    source = _object(value, path)
    return {
        field: _count(source.get(field), f"{path}.{field}")
        for field in ("root_index", "required_byte_length", "minimum_alignment")
    }


def _parameter(value: object, path: str) -> dict[str, object]:
    source = _object(value, path)
    result = {
        field: _count(source.get(field), f"{path}.{field}")
        for field in ("root_index", "byte_offset", "byte_length", "minimum_alignment")
    }
    result["key"] = _string(source.get("key"), f"{path}.key")
    return result


def _command(value: object, path: str) -> dict[str, object]:
    source = _object(value, path)
    kind = source.get("kind")
    if kind not in _KINDS:
        raise CompileReportError(f"{path}.kind: unknown command kind {kind!r}")
    barrier = source.get("barrier_before")
    if not isinstance(barrier, bool):
        raise CompileReportError(f"{path}.barrier_before: expected boolean")
    result = {"kind": kind, "barrier_before": barrier}
    if kind.startswith("dispatch_"):
        result["entry"] = _string(source.get("entry"), f"{path}.entry")
        result["argument_count"] = _count(
            source.get("argument_count"), f"{path}.argument_count"
        )
        if kind == "dispatch_direct":
            groups = source.get("workgroups")
            if not isinstance(groups, list) or len(groups) != 3:
                raise CompileReportError(f"{path}.workgroups: expected three counts")
            result["workgroups"] = [
                _count(count, f"{path}.workgroups[{i}]")
                for i, count in enumerate(groups)
            ]
        else:
            result["count_buffer"] = _buffer(
                source.get("count_buffer"), f"{path}.count_buffer"
            )
    if kind in ("fill", "copy", "timestamp"):
        result["target"] = _buffer(source.get("target"), f"{path}.target")
    if kind == "copy":
        result["source"] = _buffer(source.get("source"), f"{path}.source")
    if kind == "fill":
        result["pattern"] = _count(source.get("pattern"), f"{path}.pattern")
        result["pattern_byte_count"] = _count(
            source.get("pattern_byte_count"), f"{path}.pattern_byte_count"
        )
    return result


def _buffer_text(buffer: dict[str, object]) -> str:
    length = buffer["byte_length"]
    return (
        f"{buffer['role']}[{buffer['root_index']}]+{buffer['byte_offset']}:"
        f"{length if length is not None else 'remaining'}"
    )


def append_command_show_text(lines: list[str], view: dict[str, object]) -> None:
    """Displays observed order without inventing indirect grids or arguments."""
    for ordinal, plan in enumerate(view["rows"]):
        lines.extend(("", f"Command program[{ordinal}] @{plan['root']}"))
        kinds = ", ".join(
            f"{count} {kind}" for kind, count in plan["command_counts"].items() if count
        )
        lines.append(
            f"  commands: {plan['command_count']} ({kinds or 'empty'}); "
            f"{plan['barrier_before_count']} barriers before commands"
        )
        lines.append(
            f"  resources: {plan['fixed_buffer_count']} fixed, "
            f"{plan['rebindable_binding_count']} rebindable; "
            f"{plan['executable_count']} executables, {plan['entry_count']} entries"
        )
        for name in ("transient", "launch_counts"):
            storage = plan[name]
            lines.append(
                f"  {name}: "
                + (
                    f"rebindable[{storage['root_index']}] "
                    f"{storage['required_byte_length']} B, "
                    f"alignment {storage['minimum_alignment']}"
                    if storage is not None
                    else "none"
                )
            )
        if "commands" not in plan:
            lines.append("  command and placement details: not collected")
        for index, command in enumerate(plan.get("commands", [])):
            text = (
                f"  [{index}] "
                + ("barrier -> " if command["barrier_before"] else "")
                + command["kind"]
            )
            if "entry" in command:
                text += f" @{command['entry']} ({command['argument_count']} arguments)"
            if "workgroups" in command:
                text += (
                    " workgroups=["
                    + ", ".join(str(count) for count in command["workgroups"])
                    + "]"
                )
            if "count_buffer" in command:
                text += " counts=" + _buffer_text(command["count_buffer"])
            if "source" in command:
                text += " source=" + _buffer_text(command["source"])
            if "target" in command:
                text += " target=" + _buffer_text(command["target"])
            if "pattern" in command:
                text += (
                    f" pattern=0x{command['pattern']:08x}/"
                    f"{command['pattern_byte_count']} B"
                )
            lines.append(text)
        for index, binding in enumerate(plan.get("bindings", [])):
            lines.append(
                f"  binding[{index}]: {binding['role']}[{binding['root_index']}]"
            )
        lines.extend(
            f"  parameter root fixed[{storage['root_index']}]: "
            f"{storage['required_byte_length']} B, "
            f"alignment {storage['minimum_alignment']}"
            for storage in plan.get("parameter_roots", [])
        )
        lines.extend(
            f"  parameter {parameter['key']!r}: fixed[{parameter['root_index']}]"
            f"+{parameter['byte_offset']}:{parameter['byte_length']}, "
            f"alignment {parameter['minimum_alignment']}"
            for parameter in plan.get("parameters", [])
        )
