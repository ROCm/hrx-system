# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from copy import deepcopy

import pytest

from loom.reporting.compile_report import CompileReportError, parse_compile_report
from loom.reporting.compile_report_commands import (
    append_command_show_text,
    build_command_show,
)
from loom.reporting.compile_report_view import (
    build_compile_report_diff,
    build_compile_report_show,
    format_compile_report_show_text,
)


def _report():
    target = {
        "role": "rebindable",
        "root_index": 2,
        "byte_offset": 1 << 34,
        "byte_length": None,
    }
    source = {"role": "fixed", "root_index": 0, "byte_offset": 256, "byte_length": 0}
    commands = [
        {"kind": "fill", "target": target, "pattern": 255, "pattern_byte_count": 1},
        {"kind": "copy", "source": source, "target": target},
        {
            "kind": "dispatch_direct",
            "entry": "empty_grid",
            "argument_count": 2,
            "workgroups": [0, 2, 3],
        },
        {
            "kind": "dispatch_indirect_static",
            "entry": "stable",
            "argument_count": 1,
            "count_buffer": source,
        },
        {
            "kind": "dispatch_indirect_dynamic",
            "entry": "mutable",
            "argument_count": 1,
            "count_buffer": target,
        },
        {"kind": "barrier"},
        {"kind": "timestamp", "target": target},
    ]
    for index, command in enumerate(commands):
        command["barrier_before"] = index == 1
    storage = {
        "root_index": 2,
        "required_byte_length": 1 << 35,
        "minimum_alignment": 16,
    }
    plan = {
        "root": "program",
        "command_count": len(commands),
        "command_counts": {command["kind"]: 1 for command in commands},
        "barrier_before_count": 1,
        "fixed_buffer_count": 1,
        "rebindable_binding_count": 3,
        "executable_count": 1,
        "entry_count": 3,
        "binding_count": 2,
        "parameter_root_count": 1,
        "parameter_count": 1,
        "transient": storage,
        "launch_counts": None,
        "commands": commands,
        "bindings": [
            {"role": "fixed", "root_index": 0},
            {"role": "rebindable", "root_index": 2},
        ],
        "parameter_roots": [
            {"root_index": 0, "required_byte_length": 272, "minimum_alignment": 256}
        ],
        "parameters": [
            {
                "key": "weights",
                "root_index": 0,
                "byte_offset": 256,
                "byte_length": 16,
                "minimum_alignment": 16,
            }
        ],
    }
    return {
        "kind": "loom.compile_report",
        "schema_version": 0,
        "mode": "details",
        "status": {"code": 0, "name": "OK"},
        "entries": {"count": 0, "rows": []},
        "command_programs": [plan],
    }


def _view(report):
    return build_command_show(parse_compile_report(report))


def test_command_order_and_storage_semantics():
    report = _report()
    view = _view(report)
    assert view == {"count": 1, "rows": report["command_programs"]}
    lines = []
    append_command_show_text(lines, view)
    assert (
        "\n".join(lines)
        == """
Command program[0] @program
  commands: 7 (1 fill, 1 copy, 1 dispatch_direct, 1 dispatch_indirect_static, 1 dispatch_indirect_dynamic, 1 barrier, 1 timestamp); 1 barriers before commands
  resources: 1 fixed, 3 rebindable; 1 executables, 3 entries
  transient: rebindable[2] 34359738368 B, alignment 16
  launch_counts: none
  [0] fill target=rebindable[2]+17179869184:remaining pattern=0x000000ff/1 B
  [1] barrier -> copy source=fixed[0]+256:0 target=rebindable[2]+17179869184:remaining
  [2] dispatch_direct @empty_grid (2 arguments) workgroups=[0, 2, 3]
  [3] dispatch_indirect_static @stable (1 arguments) counts=fixed[0]+256:0
  [4] dispatch_indirect_dynamic @mutable (1 arguments) counts=rebindable[2]+17179869184:remaining
  [5] barrier
  [6] timestamp target=rebindable[2]+17179869184:remaining
  binding[0]: fixed[0]
  binding[1]: rebindable[2]
  parameter root fixed[0]: 272 B, alignment 256
  parameter 'weights': fixed[0]+256:16, alignment 16"""
    )


def test_duplicate_root_order_and_summary_distinguish_absent_details():
    report = _report()
    report["mode"] = "summary"
    plan = report["command_programs"][0]
    for field in ("commands", "bindings", "parameter_roots", "parameters"):
        del plan[field]
    report["command_programs"] = [plan, dict(plan, root="second"), deepcopy(plan)]
    view = _view(report)
    assert [row["root"] for row in view["rows"]] == ["program", "second", "program"]
    assert all("commands" not in row for row in view["rows"])
    lines = []
    append_command_show_text(lines, view)
    assert lines.count("  command and placement details: not collected") == 3


def test_no_command_report_and_empty_program_are_distinct():
    report = _report()
    del report["command_programs"]
    assert _view(report) is None
    report["command_programs"] = []
    assert _view(report) == {"count": 0, "rows": []}
    report = _report()
    plan = report["command_programs"][0]
    for field, value in plan.items():
        if isinstance(value, int):
            plan[field] = 0
        elif isinstance(value, list):
            plan[field] = []
    plan["command_counts"] = dict.fromkeys(plan["command_counts"], 0)
    plan["transient"] = None
    view = _view(report)
    lines = []
    append_command_show_text(lines, view)
    assert "  commands: 0 (empty); 0 barriers before commands" in lines
    assert "  command and placement details: not collected" not in lines


@pytest.mark.parametrize("value", [None, True, -1, "7"])
def test_rejects_invalid_counts(value):
    report = _report()
    report["command_programs"][0]["command_count"] = value
    with pytest.raises(
        CompileReportError, match=r"command_programs\[0\].command_count"
    ):
        _view(report)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("root", "", "root"),
        ("commands", [], "expected 7 rows"),
        ("bindings", None, "expected 2 rows"),
        ("transient", {}, "root_index"),
    ],
)
def test_rejects_malformed_plan(field, value, error):
    report = _report()
    report["command_programs"][0][field] = value
    with pytest.raises(CompileReportError, match=error):
        _view(report)


@pytest.mark.parametrize(
    ("index", "field", "value", "error"),
    [
        (0, "kind", "wire_opcode", "unknown command kind"),
        (0, "barrier_before", 1, "expected boolean"),
        (0, "target", {"role": "unknown"}, "expected fixed or rebindable"),
        (2, "workgroups", [1, 2], "expected three counts"),
        (2, "workgroups", [1, True, 3], "workgroups"),
        (2, "entry", None, "entry"),
        (3, "count_buffer", None, "count_buffer"),
    ],
)
def test_rejects_malformed_commands(index, field, value, error):
    report = _report()
    report["command_programs"][0]["commands"][index][field] = value
    with pytest.raises(CompileReportError, match=error):
        _view(report)


def test_rejects_inconsistent_summary_counts():
    report = _report()
    report["command_programs"][0]["command_counts"]["copy"] = 2
    with pytest.raises(CompileReportError, match="total differs from command_count"):
        _view(report)


def test_missing_nullable_observations_do_not_invent_absence_or_remaining():
    report = _report()
    del report["command_programs"][0]["transient"]
    with pytest.raises(CompileReportError, match="missing storage observation"):
        _view(report)
    report = _report()
    del report["command_programs"][0]["commands"][0]["target"]["byte_length"]
    with pytest.raises(CompileReportError, match="missing range length"):
        _view(report)


def test_command_view_does_not_describe_missing_native_entries_as_missing_commands():
    text = format_compile_report_show_text(
        build_compile_report_show(parse_compile_report(_report()))
    )
    assert "Command program[0] @program" in text
    assert "Launch geometry (compiled artifact)" not in text
    assert "entries: none" not in text


def test_native_diff_does_not_silently_ignore_command_changes():
    baseline = _report()
    candidate = deepcopy(baseline)
    candidate["command_programs"][0]["commands"][2]["workgroups"] = [4, 2, 3]
    with pytest.raises(
        CompileReportError, match="use show to inspect each command report"
    ):
        build_compile_report_diff(
            parse_compile_report(baseline), parse_compile_report(candidate)
        )
