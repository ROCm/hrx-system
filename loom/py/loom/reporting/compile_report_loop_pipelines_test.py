# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.reporting.compile_report import CompileReportError, parse_compile_report
from loom.reporting.compile_report_loop_pipelines import suggest_loop_pipelines
from loom.reporting.compile_report_view import (
    build_compile_report_show,
    format_compile_report_show_text,
)


def _report(*, details: bool = True) -> dict:
    policies = {
        "count": 2,
        "rows": [
            {
                "function": "stream",
                "loop": 0,
                "schedule": "read_ahead",
                "outcome": "pipelined",
                "depth": 3,
                "queue_records": 2,
                "values_per_record": 2,
                "read_count": 2,
            },
            {
                "function": "stream",
                "loop": 1,
                "schedule": "read_ahead",
                "outcome": "serial",
                "depth": 1,
                "queue_records": 0,
                "values_per_record": 0,
                "read_count": 0,
            },
        ],
    }
    if details:
        policies["stages"] = [
            {
                "function": "stream",
                "loop": 0,
                "position": 0,
                "op": "view.load",
                "stage": "producer",
                "iteration_lookahead": 2,
            },
            {
                "function": "stream",
                "loop": 0,
                "position": 1,
                "op": "scf.if",
                "partition": "guarded",
                "stage": "producer",
                "iteration_lookahead": 2,
            },
            {
                "function": "stream",
                "loop": 0,
                "position": 1,
                "op": "scf.if",
                "partition": "guarded",
                "stage": "consumer",
                "iteration_lookahead": 0,
            },
            {
                "function": "stream",
                "loop": 0,
                "position": 2,
                "op": "scalar.addf",
                "stage": "consumer",
                "iteration_lookahead": 0,
            },
        ]
    return {
        "kind": "loom.compile_report",
        "schema_version": 0,
        "mode": "details" if details else "summary",
        "status": {"code": 0, "name": "OK"},
        "entries": {
            "count": 1,
            "rows": [
                {
                    "index": 0,
                    "function": "stream_lowered",
                    "source_function": "stream",
                    "code_byte_count": 720,
                    "private_memory_bytes": 0,
                    "allocation_spill_count": 0,
                    "target_resources": {
                        "scalar": {"final": {"register_count": 10}},
                        "vector": {"final": {"register_count": 16}},
                        "occupancy_percent": 100,
                    },
                }
            ],
        },
        "source_low": {
            "count": 0,
            **({"rows": []} if details else {}),
            "loop_pipelines": policies,
        },
    }


@pytest.mark.parametrize("details", [False, True])
def test_show_groups_actual_schedules_and_keeps_serial_policies(details: bool) -> None:
    view = build_compile_report_show(parse_compile_report(_report(details=details)))
    policies = view["loop_pipelines"]
    assert policies["count"] == 2
    pipeline, serial = policies["rows"]
    assert pipeline["depth"] == 3
    assert serial["outcome"] == "serial"
    if details:
        assert pipeline["stages"] == [
            {
                "position": 0,
                "op": "view.load",
                "stage": "producer",
                "iteration_lookahead": 2,
            },
            {
                "position": 1,
                "op": "scf.if",
                "partition": "guarded",
                "stage": "producer",
                "iteration_lookahead": 2,
            },
            {
                "position": 1,
                "op": "scf.if",
                "partition": "guarded",
                "stage": "consumer",
                "iteration_lookahead": 0,
            },
            {
                "position": 2,
                "op": "scalar.addf",
                "stage": "consumer",
                "iteration_lookahead": 0,
            },
        ]
        assert serial["stages"] == []
    else:
        assert "stages" not in pipeline
    text = format_compile_report_show_text(view)
    assert (
        "stream loop 0: pipelined depth=3 queue_records=2 values_per_record=2 reads=2"
        in text
    )
    assert "stream loop 1: serial depth=1" in text
    assert (
        "1: scf.if partition=guarded producer iteration_lookahead=2" in text
    ) == details
    assert ("2: scalar.addf consumer iteration_lookahead=0" in text) == details


def test_suggestions_cite_applied_policy_and_exact_entry_resources() -> None:
    suggestions = suggest_loop_pipelines(parse_compile_report(_report()))
    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.entry_name == "stream"
    assert suggestion.suggestion_id == "scf.compare_pipeline_depth"
    assert "4 queued SSA values" in suggestion.action
    assert "unroll factor and workload held fixed" in suggestion.action
    assert (
        "matched comparison establishes resource and performance deltas"
        in suggestion.action
    )
    evidence = {fact.path: fact.value for fact in suggestion.evidence}
    assert evidence["source_low.loop_pipelines.rows[0].depth"] == 3
    assert (
        evidence["entries.rows[0].target_resources.vector.final.register_count"] == 16
    )
    assert evidence["entries.rows[0].private_memory_bytes"] == 0


def test_source_helpers_are_visible_without_assigning_an_entrys_costs() -> None:
    report = _report()
    report["entries"]["rows"][0]["source_function"] = "caller"
    document = parse_compile_report(report)
    assert build_compile_report_show(document)["loop_pipelines"]["count"] == 2
    assert suggest_loop_pipelines(document) == ()


def test_missing_resources_are_not_reported_as_zero() -> None:
    report = _report()
    entry = report["entries"]["rows"][0]
    for field in (
        "target_resources",
        "code_byte_count",
        "private_memory_bytes",
        "allocation_spill_count",
    ):
        del entry[field]
    (suggestion,) = suggest_loop_pipelines(parse_compile_report(report))
    assert all(fact.path.startswith("source_low.") for fact in suggestion.evidence)


def test_unannotated_and_failed_compilations_have_no_depth_advice() -> None:
    report = _report()
    report["status"] = {"code": 9, "name": "FAILED_PRECONDITION"}
    assert suggest_loop_pipelines(parse_compile_report(report)) == ()
    del report["source_low"]
    report["status"] = {"code": 0, "name": "OK"}
    document = parse_compile_report(report)
    assert "loop_pipelines" not in build_compile_report_show(document)
    assert suggest_loop_pipelines(document) == ()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("function", ""),
        ("loop", True),
        ("depth", 0),
        ("depth", -1),
        ("depth", 2.5),
        ("queue_records", 1),
        ("values_per_record", 0),
        ("read_count", 0),
        ("outcome", "serial"),
        ("schedule", "automatic"),
    ],
)
def test_rejects_malformed_policy_fields(field: str, value: object) -> None:
    report = _report()
    report["source_low"]["loop_pipelines"]["rows"][0][field] = value
    with pytest.raises(CompileReportError, match=r"loop_pipelines.rows\[0\]"):
        build_compile_report_show(parse_compile_report(report))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("function", "other"),
        ("loop", 1),
        ("position", 1),
        ("op", ""),
        ("partition", "recursive"),
        ("stage", "unknown"),
        ("stage", "consumer"),
        ("iteration_lookahead", 1),
    ],
)
def test_rejects_schedules_that_contradict_the_policy(
    field: str, value: object
) -> None:
    report = _report()
    report["source_low"]["loop_pipelines"]["stages"][0][field] = value
    with pytest.raises(CompileReportError, match=r"loop_pipelines.stages\[0\]"):
        build_compile_report_show(parse_compile_report(report))


@pytest.mark.parametrize(
    "corruption", ["count", "duplicate", "missing_stages", "serial_queue"]
)
def test_rejects_inconsistent_schedule_records(corruption: str) -> None:
    report = _report()
    policies = report["source_low"]["loop_pipelines"]
    if corruption == "count":
        policies["count"] = 3
    elif corruption == "duplicate":
        policies["rows"][1] = policies["rows"][0].copy()
    elif corruption == "missing_stages":
        policies["stages"] = []
    else:
        policies["rows"][1]["values_per_record"] = 1
    with pytest.raises(CompileReportError, match="loop_pipelines"):
        build_compile_report_show(parse_compile_report(report))


@pytest.mark.parametrize(
    ("operation", "partition"), [("index.add", None), ("scf.if", "guarded")]
)
def test_preserves_operations_executed_in_both_stages(
    operation: str, partition: str | None
) -> None:
    report = _report()
    stages = report["source_low"]["loop_pipelines"]["stages"]
    for stage in stages[1:3]:
        stage["op"] = operation
        if partition is None:
            del stage["partition"]

    document = parse_compile_report(report)
    view = build_compile_report_show(document)
    assert view["loop_pipelines"]["rows"][0]["stages"][1:3] == [
        {key: value for key, value in stage.items() if key not in ("function", "loop")}
        for stage in stages[1:3]
    ]
    text = format_compile_report_show_text(view)
    suffix = " partition=guarded" if partition else ""
    assert f"1: {operation}{suffix} producer iteration_lookahead=2" in text
    assert f"1: {operation}{suffix} consumer iteration_lookahead=0" in text
    assert len(suggest_loop_pipelines(document)) == 1


@pytest.mark.parametrize("partition", [None, "guarded"])
@pytest.mark.parametrize("corruption", ["operation", "partition", "stage", "third"])
def test_rejects_mismatched_repeated_stages(
    partition: str | None, corruption: str
) -> None:
    report = _report()
    stages = report["source_low"]["loop_pipelines"]["stages"]
    for stage in stages[1:3]:
        if partition is None:
            del stage["partition"]
    if corruption == "operation":
        stages[2]["op"] = "index.add"
    elif corruption == "partition":
        if partition is None:
            stages[2]["partition"] = "guarded"
        else:
            del stages[2]["partition"]
    elif corruption == "stage":
        stages[2]["stage"] = "producer"
        stages[2]["iteration_lookahead"] = 2
    else:
        stages.insert(3, stages[2].copy())

    with pytest.raises(
        CompileReportError,
        match="repeated position is not a matching stage pair",
    ):
        build_compile_report_show(parse_compile_report(report))
