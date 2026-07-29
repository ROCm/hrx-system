# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from copy import deepcopy

import pytest

from loom.reporting.compile_report import (
    CompileReportError,
    IncomparableCompileReportsError,
    match_compile_report_entries,
    parse_compile_report,
)
from loom.reporting.compile_report_view import (
    build_compile_report_diff,
    build_compile_report_show,
    format_compile_report_diff_text,
    format_compile_report_show_text,
)


def _compile_report() -> dict[str, object]:
    workload = {
        "workgroup_size": {"x": 128, "y": 1, "z": 1, "flat": 128},
        "workgroup_count": {"x": 12, "y": 8, "z": 1, "flat": 96},
        "dispatch_workitem_count": 12288,
    }
    entry = {
        "index": 0,
        "function": "routed_linear",
        "source_function": "routed_linear",
        "target_bundle": "gfx11_wave64",
        "target_snapshot": "gfx11_wave64",
        "target_export": "routed_linear",
        "target_export_symbol": None,
        "target_config": "gfx11_wave64",
        "workload": workload,
        "instruction_count": 100,
        "code_byte_count": 512,
        "code_storage_byte_count": 512,
        "local_memory_bytes": 4096,
        "private_memory_bytes": 0,
        "allocation_spill_count": 0,
        "allocation_materialized_copy_count": 4,
        "allocation_materialized_spill_storage_bytes": 0,
        "allocation_materialized_reload_bytes": 0,
        "schedule_node_count": 80,
        "schedule_dependency_count": 120,
        "schedule_hazard_gap_count": 2,
        "static_instruction_mix": {
            "scalar_alu_count": 10,
            "vector_alu_count": 40,
            "matrix_count": 8,
            "global_load_count": 4,
            "global_store_count": 2,
            "buffer_load_count": 6,
            "buffer_store_count": 0,
            "local_memory_count": 12,
            "conversion_count": 3,
            "register_move_count": 5,
            "barrier_count": 1,
            "branch_count": 4,
        },
        "target_resources": {
            "scalar": {
                "final": {"register_count": 32},
                "scheduled_pressure": {"peak_live_units": 28},
            },
            "vector": {
                "final": {"register_count": 96},
                "scheduled_pressure": {"peak_live_units": 88},
            },
            "resident_subgroups_per_simd": 5,
            "occupancy_percent": 31,
            "limiting_resource": "amdgpu.vgpr",
            "residency": {
                "best_tier": 16,
                "current_tier": 5,
                "next_better_tier": 6,
                "unique_limiting_resource": {
                    "name": "amdgpu.vgpr",
                    "units": 96,
                    "reduction_units_to_next_better_tier": 8,
                    "next_worse": {
                        "tier": 4,
                        "cliff_units": 104,
                        "additional_units": 8,
                    },
                },
            },
        },
        "economics": {
            "operations": {
                "dispatch": {
                    "vector_alu_count": 40960,
                    "matrix_count": 8192,
                }
            },
            "memory": {
                "dispatch_issued": {
                    "read_bytes": 3072,
                    "write_bytes": 1024,
                    "total_bytes": 4096,
                }
            },
        },
        "wait_plan": {
            "action_count": 12,
            "full_drain_count": 3,
            "partial_wait_count": 9,
        },
    }
    return {
        "kind": "loom.compile_report",
        "schema_version": 0,
        "mode": "summary",
        "artifact_kind": "hal-executable",
        "artifact_format": "hsaco",
        "backend": "amdgpu-hal",
        "status": {"code": 0, "name": "OK"},
        "target_family": "AMDGPU",
        "target_key": "gfx11-generic",
        "target_bundle": "gfx11_wave64",
        "target_snapshot": "gfx11_wave64",
        "target_config": "gfx11_wave64",
        "config_bindings": {
            "count": 2,
            "rows": [
                {"index": 0, "key": "model.output_size", "value": "768"},
                {"index": 1, "key": "model.input_size", "value": "2048"},
            ],
        },
        "workload": workload,
        "entries": {"count": 1, "rows": [entry]},
    }


def test_parses_direct_and_benchmark_envelope_reports() -> None:
    report = _compile_report()
    direct = parse_compile_report(report, source="direct.json")
    envelope = parse_compile_report(
        {
            "type": "compile_report",
            "run_id": "run0",
            "candidate_id": "candidate0",
            "compile_report": deepcopy(report),
        },
        source="envelope.json",
    )

    assert direct.container_kind == "direct"
    assert direct.config_bindings == (
        ("model.input_size", "2048"),
        ("model.output_size", "768"),
    )
    assert envelope.container_kind == "benchmark_envelope"
    assert dict(envelope.envelope_context) == {
        "run_id": "run0",
        "candidate_id": "candidate0",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report.pop("schema_version"), "schema_version"),
        (lambda report: report.__setitem__("schema_version", 1), "regenerate"),
        (
            lambda report: report["entries"].__setitem__("count", 2),
            "rows has 1 entries",
        ),
        (
            lambda report: report["config_bindings"]["rows"][1].__setitem__(
                "key", "model.output_size"
            ),
            "duplicate key",
        ),
        (
            lambda report: report.__setitem__("target_key", {"processor": "gfx1100"}),
            "target_key",
        ),
        (
            lambda report: report["entries"]["rows"][0].__setitem__(
                "workload", "96 workgroups"
            ),
            "workload",
        ),
    ],
)
def test_rejects_invalid_version_zero_documents(mutation, message: str) -> None:
    report = _compile_report()
    mutation(report)

    with pytest.raises(CompileReportError, match=message):
        parse_compile_report(report)


def test_matches_only_exact_report_and_entry_context() -> None:
    baseline = parse_compile_report(_compile_report(), source="baseline.json")
    candidate_report = _compile_report()
    candidate_report["entries"]["rows"][0]["code_byte_count"] = 480
    candidate = parse_compile_report(candidate_report, source="candidate.json")

    pairs = match_compile_report_entries(baseline, candidate)

    assert len(pairs) == 1
    assert pairs[0].identity.display_name() == "routed_linear"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report.__setitem__("target_key", "gfx1151"),
            "target_key",
        ),
        (
            lambda report: report["config_bindings"]["rows"][0].__setitem__(
                "value", "1024"
            ),
            "config_bindings",
        ),
        (
            lambda report: report["entries"]["rows"][0]["workload"][
                "workgroup_count"
            ].__setitem__("x", 6),
            "workload",
        ),
        (
            lambda report: report["entries"]["rows"][0].__setitem__(
                "source_function", "another_linear"
            ),
            "missing entry",
        ),
    ],
)
def test_rejects_incomparable_reports(mutate, message: str) -> None:
    baseline = parse_compile_report(_compile_report())
    candidate_report = _compile_report()
    mutate(candidate_report)
    candidate = parse_compile_report(candidate_report)

    with pytest.raises(IncomparableCompileReportsError, match=message):
        match_compile_report_entries(baseline, candidate)


def test_show_separates_facts_analysis_and_unavailable_evidence() -> None:
    report = _compile_report()
    del report["entries"]["rows"][0]["wait_plan"]["partial_wait_count"]
    document = parse_compile_report(report, source="report.json")

    view = build_compile_report_show(document)
    entry = view["entries"][0]
    artifact_facts = entry["artifact_facts"]
    compiler_analysis = entry["compiler_analysis"]

    assert view["missing_evidence"] == "omitted_metrics_are_unavailable"
    assert artifact_facts["code_byte_count"] == 512
    assert artifact_facts["private_memory_bytes"] == 0
    assert compiler_analysis["occupancy_percent"] == 31
    assert compiler_analysis["residency_next_better_tier"] == 6
    assert compiler_analysis["dispatch_total_bytes"] == 4096
    assert "partial_wait_count" not in compiler_analysis
    text = format_compile_report_show_text(view)
    assert "Artifact facts" in text
    assert "Compiler analysis" in text
    assert "code bytes: 512 B" in text
    assert "partial waits: unavailable" in text


def test_diff_preserves_missing_evidence_and_numeric_deltas() -> None:
    baseline = parse_compile_report(_compile_report(), source="baseline.json")
    candidate_report = _compile_report()
    candidate_entry = candidate_report["entries"]["rows"][0]
    candidate_entry["code_byte_count"] = 480
    del candidate_entry["wait_plan"]["partial_wait_count"]
    candidate = parse_compile_report(candidate_report, source="candidate.json")

    view = build_compile_report_diff(baseline, candidate)
    entry = view["entries"][0]
    code_bytes = entry["artifact_facts"]["changed"]["code_byte_count"]
    partial_waits = entry["compiler_analysis"]["incomplete"]["partial_wait_count"]

    assert code_bytes["delta"] == -32
    assert code_bytes["change_percent"] == -6.25
    assert partial_waits["candidate"] is None
    assert view["changed_entry_count"] == 1
    assert view["unchanged_entry_count"] == 0
    text = format_compile_report_diff_text(view)
    assert "512 B -> 480 B, delta -32 B (-6.25%)" in text
    assert "partial waits: 9 -> unavailable" in text


def test_diff_omits_unchanged_entries() -> None:
    baseline = parse_compile_report(_compile_report(), source="baseline.json")
    candidate = parse_compile_report(_compile_report(), source="candidate.json")

    view = build_compile_report_diff(baseline, candidate)

    assert view["entries"] == []
    assert view["changed_entry_count"] == 0
    assert view["unchanged_entry_count"] == 1
    text = format_compile_report_diff_text(view)
    assert "entries: 0 changed, 1 unchanged" in text
