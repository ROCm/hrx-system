# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from copy import deepcopy

import pytest

from loom.reporting.compile_report import (
    CompileReportComparisonMode,
    CompileReportError,
    IncomparableCompileReportsError,
    match_compile_report_entries,
    parse_compile_report,
)
from loom.reporting.compile_report_capabilities import (
    append_target_capability_diff_text,
    build_target_capability_diff,
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
        "body_instruction_count": 97,
        "entry_instruction_count": 3,
        "coissued_instruction_count": 4,
        "coissued_component_count": 8,
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
            "unknown_count": 0,
            "scalar_alu_count": 10,
            "vector_alu_count": 40,
            "matrix_count": 8,
            "mfma_count": 0,
            "smfmac_count": 0,
            "wmma_count": 8,
            "swmmac_count": 0,
            "dot_count": 0,
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
                    "wmma_count": 8192,
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
            "explicit_action_count": 2,
            "planned_action_count": 10,
            "full_drain_count": 3,
            "partial_wait_count": 9,
            "drained_count": 20,
            "max_drained_count": 4,
            "max_outstanding_before": 7,
            "max_full_drain_outstanding_before": 6,
        },
        "target_insertions": {
            "static_packet_count": 4,
            "exact_dynamic_packet_count": 4,
            "unknown_dynamic_packet_count": 0,
            "dynamic_packet_count": 384,
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


def _set_target_capabilities(
    report: dict[str, object],
    *,
    processor: str,
    descriptor_set: str,
    matrix_features: tuple[str, ...] = (),
) -> None:
    rows = [
        {
            "index": 0,
            "function": "routed_linear",
            "target_family": "AMDGPU",
            "namespace": "amdgpu",
            "key": "processor",
            "value_kind": "string",
            "value_string": processor,
        },
        {
            "index": 1,
            "function": "routed_linear",
            "target_family": "AMDGPU",
            "namespace": "amdgpu",
            "key": "descriptor_set",
            "value_kind": "string",
            "value_string": descriptor_set,
        },
        {
            "index": 2,
            "function": "routed_linear",
            "target_family": "AMDGPU",
            "namespace": "target",
            "key": "subgroup_size",
            "value_kind": "u64",
            "value_u64": 64,
        },
    ]
    for matrix_feature in matrix_features:
        rows.append(
            {
                "index": len(rows),
                "function": "routed_linear",
                "target_family": "AMDGPU",
                "namespace": "amdgpu.matrix_feature",
                "key": matrix_feature,
                "value_kind": "bool",
                "value_bool": True,
            }
        )
    report["target_capability_rows"] = {"count": len(rows), "rows": rows}


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

    match = match_compile_report_entries(baseline, candidate)

    assert len(match.pairs) == 1
    assert match.pairs[0].baseline_identity.display_name() == "routed_linear"
    assert match.pairs[0].candidate_identity.display_name() == "routed_linear"
    assert match.identity_mismatches == ()


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


def test_forced_comparison_pairs_single_entries_and_retains_identity() -> None:
    baseline_report = _compile_report()
    baseline_report["function"] = "baseline_kernel"
    baseline_report["target_export"] = "baseline_kernel"
    baseline_entry = baseline_report["entries"]["rows"][0]
    baseline_entry["function"] = "baseline_kernel"
    baseline_entry["source_function"] = "baseline_kernel"
    baseline_entry["target_export"] = "baseline_kernel"
    baseline = parse_compile_report(baseline_report, source="baseline.json")

    candidate_report = deepcopy(baseline_report)
    candidate_report["function"] = "candidate_kernel"
    candidate_report["target_export"] = "candidate_kernel"
    candidate_entry = candidate_report["entries"]["rows"][0]
    candidate_entry["function"] = "candidate_kernel"
    candidate_entry["source_function"] = "candidate_kernel"
    candidate_entry["target_export"] = "candidate_kernel"
    candidate_entry["code_byte_count"] = 480
    candidate = parse_compile_report(candidate_report, source="candidate.json")

    with pytest.raises(IncomparableCompileReportsError, match="missing entry"):
        match_compile_report_entries(baseline, candidate)

    match = match_compile_report_entries(baseline, candidate, force=True)
    assert len(match.pairs) == 1
    assert match.pairs[0].baseline_identity.display_name() == "baseline_kernel"
    assert match.pairs[0].candidate_identity.display_name() == "candidate_kernel"
    assert [mismatch.path for mismatch in match.identity_mismatches] == [
        "report.function",
        "report.target_export",
        "entry.function",
        "entry.source_function",
        "entry.target_export",
    ]

    view = build_compile_report_diff(baseline, candidate, force=True)
    assert view["forced"] is True
    assert "identity" not in view
    assert view["identities"]["baseline"]["function"] == "baseline_kernel"
    assert view["identities"]["candidate"]["function"] == "candidate_kernel"
    assert view["entry_identities"] == {
        "baseline": {"name": "baseline_kernel"},
        "candidate": {"name": "candidate_kernel"},
    }
    assert view["entries"][0]["identities"] == {
        "baseline": {"name": "baseline_kernel"},
        "candidate": {"name": "candidate_kernel"},
    }
    assert view["entries"][0]["artifact_facts"]["changed"]["code_byte_count"] == {
        "baseline": 512,
        "candidate": 480,
        "delta": -32,
        "change_percent": -6.25,
    }
    text = format_compile_report_diff_text(view)
    assert "comparison: forced single-entry observation" in text
    assert "identity contract bypassed; deltas are not causal" in text
    assert "identity mismatches: 5" in text
    assert "report.function: 'baseline_kernel' != 'candidate_kernel'" in text
    assert "Entry baseline_kernel -> candidate_kernel" in text


def test_forced_comparison_rejects_ambiguous_entry_pairing() -> None:
    baseline_report = _compile_report()
    second_entry = deepcopy(baseline_report["entries"]["rows"][0])
    second_entry["index"] = 1
    second_entry["function"] = "second_kernel"
    second_entry["source_function"] = "second_kernel"
    second_entry["target_export"] = "second_kernel"
    baseline_report["entries"] = {
        "count": 2,
        "rows": [baseline_report["entries"]["rows"][0], second_entry],
    }
    baseline = parse_compile_report(baseline_report)
    candidate = parse_compile_report(_compile_report())

    with pytest.raises(
        IncomparableCompileReportsError,
        match=r"forced comparison requires exactly one entry.*baseline has 2",
    ):
        match_compile_report_entries(baseline, candidate, force=True)


def test_target_comparison_permits_only_target_specialization_identity() -> None:
    baseline = parse_compile_report(_compile_report(), source="baseline.json")
    candidate_report = _compile_report()
    candidate_report["target_key"] = "gfx1151"
    candidate_report["target_bundle"] = "gfx11_5_wave64"
    candidate_report["target_snapshot"] = "gfx11_5_wave64"
    candidate_report["target_config"] = "gfx11_5_wave64"
    candidate_entry = candidate_report["entries"]["rows"][0]
    candidate_entry["target_bundle"] = "gfx11_5_wave64"
    candidate_entry["target_snapshot"] = "gfx11_5_wave64"
    candidate_entry["target_config"] = "gfx11_5_wave64"
    candidate = parse_compile_report(candidate_report, source="candidate.json")

    match = match_compile_report_entries(
        baseline,
        candidate,
        CompileReportComparisonMode.TARGET,
    )
    view = build_compile_report_diff(
        baseline,
        candidate,
        CompileReportComparisonMode.TARGET,
    )

    assert len(match.pairs) == 1
    assert view["comparison_mode"] == "target"
    assert view["identity"]["target_family"] == "AMDGPU"
    assert "target_key" not in view["identity"]
    assert view["targets"]["baseline"]["target_key"] == "gfx11-generic"
    assert view["targets"]["candidate"]["target_key"] == "gfx1151"
    assert view["changed_entry_count"] == 0
    assert view["unchanged_entry_count"] == 1
    text = format_compile_report_diff_text(view)
    assert "comparison: target specialization" in text
    assert "baseline target: AMDGPU/gfx11-generic via amdgpu-hal" in text
    assert "candidate target: AMDGPU/gfx1151 via amdgpu-hal" in text
    assert "reported entry evidence: unchanged" in text


def test_target_comparison_diffs_selected_capabilities() -> None:
    baseline_report = _compile_report()
    _set_target_capabilities(
        baseline_report,
        processor="gfx1100",
        descriptor_set="amdgpu.rdna3.core",
        matrix_features=("wmma-gfx11",),
    )
    baseline = parse_compile_report(baseline_report, source="baseline.json")
    candidate_report = _compile_report()
    candidate_report["target_key"] = "gfx1151"
    _set_target_capabilities(
        candidate_report,
        processor="gfx1151",
        descriptor_set="amdgpu.rdna3_5.core",
        matrix_features=("wmma-gfx12", "swmmac-gfx12"),
    )
    candidate = parse_compile_report(candidate_report, source="candidate.json")

    capability_diff = build_target_capability_diff(baseline, candidate)

    assert capability_diff is not None
    assert capability_diff["changed_count"] == 5
    assert capability_diff["unchanged_count"] == 1
    assert capability_diff["rows"][0]["identity"]["key"] == "descriptor_set"
    assert capability_diff["rows"][0]["baseline"] == {
        "kind": "string",
        "value": "amdgpu.rdna3.core",
    }
    assert capability_diff["rows"][1]["identity"]["key"] == "processor"
    assert capability_diff["rows"][2] == {
        "identity": {
            "namespace": "amdgpu.matrix_feature",
            "key": "swmmac-gfx12",
            "function": "routed_linear",
            "target_family": "AMDGPU",
        },
        "status": "added",
        "candidate": {"kind": "bool", "value": True},
    }
    assert capability_diff["rows"][3]["identity"]["key"] == "wmma-gfx11"
    assert capability_diff["rows"][3]["status"] == "removed"
    assert capability_diff["rows"][4]["identity"]["key"] == "wmma-gfx12"
    assert capability_diff["rows"][4]["status"] == "added"
    lines: list[str] = []
    append_target_capability_diff_text(lines, capability_diff)
    text = "\n".join(lines)
    assert "rows: 5 changed, 1 unchanged" in text
    assert "'amdgpu.rdna3.core' -> 'amdgpu.rdna3_5.core'" in text
    assert "amdgpu.matrix_feature.wmma-gfx11: removed true" in text
    assert "amdgpu.matrix_feature.wmma-gfx12: added true" in text

    view = build_compile_report_diff(
        baseline,
        candidate,
        CompileReportComparisonMode.TARGET,
    )
    assert view["target_capabilities"] == capability_diff


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report.__setitem__("target_family", "SPIR-V"),
            "target_family",
        ),
        (
            lambda report: report.__setitem__("backend", "spirv-hal"),
            "backend",
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
    ],
)
def test_target_comparison_rejects_non_target_changes(mutate, message: str) -> None:
    baseline = parse_compile_report(_compile_report())
    candidate_report = _compile_report()
    candidate_report["target_key"] = "gfx1151"
    mutate(candidate_report)
    candidate = parse_compile_report(candidate_report)

    with pytest.raises(IncomparableCompileReportsError, match=message):
        match_compile_report_entries(
            baseline,
            candidate,
            CompileReportComparisonMode.TARGET,
        )


def test_show_separates_facts_analysis_and_unavailable_evidence() -> None:
    report = _compile_report()
    del report["entries"]["rows"][0]["wait_plan"]["partial_wait_count"]
    report["entries"]["rows"][0]["target_insertions"][
        "unknown_dynamic_packet_count"
    ] = 1
    report["entries"]["rows"][0]["target_insertions"]["dynamic_packet_count"] = None
    document = parse_compile_report(report, source="report.json")

    view = build_compile_report_show(document)
    entry = view["entries"][0]
    artifact_facts = entry["artifact_facts"]
    compiler_analysis = entry["compiler_analysis"]

    assert view["missing_evidence"] == "omitted_metrics_are_unavailable"
    assert artifact_facts["code_byte_count"] == 512
    assert artifact_facts["private_memory_bytes"] == 0
    assert artifact_facts["body_instruction_count"] == 97
    assert artifact_facts["entry_instruction_count"] == 3
    assert artifact_facts["coissued_instruction_count"] == 4
    assert artifact_facts["coissued_component_count"] == 8
    assert artifact_facts["wmma_count"] == 8
    assert artifact_facts["unclassified_instruction_count"] == 0
    assert compiler_analysis["occupancy_percent"] == 31
    assert compiler_analysis["residency_next_better_tier"] == 6
    assert compiler_analysis["dispatch_total_bytes"] == 4096
    assert compiler_analysis["dispatch_wmma_count"] == 8192
    assert compiler_analysis["planned_wait_action_count"] == 10
    assert compiler_analysis["target_insertion_static_packet_count"] == 4
    assert compiler_analysis["target_insertion_unknown_dynamic_packet_count"] == 1
    assert "target_insertion_dynamic_packet_count" not in compiler_analysis
    assert "partial_wait_count" not in compiler_analysis
    text = format_compile_report_show_text(view)
    assert "Artifact facts" in text
    assert "Compiler analysis" in text
    assert "code bytes: 512 B" in text
    assert "WMMA instructions: 8" in text
    assert "target-planned wait actions: 10" in text
    assert "partial waits: unavailable" in text


def test_diff_preserves_missing_evidence_and_numeric_deltas() -> None:
    baseline = parse_compile_report(_compile_report(), source="baseline.json")
    candidate_report = _compile_report()
    candidate_entry = candidate_report["entries"]["rows"][0]
    candidate_entry["code_byte_count"] = 480
    candidate_entry["body_instruction_count"] = 89
    candidate_entry["static_instruction_mix"]["wmma_count"] = 6
    candidate_entry["economics"]["operations"]["dispatch"]["wmma_count"] = 6144
    candidate_entry["wait_plan"]["planned_action_count"] = 8
    candidate_entry["target_insertions"]["dynamic_packet_count"] = 256
    del candidate_entry["wait_plan"]["partial_wait_count"]
    candidate = parse_compile_report(candidate_report, source="candidate.json")

    view = build_compile_report_diff(baseline, candidate)
    entry = view["entries"][0]
    code_bytes = entry["artifact_facts"]["changed"]["code_byte_count"]
    body_instructions = entry["artifact_facts"]["changed"]["body_instruction_count"]
    wmma = entry["artifact_facts"]["changed"]["wmma_count"]
    partial_waits = entry["compiler_analysis"]["incomplete"]["partial_wait_count"]
    dynamic_wmma = entry["compiler_analysis"]["changed"]["dispatch_wmma_count"]
    planned_waits = entry["compiler_analysis"]["changed"]["planned_wait_action_count"]
    target_insertions = entry["compiler_analysis"]["changed"][
        "target_insertion_dynamic_packet_count"
    ]

    assert code_bytes["delta"] == -32
    assert code_bytes["change_percent"] == -6.25
    assert body_instructions["delta"] == -8
    assert wmma["delta"] == -2
    assert dynamic_wmma["delta"] == -2048
    assert planned_waits["delta"] == -2
    assert target_insertions["delta"] == -128
    assert partial_waits["candidate"] is None
    assert view["changed_entry_count"] == 1
    assert view["unchanged_entry_count"] == 0
    text = format_compile_report_diff_text(view)
    assert "512 B -> 480 B, delta -32 B (-6.25%)" in text
    assert "partial waits: 9 -> unavailable" in text


def test_diff_treats_explicit_null_as_unavailable_evidence() -> None:
    baseline = parse_compile_report(_compile_report(), source="baseline.json")
    candidate_report = _compile_report()
    candidate_report["entries"]["rows"][0]["target_insertions"][
        "dynamic_packet_count"
    ] = None
    candidate = parse_compile_report(candidate_report, source="candidate.json")

    view = build_compile_report_diff(baseline, candidate)
    target_insertions = view["entries"][0]["compiler_analysis"]["incomplete"][
        "target_insertion_dynamic_packet_count"
    ]

    assert target_insertions == {"baseline": 384, "candidate": None}


def test_diff_omits_unchanged_entries() -> None:
    baseline = parse_compile_report(_compile_report(), source="baseline.json")
    candidate = parse_compile_report(_compile_report(), source="candidate.json")

    view = build_compile_report_diff(baseline, candidate)

    assert view["entries"] == []
    assert view["changed_entry_count"] == 0
    assert view["unchanged_entry_count"] == 1
    text = format_compile_report_diff_text(view)
    assert "entries: 0 changed, 1 unchanged" in text
