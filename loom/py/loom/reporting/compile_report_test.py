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
        "dynamic_instruction_mix": {
            "descriptor_count": 100,
            "scalar_alu_count": 10,
            "vector_alu_count": 40,
            "matrix_count": 8,
            "mfma_count": 0,
            "smfmac_count": 0,
            "wmma_count": 8,
            "swmmac_count": 0,
            "dot_count": 0,
            "atomic_count": 0,
            "branch_count": 4,
            "barrier_count": 1,
            "control_count": 4,
            "conversion_count": 3,
            "cache_count": 0,
            "register_move_count": 5,
            "global_load_count": 4,
            "global_store_count": 2,
            "buffer_load_count": 6,
            "buffer_store_count": 0,
            "flat_memory_count": 0,
            "local_memory_count": 12,
            "scalar_memory_count": 0,
            "private_memory_count": 0,
            "generic_memory_count": 0,
            "memory_read_byte_count": 32,
            "memory_write_byte_count": 16,
            "memory_read_unknown_width_count": 0,
            "memory_write_unknown_width_count": 0,
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
            "subgroup_size": 64,
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
                "per_workitem": {
                    "scalar_alu_count": 10,
                    "vector_alu_count": 40,
                    "matrix_count": 8,
                    "wmma_count": 8,
                },
                "dispatch": {
                    "scalar_alu_count": 122880,
                    "vector_alu_count": 491520,
                    "matrix_count": 98304,
                    "wmma_count": 98304,
                },
            },
            "memory": {
                "per_workitem_issued": {
                    "read_bytes": 32,
                    "write_bytes": 16,
                    "total_bytes": 48,
                },
                "dispatch_issued": {
                    "read_bytes": 393216,
                    "write_bytes": 196608,
                    "total_bytes": 589824,
                },
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
        "target_resources": {"subgroup_size": 64},
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


def _set_move_causes(
    report: dict[str, object],
    causes: tuple[tuple[str, int, int], ...],
) -> None:
    report["mode"] = "details"
    report["entries"]["rows"][0]["move_causes"] = {
        "kind_count": len(causes),
        "packet_count": sum(packet_count for _, packet_count, _ in causes),
        "unit_count": sum(unit_count for _, _, unit_count in causes),
        "causes": [
            {
                "cause": cause,
                "packet_count": packet_count,
                "unit_count": unit_count,
            }
            for cause, packet_count, unit_count in causes
        ],
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
    execution_economics = entry["execution_economics"]

    assert view["missing_evidence"] == "omitted_metrics_are_unavailable"
    assert view["workload"] == {
        "workgroup_size": {"x": 128, "y": 1, "z": 1, "flat": 128},
        "workitems_per_workgroup": 128,
        "workgroup_count": {"x": 12, "y": 8, "z": 1, "flat": 96},
        "dispatch_workgroup_count": 96,
        "subgroup_size": 64,
        "subgroups_per_workgroup": 2,
        "dispatch_subgroup_count": 192,
        "dispatch_workitem_count": 12288,
    }
    assert "workload" not in entry
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
    assert execution_economics["fixed_trip_multiplicity_coverage"] == "exact"
    assert execution_economics["issued_byte_coverage"] == "exact"
    assert execution_economics["primary_scope"] == "workgroup"
    workgroup_metrics = execution_economics["scopes"]["workgroup"]["metrics"]
    assert workgroup_metrics["issued_total_bytes"] == 6144
    assert workgroup_metrics["wmma_count"] == 1024
    assert compiler_analysis["planned_wait_action_count"] == 10
    assert compiler_analysis["target_insertion_static_packet_count"] == 4
    assert compiler_analysis["target_insertion_unknown_dynamic_packet_count"] == 1
    assert "target_insertion_dynamic_packet_count" not in compiler_analysis
    assert "partial_wait_count" not in compiler_analysis
    text = format_compile_report_show_text(view)
    assert "Artifact facts" in text
    assert "Compiler analysis" in text
    assert "Compiled execution economics (compiler analysis)" in text
    assert "fixed-trip multiplicity coverage: exact" in text
    assert "all statically reachable blocks (path envelope)" in text
    assert "scope: one workgroup (128 workitems)" in text
    assert "Launch geometry (compiled artifact)" in text
    assert "dispatch subgroups: 192" in text
    assert "code bytes: 512 B" in text
    assert "WMMA instructions: 8" in text
    assert "target-planned wait actions: 10" in text
    assert "partial waits: unavailable" in text


def test_show_execution_economics_without_static_dispatch_geometry() -> None:
    report = _compile_report()
    workload = report["workload"]
    del workload["workgroup_count"]
    del workload["dispatch_workitem_count"]
    entry = report["entries"]["rows"][0]
    del entry["economics"]["operations"]["dispatch"]
    del entry["economics"]["memory"]["dispatch_issued"]
    document = parse_compile_report(report, source="dynamic.json")

    view = build_compile_report_show(document)

    economics = view["entries"][0]["execution_economics"]
    assert economics["primary_scope"] == "workgroup"
    assert set(economics["scopes"]) == {"workitem", "workgroup"}
    assert economics["scopes"]["workitem"]["metrics"]["wmma_count"] == 8
    assert economics["scopes"]["workgroup"]["metrics"]["wmma_count"] == 1024
    text = format_compile_report_show_text(view)
    assert "scope: one workgroup (128 workitems)" in text
    assert "dispatch workitems: unavailable" in text


def test_show_execution_economics_without_any_static_launch_geometry() -> None:
    report = _compile_report()
    report["workload"].clear()
    entry = report["entries"]["rows"][0]
    del entry["economics"]["operations"]["dispatch"]
    del entry["economics"]["memory"]["dispatch_issued"]
    document = parse_compile_report(report, source="dynamic.json")

    view = build_compile_report_show(document)

    economics = view["entries"][0]["execution_economics"]
    assert economics["primary_scope"] == "workitem"
    assert set(economics["scopes"]) == {"workitem"}
    assert economics["scopes"]["workitem"]["metrics"]["issued_total_bytes"] == 48
    text = format_compile_report_show_text(view)
    assert "scope: one workitem" in text
    assert "workgroup size: unavailable" in text


def test_forced_diff_expands_workload_before_artifact_metrics() -> None:
    baseline_report = _compile_report()
    baseline = parse_compile_report(baseline_report, source="baseline.json")
    candidate_report = deepcopy(baseline_report)
    for workload in (
        candidate_report["workload"],
        candidate_report["entries"]["rows"][0]["workload"],
    ):
        workload["workgroup_count"] = {
            "x": 48,
            "y": 8,
            "z": 1,
            "flat": 384,
        }
        workload["dispatch_workitem_count"] = 49152
    del candidate_report["entries"]["rows"][0]["economics"]
    candidate_report["entries"]["rows"][0]["body_instruction_count"] = 48
    candidate = parse_compile_report(candidate_report, source="candidate.json")

    view = build_compile_report_diff(baseline, candidate, force=True)

    workload = view["workload"]
    assert workload["changed"]["dispatch_workgroup_count"] == {
        "baseline": 96,
        "candidate": 384,
        "delta": 288,
        "change_percent": 300.0,
    }
    assert workload["changed"]["dispatch_subgroup_count"]["candidate"] == 768
    assert workload["changed"]["dispatch_workitem_count"]["candidate"] == 49152
    assert "workload" not in view["entries"][0]
    text = format_compile_report_diff_text(view)
    assert "workload mismatches: 2; expanded below" in text
    assert "report.workload:" not in text
    assert "entry.workload:" not in text
    assert text.index("Launch geometry (compiled artifact)") < text.index(
        "Artifact facts"
    )
    assert "dispatch subgroups: 192 -> 768, delta +576 (+300.00%)" in text


def test_show_rejects_inconsistent_workload_products() -> None:
    report = _compile_report()
    report["workload"]["workgroup_count"]["flat"] = 95
    document = parse_compile_report(report, source="report.json")

    with pytest.raises(CompileReportError, match=r"workgroup_count.flat: expected 96"):
        build_compile_report_show(document)


def test_show_explains_unavailable_fixed_trip_economics() -> None:
    report = _compile_report()
    entry = report["entries"]["rows"][0]
    del entry["dynamic_instruction_mix"]
    del entry["economics"]
    document = parse_compile_report(report, source="report.json")

    view = build_compile_report_show(document)

    economics = view["entries"][0]["execution_economics"]
    assert economics == {
        "fixed_trip_multiplicity_coverage": "unavailable",
        "issued_byte_coverage": "unavailable",
        "reason": "exact_fixed_trip_multiplicities_unavailable",
    }
    text = format_compile_report_show_text(view)
    assert "fixed-trip multiplicity coverage: unavailable" in text
    assert "exact fixed-trip multiplicities were not proven" in text
    assert "estimated dispatch" not in text


def test_show_qualifies_unknown_issued_widths() -> None:
    report = _compile_report()
    entry = report["entries"]["rows"][0]
    entry["dynamic_instruction_mix"]["memory_read_unknown_width_count"] = 2
    document = parse_compile_report(report, source="report.json")

    view = build_compile_report_show(document)

    economics = view["entries"][0]["execution_economics"]
    assert economics["issued_byte_coverage"] == "partial"
    assert (
        economics["scopes"]["workgroup"]["metrics"]["issued_read_unknown_width_count"]
        == 256
    )
    text = format_compile_report_show_text(view)
    assert "issued byte coverage: partial" in text
    assert "reads with unknown issued width: 256" in text


def test_show_rejects_inconsistent_retained_static_dispatch_economics() -> None:
    report = _compile_report()
    report["entries"]["rows"][0]["economics"]["operations"]["dispatch"][
        "wmma_count"
    ] = 1
    document = parse_compile_report(report, source="report.json")

    with pytest.raises(CompileReportError, match=r"wmma_count: expected 98304"):
        build_compile_report_show(document)


def test_diff_preserves_fixed_trip_economics_coverage_loss() -> None:
    baseline = parse_compile_report(_compile_report(), source="baseline.json")
    candidate_report = _compile_report()
    candidate_entry = candidate_report["entries"]["rows"][0]
    del candidate_entry["dynamic_instruction_mix"]
    del candidate_entry["economics"]
    candidate = parse_compile_report(candidate_report, source="candidate.json")

    view = build_compile_report_diff(baseline, candidate)

    assert view["changed_entry_count"] == 1
    economics = view["entries"][0]["execution_economics"]
    assert economics["fixed_trip_multiplicity_coverage"] == {
        "baseline": "exact",
        "candidate": "unavailable",
    }
    assert economics["incomplete"]["scalar_alu_count"] == {
        "baseline": 10,
        "candidate": None,
    }
    text = format_compile_report_diff_text(view)
    assert "fixed-trip multiplicity coverage: exact -> unavailable" in text
    assert "exact fixed-trip multiplicities were not proven" in text


def test_show_and_diff_preserve_move_cause_breakdown() -> None:
    baseline_report = _compile_report()
    _set_move_causes(
        baseline_report,
        (
            ("constant_materialization", 16, 16),
            ("operand_bank_materialization", 10, 10),
        ),
    )
    baseline = parse_compile_report(baseline_report, source="baseline.json")
    candidate_report = _compile_report()
    _set_move_causes(
        candidate_report,
        (
            ("constant_materialization", 19, 19),
            ("low_slice", 2, 2),
            ("operand_bank_materialization", 10, 10),
        ),
    )
    candidate = parse_compile_report(candidate_report, source="candidate.json")

    show = build_compile_report_show(baseline)
    assert show["entries"][0]["move_causes"] == {
        "kind_count": 2,
        "packet_count": 26,
        "unit_count": 26,
        "causes": [
            {
                "cause": "constant_materialization",
                "packet_count": 16,
                "unit_count": 16,
            },
            {
                "cause": "operand_bank_materialization",
                "packet_count": 10,
                "unit_count": 10,
            },
        ],
    }
    show_text = format_compile_report_show_text(show)
    assert "Move causes (compiler analysis)" in show_text
    assert "operand bank materialization: 10 packets, 10 units" in show_text

    diff = build_compile_report_diff(baseline, candidate)
    assert diff["changed_entry_count"] == 1
    assert diff["unchanged_entry_count"] == 0
    move_diff = diff["entries"][0]["move_causes"]
    assert move_diff["changed_cause_count"] == 2
    assert move_diff["unchanged_cause_count"] == 1
    assert move_diff["causes"][0]["cause"] == "constant_materialization"
    assert move_diff["causes"][0]["packet_count"]["delta"] == 3
    assert move_diff["causes"][1] == {
        "cause": "low_slice",
        "status": "added",
        "candidate": {
            "cause": "low_slice",
            "packet_count": 2,
            "unit_count": 2,
        },
    }
    assert move_diff["causes"][2]["status"] == "unchanged"
    diff_text = format_compile_report_diff_text(diff)
    assert "low slice: added 2 packets, 2 units" in diff_text
    assert (
        "operand bank materialization: unchanged at 10 packets, unchanged at 10 units"
    ) in diff_text


def test_rejects_inconsistent_detailed_move_cause_totals() -> None:
    report = _compile_report()
    _set_move_causes(report, (("constant_materialization", 16, 16),))
    report["entries"]["rows"][0]["move_causes"]["packet_count"] = 17
    document = parse_compile_report(report, source="report.json")

    with pytest.raises(CompileReportError, match="detailed causes total 16"):
        build_compile_report_show(document)


def test_summary_move_causes_preserve_aggregate_without_false_diff() -> None:
    report = _compile_report()
    report["entries"]["rows"][0]["move_causes"] = {
        "kind_count": 2,
        "packet_count": 26,
        "unit_count": 26,
    }
    baseline = parse_compile_report(deepcopy(report), source="baseline.json")
    candidate = parse_compile_report(report, source="candidate.json")

    show = build_compile_report_show(baseline)
    assert show["entries"][0]["move_causes"] == {
        "kind_count": 2,
        "packet_count": 26,
        "unit_count": 26,
    }
    assert "cause breakdown: unavailable" in format_compile_report_show_text(show)
    diff = build_compile_report_diff(baseline, candidate)
    assert diff["entries"] == []
    assert diff["unchanged_entry_count"] == 1


def test_diff_preserves_missing_evidence_and_numeric_deltas() -> None:
    baseline = parse_compile_report(_compile_report(), source="baseline.json")
    candidate_report = _compile_report()
    candidate_entry = candidate_report["entries"]["rows"][0]
    candidate_entry["code_byte_count"] = 480
    candidate_entry["body_instruction_count"] = 89
    candidate_entry["static_instruction_mix"]["wmma_count"] = 6
    candidate_entry["dynamic_instruction_mix"]["matrix_count"] = 6
    candidate_entry["dynamic_instruction_mix"]["wmma_count"] = 6
    candidate_entry["economics"]["operations"]["per_workitem"]["matrix_count"] = 6
    candidate_entry["economics"]["operations"]["per_workitem"]["wmma_count"] = 6
    candidate_entry["economics"]["operations"]["dispatch"]["matrix_count"] = 73728
    candidate_entry["economics"]["operations"]["dispatch"]["wmma_count"] = 73728
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
    dynamic_wmma = entry["execution_economics"]["changed"]["wmma_count"]
    planned_waits = entry["compiler_analysis"]["changed"]["planned_wait_action_count"]
    target_insertions = entry["compiler_analysis"]["changed"][
        "target_insertion_dynamic_packet_count"
    ]

    assert code_bytes["delta"] == -32
    assert code_bytes["change_percent"] == -6.25
    assert body_instructions["delta"] == -8
    assert wmma["delta"] == -2
    assert dynamic_wmma["delta"] == -256
    assert planned_waits["delta"] == -2
    assert target_insertions["delta"] == -128
    assert partial_waits["candidate"] is None
    assert view["changed_entry_count"] == 1
    assert view["unchanged_entry_count"] == 0
    text = format_compile_report_diff_text(view)
    assert "512 B -> 480 B, delta -32 B (-6.25%)" in text
    assert "partial waits: 9 -> unavailable" in text
    assert "WMMA operation effects: 1,024 -> 768, delta -256 (-25.00%)" in text


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
