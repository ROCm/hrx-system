# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import copy

import pytest

from loom.reporting.compile_report import CompileReportError, parse_compile_report
from loom.reporting.compile_report_suggestions import (
    CompileReportSuggestionOptions,
)
from loom.target.arch.amd.xdna.aie2p.compile_report_suggestions import (
    AIE2P_COMPILE_REPORT_SUGGESTION_PROVIDER,
)
from loom.target.arch.compile_report_suggestions import suggest_compile_report


def _compile_report() -> dict[str, object]:
    return {
        "kind": "loom.compile_report",
        "schema_version": 0,
        "mode": "details",
        "status": {"code": 0, "name": "OK"},
        "target_family": "amd.xdna.aie2p",
        "entries": {
            "count": 1,
            "rows": [
                {
                    "index": 0,
                    "function": "projection",
                    "allocation_spill_count": 2,
                    "allocation_materialized_reload_bytes": 256,
                    "body_instruction_count": 128,
                    "coissued_instruction_count": 16,
                    "coissued_component_count": 32,
                }
            ],
        },
        "pipeline_plans": [
            {
                "root": "q5_gate_up",
                "realization": "spatial-program",
                "workers": {
                    "count": 1,
                    "rows": [
                        {
                            "worker_index": 0,
                            "entry": "projection",
                            "code_byte_count": 14336,
                            "code_capacity_byte_count": 16384,
                            "worker_storage_byte_count": 1024,
                            "channel_storage_byte_count": 1024,
                            "local_memory_byte_count": 32768,
                            "local_memory_capacity_byte_count": 65536,
                            "maximum_bank_storage_byte_count": 14336,
                            "bank_storage_capacity_byte_count": 16384,
                        }
                    ],
                },
                "channels": {
                    "count": 1,
                    "rows": [
                        {
                            "channel_index": 0,
                            "receiver": {
                                "owner": "worker",
                                "owner_index": 0,
                                "port": 0,
                            },
                            "storage": {
                                "schema": "ggml.q5_k",
                                "transform": "transpose-4x",
                                "transform_extra_byte_count": 704,
                            },
                            "external_transfer": {
                                "task_byte_count": 352,
                                "task_repeat_count": 16,
                            },
                        }
                    ],
                },
            }
        ],
    }


def test_suggests_ordered_experiments_from_exact_aie2p_evidence() -> None:
    document = parse_compile_report(_compile_report(), source="report.json")

    result = AIE2P_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(document)

    assert result.unavailable_reason is None
    assert [suggestion.suggestion_id for suggestion in result.suggestions] == [
        "aie2p.spill_traffic",
        "aie2p.vliw_coissue_density",
        "aie2p.code_headroom",
        "aie2p.bank_pressure",
        "aie2p.storage_transform",
    ]
    code_evidence = result.suggestions[2].evidence
    assert code_evidence[0].path == (
        "pipeline_plans[0].workers.rows[0].code_byte_count"
    )
    assert code_evidence[0].value == 14336


def test_experimental_mode_surfaces_small_repeated_dma_tasks() -> None:
    document = parse_compile_report(_compile_report())

    result = AIE2P_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(
        document,
        CompileReportSuggestionOptions(include_experimental=True),
    )

    suggestion = result.suggestions[-1]
    assert suggestion.suggestion_id == "aie2p.dma_record_coalescing"
    assert suggestion.confidence.value == "experimental"
    evidence = {item.path: item.value for item in suggestion.evidence}
    assert (
        evidence[
            "pipeline_plans[0].channels.rows[0].external_transfer.task_repeat_count"
        ]
        == 16
    )


def test_empty_packets_are_scoped_to_one_repeated_native_loop() -> None:
    report = _compile_report()
    report["pipeline_plans"][0]["placed_functions"] = [
        {
            "function": "mix",
            "worker_index": 3,
            "packet_loops": [
                {
                    "byte_offset": 16,
                    "packet_count": 26,
                    "trip_count": 256,
                    "empty_packet_count": 11,
                },
                # The same static footprint executed once.
                {
                    "byte_offset": 240,
                    "packet_count": 26,
                    "trip_count": 1,
                    "empty_packet_count": 11,
                },
                # A long running loop with nearly every issue position occupied.
                {
                    "byte_offset": 480,
                    "packet_count": 93,
                    "trip_count": 256,
                    "empty_packet_count": 2,
                },
                # A short repeat and a report predating the occupancy field.
                {
                    "byte_offset": 960,
                    "packet_count": 8,
                    "trip_count": 2,
                    "empty_packet_count": 4,
                },
                {"byte_offset": 1008, "packet_count": 26, "trip_count": 256},
            ],
        }
    ]
    result = suggest_compile_report(parse_compile_report(report))
    (finding,) = [
        s for s in result.suggestions if s.suggestion_id == "aie2p.empty_loop_packets"
    ]
    assert finding.entry_name == "mix +0x10 (worker 3)"
    assert finding.example_magnitude == 2816
    assert "2,816 empty packets per loop entry" in finding.action
    assert finding.evidence[1].path.endswith("packet_loops[0].empty_packet_count")
    assert finding.evidence[1].value == 11


def test_healthy_direct_pipeline_has_no_findings() -> None:
    report = copy.deepcopy(_compile_report())
    entry = report["entries"]["rows"][0]
    entry["allocation_spill_count"] = 0
    entry["allocation_materialized_reload_bytes"] = 0
    entry["coissued_instruction_count"] = 64
    entry["coissued_component_count"] = 128
    worker = report["pipeline_plans"][0]["workers"]["rows"][0]
    worker["code_byte_count"] = 8192
    worker["maximum_bank_storage_byte_count"] = 512
    channel = report["pipeline_plans"][0]["channels"]["rows"][0]
    channel["storage"]["transform"] = "none"
    channel["storage"]["transform_extra_byte_count"] = 0
    channel["external_transfer"]["task_repeat_count"] = 1

    result = AIE2P_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(
        parse_compile_report(report),
        CompileReportSuggestionOptions(include_experimental=True),
    )

    assert result.suggestions == ()


def test_summary_surfaces_aggregate_physical_pressure() -> None:
    report = copy.deepcopy(_compile_report())
    report["mode"] = "summary"
    pipeline_plan = report["pipeline_plans"][0]
    pipeline_plan["maximum_worker_code_byte_count"] = 14336
    pipeline_plan["minimum_worker_code_headroom_byte_count"] = 2048
    pipeline_plan["maximum_bank_storage_byte_count"] = 14336
    pipeline_plan["bank_storage_capacity_byte_count"] = 16384
    del pipeline_plan["workers"]["rows"]
    del pipeline_plan["channels"]["rows"]

    result = AIE2P_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(
        parse_compile_report(report)
    )

    suggestion_ids = [item.suggestion_id for item in result.suggestions]
    assert suggestion_ids[-2:] == ["aie2p.code_headroom", "aie2p.bank_pressure"]


def test_rejects_impossible_physical_capacity_evidence() -> None:
    report = _compile_report()
    worker = report["pipeline_plans"][0]["workers"]["rows"][0]
    worker["code_byte_count"] = 16385
    document = parse_compile_report(report)

    with pytest.raises(CompileReportError, match="code_byte_count: exceeds capacity"):
        AIE2P_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(document)


def test_arch_dispatch_selects_aie2p_provider() -> None:
    result = suggest_compile_report(parse_compile_report(_compile_report()))

    assert result.provider_name == "amd.xdna.aie2p"
    assert result.unavailable_reason is None


@pytest.mark.parametrize("mode", ["summary", "details"])
def test_preserves_evidence_for_each_pipeline_export(mode: str) -> None:
    report = _compile_report()
    report["mode"] = mode
    plans = report["pipeline_plans"]
    plans.append(copy.deepcopy(plans[0]))
    plans[1]["root"] = "q4_gate_up"
    plans[1]["workers"]["rows"][0]["entry"] = "q4_projection"
    plans[1]["workers"]["rows"][0]["code_byte_count"] = 12288
    if mode == "summary":
        for plan in plans:
            code_bytes = plan["workers"]["rows"][0]["code_byte_count"]
            plan["maximum_worker_code_byte_count"] = code_bytes
            plan["minimum_worker_code_headroom_byte_count"] = 16384 - code_bytes
            plan["maximum_bank_storage_byte_count"] = 14336
            plan["bank_storage_capacity_byte_count"] = 16384
            del plan["workers"]["rows"]
            del plan["channels"]["rows"]

    result = suggest_compile_report(parse_compile_report(report))
    code_findings = [
        item
        for item in result.suggestions
        if item.suggestion_id == "aie2p.code_headroom"
    ]

    for index, finding in enumerate(code_findings):
        plan_path = f"pipeline_plans[{index}]"
        expected_path = (
            f"{plan_path}.maximum_worker_code_byte_count"
            if mode == "summary"
            else f"{plan_path}.workers.rows[0].code_byte_count"
        )
        assert finding.evidence[0].path == expected_path
    assert [item.evidence[0].value for item in code_findings] == [14336, 12288]
    assert [item.entry_name for item in code_findings] == (
        ["q5_gate_up", "q4_gate_up"]
        if mode == "summary"
        else ["projection", "q4_projection"]
    )
    if mode == "details":
        storage_findings = [
            item
            for item in result.suggestions
            if item.suggestion_id == "aie2p.storage_transform"
        ]
        assert [item.entry_name for item in storage_findings] == [
            "q5_gate_up:worker[0]",
            "q4_gate_up:worker[0]",
        ]
        assert [item.evidence[0].path for item in storage_findings] == [
            "pipeline_plans[0].channels.rows[0].storage.schema",
            "pipeline_plans[1].channels.rows[0].storage.schema",
        ]


@pytest.mark.parametrize("plans", [None, {}, "pipeline"])
def test_rejects_non_array_pipeline_plans(plans: object) -> None:
    report = _compile_report()
    report["pipeline_plans"] = plans
    with pytest.raises(CompileReportError, match="pipeline_plans: expected array"):
        suggest_compile_report(parse_compile_report(report))


def test_validates_rows_in_later_pipeline_exports() -> None:
    report = _compile_report()
    plans = report["pipeline_plans"]
    plans.append(copy.deepcopy(plans[0]))
    plans[1]["workers"]["rows"][0]["worker_index"] = 1
    with pytest.raises(
        CompileReportError,
        match=r"pipeline_plans\[1\]\.workers\.rows\[0\]\.worker_index: expected 0",
    ):
        suggest_compile_report(parse_compile_report(report))


def test_configuration_reports_memory_only_tiles_without_inventing_channels() -> None:
    report = _compile_report()
    report["entries"] = {"count": 0, "rows": []}
    plan = report["pipeline_plans"][0]
    plan["realization"] = "resident-configuration"
    del plan["channels"]
    worker = plan["workers"]["rows"][0]
    worker["code_byte_count"] = 512
    # The same occupied bank may be visible through several loaded workers.
    # Physical memory rows own pressure suggestions when they are available.
    plan["memories"] = {
        "count": 2,
        "rows": [
            {
                "memory_index": 0,
                "placement": {"rank": 2, "x": 0, "y": 3},
                "high_water_byte_count": 32768,
                "capacity_byte_count": 65536,
                "maximum_bank_storage_byte_count": 14336,
                "bank_storage_capacity_byte_count": 16384,
            },
            {
                "memory_index": 1,
                "placement": {"rank": 2, "x": 1, "y": 1},
                "high_water_byte_count": 458752,
                "capacity_byte_count": 524288,
                "maximum_bank_storage_byte_count": 65536,
                "bank_storage_capacity_byte_count": 65536,
            },
        ],
    }

    result = suggest_compile_report(parse_compile_report(report))

    suggestions = result.suggestions
    assert [(item.suggestion_id, item.entry_name) for item in suggestions] == [
        ("aie2p.bank_pressure", "q5_gate_up:tile[0,3]"),
        ("aie2p.bank_pressure", "q5_gate_up:tile[1,1]"),
        ("aie2p.local_memory_pressure", "q5_gate_up:tile[1,1]"),
    ]
    assert suggestions[-1].evidence[0].path == (
        "pipeline_plans[0].memories.rows[1].high_water_byte_count"
    )
