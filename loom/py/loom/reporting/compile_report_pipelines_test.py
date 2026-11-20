# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from copy import deepcopy

import pytest

from loom.reporting.compile_report import CompileReportError, parse_compile_report
from loom.reporting.compile_report_suggestions import (
    CompileReportSuggestionResult,
    build_compile_report_suggestions,
    format_compile_report_suggestions_text,
)
from loom.reporting.compile_report_view import (
    build_compile_report_show,
    format_compile_report_show_text,
)


def _report() -> dict[str, object]:
    # A single image loaded twice and a third tile used only for retained data.
    # Physical occupancy and address extent are independently reported facts.
    return {
        "kind": "loom.compile_report",
        "schema_version": 0,
        "mode": "details",
        "status": {"code": 0, "name": "OK"},
        "entries": {"count": 0, "rows": []},
        "pipeline_plans": [
            {
                "root": "retained_weights",
                "realization": "resident-configuration",
                "program_count": 1,
                "program_code_byte_count": 24,
                "worker_code_byte_count": 48,
                "reserved_storage_byte_count": 98320,
                "worker_storage_byte_count": 128,
                "workers": {
                    "count": 2,
                    "rows": [
                        {
                            "worker_index": index,
                            "entry": "reduce",
                            "placement": {"rank": 2, "x": index, "y": 3},
                            "code_byte_count": 24,
                            "code_capacity_byte_count": 16384,
                            "worker_storage_byte_count": 64,
                            "local_memory_byte_count": extent,
                            "local_memory_capacity_byte_count": 65536,
                        }
                        for index, extent in enumerate((32896, 65536))
                    ],
                },
                "memories": {
                    "count": 3,
                    "rows": [
                        {
                            "memory_index": index,
                            "placement": {"rank": 2, "x": column, "y": row},
                            "reserved_byte_count": reserved,
                            "program_data_byte_count": program,
                            "occupied_byte_count": reserved + program,
                            "high_water_byte_count": extent,
                            "capacity_byte_count": capacity,
                            "maximum_bank_storage_byte_count": bank,
                            "bank_storage_capacity_byte_count": bank_capacity,
                        }
                        for index, (
                            column,
                            row,
                            reserved,
                            program,
                            extent,
                            capacity,
                            bank,
                            bank_capacity,
                        ) in enumerate(
                            (
                                (0, 1, 72, 0, 131104, 524288, 32, 65536),
                                (0, 3, 32776, 64, 32896, 65536, 16384, 16384),
                                (1, 3, 65472, 64, 65536, 65536, 16384, 16384),
                            )
                        )
                    ],
                },
            }
        ],
    }


def test_show_separates_images_placements_and_shared_storage() -> None:
    view = build_compile_report_show(parse_compile_report(_report()))
    plan = view["pipelines"]["rows"][0]
    assert plan["program_code_byte_count"] == 24
    assert plan["worker_code_byte_count"] == 48
    assert plan["memories"]["rows"][0]["occupied_byte_count"] == 72
    assert plan["memories"]["rows"][0]["high_water_byte_count"] == 131104
    assert "channel_count" not in plan
    text = format_compile_report_show_text(view)
    assert "Pipeline retained_weights (resident-configuration)" in text
    assert "1 images; 24 unique code bytes; 48 loaded code bytes" in text
    assert "logical channels: unavailable" in text
    assert "worker[1] reduce at [1,3]: 24/16384 B code; 64 B program data" in text
    assert "memory[0] at [0,1]: 72/524288 B occupied" in text
    assert (
        "32840/65536 B occupied; 32776 B reserved + 64 B program data; extent 32896 B"
        in text
    )


def test_summary_retains_counts_without_inventing_detail_rows() -> None:
    report = _report()
    report["mode"] = "summary"
    plan = report["pipeline_plans"][0]
    del plan["workers"]["rows"]
    del plan["memories"]["rows"]
    plan["channels"] = {"count": 0}
    view = build_compile_report_show(parse_compile_report(report))
    selected = view["pipelines"]["rows"][0]
    assert selected["memories"] == {"count": 3}
    assert selected["workers"] == {"count": 2}
    text = format_compile_report_show_text(view)
    assert "logical channels: 0" in text
    assert "physical memory owners: 3" in text
    assert "worker[" not in text
    assert "memory[" not in text


def test_dma_traffic_distinguishes_unknown_from_zero_for_each_export() -> None:
    report = _report()
    report["pipeline_plans"].append(
        {
            "root": "weight_input",
            "realization": "spatial-program",
            "external_dma_byte_count": 4608,
            "routed_dma_byte_count": 0,
            "channel_storage_byte_count": 2304,
            "channels": {
                "count": 1,
                "rows": [
                    {
                        "channel_index": 0,
                        "transport": "external-dma",
                        "sender": {"owner": "binding", "owner_index": 2, "port": 0},
                        "receiver": {"owner": "worker", "owner_index": 0, "port": 1},
                        "capacity": 2,
                        "record_count": 4,
                        "record_byte_count": 1152,
                        "activation_byte_count": 4608,
                        "local_storage_byte_count": 2304,
                        "external_transfer": {
                            "binding_ordinal": 2,
                            "task_byte_count": 1152,
                            "task_repeat_count": 4,
                            "activation_byte_count": 4608,
                        },
                    }
                ],
            },
        }
    )
    document = parse_compile_report(report)
    show = build_compile_report_show(document)
    suggest = build_compile_report_suggestions(
        document, CompileReportSuggestionResult("test", None)
    )
    for view, render in (
        (show, format_compile_report_show_text),
        (suggest, format_compile_report_suggestions_text),
    ):
        resident, spatial = view["pipelines"]["rows"]
        assert resident["external_dma_byte_count"] is None
        assert resident["routed_dma_byte_count"] is None
        assert spatial["external_dma_byte_count"] == 4608
        assert spatial["routed_dma_byte_count"] == 0
        text = render(view)
        assert text.count("Programmed workers: 2") == 1
        assert "Planned external DMA per activation: unavailable" in text
        assert "Planned external DMA per activation: 4608 B" in text
        assert "Planned routed on-array DMA per activation: 0 B" in text
        assert text.count("Incomplete transfer evidence") == 1
    assert "rows" not in suggest["pipelines"]["rows"][0]["workers"]
    assert "channel_rows" not in suggest["pipelines"]["rows"][1]
    text = format_compile_report_show_text(show)
    assert (
        "binding[2]:0 -> worker[0]:1: 4 records x 1152 B "
        "= 4608 B per activation, 2 slots, 2304 B storage" in text
    )
    assert "binding[2] DMA: 4 tasks x 1152 B = 4608 B per activation" in text


def test_phase_copies_rank_per_call_and_keep_runtime_counts_unknown() -> None:
    report = _report()
    read = dict(
        root="stream",
        phase="restore",
        occurrence=2,
        binding=4,
        direction="read",
        bytes_per_copy=1024,
        copies_per_phase_call=4,
    )
    write = dict(read, phase="save", occurrence=3, direction="write")
    report["pipeline_external_copies"] = [
        read,
        read,
        write,
        # The same child is also called with a different input binding.
        dict(read, occurrence=7, binding=6),
        dict(read, occurrence=7, binding=6, copies_per_phase_call=None),
        # This site has a runtime-sized record but never executes.
        dict(write, bytes_per_copy=None, copies_per_phase_call=0),
        dict(write, bytes_per_copy=0, copies_per_phase_call=None),
        dict(read, root="sink", bytes_per_copy=None),
    ]
    document = parse_compile_report(report)
    show = build_compile_report_show(document)["pipeline_phase_copies"]
    summary = build_compile_report_suggestions(
        document, CompileReportSuggestionResult("test", None)
    )
    stream, sink = summary["pipeline_phase_copies"]
    assert stream["group_count"] == 5
    assert stream["unknown_group_count"] == 1
    assert "groups" not in stream
    assert [
        (row["phase"], row["copies_per_phase_call"], row["bytes_per_phase_call"])
        for row in stream["largest_counted"]
    ] == [("restore", 8, 8192), ("save", 4, 4096)]
    assert [row["bytes_per_phase_call"] for row in show[0]["groups"]] == [
        8192,
        4096,
        None,
        0,
        0,
    ]
    assert sink["unknown_group_count"] == 1
    assert sink["largest_counted"] == []
    assert "range_check" not in sink
    assert (
        "Check whether they save and restore the same byte ranges"
        in stream["range_check"]
    )
    text = format_compile_report_suggestions_text(summary)
    assert "read binding[4] restore (call site 2): 8 x 1,024 B = 8,192 B" in text
    assert "write binding[4] save (call site 3): 4 x 1,024 B = 4,096 B" in text
    assert "1 of 5 groups lack a proven byte total" in text

    # Ordinary streaming has independent input and output bindings.
    write["binding"] = 1
    independent = build_compile_report_show(parse_compile_report(report))
    assert "range_check" not in independent["pipeline_phase_copies"][0]


def test_phase_copy_totals_span_different_external_runs() -> None:
    site = dict(
        root="pipeline",
        phase="projection",
        occurrence=0,
        binding=4,
        direction="read",
        bytes_per_copy=131072,
        copies_per_phase_call=1,
    )
    report = _report()
    report["pipeline_external_copies"] = [
        dict(site, contiguous_run_bytes=2048),
        dict(site, contiguous_run_bytes=2048),
        dict(site, direction="write", contiguous_run_bytes=64),
        dict(site, direction="write", contiguous_run_bytes=2048),
        dict(site, binding=3),
        dict(site, binding=3, contiguous_run_bytes=64),
    ]
    view = build_compile_report_show(parse_compile_report(report))
    read, write, mixed = view["pipeline_phase_copies"][0]["groups"]
    assert [g["bytes_per_phase_call"] for g in (read, write, mixed)] == [262144] * 3
    assert read["contiguous_run_bytes"] == 2048
    assert "contiguous_run_bytes" not in write
    assert "contiguous_run_bytes" not in mixed
    text = format_compile_report_show_text(view)
    assert "262,144 B (2,048 consecutive B per run)" in text
    assert text.count("consecutive B per run") == 1


@pytest.mark.parametrize(
    ("width", "run"), [(1024, 0), (1024, 48), (1024, 1024), (None, 4)]
)
def test_invalid_strided_external_run_names_its_source(
    width: int | None, run: int
) -> None:
    report = _report()
    report["pipeline_external_copies"] = [
        dict(
            root="stream",
            phase="copy",
            occurrence=0,
            binding=0,
            direction="read",
            bytes_per_copy=width,
            copies_per_phase_call=1,
            contiguous_run_bytes=run,
        )
    ]
    with pytest.raises(
        CompileReportError, match=r"pipeline_external_copies\[0\].contiguous_run_bytes"
    ):
        build_compile_report_show(parse_compile_report(report))


def test_show_keeps_exports_separate_and_preserves_placement_rank() -> None:
    report = _report()
    plans = report["pipeline_plans"]
    plans.append(deepcopy(plans[0]))
    plans[1]["root"] = "second_pipeline"
    plans[1]["workers"]["rows"][0]["placement"] = {"rank": 3, "x": 2, "y": 4, "z": 6}
    plans[1]["workers"]["rows"][1]["placement"] = {"rank": 1, "x": 7}
    view = build_compile_report_show(parse_compile_report(report))
    assert view["pipelines"]["count"] == 2
    text = format_compile_report_show_text(view)
    assert "Pipeline second_pipeline" in text
    assert "reduce at [2,4,6]" in text
    assert "reduce at [7]" in text


@pytest.mark.parametrize("value", [None, -1, True, "72"])
def test_rejects_invalid_memory_counts_with_a_precise_path(value: object) -> None:
    report = _report()
    report["pipeline_plans"][0]["memories"]["rows"][0]["occupied_byte_count"] = value
    with pytest.raises(
        CompileReportError, match=r"memories.rows\[0\].occupied_byte_count"
    ):
        build_compile_report_show(parse_compile_report(report))


def test_rejects_incomplete_physical_inventory() -> None:
    report = _report()
    report["pipeline_plans"][0]["memories"]["count"] = 4
    with pytest.raises(CompileReportError, match="expected 4 rows"):
        build_compile_report_show(parse_compile_report(report))


@pytest.mark.parametrize("value", [None, {}, "pipeline"])
def test_rejects_invalid_plan_collection(value: object) -> None:
    report = _report()
    report["pipeline_plans"] = value
    with pytest.raises(CompileReportError, match="pipeline_plans: expected array"):
        build_compile_report_show(parse_compile_report(report))
