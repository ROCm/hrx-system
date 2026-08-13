# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from copy import deepcopy

import pytest

from loom.reporting.compile_report import CompileReportError, parse_compile_report
from loom.reporting.compile_report_view import (
    build_compile_report_diff,
    build_compile_report_show,
    format_compile_report_diff_text,
    format_compile_report_show_text,
)


def _exact_access(
    *,
    subgroup_size: int = 32,
    packet_bytes: int = 8,
    lane_stride_bytes: int = 64,
    coverage: str = "gapped",
) -> dict[str, object]:
    requested_bytes = subgroup_size * packet_bytes
    if coverage == "dense":
        assert lane_stride_bytes <= packet_bytes
        span_bytes = (subgroup_size - 1) * lane_stride_bytes + packet_bytes
        unique_bytes = span_bytes
        region_count = 1
        maximum_gap_bytes = 0
    else:
        assert lane_stride_bytes > packet_bytes
        span_bytes = (subgroup_size - 1) * lane_stride_bytes + packet_bytes
        unique_bytes = requested_bytes
        region_count = subgroup_size
        maximum_gap_bytes = lane_stride_bytes - packet_bytes
    return {
        "proof": "exact",
        "address": {
            "lane_address_proof": "compiled-fragment-lane-register-layout",
            "active_lane_proof": "subgroup-uniform-control-full-wave",
            "lane_mapping": "linear",
            "subgroup_size": subgroup_size,
            "per_lane_packet_bytes": packet_bytes,
            "linear_lane_stride_bytes": lane_stride_bytes,
            "lane_terms": [
                {
                    "divisor": 1,
                    "modulus": 0,
                    "byte_stride": lane_stride_bytes,
                }
            ],
        },
        "geometry": {
            "interval_coverage": coverage,
            "subgroup_requested_bytes": requested_bytes,
            "subgroup_unique_bytes": unique_bytes,
            "subgroup_span_bytes": span_bytes,
            "maximum_adjacent_lane_delta_bytes": lane_stride_bytes,
            "maximum_uncovered_gap_bytes": maximum_gap_bytes,
            "distinct_lane_address_count": subgroup_size,
            "contiguous_region_count": region_count,
        },
    }


def _unknown_access(*, subgroup_size: int = 32) -> dict[str, object]:
    return {
        "proof": "unknown",
        "unknown_reason": "active-lane-control-not-uniform",
        "address": {
            "lane_address_proof": "compiled-fragment-lane-register-layout",
            "active_lane_proof": "unproven",
            "lane_mapping": "digit-terms",
            "subgroup_size": subgroup_size,
            "per_lane_packet_bytes": 16,
            "linear_lane_stride_bytes": 0,
            "lane_terms": [{"divisor": 1, "modulus": 16, "byte_stride": 4096}],
        },
    }


def _summary(
    access: dict[str, object],
    *,
    modeled_packets: int = 1,
    dynamic_packets: int = 1,
    dynamic_proof_exact: bool = True,
) -> dict[str, object]:
    is_exact = access["proof"] == "exact"
    geometry = access.get("geometry", {})
    is_dense = is_exact and geometry.get("interval_coverage") == "dense"
    is_gapped = is_exact and geometry.get("interval_coverage") == "gapped"
    is_overlapping = is_exact and (
        geometry["subgroup_requested_bytes"] > geometry["subgroup_unique_bytes"]
    )
    exact_dynamic_packets = modeled_packets if dynamic_proof_exact else 0
    unknown_dynamic_packets = modeled_packets - exact_dynamic_packets
    if not is_exact:
        dynamic_packets = 0
        exact_dynamic_packets = 0
        unknown_dynamic_packets = modeled_packets
    return {
        "modeled_packet_count": modeled_packets,
        "exact_packet_count": modeled_packets if is_exact else 0,
        "unknown_packet_count": 0 if is_exact else modeled_packets,
        "structural": {
            "dense_packet_count": modeled_packets if is_dense else 0,
            "gapped_packet_count": modeled_packets if is_gapped else 0,
            "overlapping_packet_count": modeled_packets if is_overlapping else 0,
        },
        "dynamic": {
            "exact_packet_count": exact_dynamic_packets,
            "unknown_packet_count": unknown_dynamic_packets,
            "packet_count": dynamic_packets,
            "dense_packet_count": dynamic_packets if is_dense else 0,
            "gapped_packet_count": dynamic_packets if is_gapped else 0,
            "overlapping_packet_count": dynamic_packets if is_overlapping else 0,
        },
    }


def _group(
    access: dict[str, object],
    *,
    packet: str = "amdgpu.global_load_b64_saddr",
    strategy: str = "fragment_load",
    modeled_packets: int = 1,
    dynamic_packets: int = 1,
) -> dict[str, object]:
    return {
        "index": 0,
        "function": "dflash_attention",
        "source_op": "vector.fragment.load",
        "source_op_kind": 3603,
        "source_root": "input",
        "source_root_argument_index": 0,
        "memory_space": "global",
        "operation": "load",
        "packet": packet,
        "strategy": strategy,
        "access": access,
        "summary": _summary(
            access,
            modeled_packets=modeled_packets,
            dynamic_packets=dynamic_packets,
        ),
    }


def _sum_summaries(groups: list[dict[str, object]]) -> dict[str, object]:
    summary = {
        "modeled_packet_count": 0,
        "exact_packet_count": 0,
        "unknown_packet_count": 0,
        "structural": {
            "dense_packet_count": 0,
            "gapped_packet_count": 0,
            "overlapping_packet_count": 0,
        },
        "dynamic": {
            "exact_packet_count": 0,
            "unknown_packet_count": 0,
            "packet_count": 0,
            "dense_packet_count": 0,
            "gapped_packet_count": 0,
            "overlapping_packet_count": 0,
        },
    }
    for group in groups:
        group_summary = group["summary"]
        for field in (
            "modeled_packet_count",
            "exact_packet_count",
            "unknown_packet_count",
        ):
            summary[field] += group_summary[field]
        for category in ("structural", "dynamic"):
            for field, value in group_summary[category].items():
                summary[category][field] += value
    return summary


def _compile_report(
    groups: list[dict[str, object]],
    *,
    target_key: str = "gfx1250-a0",
    subgroup_size: int = 32,
) -> dict[str, object]:
    workload = {
        "workgroup_size": {"x": 128, "y": 1, "z": 1, "flat": 128},
        "workgroup_count": {"x": 1, "y": 1, "z": 1, "flat": 1},
        "dispatch_workitem_count": 128,
    }
    groups = deepcopy(groups)
    for index, group in enumerate(groups):
        group["index"] = index
    return {
        "kind": "loom.compile_report",
        "schema_version": 0,
        "mode": "summary",
        "artifact_kind": "hal-executable",
        "artifact_format": "hsaco",
        "backend": "amdgpu-hal",
        "status": {"code": 0, "name": "OK"},
        "target_family": "AMDGPU",
        "target_key": target_key,
        "target_bundle": "test_target",
        "target_snapshot": "test_target",
        "target_config": "test_target",
        "target_resources": {"subgroup_size": subgroup_size},
        "workload": workload,
        "source_low": {
            "memory": {
                "subgroup_access": _sum_summaries(groups),
                "subgroup_access_group_count": len(groups),
                "subgroup_access_groups": groups,
            }
        },
        "entries": {
            "count": 1,
            "rows": [
                {
                    "index": 0,
                    "function": "dflash_attention",
                    "source_function": "dflash_attention",
                    "target_bundle": "test_target",
                    "target_snapshot": "test_target",
                    "target_export": "dflash_attention",
                    "target_export_symbol": None,
                    "target_config": "test_target",
                    "target_resources": {"subgroup_size": subgroup_size},
                    "workload": workload,
                }
            ],
        },
    }


@pytest.mark.parametrize(
    ("target_key", "subgroup_size"),
    [
        ("gfx942", 64),
        ("gfx1250-a0", 32),
    ],
)
def test_show_surfaces_exact_geometry_and_lane_formula(
    target_key: str,
    subgroup_size: int,
) -> None:
    report = _compile_report(
        [
            _group(
                _exact_access(subgroup_size=subgroup_size),
                dynamic_packets=16,
            )
        ],
        target_key=target_key,
        subgroup_size=subgroup_size,
    )
    document = parse_compile_report(report, source="report.json")

    view = build_compile_report_show(document)

    subgroup_access = view["subgroup_access"]
    assert subgroup_access["summary"]["gapped_packet_count"] == 1
    group = subgroup_access["groups"][0]
    expected_span_bytes = (subgroup_size - 1) * 64 + 8
    assert group["access"]["geometry"]["subgroup_span_bytes"] == (expected_span_bytes)
    text = format_compile_report_show_text(view)
    assert "Wave memory access (compiler analysis)" in text
    assert "shapes: 0 dense, 1 gapped, 0 overlapping" in text
    assert f"{subgroup_size} lanes, 8 B/lane; linear offset: lane * 64 B" in text
    assert (
        f"requested {subgroup_size * 8:,} B, unique {subgroup_size * 8:,} B, "
        f"span {expected_span_bytes:,} B"
    ) in text
    assert (
        f"{subgroup_size} distinct starts, {subgroup_size} regions, max gap 56 B"
    ) in text


def test_show_rejects_geometry_for_another_target_subgroup_size() -> None:
    report = _compile_report(
        [_group(_exact_access(subgroup_size=32))],
        target_key="gfx942",
        subgroup_size=64,
    )
    document = parse_compile_report(report, source="report.json")

    with pytest.raises(
        CompileReportError,
        match="expected target subgroup size 64, got 32",
    ):
        build_compile_report_show(document)


def test_show_falls_back_to_report_target_subgroup_size() -> None:
    report = _compile_report(
        [_group(_exact_access(subgroup_size=32))],
        target_key="gfx942",
        subgroup_size=64,
    )
    report["source_low"]["memory"]["subgroup_access_groups"][0]["function"] = (
        "unmatched_kernel"
    )
    document = parse_compile_report(report, source="report.json")

    with pytest.raises(
        CompileReportError,
        match="expected target subgroup size 64, got 32",
    ):
        build_compile_report_show(document)


def test_show_surfaces_unknown_proof_reason() -> None:
    report = _compile_report([_group(_unknown_access(), dynamic_packets=0)])
    document = parse_compile_report(report, source="report.json")

    view = build_compile_report_show(document)

    group = view["subgroup_access"]["groups"][0]
    assert group["access"]["proof"] == "unknown"
    text = format_compile_report_show_text(view)
    assert "proof: unknown (active-lane-control-not-uniform)" in text
    assert "geometry:" not in text


def test_diff_keeps_packet_variants_under_one_source() -> None:
    baseline_access = _exact_access(
        packet_bytes=2,
        lane_stride_bytes=2,
        coverage="dense",
    )
    baseline_group = _group(
        baseline_access,
        packet="amdgpu.global_load_b16_d16_saddr",
        strategy="scalar_fragment_load",
        modeled_packets=128,
        dynamic_packets=128,
    )
    baseline = parse_compile_report(
        _compile_report([baseline_group]), source="baseline.json"
    )
    candidate_access = _exact_access(
        packet_bytes=16,
        lane_stride_bytes=4096,
        coverage="gapped",
    )
    candidate_group = _group(
        candidate_access,
        packet="amdgpu.global_load_b128_saddr",
        strategy="wide_fragment_load",
        modeled_packets=16,
        dynamic_packets=16,
    )
    candidate = parse_compile_report(
        _compile_report([candidate_group]), source="candidate.json"
    )

    view = build_compile_report_diff(baseline, candidate)

    subgroup_access = view["subgroup_access"]
    assert subgroup_access["changed_source_count"] == 1
    source = subgroup_access["sources"][0]
    assert source["identity"]["source_root"] == "input"
    assert len(source["removed_variants"]) == 1
    assert len(source["added_variants"]) == 1
    assert (
        source["added_variants"][0]["access"]["geometry"]["maximum_uncovered_gap_bytes"]
        == 4080
    )
    text = format_compile_report_diff_text(view)
    assert "Wave memory access diff (compiler analysis)" in text
    assert "sources: 1 changed, 0 unchanged" in text
    assert "[amdgpu.global_load_b16_d16_saddr; strategy=scalar_fragment_load]" in text
    assert "[amdgpu.global_load_b128_saddr; strategy=wide_fragment_load]" in text
    assert "geometry: gapped; requested 512 B, unique 512 B" in text
    assert "max gap 4,080 B" in text


def test_diff_preserves_multiple_variants_without_guessing_a_pair() -> None:
    access = _exact_access(packet_bytes=2, lane_stride_bytes=2, coverage="dense")
    baseline_groups = [
        _group(
            access,
            packet="amdgpu.global_load_b16_d16_saddr",
            strategy="scalar_fragment_load",
        ),
        _group(
            access,
            packet="amdgpu.global_load_b16_d16_hi_saddr",
            strategy="scalar_fragment_load",
        ),
    ]
    candidate_group = _group(
        _exact_access(
            packet_bytes=16,
            lane_stride_bytes=4096,
            coverage="gapped",
        ),
        packet="amdgpu.global_load_b128_saddr",
        strategy="wide_fragment_load",
    )
    baseline = parse_compile_report(
        _compile_report(baseline_groups), source="baseline.json"
    )
    candidate = parse_compile_report(
        _compile_report([candidate_group]), source="candidate.json"
    )

    source = build_compile_report_diff(baseline, candidate)["subgroup_access"][
        "sources"
    ][0]

    assert len(source["removed_variants"]) == 2
    assert len(source["added_variants"]) == 1
    assert source["changed_variants"] == []


def test_diff_reports_execution_count_change_on_same_variant() -> None:
    access = _exact_access(packet_bytes=2, lane_stride_bytes=2, coverage="dense")
    baseline = parse_compile_report(
        _compile_report([_group(access, modeled_packets=1, dynamic_packets=1)]),
        source="baseline.json",
    )
    candidate = parse_compile_report(
        _compile_report([_group(access, modeled_packets=2, dynamic_packets=5)]),
        source="candidate.json",
    )

    source = build_compile_report_diff(baseline, candidate)["subgroup_access"][
        "sources"
    ][0]

    assert source["added_variants"] == []
    assert source["removed_variants"] == []
    assert len(source["changed_variants"]) == 1
    assert (
        source["changed_variants"][0]["summary"]["changed"]["dynamic_packet_count"][
            "delta"
        ]
        == 4
    )


def test_show_rejects_duplicate_semantic_access_variant() -> None:
    group = _group(_exact_access())
    report = _compile_report([group, deepcopy(group)])
    document = parse_compile_report(report, source="report.json")

    with pytest.raises(CompileReportError, match="duplicate semantic access variant"):
        build_compile_report_show(document)


def test_show_rejects_geometry_summary_disagreement() -> None:
    report = _compile_report([_group(_exact_access())])
    group_summary = report["source_low"]["memory"]["subgroup_access_groups"][0][
        "summary"
    ]
    group_summary["structural"]["dense_packet_count"] = 1
    group_summary["structural"]["gapped_packet_count"] = 0
    report["source_low"]["memory"]["subgroup_access"] = deepcopy(group_summary)
    document = parse_compile_report(report, source="report.json")

    with pytest.raises(CompileReportError, match="disagrees with retained exact"):
        build_compile_report_show(document)
