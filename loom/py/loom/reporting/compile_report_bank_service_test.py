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


def _bank_service_summary(
    *,
    exact_packets: int = 1,
    unknown_packets: int = 0,
    conflict_free_packets: int = 0,
    conflicted_packets: int = 1,
    required_rounds: int = 4,
    uncontended_rounds: int = 2,
    extra_rounds: int = 2,
) -> dict[str, object]:
    return {
        "modeled_packet_count": exact_packets + unknown_packets,
        "exact_packet_count": exact_packets,
        "unknown_packet_count": unknown_packets,
        "structural": {
            "conflict_free_packet_count": conflict_free_packets,
            "conflicted_packet_count": conflicted_packets,
            "required_round_count": required_rounds,
            "uncontended_round_count": uncontended_rounds,
            "extra_round_count": extra_rounds,
            "maximum_request_multiplicity": 2,
        },
        "dynamic": {
            "exact_packet_count": exact_packets,
            "unknown_packet_count": unknown_packets,
            "packet_count": exact_packets,
            "required_round_count": required_rounds,
            "uncontended_round_count": uncontended_rounds,
            "extra_round_count": extra_rounds,
        },
    }


def _bank_service_group() -> dict[str, object]:
    return {
        "index": 0,
        "function": "routed_linear",
        "source_op": "vector.fragment.load",
        "source_op_kind": 80,
        "source_root": "scratch",
        "memory_space": "workgroup",
        "operation": "load",
        "packet": "amdgpu.ds_read_b128",
        "strategy": None,
        "model": {
            "key": "amdgpu.lds.wave32.b128.quad-phases.read.count-each",
            "revision": "llvm-project@bank-model",
            "evidence": "silicon-calibrated-vendor-model",
            "request_policy": "count-each",
            "wave_size": 32,
            "bank_count": 32,
            "bank_word_bytes": 4,
            "packet_bank_words": 4,
        },
        "summary": _bank_service_summary(),
    }


def _compile_report() -> dict[str, object]:
    workload = {
        "workgroup_size": {"x": 128, "y": 1, "z": 1, "flat": 128},
        "workgroup_count": {"x": 1, "y": 1, "z": 1, "flat": 1},
        "dispatch_workitem_count": 128,
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
        "target_key": "gfx1250-a0",
        "target_bundle": "gfx1250-a0",
        "target_snapshot": "gfx1250-a0",
        "target_config": "gfx1250-a0",
        "workload": workload,
        "source_low": {
            "memory": {
                "bank_service": _bank_service_summary(),
                "bank_service_group_count": 1,
                "bank_service_groups": [_bank_service_group()],
            }
        },
        "entries": {
            "count": 1,
            "rows": [
                {
                    "index": 0,
                    "function": "routed_linear",
                    "source_function": "routed_linear",
                    "target_bundle": "gfx1250-a0",
                    "target_snapshot": "gfx1250-a0",
                    "target_export": "routed_linear",
                    "target_export_symbol": None,
                    "target_config": "gfx1250-a0",
                    "workload": workload,
                }
            ],
        },
    }


def test_show_preserves_service_proof_and_model_provenance() -> None:
    document = parse_compile_report(_compile_report(), source="report.json")

    view = build_compile_report_show(document)

    bank_service = view["bank_service"]
    assert bank_service["summary"]["conflicted_packet_count"] == 1
    assert (
        bank_service["groups"][0]["model"]["evidence"]
        == "silicon-calibrated-vendor-model"
    )
    text = format_compile_report_show_text(view)
    assert "Bank service (compiler analysis)" in text
    assert "structural extra rounds: 2" in text
    assert "vector.fragment.load scratch [amdgpu.ds_read_b128]" in text


def test_diff_matches_semantic_groups_and_reports_service_delta() -> None:
    baseline = parse_compile_report(_compile_report(), source="baseline.json")
    candidate_report = _compile_report()
    candidate_memory = candidate_report["source_low"]["memory"]
    candidate_memory["bank_service"]["structural"]["extra_round_count"] = 1
    candidate_memory["bank_service"]["dynamic"]["extra_round_count"] = 1
    candidate_summary = candidate_memory["bank_service_groups"][0]["summary"]
    candidate_summary["structural"]["extra_round_count"] = 1
    candidate_summary["dynamic"]["extra_round_count"] = 1
    candidate = parse_compile_report(candidate_report, source="candidate.json")

    view = build_compile_report_diff(baseline, candidate)

    bank_service = view["bank_service"]
    assert bank_service["changed_group_count"] == 1
    group = bank_service["groups"][0]
    assert group["proof_loss"] is False
    assert group["service"]["changed"]["extra_round_count"]["delta"] == -1
    text = format_compile_report_diff_text(view)
    assert "Bank service diff (compiler analysis)" in text
    assert "structural extra rounds: 2 -> 1, delta -1" in text


def test_diff_calls_out_proof_and_model_regressions() -> None:
    baseline = parse_compile_report(_compile_report(), source="baseline.json")
    candidate_report = _compile_report()
    memory = candidate_report["source_low"]["memory"]
    memory["bank_service"] = _bank_service_summary(
        exact_packets=0,
        unknown_packets=1,
        conflicted_packets=0,
        required_rounds=0,
        uncontended_rounds=0,
        extra_rounds=0,
    )
    group = memory["bank_service_groups"][0]
    group["summary"] = deepcopy(memory["bank_service"])
    group["unknown_evidence"] = {
        "reason": "active-lane-control-not-uniform",
        "mixed_reasons": False,
    }
    group["model"]["revision"] = "llvm-project@new-bank-model"
    candidate = parse_compile_report(candidate_report, source="candidate.json")

    view = build_compile_report_diff(baseline, candidate)

    bank_service = view["bank_service"]
    group_diff = bank_service["groups"][0]
    assert group_diff["proof_loss"] is True
    assert group_diff["model"]["revision"] == {
        "baseline": "llvm-project@bank-model",
        "candidate": "llvm-project@new-bank-model",
    }
    text = format_compile_report_diff_text(view)
    assert "exact proof coverage regressed" in text
    assert "unknown evidence" in text


def test_diff_rejects_duplicate_semantic_group_identity() -> None:
    baseline_report = _compile_report()
    duplicate = deepcopy(
        baseline_report["source_low"]["memory"]["bank_service_groups"][0]
    )
    duplicate["index"] = 1
    baseline_report["source_low"]["memory"]["bank_service_groups"].append(duplicate)
    baseline_report["source_low"]["memory"]["bank_service_group_count"] = 2
    baseline = parse_compile_report(baseline_report, source="baseline.json")
    candidate = parse_compile_report(_compile_report(), source="candidate.json")

    with pytest.raises(CompileReportError, match="duplicate semantic group identity"):
        build_compile_report_diff(baseline, candidate)
