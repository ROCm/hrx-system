# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loom.tools.compile_report import main


def _write_report(
    path: Path,
    *,
    code_byte_count: int = 512,
    target_family: str = "AMDGPU",
    target_key: str = "gfx11-generic",
    target_record: str = "gfx11-generic",
    experimental_bank_conflict: bool = False,
) -> None:
    workload = {
        "workgroup_size": {"x": 64, "y": 1, "z": 1, "flat": 64},
        "workgroup_count": {"x": 4, "y": 1, "z": 1, "flat": 4},
        "dispatch_workitem_count": 256,
    }
    report = {
        "kind": "loom.compile_report",
        "schema_version": 0,
        "mode": "summary",
        "artifact_kind": "hal-executable",
        "artifact_format": "elf",
        "backend": "amdgpu-hal",
        "status": {"code": 0, "name": "OK"},
        "target_family": target_family,
        "target_key": target_key,
        "target_bundle": target_record,
        "target_snapshot": target_record,
        "target_config": target_record,
        "workload": workload,
        "entries": {
            "count": 1,
            "rows": [
                {
                    "index": 0,
                    "function": "kernel",
                    "source_function": "kernel",
                    "target_bundle": target_record,
                    "target_snapshot": target_record,
                    "target_export": "kernel",
                    "target_export_symbol": None,
                    "target_config": target_record,
                    "workload": workload,
                    "instruction_count": 100,
                    "code_byte_count": code_byte_count,
                    "code_storage_byte_count": code_byte_count,
                    "local_memory_bytes": 0,
                    "private_memory_bytes": 0,
                }
            ],
        },
    }
    if experimental_bank_conflict:
        bank_service = {
            "modeled_packet_count": 1,
            "exact_packet_count": 1,
            "unknown_packet_count": 0,
            "structural": {
                "conflict_free_packet_count": 0,
                "conflicted_packet_count": 1,
                "required_round_count": 16,
                "uncontended_round_count": 8,
                "extra_round_count": 8,
                "maximum_request_multiplicity": 2,
            },
            "dynamic": {
                "exact_packet_count": 1,
                "unknown_packet_count": 0,
                "packet_count": 1,
                "required_round_count": 16,
                "uncontended_round_count": 8,
                "extra_round_count": 8,
            },
        }
        report["source_low"] = {
            "memory": {
                "bank_service": bank_service,
                "bank_service_group_count": 1,
                "bank_service_groups": [
                    {
                        "index": 0,
                        "function": "kernel",
                        "source_op": "vector.fragment.load",
                        "source_op_kind": 80,
                        "source_root": "scratch",
                        "memory_space": "workgroup",
                        "operation": "load",
                        "packet": "amdgpu.ds_read_b128",
                        "strategy": None,
                        "model": {
                            "key": (
                                "amdgpu.lds.wave32.b128.quad-phases.read.count-each"
                            ),
                            "revision": "ROCm/rocm-libraries@model",
                            "evidence": "vendor-software-model-unvalidated",
                            "request_policy": "count-each",
                            "wave_size": 32,
                            "bank_count": 32,
                            "bank_word_bytes": 4,
                            "packet_bank_words": 4,
                        },
                        "summary": bank_service,
                    }
                ],
            }
        }
    path.write_text(json.dumps(report), encoding="utf-8")


def test_show_json_reads_direct_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "report.json"
    _write_report(report_path)

    assert main(["show", str(report_path), "--format=json"]) == 0

    captured = capsys.readouterr()
    view = json.loads(captured.out)
    assert captured.err == ""
    assert view["kind"] == "loom.compile_report.show"
    assert view["schema_version"] == 0
    assert view["missing_evidence"] == "omitted_metrics_are_unavailable"
    assert view["entries"][0]["identity"] == {"name": "kernel"}


def test_diff_text_reports_changed_metric(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write_report(baseline_path, code_byte_count=512)
    _write_report(candidate_path, code_byte_count=480)

    assert main(["diff", str(baseline_path), str(candidate_path)]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "code bytes: 512 B -> 480 B" in captured.out


def test_diff_force_compares_single_mismatched_entries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write_report(baseline_path, code_byte_count=512)
    _write_report(candidate_path, code_byte_count=480)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_entry = candidate["entries"]["rows"][0]
    candidate_entry["function"] = "candidate_kernel"
    candidate_entry["source_function"] = "candidate_kernel"
    candidate_entry["target_export"] = "candidate_kernel"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    assert main(["diff", str(baseline_path), str(candidate_path)]) == 2
    strict = capsys.readouterr()
    assert strict.out == ""
    assert "missing entry" in strict.err

    assert (
        main(
            [
                "diff",
                str(baseline_path),
                str(candidate_path),
                "--force",
                "--format=json",
            ]
        )
        == 0
    )
    forced = capsys.readouterr()
    view = json.loads(forced.out)
    assert forced.err == ""
    assert view["forced"] is True
    assert [row["path"] for row in view["identity_mismatches"]] == [
        "entry.function",
        "entry.source_function",
        "entry.target_export",
    ]
    assert view["entries"][0]["identities"] == {
        "baseline": {"name": "kernel"},
        "candidate": {"name": "candidate_kernel"},
    }
    assert (
        view["entries"][0]["artifact_facts"]["changed"]["code_byte_count"]["delta"]
        == -32
    )


def test_diff_json_preserves_qualified_target_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    target_key = "gfx942:sramecc+:xnack-"
    _write_report(
        baseline_path,
        code_byte_count=512,
        target_key=target_key,
        target_record="gfx942",
    )
    _write_report(
        candidate_path,
        code_byte_count=480,
        target_key=target_key,
        target_record="gfx942",
    )

    assert (
        main(
            [
                "diff",
                str(baseline_path),
                str(candidate_path),
                "--format=json",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    view = json.loads(captured.out)
    assert captured.err == ""
    assert view["identity"]["target_key"] == target_key
    assert view["changed_entry_count"] == 1


def test_diff_target_comparison_preserves_both_specializations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write_report(
        baseline_path,
        target_key="gfx1100",
        target_record="amdgpu.rdna3.core",
    )
    _write_report(
        candidate_path,
        target_key="gfx1151",
        target_record="amdgpu.rdna3_5.core",
    )

    assert (
        main(
            [
                "diff",
                str(baseline_path),
                str(candidate_path),
                "--comparison=target",
                "--format=json",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    view = json.loads(captured.out)
    assert captured.err == ""
    assert view["comparison_mode"] == "target"
    assert view["targets"]["baseline"]["target_key"] == "gfx1100"
    assert view["targets"]["candidate"]["target_key"] == "gfx1151"
    assert view["changed_entry_count"] == 0
    assert view["unchanged_entry_count"] == 1


def test_diff_target_comparison_rejects_workload_changes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write_report(
        baseline_path,
        target_key="gfx1100",
        target_record="amdgpu.rdna3.core",
    )
    _write_report(
        candidate_path,
        target_key="gfx1151",
        target_record="amdgpu.rdna3_5.core",
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["workload"]["workgroup_count"]["x"] = 8
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    assert (
        main(
            [
                "diff",
                str(baseline_path),
                str(candidate_path),
                "--comparison=target",
            ]
        )
        == 2
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "workload" in captured.err


def test_rejects_unversioned_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "report.json"
    _write_report(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    del report["schema_version"]
    report_path.write_text(json.dumps(report), encoding="utf-8")

    assert main(["show", str(report_path)]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "schema_version" in captured.err


def test_suggest_json_uses_registered_target_provider(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "report.json"
    _write_report(report_path)

    assert main(["suggest", str(report_path), "--format=json"]) == 0

    captured = capsys.readouterr()
    view = json.loads(captured.out)
    assert captured.err == ""
    assert view["kind"] == "loom.compile_report.suggest"
    assert view["provider"] == "amdgpu"
    assert view["status"] == "available"
    assert view["findings"] == []


def test_suggest_experimental_bank_model_requires_explicit_opt_in(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        target_key="gfx1250-a0",
        target_record="gfx1250-a0",
        experimental_bank_conflict=True,
    )

    assert main(["suggest", str(report_path), "--format=json"]) == 0
    default_view = json.loads(capsys.readouterr().out)
    assert "amdgpu.lds_bank_service" not in {
        finding["id"] for finding in default_view["findings"]
    }

    assert (
        main(
            [
                "suggest",
                str(report_path),
                "--include-experimental",
                "--format=json",
            ]
        )
        == 0
    )
    experimental_view = json.loads(capsys.readouterr().out)
    finding = next(
        finding
        for finding in experimental_view["findings"]
        if finding["id"] == "amdgpu.lds_bank_service"
    )
    assert finding["confidence"] == "experimental"


@pytest.mark.parametrize(
    ("target_key", "target_record"),
    [
        ("gfx1250-a0", "gfx1250-a0"),
        ("gfx942:sramecc+:xnack-", "gfx942"),
    ],
)
def test_suggest_json_resolves_structured_amdgpu_target_keys(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    target_key: str,
    target_record: str,
) -> None:
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        target_key=target_key,
        target_record=target_record,
    )

    assert main(["suggest", str(report_path), "--format=json"]) == 0

    captured = capsys.readouterr()
    view = json.loads(captured.out)
    assert captured.err == ""
    assert view["provider"] == "amdgpu"
    assert view["status"] == "available"
    assert view["target"] == {"family": "AMDGPU", "key": target_key}


def test_suggest_json_reports_unknown_target_without_guessing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "report.json"
    _write_report(report_path, target_key="gfx9999", target_record="gfx9999")

    assert main(["suggest", str(report_path), "--format=json"]) == 0

    captured = capsys.readouterr()
    view = json.loads(captured.out)
    assert captured.err == ""
    assert view["status"] == "unavailable"
    assert view["reason"] == "unknown_target_key"
    assert view["findings"] == []


def test_suggest_json_reports_invalid_target_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        target_key="gfx942:xnack+:xnack-",
        target_record="gfx942",
    )

    assert main(["suggest", str(report_path), "--format=json"]) == 0

    captured = capsys.readouterr()
    view = json.loads(captured.out)
    assert captured.err == ""
    assert view["status"] == "unavailable"
    assert view["reason"] == "invalid_target_key"
    assert view["findings"] == []


def test_suggest_json_reports_unsupported_family_without_provider(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        target_family="TEST",
        target_key="test0",
        target_record="test0",
    )

    assert main(["suggest", str(report_path), "--format=json"]) == 0

    captured = capsys.readouterr()
    view = json.loads(captured.out)
    assert captured.err == ""
    assert view["status"] == "unavailable"
    assert view["reason"] == "unsupported_target_family"
    assert "provider" not in view
    assert view["findings"] == []
