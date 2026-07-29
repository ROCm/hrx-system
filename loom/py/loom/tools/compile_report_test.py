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
    target_key: str = "gfx11-generic",
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
        "artifact_format": "hsaco",
        "backend": "amdgpu-hal",
        "status": {"code": 0, "name": "OK"},
        "target_family": "AMDGPU",
        "target_key": target_key,
        "target_bundle": "gfx11",
        "target_snapshot": "gfx11",
        "target_config": "gfx11",
        "workload": workload,
        "entries": {
            "count": 1,
            "rows": [
                {
                    "index": 0,
                    "function": "kernel",
                    "source_function": "kernel",
                    "target_bundle": "gfx11",
                    "target_snapshot": "gfx11",
                    "target_export": "kernel",
                    "target_export_symbol": None,
                    "target_config": "gfx11",
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


def test_suggest_json_uses_exact_target_provider(
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


def test_suggest_json_reports_unknown_target_without_guessing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "report.json"
    _write_report(report_path, target_key="gfx9999")

    assert main(["suggest", str(report_path), "--format=json"]) == 0

    captured = capsys.readouterr()
    view = json.loads(captured.out)
    assert captured.err == ""
    assert view["status"] == "unavailable"
    assert view["reason"] == "unknown_target_key"
    assert view["findings"] == []
