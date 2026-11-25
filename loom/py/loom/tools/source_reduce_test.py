# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the loom-reduce plan applier."""

from __future__ import annotations

import json
from pathlib import Path

from loom.tools import source_reduce

SOURCE = """\
func.def @first(%row: index, %stride: index, %base: index) -> (index) {
  %scaled = index.mul %row, %stride : index
  %address = index.add %base, %scaled : index
  func.return %address : index
}

func.def @second(%lane: index, %pitch: index, %origin: index) -> (index) {
  %lane_bytes = index.mul %lane, %pitch : index
  %start = index.add %origin, %lane_bytes : index
  func.return %start : index
}
"""


def _binding(name: str, type_name: str = "index") -> dict:
    return {"name": name, "type": type_name}


def _occurrence(
    start_line: int,
    end_line: int,
    inputs: list[dict],
    outputs: list[dict],
    *,
    callable_copy: bool = True,
    end_column: int = 44,
) -> dict:
    return {
        "file": "kernel.loom",
        "start_line": start_line,
        "start_column": 3,
        "end_line": end_line,
        "end_column": end_column,
        "callable": callable_copy,
        "inputs": inputs,
        "outputs": outputs,
        "motif_definition": False,
    }


def _finding(occurrences: list[dict], **overrides) -> dict:
    finding = {
        "rule": "loom-analyze/duplicate-sequence",
        "op_count": 2,
        "fingerprint": "0123456789abcdef",
        "parameterized_by": "",
        "applicable": True,
        "selected": True,
        "occurrences": occurrences,
    }
    finding.update(overrides)
    return finding


def _line_end(line: int) -> int:
    """Returns the exclusive one-based end column of a SOURCE line."""

    return len(SOURCE.splitlines()[line - 1]) + 1


def _duplicate_finding(**overrides) -> dict:
    return _finding(
        [
            _occurrence(
                2,
                3,
                [_binding("%row"), _binding("%stride"), _binding("%base")],
                [_binding("%address")],
                end_column=_line_end(3),
            ),
            _occurrence(
                8,
                9,
                [_binding("%lane"), _binding("%pitch"), _binding("%origin")],
                [_binding("%start")],
                end_column=_line_end(9),
            ),
        ],
        **overrides,
    )


def _plan(*findings: dict) -> dict:
    return {"tool": "loom-analyze", "summary": {"ops": 6}, "findings": list(findings)}


def _reduce(tmp_path: Path, plan: dict, verify=None) -> source_reduce.ReduceResult:
    root = tmp_path / "source"
    root.mkdir()
    (root / "kernel.loom").write_text(SOURCE, encoding="utf-8")
    return source_reduce.reduce_sources(
        plan, root=root, output_dir=tmp_path / "out", verify=verify
    )


def test_replaces_copies_with_calls_to_one_motif(tmp_path: Path) -> None:
    result = _reduce(tmp_path, _plan(_duplicate_finding()))
    text = (tmp_path / "out" / "kernel.loom").read_text(encoding="utf-8")
    name = result.jobs[0].name
    assert result.jobs[0].status == "applied"
    assert (
        f"  %address = func.call @{name}(%row, %stride, %base) : "
        "(index, index, index) -> (index)\n" in text
    )
    assert (
        f"  %start = func.call @{name}(%lane, %pitch, %origin) : "
        "(index, index, index) -> (index)\n" in text
    )
    assert (
        f"func.def inline @{name}(%row: index, %stride: index, %base: index)"
        " -> (index) {\n"
        "  %scaled = index.mul %row, %stride : index\n"
        "  %address = index.add %base, %scaled : index\n"
        "  func.return %address : index\n"
        "}\n" in text
    )
    assert result.changed_files == ["kernel.loom"]


def test_copy_without_outputs_calls_without_results(tmp_path: Path) -> None:
    finding = _duplicate_finding()
    for occurrence in finding["occurrences"]:
        occurrence["outputs"] = []
    result = _reduce(tmp_path, _plan(finding))
    text = (tmp_path / "out" / "kernel.loom").read_text(encoding="utf-8")
    name = result.jobs[0].name
    assert (
        f"  func.call @{name}(%row, %stride, %base) : (index, index, index)\n" in text
    )
    assert (
        f"func.def inline @{name}(%row: index, %stride: index, %base: index) {{" in text
    )
    assert "  func.return\n}" in text


def test_non_callable_copies_stay_inline(tmp_path: Path) -> None:
    finding = _duplicate_finding()
    finding["occurrences"][1]["callable"] = False
    result = _reduce(tmp_path, _plan(finding))
    assert result.jobs == []
    assert (tmp_path / "out" / "kernel.loom").read_text(encoding="utf-8") == SOURCE


def test_skips_parameterized_and_reimplemented_findings(tmp_path: Path) -> None:
    parameterized = _duplicate_finding(parameterized_by="constants")
    reimplemented = _duplicate_finding(rule="loom-analyze/reimplemented-motif")
    result = _reduce(tmp_path, _plan(parameterized, reimplemented))
    reasons = sorted(skip["reason"] for skip in result.skipped)
    assert reasons == [
        "calling an existing motif needs --scope=corpus",
        "copies differ in constants or types",
    ]
    assert result.changed_files == []


def test_unselected_findings_are_ignored(tmp_path: Path) -> None:
    result = _reduce(tmp_path, _plan(_duplicate_finding(selected=False)))
    assert result.jobs == []


def test_parameter_type_must_reference_earlier_parameters(tmp_path: Path) -> None:
    finding = _duplicate_finding()
    finding["occurrences"][0]["inputs"][0]["type"] = "view<[%missing]xf32>"
    result = _reduce(tmp_path, _plan(finding))
    assert result.jobs[0].status == "skipped"
    assert "earlier parameter" in result.jobs[0].reason


def test_failed_verification_restores_the_file(tmp_path: Path) -> None:
    result = _reduce(
        tmp_path, _plan(_duplicate_finding()), verify=lambda paths: set(paths)
    )
    assert result.jobs[0].status == "reverted"
    assert result.changed_files == []
    assert (tmp_path / "out" / "kernel.loom").read_text(encoding="utf-8") == SOURCE


def test_cli_writes_report_and_requires_empty_output(tmp_path: Path, capsys) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "kernel.loom").write_text(SOURCE, encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan(_duplicate_finding())), encoding="utf-8")
    report_path = tmp_path / "report.json"
    arguments = [
        f"--plan={plan_path}",
        f"--root={root}",
        f"--output-dir={tmp_path / 'out'}",
        f"--report={report_path}",
        "--no-verify",
    ]
    assert source_reduce.main(arguments) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["motifs"] == 1
    assert report["summary"]["replaced_copies"] == 2
    assert "1 motif(s) replace 2 copies" in capsys.readouterr().out

    assert source_reduce.main(arguments) == 2
    assert "is not empty" in capsys.readouterr().err


def test_cli_rejects_a_non_plan_file(tmp_path: Path, capsys) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    arguments = [
        f"--plan={plan_path}",
        f"--root={tmp_path}",
        f"--output-dir={tmp_path / 'out'}",
        "--no-verify",
    ]
    assert source_reduce.main(arguments) == 2
    assert "is not a loom-analyze" in capsys.readouterr().err
