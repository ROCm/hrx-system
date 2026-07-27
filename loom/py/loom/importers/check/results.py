# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Importer check result reporting."""

from __future__ import annotations

import difflib
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One importer check case result."""

    path: Path
    case_index: int
    returncode: int
    stdout: str
    stderr: str
    input: str = ""
    expected: str = ""
    mismatch: str | None = None
    diff: str | None = None
    updated: bool = False
    skipped: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.skipped or (self.returncode == 0 and self.mismatch is None)

    @property
    def status(self) -> str:
        if self.skipped:
            return "skipped"
        if self.updated:
            return "updated"
        if self.returncode == 0:
            return "passed" if self.mismatch is None else "failed"
        if self.returncode < 0:
            return "crashed"
        return "failed"

    def as_json_object(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "case_index": self.case_index,
            "returncode": self.returncode,
            "status": self.status,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "input": self.input,
            "expected": self.expected,
            "mismatch": self.mismatch,
            "diff": self.diff,
            "updated": self.updated,
            "skipped": self.skipped,
            "metadata": dict(self.metadata),
        }


def unified_diff(
    expected: str,
    actual: str,
    *,
    fromfile: str,
    tofile: str,
) -> str:
    return "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )


def results_to_json(results: Sequence[CheckResult]) -> str:
    return json.dumps(
        {
            "results": [result.as_json_object() for result in results],
            "summary": summarize_results(results),
        },
        indent=2,
        sort_keys=True,
    )


def summarize_results(results: Sequence[CheckResult]) -> dict[str, int]:
    summary = {
        "passed": 0,
        "updated": 0,
        "skipped": 0,
        "failed": 0,
        "crashed": 0,
    }
    for result in results:
        summary[result.status] += 1
    return summary


def dump_check_results(results: Sequence[CheckResult], dump_root: Path) -> None:
    """Writes per-case artifacts for debugging importer checks."""

    dump_root.mkdir(parents=True, exist_ok=True)
    for result in results:
        case_dir = dump_root / _dump_case_directory(result)
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_text(case_dir / "input.txt", result.input)
        _write_text(case_dir / "expected.txt", result.expected)
        _write_text(case_dir / "stdout.txt", result.stdout)
        _write_text(case_dir / "stderr.txt", result.stderr)
        if result.diff:
            _write_text(case_dir / "diff.patch", result.diff)
        _write_text(
            case_dir / "result.json",
            json.dumps(result.as_json_object(), indent=2, sort_keys=True) + "\n",
        )


def _dump_case_directory(result: CheckResult) -> str:
    digest = hashlib.sha256(str(result.path).encode()).hexdigest()[:12]
    path_fragment = _safe_path_fragment(str(result.path))
    case_fragment = "file" if result.case_index < 0 else f"case{result.case_index}"
    return f"{path_fragment}-{digest}/{case_fragment}"


def _safe_path_fragment(text: str) -> str:
    return (
        "".join(
            char if char.isalnum() or char in ("-", "_", ".") else "_" for char in text
        ).strip("_")
        or "input"
    )


def _write_text(path: Path, text: str) -> None:
    path.write_text(text)
