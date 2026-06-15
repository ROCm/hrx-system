# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from tempfile import TemporaryDirectory

from loom.importers.check.results import (
    CheckResult,
    dump_check_results,
    results_to_json,
    summarize_results,
)


def test_summarize_results_classifies_failures_crashes_and_skips() -> None:
    results = [
        CheckResult(Path("ok.mlir"), 0, 0, "", ""),
        CheckResult(Path("updated.mlir"), 1, 0, "", "", updated=True),
        CheckResult(Path("skipped.mlir"), -1, 0, "", "", skipped=True),
        CheckResult(Path("bad.mlir"), 2, 1, "", ""),
        CheckResult(Path("mismatch.mlir"), 3, 0, "", "", mismatch="different"),
        CheckResult(Path("segv.mlir"), 4, -11, "", ""),
    ]

    assert summarize_results(results) == {
        "passed": 1,
        "updated": 1,
        "skipped": 1,
        "failed": 2,
        "crashed": 1,
    }


def test_dump_check_results_writes_case_artifacts() -> None:
    with TemporaryDirectory() as directory:
        result = CheckResult(
            Path("kernels.mlir"),
            0,
            0,
            "actual\n",
            "",
            input="input\n",
            expected="expected\n",
            diff="diff\n",
        )
        dump_check_results([result], Path(directory))
        case_dirs = list(Path(directory).glob("*/case0"))

        assert len(case_dirs) == 1
        assert (case_dirs[0] / "input.txt").read_text() == "input\n"
        assert (case_dirs[0] / "expected.txt").read_text() == "expected\n"
        assert (case_dirs[0] / "stdout.txt").read_text() == "actual\n"
        assert (case_dirs[0] / "diff.patch").read_text() == "diff\n"


def test_results_json_preserves_metadata() -> None:
    rendered = results_to_json(
        [
            CheckResult(
                Path("kernels.py"),
                0,
                0,
                "actual\n",
                "",
                metadata={"tilelang_oracle": {"status": "unavailable"}},
            )
        ]
    )

    assert '"tilelang_oracle"' in rendered
    assert '"status": "unavailable"' in rendered
