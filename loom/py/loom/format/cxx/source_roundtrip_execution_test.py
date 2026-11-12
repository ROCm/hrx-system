# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Executes authored programs before and after public C++ projection."""

import json
import subprocess
import sys
from pathlib import Path

from loom.format.cxx import load_module, render_module


def import_source(importer: str, source: bytes) -> bytes:
    result = subprocess.run(
        [importer, "--cleanup=false", "--to=bc", "-"],
        input=source,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert result.stderr == b"", result.stderr.decode()
    return result.stdout


def main(
    importer: str,
    runner: str,
    source_path: str,
    checks_path: str,
    cases: str,
    expectations: str,
) -> None:
    source = Path(source_path).read_bytes()
    checks = Path(checks_path).read_bytes()
    original = import_source(importer, source)
    projected = render_module(load_module(original)).encode()
    for label, program in (("original", source), ("exported", projected)):
        imported = import_source(importer, program + b"\n" + checks)
        executed = subprocess.run(
            [runner, "--input-format=loombc", "-"],
            input=imported,
            capture_output=True,
            check=False,
        )
        assert executed.returncode == 0, (
            label + ": " + executed.stdout.decode() + executed.stderr.decode()
        )
        report = json.loads(executed.stdout)
        assert report["case_count"] == int(cases), report
        assert report["sample_count"] == int(cases), report
        assert report["skipped_case_count"] == 0, report
        assert report["planning_issue_count"] == 0, report
        assert sum(
            sample["expectations"]["expectation_count"] for sample in report["samples"]
        ) == int(expectations), report
        sys.stdout.write(
            f"{label}: {cases} cases, {expectations} expectations passed\n"
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
