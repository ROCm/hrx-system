# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Executes source observations after public C++ export and reimport."""

import argparse
import io
import json
import re
import subprocess
import tempfile
from pathlib import Path

from loom.tools import export_cxx


def invoke(tool: str, arguments: list[str], source: bytes) -> bytes:
    result = subprocess.run(
        [str(Path(tool).resolve()), *arguments],
        input=source,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"{tool} exited with {result.returncode}:\n"
            + result.stdout.decode("utf-8", errors="replace")
            + result.stderr.decode("utf-8", errors="replace")
        )
    return result.stdout


def export(module: bytes) -> bytes:
    output = io.BytesIO()
    error = io.StringIO()
    exit_code = export_cxx.run(
        "-",
        "-",
        stdin=io.BytesIO(module),
        stdout=output,
        stderr=error,
    )
    if exit_code:
        raise RuntimeError(
            f"loom-export-cxx exited with {exit_code}:\n{error.getvalue()}"
        )
    return output.getvalue()


def execute_assertions(
    importer: str, runner: str, subject_path: str, assertions_path: str
) -> None:
    """Runs the same named checks against authored and exported subjects."""
    source = Path(subject_path).read_bytes()
    module = invoke(importer, ["--cleanup=false", "--to=bc", "-"], source)
    for phase, subject in (("authored", source), ("exported", export(module))):
        try:
            with tempfile.TemporaryDirectory() as directory:
                Path(directory, Path(subject_path).name).write_bytes(subject)
                checked_module = invoke(
                    importer,
                    [f"--I={directory}", "--cleanup=false", "--to=bc", assertions_path],
                    b"",
                )
            report = json.loads(
                invoke(runner, ["--input-format=loombc", "-"], checked_module)
            )
            if (
                not report["sample_count"]
                or report["skipped_case_count"]
                or report["planning_issue_count"]
            ):
                raise RuntimeError(f"checks did not all execute: {report}")
            expectation_count = sum(
                sample["expectations"]["expectation_count"]
                for sample in report["samples"]
            )
            print(
                f"{subject_path} ({phase}): {report['case_count']} named cases, "
                f"{expectation_count} expectations passed"
            )
        except RuntimeError as error:
            raise RuntimeError(f"{subject_path} ({phase}): {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importer", required=True)
    parser.add_argument("--checker")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--runner")
    parser.add_argument(
        "--assertions",
        nargs=2,
        action="append",
        default=[],
        metavar=("SUBJECT", "CHECKS"),
    )
    args = parser.parse_args()
    if not args.source and not args.assertions:
        parser.error("at least one --source or --assertions is required")
    if args.source and not args.checker:
        parser.error("--source requires --checker")
    if args.assertions and not args.runner:
        parser.error("--assertions requires --runner")
    fixtures: list[bytes] = []
    origins: list[tuple[str, int]] = []
    for source_path in args.source:
        text = Path(source_path).read_text(encoding="utf-8")
        cases = re.split(r"^// ====\s*$", text, flags=re.MULTILINE)
        for ordinal, case in enumerate(cases, 1):
            source, expected = case.split("// ----\n")
            try:
                module = source.encode("utf-8")
                if source_path.endswith(".cxx-test"):
                    module = invoke(
                        args.importer,
                        ["--root=run", "--cleanup=false", "--to=bc", "-"],
                        module,
                    )
                fixtures.append(
                    b"// RUN: emit vm-object-lifetime\n// INPUT: cxx root=run\n"
                    + export(module)
                    + b"\n// ----\n"
                    + expected.encode("utf-8")
                )
            except RuntimeError as error:
                raise RuntimeError(f"{source_path}, case {ordinal}: {error}") from error
            origins.append((source_path, ordinal))

    if fixtures:
        try:
            invoke(
                args.checker,
                ["--input-format=cxx", "-"],
                b"\n// ====\n".join(fixtures),
            )
        except RuntimeError as error:
            failed_indices = {
                int(match) - 1 for match in re.findall(r":: case (\d+)", str(error))
            }
            failed_origins = [
                f"{origins[index][0]}, case {origins[index][1]}"
                for index in sorted(failed_indices)
                if 0 <= index < len(origins)
            ]
            prefix = ", ".join(failed_origins) or "combined roundtrip fixture"
            raise RuntimeError(f"{prefix}: {error}") from error
        print(f"{len(fixtures)} exported programs preserve their observations")
    for subject_path, assertions_path in args.assertions:
        execute_assertions(args.importer, args.runner, subject_path, assertions_path)


if __name__ == "__main__":
    main()
