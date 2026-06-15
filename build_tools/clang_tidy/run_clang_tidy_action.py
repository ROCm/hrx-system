#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_arguments(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" not in argv:
        raise ValueError("missing '--' before compile arguments")
    delimiter = argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("--clang-tidy", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checks", required=True)
    parser.add_argument("--warnings-as-errors")
    parser.add_argument("--export-fixes", type=Path)
    parser.add_argument("--line-filter")
    parser.add_argument(
        "--allow-diagnostics",
        action="store_true",
        help="Return success when clang-tidy produced diagnostic fix artifacts.",
    )
    args = parser.parse_args(argv[:delimiter])
    return args, argv[delimiter + 1 :]


def write_empty_fixes(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\nMainSourceFile: ''\nDiagnostics: []\n...\n",
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    try:
        args, compile_args = parse_arguments(argv)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    command = [
        str(args.clang_tidy),
        f"--load={args.plugin}",
        f"--checks={args.checks}",
        str(args.source),
    ]
    if args.warnings_as_errors:
        command.append(f"--warnings-as-errors={args.warnings_as_errors}")
    if args.export_fixes:
        command.append(f"--export-fixes={args.export_fixes}")
    if args.line_filter:
        command.append(f"--line-filter={args.line_filter}")
    command += ["--", *compile_args]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = completed.stdout
    args.output.write_text(output, encoding="utf-8")
    if (
        args.export_fixes
        and completed.returncode == 0
        and not args.export_fixes.exists()
    ):
        write_empty_fixes(args.export_fixes)
    if completed.returncode != 0:
        if args.allow_diagnostics and args.export_fixes and args.export_fixes.exists():
            return 0
        print("clang-tidy command failed:", file=sys.stderr)
        print(shlex.join(command), file=sys.stderr)
        if output:
            print(output, file=sys.stderr, end="" if output.endswith("\n") else "\n")
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
