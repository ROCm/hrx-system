# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command-line entry point for ``loom-compile-report``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

from loom.reporting.compile_report import (
    CompileReportComparisonMode,
    CompileReportError,
    IncomparableCompileReportsError,
    load_compile_report,
)
from loom.reporting.compile_report_view import (
    build_compile_report_diff,
    build_compile_report_show,
    format_compile_report_diff_text,
    format_compile_report_show_text,
)

_AGENTS_MARKDOWN = """## loom-compile-report

`loom-compile-report` turns version-zero compile report JSON into bounded views
for one compiler checkout. Start here before querying the complete report tree.

### Read, compare, and choose an experiment

```shell
loom-compile-report show kernel.report.json
loom-compile-report diff baseline.report.json candidate.report.json
loom-compile-report diff gfx11.report.json gfx1151.report.json \\
  --comparison=target
loom-compile-report suggest kernel.report.json
```

`show` separates emitted artifact facts from compiler analysis and marks omitted
metrics as unavailable. `diff` uses an exact compilation identity by default, so
a changed source, workload, target, backend, or configuration fails instead of
producing a false causal comparison. `--comparison=target` admits target
specialization changes within one target and backend family. `--force` retains
identity mismatches for historical inspection; its result is observational.

`suggest` asks the selected target provider for evidence-backed experiments.
Suggestions are hypotheses to recompile, retest, and measure, not performance
claims.

### Use the bounded JSON views

```shell
loom-compile-report show kernel.report.json --format=json | \\
  jq '{status, identity, workload, entries, missing_evidence}'
loom-compile-report diff baseline.report.json candidate.report.json \\
  --format=json | \\
  jq '{identity_mismatches, changed_entry_count, entries}'
loom-compile-report suggest kernel.report.json --format=json | \\
  jq '{status, provider, findings}'
```

The JSON views are smaller and more stable than the complete compiler report.
Use raw report fields only after a bounded view identifies a question that needs
row-level scheduling, allocation, memory, or legalization evidence. The detailed
workflow and advanced `jq` cuts live in
`loom/docs/src/workflows/compile-reports.md` and
`loom/docs/src/workflows/compile-report-queries.md`.
"""


def main(argv: list[str] | None = None) -> int:
    """Runs the compile report analysis tool."""
    parser = _create_argument_parser()
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if arguments == ["--agents_md"]:
        sys.stdout.write(_AGENTS_MARKDOWN)
        return 0
    if "--agents_md" in arguments:
        parser.error("--agents_md must be used alone")
    args = parser.parse_args(arguments)
    try:
        return run(args, stdout=sys.stdout)
    except (CompileReportError, IncomparableCompileReportsError) as exc:
        sys.stderr.write(f"loom-compile-report: error: {exc}\n")
        return 2


def run(args: argparse.Namespace, *, stdout: TextIO) -> int:
    """Runs parsed CLI arguments and returns a process exit code."""
    if args.command == "show":
        document = load_compile_report(args.report)
        view = build_compile_report_show(document)
        text = format_compile_report_show_text(view)
    elif args.command == "diff":
        baseline = load_compile_report(args.baseline)
        candidate = load_compile_report(args.candidate)
        view = build_compile_report_diff(
            baseline,
            candidate,
            CompileReportComparisonMode(args.comparison),
            force=args.force,
        )
        text = format_compile_report_diff_text(view)
    elif args.command == "suggest":
        from loom.reporting.compile_report_suggestions import (
            CompileReportSuggestionOptions,
            build_compile_report_suggestions,
            format_compile_report_suggestions_text,
        )
        from loom.target.arch.compile_report_suggestions import suggest_compile_report

        document = load_compile_report(args.report)
        result = suggest_compile_report(
            document,
            CompileReportSuggestionOptions(
                include_experimental=args.include_experimental,
            ),
        )
        view = build_compile_report_suggestions(document, result)
        text = format_compile_report_suggestions_text(view)
    else:
        raise AssertionError(f"unhandled command: {args.command}")

    if args.output_format == "json":
        json.dump(view, stdout, sort_keys=True, separators=(",", ":"))
        stdout.write("\n")
    else:
        stdout.write(text)
    return 0


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loom-compile-report",
        description=(
            "Renders and compares ephemeral version-zero Loom compile reports."
        ),
    )
    parser.add_argument(
        "--agents_md",
        action="store_true",
        help="Print a compact Markdown workflow for coding agents and exit.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_parser = subparsers.add_parser(
        "show",
        help="Renders emitted artifact facts and compiler analysis.",
    )
    show_parser.add_argument("report", type=Path, help="Compile report JSON path.")
    _add_output_format_argument(show_parser)

    diff_parser = subparsers.add_parser(
        "diff",
        help="Compares reports under a strict identity contract.",
    )
    diff_parser.add_argument("baseline", type=Path, help="Baseline report path.")
    diff_parser.add_argument("candidate", type=Path, help="Candidate report path.")
    identity_group = diff_parser.add_mutually_exclusive_group()
    identity_group.add_argument(
        "--comparison",
        choices=tuple(mode.value for mode in CompileReportComparisonMode),
        default=CompileReportComparisonMode.EXACT.value,
        help=(
            "Identity contract. 'exact' requires identical compilation identity; "
            "'target' permits only target specialization identity to vary within "
            "one target and backend family. Defaults to exact."
        ),
    )
    identity_group.add_argument(
        "--force",
        action="store_true",
        help=(
            "Compare one entry from each report despite identity mismatches. "
            "The output retains every mismatch and is observational, not causal."
        ),
    )
    _add_output_format_argument(diff_parser)

    suggest_parser = subparsers.add_parser(
        "suggest",
        help="Suggests target-owned optimization experiments.",
    )
    suggest_parser.add_argument("report", type=Path, help="Compile report JSON path.")
    suggest_parser.add_argument(
        "--include-experimental",
        action="store_true",
        help=(
            "Include exploratory experiments whose model or policy has not "
            "earned default confidence."
        ),
    )
    _add_output_format_argument(suggest_parser)
    return parser


def _add_output_format_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("text", "json"),
        default="text",
        help="Output format. Defaults to text.",
    )


if __name__ == "__main__":
    sys.exit(main())
