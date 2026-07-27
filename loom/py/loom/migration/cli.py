# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command-line entry point for ``loom-migrate``."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from loom.diagnostics import DiagnosticSeverity, SourceRange
from loom.migration.driver import (
    MigrationFileDiagnostic,
    MigrationFileKind,
    MigrationRunResult,
    classify_file,
    discover_migration_files,
    migrate_files,
)


def main(argv: list[str] | None = None) -> int:
    """Runs the command-line migration tool."""
    parser = _create_arg_parser()
    args = parser.parse_args(argv)
    try:
        return run(args, stdout=sys.stdout, stderr=sys.stderr)
    except _UsageError as exc:
        parser.error(str(exc))


def run(args: argparse.Namespace, *, stdout: TextIO, stderr: TextIO) -> int:
    """Runs parsed CLI arguments and returns a process exit code."""
    paths = _resolve_paths(args)
    run_result = migrate_files(paths)
    if args.check:
        run_result = _with_check_diagnostics(run_result)

    if args.in_place and run_result.ok:
        _write_in_place(run_result)
    if args.output is not None and run_result.ok:
        _write_output(run_result, args.output)

    if args.json:
        json.dump(run_result.to_json_object(), stdout, indent=2, sort_keys=True)
        stdout.write("\n")
    else:
        _write_text_diagnostics(run_result, stderr)
        if _should_print_source(args, run_result):
            text = run_result.files[0].text
            if text is not None:
                stdout.write(text)

    return 0 if run_result.ok else 1


def _create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loom-migrate",
        description="Migrates Loom source files to the current textual format.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("input", nargs="?", type=Path, help="Input file.")
    input_group.add_argument("--root", type=Path, help="Root directory to traverse.")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--check",
        action="store_true",
        help="Fail if any input needs migration or has diagnostics.",
    )
    mode_group.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite source files in place.",
    )
    mode_group.add_argument(
        "--output",
        type=Path,
        help="Writes a single migrated source file to this path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Writes structured JSON diagnostics and per-file results.",
    )
    return parser


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    if args.root is not None:
        if args.output is not None:
            raise _UsageError("--output can only be used with a single input file")
        if not args.check and not args.in_place:
            raise _UsageError("--root requires --check or --in-place")
        if not args.root.exists():
            raise _UsageError(f"root path does not exist: {args.root}")
        return discover_migration_files(args.root)

    input_path = args.input
    if input_path is None:
        raise _UsageError("expected input file or --root")
    if not input_path.exists():
        raise _UsageError(f"input file does not exist: {input_path}")
    if args.output is not None and not _is_text_migration_kind(
        classify_file(input_path)
    ):
        raise _UsageError("--output can only be used with a Loom text input")
    return (input_path,)


def _with_check_diagnostics(run_result: MigrationRunResult) -> MigrationRunResult:
    file_results = []
    for file_result in run_result.files:
        diagnostics = list(file_result.diagnostics)
        if file_result.changed:
            diagnostics.append(
                MigrationFileDiagnostic(
                    severity=DiagnosticSeverity.ERROR,
                    message="file requires Loom source migration",
                    file=file_result.path,
                    rule_id="loom.migrate.check",
                    source_range=SourceRange(filename=file_result.path),
                    fixup_hint="Run loom-migrate --in-place on this file.",
                )
            )
        file_results.append(replace(file_result, diagnostics=tuple(diagnostics)))
    return MigrationRunResult(tuple(file_results))


def _write_in_place(run_result: MigrationRunResult) -> None:
    for file_result in run_result.files:
        if (
            _is_text_migration_kind(file_result.kind)
            and file_result.changed
            and file_result.text is not None
        ):
            file_result.path.write_text(file_result.text, encoding="utf-8")


def _write_output(run_result: MigrationRunResult, output_path: Path) -> None:
    if len(run_result.files) != 1:
        raise _UsageError("--output can only be used with one input file")
    file_result = run_result.files[0]
    if not _is_text_migration_kind(file_result.kind):
        raise _UsageError("--output can only be used with a Loom text input")
    if file_result.text is not None:
        output_path.write_text(file_result.text, encoding="utf-8")


def _write_text_diagnostics(run_result: MigrationRunResult, stderr: TextIO) -> None:
    for diagnostic in run_result.diagnostics:
        stderr.write(_format_text_diagnostic(diagnostic) + "\n")


def _format_text_diagnostic(diagnostic: MigrationFileDiagnostic) -> str:
    source_range = diagnostic.source_range
    if source_range is not None and source_range.has_location:
        location = source_range.display()
    elif diagnostic.file is not None:
        location = str(diagnostic.file)
    else:
        location = "<unknown>"
    rule_suffix = f" [{diagnostic.rule_id}]" if diagnostic.rule_id else ""
    return f"{location}: {diagnostic.severity.value}: {diagnostic.message}{rule_suffix}"


def _should_print_source(
    args: argparse.Namespace, run_result: MigrationRunResult
) -> bool:
    if args.check or args.in_place or args.output is not None or args.json:
        return False
    if len(run_result.files) != 1 or not run_result.ok:
        return False
    return _is_text_migration_kind(run_result.files[0].kind)


def _is_text_migration_kind(kind: MigrationFileKind) -> bool:
    return kind in (MigrationFileKind.LOOM_SOURCE, MigrationFileKind.LOOM_TEST)


class _UsageError(ValueError):
    pass


if __name__ == "__main__":
    sys.exit(main())
