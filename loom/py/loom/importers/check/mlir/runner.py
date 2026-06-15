# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs MLIR importer checks using the inline loom-check case format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loom.diagnostics import LoomDiagnosticError
from loom.importers.check.annotations import (
    parse_expected_diagnostics,
    source_diagnostic_check_result,
)
from loom.importers.check.cases import (
    DEFAULT_CHECK_SYNTAX,
    CheckCase,
    case_matches_filter,
    parse_inline_cases,
)
from loom.importers.check.loom_verifier import LoomOutputVerifier
from loom.importers.check.results import CheckResult, unified_diff
from loom.importers.check.updates import format_updated_source


@dataclass(frozen=True, slots=True)
class MlirCheckOptions:
    """Options for inline MLIR importer checks."""

    kernel: str | None = None
    prefer_abi3_extensions: bool = False
    update: bool = False
    case_filter: str | None = None
    output_verifier: LoomOutputVerifier | None = None


def run_mlir_check(
    path: Path,
    *,
    options: MlirCheckOptions,
) -> tuple[CheckResult, ...]:
    source = path.read_text()
    cases = parse_check_cases(path, source, default_kernel=options.kernel)
    selected_cases = tuple(
        case for case in cases if case_matches_filter(case, options.case_filter)
    )
    results = tuple(import_mlir_case(case, options=options) for case in selected_cases)
    if options.update:
        updated_source = format_updated_source(source, cases, results)
        if updated_source != source:
            path.write_text(updated_source)
    return results


def parse_check_cases(
    path: Path,
    source: str,
    *,
    default_kernel: str | None = None,
) -> tuple[CheckCase, ...]:
    raw_cases = parse_inline_cases(
        path,
        source,
        syntax=DEFAULT_CHECK_SYNTAX,
        default_run="mlir",
    )
    cases: list[CheckCase] = []
    for raw_case in raw_cases:
        run = raw_case.run or "mlir"
        run_options = parse_mlir_run(run)
        kernel = run_options.kernel or default_kernel
        effective_run = "mlir" if kernel is None else f"mlir --kernel {kernel}"
        cases.append(
            CheckCase(
                path=path,
                index=raw_case.index,
                source=raw_case.source,
                input=raw_case.input,
                expected=raw_case.expected,
                has_expected=raw_case.has_expected,
                run=effective_run,
                line_start=raw_case.line_start,
                line_end=raw_case.line_end,
            )
        )
    return tuple(cases)


@dataclass(frozen=True, slots=True)
class MlirRunOptions:
    """Per-case options parsed from `// RUN: mlir ...`."""

    kernel: str | None = None


def parse_mlir_run(run: str) -> MlirRunOptions:
    pieces = run.split()
    if not pieces:
        return MlirRunOptions()
    if pieces[0] != "mlir":
        raise ValueError(f"unsupported importer check RUN mode `{pieces[0]}`")
    kernel: str | None = None
    index = 1
    while index < len(pieces):
        piece = pieces[index]
        if piece == "--kernel":
            index += 1
            if index == len(pieces):
                raise ValueError("RUN: mlir --kernel requires a value")
            kernel = pieces[index]
        else:
            raise ValueError(f"unsupported RUN: mlir option `{piece}`")
        index += 1
    return MlirRunOptions(kernel=kernel)


def import_mlir_case(
    case: CheckCase,
    *,
    options: MlirCheckOptions,
) -> CheckResult:
    from loom.importers.core import kernel_module_ops, print_loom_module
    from loom.importers.mlir.importer import (
        MlirImportOptions,
        import_mlir_module,
    )

    run_options = parse_mlir_run(case.run or "mlir")
    expected_diagnostics = parse_expected_diagnostics(
        case.input,
        path=case.path,
        line_start=case.line_start,
        comment_prefix="//",
    )
    try:
        result = import_mlir_module(
            case.input,
            options=MlirImportOptions(
                kernel=run_options.kernel,
                include_report=True,
                prefer_abi3_extensions=options.prefer_abi3_extensions,
            ),
        )
        target_format = getattr(result.report, "target_format", None)
        stdout = print_loom_module(
            result.module,
            ops=kernel_module_ops(target_format or "unknown"),
        )
    except LoomDiagnosticError as exc:
        if expected_diagnostics:
            return source_diagnostic_check_result(
                case,
                expected_diagnostics=expected_diagnostics,
                actual_diagnostics=exc.diagnostics,
            )
        return CheckResult(
            path=case.path,
            case_index=case.index,
            returncode=1,
            stdout="",
            stderr=f"{exc}\n",
            input=case.input,
            expected=case.expected,
        )
    except Exception as exc:
        return CheckResult(
            path=case.path,
            case_index=case.index,
            returncode=1,
            stdout="",
            stderr=f"{type(exc).__name__}: {exc}\n",
            input=case.input,
            expected=case.expected,
        )

    if expected_diagnostics:
        return source_diagnostic_check_result(
            case,
            expected_diagnostics=expected_diagnostics,
            actual_diagnostics=(),
            stdout=stdout,
        )
    if options.output_verifier is not None:
        verify_result = options.output_verifier.verify(case, stdout)
        if verify_result is not None:
            return verify_result
    if stdout == case.expected:
        return CheckResult(
            path=case.path,
            case_index=case.index,
            returncode=0,
            stdout=stdout,
            stderr="",
            input=case.input,
            expected=case.expected,
        )
    if options.update:
        return CheckResult(
            path=case.path,
            case_index=case.index,
            returncode=0,
            stdout=stdout,
            stderr="",
            input=case.input,
            expected=case.expected,
            updated=True,
        )
    return CheckResult(
        path=case.path,
        case_index=case.index,
        returncode=0,
        stdout=stdout,
        stderr="",
        input=case.input,
        expected=case.expected,
        mismatch="output differs from expected output",
        diff=unified_diff(
            case.expected,
            stdout,
            fromfile=f"{case.path}:case{case.index}:expected",
            tofile=f"{case.path}:case{case.index}:actual",
        ),
    )
