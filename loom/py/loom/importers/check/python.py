# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Python importer check fixture runner."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType
from typing import Any

from loom.diagnostics import Diagnostic, LoomDiagnosticError, SourceRange
from loom.importers.check.annotations import (
    parse_expected_diagnostics,
    source_diagnostic_check_result,
)
from loom.importers.check.cases import (
    CheckCase,
    InlineCheckSyntax,
    case_matches_filter,
    parse_inline_cases,
)
from loom.importers.check.loom_verifier import LoomOutputVerifier
from loom.importers.check.results import CheckResult, unified_diff
from loom.importers.check.updates import format_updated_source

PYTHON_CHECK_SYNTAX = InlineCheckSyntax(
    case_separator_prefix="# ====",
    expected_separator="# ----",
    comment_prefix="#",
    case_separator_spacer="\n\n",
)


@dataclass(frozen=True, slots=True)
class PythonCheckOptions:
    """Options for Python importer check fixtures."""

    update: bool = False
    case_filter: str | None = None
    output_verifier: LoomOutputVerifier | None = None


@dataclass(frozen=True, slots=True)
class PythonCheckCase:
    """One executable Python check case mapped to an inline source segment."""

    check_case: CheckCase
    function: Callable[..., Any]


@dataclass(frozen=True, slots=True)
class PythonCheckInvocation:
    """Structured result from invoking one Python check case."""

    stdout: str
    stderr: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)


class PythonCheckError(ValueError):
    """Raised for structurally invalid Python importer check files."""


class PythonCheckSkip(Exception):
    """Raised by an importer check case to skip that case."""

    def __init__(
        self,
        reason: str,
        *,
        stdout: str = "",
        stderr: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason: str = reason
        self.stdout: str = stdout
        self.stderr: str = stderr
        self.metadata: Mapping[str, object] = dict(metadata or {})


def run_python_check(
    path: Path,
    *,
    options: PythonCheckOptions,
    is_case: Callable[[object], bool],
    invoke: Callable[[PythonCheckCase], str | PythonCheckInvocation],
    case_labels: Callable[[PythonCheckCase], Sequence[str]] | None = None,
) -> tuple[CheckResult, ...]:
    source = path.read_text()
    check_cases = parse_python_check_cases(path, source)
    try:
        module = load_python_module(path, source)
        python_cases = discover_python_cases(
            module,
            check_cases=check_cases,
            is_case=is_case,
        )
    except Exception as exc:
        return (
            CheckResult(
                path=path,
                case_index=-1,
                returncode=1,
                stdout="",
                stderr=f"{type(exc).__name__}: {exc}\n",
            ),
        )

    selected_cases = _filter_python_cases(
        python_cases,
        options.case_filter,
        case_labels=case_labels,
    )
    results = tuple(
        run_python_case(python_case, options=options, invoke=invoke)
        for python_case in selected_cases
    )
    if options.update:
        updated_source = format_updated_source(
            source,
            check_cases,
            results,
            syntax=PYTHON_CHECK_SYNTAX,
            expected_encoder=encode_python_expected,
        )
        if updated_source != source:
            path.write_text(updated_source)
    return results


def parse_python_check_cases(path: Path, source: str) -> tuple[CheckCase, ...]:
    raw_cases = parse_inline_cases(
        path,
        source,
        syntax=PYTHON_CHECK_SYNTAX,
        allow_preamble=True,
    )
    return tuple(
        CheckCase(
            path=case.path,
            index=case.index,
            source=case.source,
            input=case.input,
            expected=(
                decode_python_expected(case.expected)
                if case.has_expected
                else case.expected
            ),
            has_expected=case.has_expected,
            run=case.run,
            line_start=case.line_start,
            line_end=case.line_end,
            raw_source=case.raw_source,
        )
        for case in raw_cases
    )


def load_python_module(path: Path, source: str) -> ModuleType:
    module_name = _module_name(path)
    module = ModuleType(module_name)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    try:
        code = compile(source, str(path), "exec")
        exec(code, module.__dict__)
    finally:
        sys.modules.pop(module_name, None)
    return module


def discover_python_cases(
    module: ModuleType,
    *,
    check_cases: tuple[CheckCase, ...],
    is_case: Callable[[object], bool],
) -> tuple[PythonCheckCase, ...]:
    case_by_index: dict[int, PythonCheckCase] = {}
    for value in vars(module).values():
        if not is_case(value):
            continue
        if not callable(value):
            raise PythonCheckError("decorated Python check case is not callable")
        code = getattr(value, "__code__", None)
        if code is None:
            raise PythonCheckError("decorated Python check case has no code object")
        line_number = code.co_firstlineno
        check_case = _check_case_for_line(check_cases, line_number)
        if check_case.index in case_by_index:
            raise PythonCheckError(
                f"multiple decorated cases in case{check_case.index}"
            )
        case_by_index[check_case.index] = PythonCheckCase(
            check_case=check_case,
            function=value,
        )
    missing = [
        f"case{case.index}" for case in check_cases if case.index not in case_by_index
    ]
    if missing:
        raise PythonCheckError(
            f"missing decorated Python check function for {', '.join(missing)}"
        )
    return tuple(case_by_index[index] for index in sorted(case_by_index))


def run_python_case(
    python_case: PythonCheckCase,
    *,
    options: PythonCheckOptions,
    invoke: Callable[[PythonCheckCase], str | PythonCheckInvocation],
) -> CheckResult:
    check_case = python_case.check_case
    expected_diagnostics = parse_expected_diagnostics(
        check_case.input,
        path=check_case.path,
        line_start=check_case.line_start,
        comment_prefix="#",
    )
    try:
        invocation = _normalize_invocation(invoke(python_case))
    except PythonCheckSkip as exc:
        return CheckResult(
            path=check_case.path,
            case_index=check_case.index,
            returncode=0,
            stdout=exc.stdout,
            stderr=exc.stderr or f"{exc.reason}\n",
            input=check_case.input,
            expected=check_case.expected,
            skipped=True,
            metadata=exc.metadata,
        )
    except LoomDiagnosticError as exc:
        diagnostics = _diagnostics_with_case_location(exc.diagnostics, python_case)
        if expected_diagnostics:
            return source_diagnostic_check_result(
                check_case,
                expected_diagnostics=expected_diagnostics,
                actual_diagnostics=diagnostics,
            )
        return CheckResult(
            path=check_case.path,
            case_index=check_case.index,
            returncode=1,
            stdout="",
            stderr="".join(f"{diagnostic}\n" for diagnostic in diagnostics),
            input=check_case.input,
            expected=check_case.expected,
        )
    except Exception as exc:
        return CheckResult(
            path=check_case.path,
            case_index=check_case.index,
            returncode=1,
            stdout="",
            stderr=f"{type(exc).__name__}: {exc}\n",
            input=check_case.input,
            expected=check_case.expected,
        )
    stdout = invocation.stdout
    if expected_diagnostics:
        return source_diagnostic_check_result(
            check_case,
            expected_diagnostics=expected_diagnostics,
            actual_diagnostics=(),
            stdout=stdout,
        )
    if options.output_verifier is not None:
        verify_result = options.output_verifier.verify(check_case, stdout)
        if verify_result is not None:
            return verify_result
    if stdout == check_case.expected:
        return CheckResult(
            path=check_case.path,
            case_index=check_case.index,
            returncode=0,
            stdout=stdout,
            stderr=invocation.stderr,
            input=check_case.input,
            expected=check_case.expected,
            metadata=invocation.metadata,
        )
    if options.update:
        return CheckResult(
            path=check_case.path,
            case_index=check_case.index,
            returncode=0,
            stdout=stdout,
            stderr=invocation.stderr,
            input=check_case.input,
            expected=check_case.expected,
            updated=True,
            metadata=invocation.metadata,
        )
    return CheckResult(
        path=check_case.path,
        case_index=check_case.index,
        returncode=0,
        stdout=stdout,
        stderr=invocation.stderr,
        input=check_case.input,
        expected=check_case.expected,
        mismatch="output differs from expected output",
        metadata=invocation.metadata,
        diff=unified_diff(
            check_case.expected,
            stdout,
            fromfile=f"{check_case.path}:case{check_case.index}:expected",
            tofile=f"{check_case.path}:case{check_case.index}:actual",
        ),
    )


def _normalize_invocation(result: str | PythonCheckInvocation) -> PythonCheckInvocation:
    if isinstance(result, PythonCheckInvocation):
        return result
    return PythonCheckInvocation(stdout=result)


def _filter_python_cases(
    python_cases: tuple[PythonCheckCase, ...],
    case_filter: str | None,
    *,
    case_labels: Callable[[PythonCheckCase], Sequence[str]] | None,
) -> tuple[PythonCheckCase, ...]:
    if not case_filter:
        return python_cases
    filtered_cases: list[PythonCheckCase] = []
    for python_case in python_cases:
        labels = [python_case.function.__name__]
        if case_labels is not None:
            labels.extend(case_labels(python_case))
        if case_matches_filter(
            python_case.check_case,
            case_filter,
            labels=labels,
        ):
            filtered_cases.append(python_case)
    return tuple(filtered_cases)


def encode_python_expected(text: str) -> str:
    delimiter = _select_python_expected_delimiter(text)
    return f"r{delimiter}\n{text}{delimiter}\n"


def decode_python_expected(text: str) -> str:
    triple_quoted = _decode_triple_quoted_expected(text)
    if triple_quoted is not None:
        return triple_quoted
    lines: list[str] = []
    for line in text.splitlines(keepends=True):
        if not line.strip():
            lines.append(line)
            continue
        if line.startswith("# "):
            lines.append(line[2:])
            continue
        if line.strip() == "#":
            lines.append("\n")
            continue
        raise PythonCheckError(
            "Python expected output must be a triple-quoted block or comments"
        )
    return "".join(lines)


def _select_python_expected_delimiter(text: str) -> str:
    if '"""' not in text:
        return '"""'
    if "'''" not in text:
        return "'''"
    raise PythonCheckError(
        "Python expected output cannot contain both triple-quote delimiters"
    )


def _decode_triple_quoted_expected(text: str) -> str | None:
    lines = text.splitlines(keepends=True)
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return None
    delimiter = _python_expected_delimiter(lines[0])
    if delimiter is None:
        return None
    body_with_close = "".join(lines[1:])
    close_position = body_with_close.rfind(delimiter)
    if close_position < 0:
        raise PythonCheckError("Python expected output is missing its closing quote")
    trailing_text = body_with_close[close_position + len(delimiter) :]
    if trailing_text.strip():
        raise PythonCheckError(
            "Python expected output has text after its closing quote"
        )
    return body_with_close[:close_position]


def _python_expected_delimiter(line: str) -> str | None:
    stripped = line.strip()
    if stripped in ('"""', 'r"""', 'R"""'):
        return '"""'
    if stripped in ("'''", "r'''", "R'''"):
        return "'''"
    return None


def _check_case_for_line(
    check_cases: tuple[CheckCase, ...],
    line_number: int,
) -> CheckCase:
    for check_case in check_cases:
        if check_case.line_start <= line_number <= check_case.line_end:
            return check_case
    raise PythonCheckError(f"decorated case at line {line_number} is outside a case")


def _module_name(path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    return f"_loom_import_check_{digest}"


def _diagnostics_with_case_location(
    diagnostics: tuple[Diagnostic, ...],
    python_case: PythonCheckCase,
) -> tuple[Diagnostic, ...]:
    fallback = SourceRange.line(
        python_case.check_case.path,
        python_case.function.__code__.co_firstlineno,
    )
    return tuple(
        _diagnostic_with_fallback_location(diagnostic, fallback)
        for diagnostic in diagnostics
    )


def _diagnostic_with_fallback_location(
    diagnostic: Diagnostic,
    fallback: SourceRange,
) -> Diagnostic:
    location = diagnostic.primary_location
    if location is not None and location.has_location:
        return diagnostic
    return replace(diagnostic, source_location=fallback)
