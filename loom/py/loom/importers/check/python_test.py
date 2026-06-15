# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from tempfile import TemporaryDirectory

from loom.diagnostics import (
    Diagnostic,
    DiagnosticParam,
    DiagnosticSeverity,
    LoomDiagnosticError,
    SourceRange,
)
from loom.error.type import ERR_TYPE_001
from loom.importers.check.python import (
    PythonCheckCase,
    PythonCheckInvocation,
    PythonCheckOptions,
    PythonCheckSkip,
    decode_python_expected,
    encode_python_expected,
    run_python_check,
)


def test_python_expected_codec_emits_triple_quoted_blocks() -> None:
    encoded = encode_python_expected("func @main\n\n  return\n")

    assert encoded == 'r"""\nfunc @main\n\n  return\n"""\n'
    assert decode_python_expected(encoded) == "func @main\n\n  return\n"


def test_python_expected_codec_reads_legacy_commented_blocks() -> None:
    assert (
        decode_python_expected("# func @main\n#\n#   return\n")
        == "func @main\n\n  return\n"
    )


def test_python_expected_codec_falls_back_to_single_quote_delimiter() -> None:
    encoded = encode_python_expected('say """hello"""\n')

    assert encoded == "r'''\nsay \"\"\"hello\"\"\"\n'''\n"
    assert decode_python_expected(encoded) == 'say """hello"""\n'


def test_run_python_check_executes_decorated_cases() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cases.py"
        path.write_text(
            """
def case(func):
    func.__case__ = True
    return func

@case
def first():
    return "first\\n"
# ----
# first
# ====
@case
def second():
    return "second\\n"
# ----
# second
"""
        )

        results = run_python_check(
            path,
            options=PythonCheckOptions(),
            is_case=lambda value: bool(getattr(value, "__case__", False)),
            invoke=lambda case: case.function(),
        )

    assert [result.status for result in results] == ["passed", "passed"]


def test_run_python_check_shares_preamble_across_cases() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cases.py"
        path.write_text(
            """
def case(func):
    func.__case__ = True
    return func


class Shared:
    value = "shared"


# ====
@case
def first():
    return f"{Shared.value}:first\\n"
# ----
# shared:first

# ====
@case
def second():
    return f"{Shared.value}:second\\n"
# ----
# shared:second
"""
        )

        results = run_python_check(
            path,
            options=PythonCheckOptions(),
            is_case=lambda value: bool(getattr(value, "__case__", False)),
            invoke=lambda case: case.function(),
        )

    assert [result.status for result in results] == ["passed", "passed"]


def test_run_python_check_updates_expected_blocks() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cases.py"
        path.write_text(
            """
def case(func):
    func.__case__ = True
    return func

@case
def first():
    return "new\\n"
# ----
# old
"""
        )

        results = run_python_check(
            path,
            options=PythonCheckOptions(update=True),
            is_case=lambda value: bool(getattr(value, "__case__", False)),
            invoke=lambda case: case.function(),
        )
        updated_source = path.read_text()

    assert results[0].updated
    assert 'r"""\nnew\n"""\n' in updated_source
    assert "# old\n" not in updated_source


def test_run_python_check_matches_source_diagnostic_annotations() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cases.py"
        path.write_text(
            """
def case(func):
    func.__case__ = True
    return func

@case
def broken():
    # ERROR@+1: TYPE/001 {field_a="lhs"} "same type"
    return "ignored\\n"
"""
        )

        def invoke(case: PythonCheckCase) -> str:
            lines = case.check_case.input.splitlines()
            target_line = case.check_case.line_start + lines.index(
                '    return "ignored\\n"'
            )
            diagnostic = Diagnostic(
                severity=DiagnosticSeverity.ERROR,
                error_def=ERR_TYPE_001,
                message="values must have the same type",
                source_location=SourceRange.line(case.check_case.path, target_line),
                params=(DiagnosticParam("field_a", "lhs"),),
            )
            raise LoomDiagnosticError((diagnostic,))

        results = run_python_check(
            path,
            options=PythonCheckOptions(),
            is_case=lambda value: bool(getattr(value, "__case__", False)),
            invoke=invoke,
        )

    assert results[0].passed


def test_run_python_check_filters_by_function_name() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cases.py"
        path.write_text(
            """
def case(func):
    func.__case__ = True
    return func

@case
def first():
    return "first\\n"
# ----
# first

# ====
@case
def second():
    return "second\\n"
# ----
# second
"""
        )

        results = run_python_check(
            path,
            options=PythonCheckOptions(case_filter="second"),
            is_case=lambda value: bool(getattr(value, "__case__", False)),
            invoke=lambda case: case.function(),
        )

    assert len(results) == 1
    assert results[0].case_index == 1
    assert results[0].passed


def test_invoke_receives_python_check_case() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cases.py"
        path.write_text(
            """
def case(func):
    func.__case__ = True
    return func

@case
def named():
    return "ignored\\n"
# ----
# cases.py:0
"""
        )

        def invoke(case: PythonCheckCase) -> str:
            return f"{case.check_case.path.name}:{case.check_case.index}\n"

        results = run_python_check(
            path,
            options=PythonCheckOptions(),
            is_case=lambda value: bool(getattr(value, "__case__", False)),
            invoke=invoke,
        )

    assert results[0].passed


def test_run_python_check_preserves_invocation_metadata() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cases.py"
        path.write_text(
            """
def case(func):
    func.__case__ = True
    return func

@case
def named():
    return "ignored\\n"
# ----
# actual
"""
        )

        def invoke(_case: PythonCheckCase) -> PythonCheckInvocation:
            return PythonCheckInvocation(
                stdout="actual\n",
                stderr="sidecar\n",
                metadata={"artifact": "captured"},
            )

        results = run_python_check(
            path,
            options=PythonCheckOptions(),
            is_case=lambda value: bool(getattr(value, "__case__", False)),
            invoke=invoke,
        )

    assert results[0].passed
    assert results[0].stderr == "sidecar\n"
    assert results[0].metadata == {"artifact": "captured"}


def test_run_python_check_supports_case_skip() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cases.py"
        path.write_text(
            """
def case(func):
    func.__case__ = True
    return func

@case
def named():
    return "ignored\\n"
# ----
# ignored
"""
        )

        def invoke(_case: PythonCheckCase) -> str:
            raise PythonCheckSkip(
                "oracle unavailable",
                metadata={"dependency": "hipcc"},
            )

        results = run_python_check(
            path,
            options=PythonCheckOptions(),
            is_case=lambda value: bool(getattr(value, "__case__", False)),
            invoke=invoke,
        )

    assert results[0].skipped
    assert results[0].status == "skipped"
    assert results[0].stderr == "oracle unavailable\n"
    assert results[0].metadata == {"dependency": "hipcc"}
