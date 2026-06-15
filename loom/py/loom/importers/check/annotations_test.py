# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

from loom.diagnostics import (
    Diagnostic,
    DiagnosticParam,
    DiagnosticSeverity,
    SourceRange,
)
from loom.error.dominance import ERR_DOMINANCE_011
from loom.error.type import ERR_TYPE_001
from loom.importers.check.annotations import (
    match_source_diagnostics,
    parse_expected_diagnostics,
)


def test_parse_expected_diagnostics_uses_comment_prefix_and_offsets() -> None:
    expected = parse_expected_diagnostics(
        '# ERROR@+2: TYPE/001 "lhs" "rhs"\nvalue = 1\nbad()\n',
        path=Path("case.py"),
        line_start=10,
        comment_prefix="#",
    )

    assert len(expected) == 1
    assert expected[0].severity == DiagnosticSeverity.ERROR
    assert expected[0].domain == "TYPE"
    assert expected[0].code == 1
    assert expected[0].annotation.start_line == 10
    assert expected[0].target.start_line == 12
    assert expected[0].message_substrings == ("lhs", "rhs")


def test_parse_expected_diagnostics_supports_domain_only_and_unsigned_offsets() -> None:
    expected = parse_expected_diagnostics(
        "# ERROR@1: TYPE\nbad()\n",
        path=Path("case.py"),
        line_start=1,
        comment_prefix="#",
    )

    assert expected[0].domain == "TYPE"
    assert expected[0].code == 0
    assert expected[0].target.start_line == 2


def test_parse_expected_diagnostics_ignores_natural_language_comments() -> None:
    assert (
        parse_expected_diagnostics(
            "# ERROR is a word in this comment\nbad()\n",
            path=Path("case.py"),
            line_start=1,
            comment_prefix="#",
        )
        == ()
    )


def test_parse_expected_diagnostics_supports_structured_param_matchers() -> None:
    expected = parse_expected_diagnostics(
        '# ERROR@+1: DOMINANCE/011 {op_name="test.add", field_name="lhs"}\nbad()\n',
        path=Path("case.py"),
        line_start=1,
        comment_prefix="#",
    )

    assert expected[0].param_matches[0].name == "op_name"
    assert expected[0].param_matches[0].value == "test.add"
    assert expected[0].param_matches[1].name == "field_name"
    assert expected[0].param_matches[1].value == "lhs"


def test_match_source_diagnostics_accepts_matching_core_diagnostic() -> None:
    expected = parse_expected_diagnostics(
        '# ERROR@+1: TYPE/001 "same type"\nbad()\n',
        path=Path("case.py"),
        line_start=1,
        comment_prefix="#",
    )
    actual = (
        Diagnostic(
            severity=DiagnosticSeverity.ERROR,
            error_def=ERR_TYPE_001,
            message="values must have the same type",
            source_location=SourceRange.line(Path("case.py"), 2),
        ),
    )

    assert match_source_diagnostics(expected, actual).passed


def test_match_source_diagnostics_rejects_unexpected_diagnostic() -> None:
    actual = (
        Diagnostic(
            severity=DiagnosticSeverity.ERROR,
            message="boom",
            source_location=SourceRange.line(Path("case.py"), 2),
        ),
    )

    result = match_source_diagnostics((), actual)

    assert not result.passed
    assert "unexpected diagnostic" in result.message()


def test_match_source_diagnostics_accepts_structured_params() -> None:
    expected = parse_expected_diagnostics(
        '# ERROR@+1: DOMINANCE/011 {op_name="test.add", field_name="lhs"}\nbad()\n',
        path=Path("case.py"),
        line_start=1,
        comment_prefix="#",
    )
    actual = (
        Diagnostic(
            severity=DiagnosticSeverity.ERROR,
            error_def=ERR_DOMINANCE_011,
            message="lowering failed",
            source_location=SourceRange.line(Path("case.py"), 2),
            params=(
                DiagnosticParam("op_name", "test.add"),
                DiagnosticParam("field_name", "lhs"),
            ),
        ),
    )

    assert match_source_diagnostics(expected, actual).passed


def test_match_source_diagnostics_rejects_unmatched_annotation() -> None:
    expected = parse_expected_diagnostics(
        '# ERROR@+1: TYPE/001 "same type"\nbad()\n',
        path=Path("case.py"),
        line_start=1,
        comment_prefix="#",
    )

    result = match_source_diagnostics(expected, ())

    assert not result.passed
    assert "expected diagnostic was not produced" in result.message()
