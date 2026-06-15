# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

from loom.importers.check.cases import (
    CheckCase,
    InlineCheckSyntax,
    case_matches_filter,
    parse_inline_cases,
    rewrite_inline_case_inputs,
    split_expected,
    split_raw_cases,
)


def test_split_raw_cases_uses_loom_check_case_separator() -> None:
    source = "case 0\n// ====\ncase 1\n"

    assert split_raw_cases(source) == ("case 0\n", "case 1\n")


def test_split_raw_cases_ignores_separator_spacer() -> None:
    source = "case 0\n\n// ====\ncase 1\n"

    assert split_raw_cases(source) == ("case 0\n", "case 1\n")


def test_split_raw_cases_ignores_multiple_separator_spacer_lines() -> None:
    source = "case 0\n\n\n// ====\ncase 1\n"

    assert split_raw_cases(source) == ("case 0\n", "case 1\n")


def test_split_expected_uses_loom_check_expected_separator() -> None:
    source, expected, has_expected = split_expected("input\n// ----\nexpected\n")

    assert source == "input\n"
    assert expected == "expected\n"
    assert has_expected


def test_parse_inline_cases_uses_configured_comment_syntax() -> None:
    syntax = InlineCheckSyntax(
        case_separator_prefix="# ====",
        expected_separator="# ----",
        comment_prefix="#",
    )
    cases = parse_inline_cases(
        Path("kernels.py"),
        "case_0()\n# ----\nexpected 0\n\n# ====\ncase_1()\n# ----\nexpected 1\n",
        syntax=syntax,
    )

    assert len(cases) == 2
    assert cases[0] == CheckCase(
        path=Path("kernels.py"),
        index=0,
        source="case_0()\n",
        input="case_0()\n",
        expected="expected 0\n",
        has_expected=True,
    )
    assert cases[1].input == "case_1()\n"


def test_parse_inline_cases_can_ignore_shared_preamble() -> None:
    syntax = InlineCheckSyntax(
        case_separator_prefix="# ====",
        expected_separator="# ----",
        comment_prefix="#",
    )
    cases = parse_inline_cases(
        Path("kernels.py"),
        "shared = 1\n\n# ====\ncase_0()\n# ----\nexpected 0\n",
        syntax=syntax,
        allow_preamble=True,
    )

    assert len(cases) == 1
    assert cases[0] == CheckCase(
        path=Path("kernels.py"),
        index=0,
        source="case_0()\n",
        input="case_0()\n",
        expected="expected 0\n",
        has_expected=True,
        line_start=4,
        line_end=6,
    )


def test_parse_inline_cases_keeps_first_case_without_preamble_mode() -> None:
    syntax = InlineCheckSyntax(
        case_separator_prefix="# ====",
        expected_separator="# ----",
        comment_prefix="#",
    )
    cases = parse_inline_cases(
        Path("kernels.py"),
        "case_0()\n# ----\nexpected 0\n\n# ====\ncase_1()\n# ----\nexpected 1\n",
        syntax=syntax,
        allow_preamble=True,
    )

    assert len(cases) == 2
    assert cases[0].input == "case_0()\n"
    assert cases[1].input == "case_1()\n"


def test_parse_inline_cases_tracks_input_and_expected_spans() -> None:
    source = (
        "// RUN: roundtrip\n"
        "// descriptive harness comment\n"
        "\n"
        "func.def @f() {\n"
        "  func.return\n"
        "}\n"
        "// ----\n"
        "expected output\n"
    )

    cases = parse_inline_cases(Path("case.loom-test"), source)

    assert len(cases) == 1
    assert cases[0].input_span is not None
    assert cases[0].input_span.text_from(source) == (
        "func.def @f() {\n  func.return\n}\n"
    )
    assert cases[0].expected_span is not None
    assert cases[0].expected_span.text_from(source) == "expected output\n"


def test_rewrite_inline_case_inputs_preserves_expected_sections() -> None:
    source = (
        "// RUN: roundtrip\n"
        "\n"
        "func.def @f() {\n"
        "  func.return\n"
        "}\n"
        "// ----\n"
        "func.def @f() {\n"
        "  func.return\n"
        "}\n"
    )

    result = rewrite_inline_case_inputs(
        Path("case.loom-test"),
        source,
        lambda _case, input_text: input_text.replace("@f", "@g"),
    )

    assert result.changed
    assert len(result.changed_cases) == 1
    assert result.source == (
        "// RUN: roundtrip\n"
        "\n"
        "func.def @g() {\n"
        "  func.return\n"
        "}\n"
        "// ----\n"
        "func.def @f() {\n"
        "  func.return\n"
        "}\n"
    )


def test_case_matches_filter_checks_path_case_run_and_labels() -> None:
    check_case = CheckCase(
        path=Path("kernels/vector.mlir"),
        index=2,
        source="",
        input="",
        expected="",
        run="mlir --kernel reduce",
    )

    assert case_matches_filter(check_case, "vector.mlir")
    assert case_matches_filter(check_case, "case2")
    assert case_matches_filter(check_case, "reduce")
    assert case_matches_filter(check_case, "arith", labels=("arith",))
    assert not case_matches_filter(check_case, "missing")
