# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

from loom.importers.check.cases import InlineCheckSyntax, parse_inline_cases
from loom.importers.check.results import CheckResult
from loom.importers.check.updates import format_updated_case, format_updated_source


def test_format_updated_case_replaces_expected_section() -> None:
    assert (
        format_updated_case("input\n// ----\nold\n", "new\n") == "input\n// ----\nnew\n"
    )


def test_format_updated_source_puts_blank_line_before_case_separator() -> None:
    cases = parse_inline_cases(
        Path("kernels.mlir"),
        "input\n// ----\nold\n// ====\ninput2\n// ----\nold2\n",
    )
    results = [
        CheckResult(Path("kernels.mlir"), 0, 0, "new\n", ""),
        CheckResult(Path("kernels.mlir"), 1, 0, "new2\n", ""),
    ]

    assert (
        format_updated_source("", cases, results)
        == "input\n// ----\nnew\n\n// ====\ninput2\n// ----\nnew2\n"
    )


def test_format_updated_source_keeps_unselected_cases() -> None:
    source = "input\n// ----\nold\n\n// ====\ninput2\n// ----\nold2\n"
    cases = parse_inline_cases(
        Path("kernels.mlir"),
        source,
    )
    results = [CheckResult(Path("kernels.mlir"), 1, 0, "new2\n", "")]

    assert (
        format_updated_source(source, cases, results)
        == "input\n// ----\nold\n\n// ====\ninput2\n// ----\nnew2\n"
    )


def test_format_updated_source_keeps_skipped_cases() -> None:
    source = "input\n// ----\nold\n"
    cases = parse_inline_cases(
        Path("kernels.mlir"),
        source,
    )
    results = [CheckResult(Path("kernels.mlir"), 0, 0, "", "", skipped=True)]

    assert format_updated_source(source, cases, results) == source


def test_format_updated_source_preserves_python_preamble() -> None:
    syntax = InlineCheckSyntax(
        case_separator_prefix="# ====",
        expected_separator="# ----",
        comment_prefix="#",
    )
    source = "shared = 1\n\n# ====\ncase_0()\n# ----\n# old\n"
    cases = parse_inline_cases(
        Path("kernels.py"),
        source,
        syntax=syntax,
        allow_preamble=True,
    )
    results = [CheckResult(Path("kernels.py"), 0, 0, "new\n", "")]

    assert (
        format_updated_source(
            source,
            cases,
            results,
            syntax=syntax,
            expected_encoder=lambda text: f"# {text}",
        )
        == "shared = 1\n\n# ====\ncase_0()\n# ----\n# new\n"
    )


def test_format_updated_source_does_not_add_expected_section_for_empty_stdout() -> None:
    source = "# ERROR@+1: TYPE/001\nbad()\n"
    cases = parse_inline_cases(
        Path("kernels.py"),
        source,
        syntax=InlineCheckSyntax(
            case_separator_prefix="# ====",
            expected_separator="# ----",
            comment_prefix="#",
        ),
    )
    results = [CheckResult(Path("kernels.py"), 0, 0, "", "matched diagnostic\n")]

    assert (
        format_updated_source(
            source,
            cases,
            syntax=InlineCheckSyntax(
                case_separator_prefix="# ====",
                expected_separator="# ----",
                comment_prefix="#",
            ),
            results=results,
        )
        == source
    )


def test_format_updated_source_uses_custom_case_separator_spacer() -> None:
    syntax = InlineCheckSyntax(
        case_separator_prefix="# ====",
        expected_separator="# ----",
        comment_prefix="#",
        case_separator_spacer="\n\n",
    )
    source = "# ====\ncase_0()\n# ----\n# old\n\n# ====\ncase_1()\n# ----\n# old\n"
    cases = parse_inline_cases(
        Path("kernels.py"),
        source,
        syntax=syntax,
    )
    results = [
        CheckResult(Path("kernels.py"), 0, 0, "new\n", ""),
        CheckResult(Path("kernels.py"), 1, 0, "new2\n", ""),
    ]

    assert (
        format_updated_source(
            source,
            cases,
            results,
            syntax=syntax,
            expected_encoder=lambda text: f"# {text}",
        )
        == "# ====\ncase_0()\n# ----\n# new\n\n\n# ====\ncase_1()\n# ----\n# new2\n"
    )
