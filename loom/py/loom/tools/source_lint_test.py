# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

from loom.tools.source_lint import CONSTANT_NAME_RULE, lint_source, main


def _finding_names(source: str, suffix: str = ".loom") -> list[str]:
    return [finding.name for finding in lint_source(Path("source" + suffix), source)]


@pytest.mark.parametrize(
    "name",
    [
        "two",
        "fivehundredtwelve",
        "thirty_two",
        "zero_f32x4",
        "four_bytes",
        "negative_one",
        "i32_fivehundredtwelve_bytes",
        "all_ones",
        "ones",
        "zeroes",
        "sixes",
        "twenties",
        "one_hundred_and_one",
        "thousand",
    ],
)
def test_spelled_numeric_constant_names_fail(name: str) -> None:
    source = f"%{name} = index.constant 1 : index\n"

    assert _finding_names(source) == [name]


@pytest.mark.parametrize(
    "name",
    [
        "batch_size",
        "c512",
        "c0_i32",
        "zero_point",
        "zero_exponent",
        "two_pi",
        "positive_signed_zero",
        "all_bits_set",
        "f32_sign_threshold",
        "nonzero",
        "fourth_word_ordinal",
        "and",
        "and_zero",
    ],
)
def test_semantic_and_compact_numeric_names_pass(name: str) -> None:
    source = f"%{name} = scalar.constant 1 : i32\n"

    assert _finding_names(source) == []


def test_constant_operation_and_assignment_may_span_lines() -> None:
    source = """module {
  %fivehundredtwelve
      =
      index.constant 512 : index
}
"""

    findings = lint_source(Path("multiline.loom"), source)

    assert [(finding.line, finding.column, finding.name) for finding in findings] == [
        (2, 3, "fivehundredtwelve")
    ]


def test_all_constant_operation_spellings_share_the_rule() -> None:
    source = """%one = scalar.constant 1 : i32
%two = vector.constant 2 : vector<4xi32>
%three = test.clause_constant value(3) : i32
%four = test.effectful_constant 4 : i32
"""

    assert _finding_names(source) == ["one", "two", "three", "four"]


def test_comments_and_string_literals_are_not_authored_operations() -> None:
    source = """// %one = index.constant 1 : index
test.string "%two = index.constant 2 : index // still a string"
test.string "escaped \\" %three = index.constant 3 : index"
%batch_size = index.constant 512 : index // %four = index.constant 4 : index
"""

    assert _finding_names(source) == []


def test_loom_test_expected_output_is_excluded_and_cases_restore_input() -> None:
    source = """%two = index.constant 2 : index
// ----
%fivehundredtwelve = index.constant 512 : index
// ==== second case
%thirty_two = index.constant 32 : index
// ----
%sixty_four = index.constant 64 : index
"""

    findings = lint_source(Path("cases.loom-test"), source)

    assert [(finding.line, finding.name) for finding in findings] == [
        (1, "two"),
        (5, "thirty_two"),
    ]


def test_cli_reports_source_location_rule_and_remedy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_path = tmp_path / "bad.loom"
    source_path.write_text(
        "module {\n  %fivehundredtwelve = index.constant 512 : index\n}\n",
        encoding="utf-8",
    )

    assert main([str(source_path)]) == 1

    diagnostic = capsys.readouterr().err
    assert f"{source_path}:2:3: error:" in diagnostic
    assert "%fivehundredtwelve" in diagnostic
    assert "%c<literal>" in diagnostic
    assert f"[{CONSTANT_NAME_RULE}]" in diagnostic


def test_cli_accepts_clean_batches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first_path = tmp_path / "first.loom"
    first_path.write_text("%c1 = index.constant 1 : index\n", encoding="utf-8")
    second_path = tmp_path / "second.loom-test"
    second_path.write_text("%batch_size = index.constant 1 : index\n", encoding="utf-8")

    assert main([str(first_path), str(second_path)]) == 0
    assert capsys.readouterr().err == ""


def test_cli_rejects_unsupported_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_path = tmp_path / "source.txt"
    source_path.write_text("%one = index.constant 1 : index\n", encoding="utf-8")

    assert main([str(source_path)]) == 2
    assert "unsupported source suffix" in capsys.readouterr().err


def test_cli_rejects_missing_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_path = tmp_path / "missing.loom"

    assert main([str(source_path)]) == 2
    assert "source file does not exist" in capsys.readouterr().err
