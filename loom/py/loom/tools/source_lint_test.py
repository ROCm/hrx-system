# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

from loom.tools.source_lint import (
    CONSTANT_NAME_RULE,
    GENERATED_NAME_RULE,
    lint_source,
    main,
)


def _finding_names(source: str, suffix: str = ".loom") -> list[str]:
    return [
        finding.name.removeprefix("%")
        for finding in lint_source(Path("source" + suffix), source)
    ]


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
        (2, 3, "%fivehundredtwelve")
    ]


def test_all_constant_operation_spellings_share_the_rule() -> None:
    source = """%one = scalar.constant 1 : i32
%two = vector.splati 2 : vector<4xi32>
%three = vector.splatf 3.0 : vector<4xf32>
%four = test.clause_constant value(4) : i32
%five = test.effectful_constant 5 : i32
"""

    assert _finding_names(source) == ["one", "two", "three", "four", "five"]


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
        (1, "%two"),
        (5, "%thirty_two"),
    ]


@pytest.mark.parametrize("suffix", [".loom", ".loom-test"])
@pytest.mark.parametrize(
    ("source", "name"),
    [
        ("%double$17$0 = index.constant 2 : index", "%double$17$0"),
        ("func.return %value$copy : i32", "%value$copy"),
        ("%17$0 = index.constant 17 : index", "%17$0"),
        ("%$temporary = index.constant 17 : index", "%$temporary"),
        ("%trailing$ = index.constant 17 : index", "%trailing$"),
        ("func.decl @pipeline$config(%value: i32)", "@pipeline$config"),
        ("func.decl @root::@nested$worker()", "@nested$worker"),
        ("func.decl @named(%argument$0: i32)", "%argument$0"),
        ("cfg.br ^block$1", "^block$1"),
        ("#layout$0 = #encoding.layout.dense", "#layout$0"),
        ("{field$name = 1}", "field$name"),
        ("func.decl @shaped(%input: tensor<[%extent$0]xf32>)", "%extent$0"),
    ],
)
def test_dollar_names_fail_in_every_authored_position(
    suffix: str, source: str, name: str
) -> None:
    findings = lint_source(Path("input" + suffix), source)

    assert [(finding.name, finding.rule) for finding in findings] == [
        (name, GENERATED_NAME_RULE)
    ]
    assert findings[0].column == source.index(name) + 1


def test_dollar_rule_accepts_comments_strings_and_semantic_names() -> None:
    source = r"""// %comment$0 and @helper$config are compiler output.
test.string "operands $2, $0"
test.string "escaped \" @not_a_symbol$1 // still a string"
%batch_dim = index.constant 57 : index // %double$17$0
func.decl @pipeline_worker(%input_count: index)
"""

    assert lint_source(Path("strings.loom"), source) == []


def test_dollar_output_is_allowed_but_each_case_restores_input_checks() -> None:
    source = """// RUN: roundtrip
%batch_dim = index.constant 57 : index
// ----
%batch_dim$0 = index.constant 57 : index
// ==== name resolution still requires authored role names
// RUN: roundtrip
%double$17$0 = index.constant 2 : index
// ----
%double$17$0 = index.constant 2 : index
// ====
func.decl @pipeline$config()
"""

    findings = lint_source(Path("cases.loom-test"), source)

    assert [(finding.line, finding.name) for finding in findings] == [
        (7, "%double$17$0"),
        (11, "@pipeline$config"),
    ]


def test_loom_source_has_no_expected_output_section() -> None:
    source = "// ----\n%output$0 = index.constant 0 : index\n"

    assert _finding_names(source) == ["output$0"]


def test_comment_markers_inside_multiline_strings_are_literal_contents() -> None:
    source = """test.string "multiline
// ----
%literal$0
// ==== still a string
@literal$config"
%input$0 = index.constant 0 : index
"""

    findings = lint_source(Path("multiline.loom"), source)

    assert [(finding.line, finding.name) for finding in findings] == [(6, "%input$0")]


def test_rules_report_mixed_findings_in_source_order_with_crlf() -> None:
    source = (
        "  %input$0 = index.constant 0 : index\r\n"
        "  %one = index.constant 1 : index\r\n"
        "func.decl @helper$config(%arg$2: i32)\r\n"
        "  %two = index.constant 2 : index\r\n"
    )

    findings = lint_source(Path("ordered.loom"), source)

    assert [(finding.line, finding.column, finding.rule) for finding in findings] == [
        (1, 3, GENERATED_NAME_RULE),
        (2, 3, CONSTANT_NAME_RULE),
        (3, 11, GENERATED_NAME_RULE),
        (3, 26, GENERATED_NAME_RULE),
        (4, 3, CONSTANT_NAME_RULE),
    ]


def test_dense_findings_retain_every_source_position() -> None:
    count = 4096
    source = "%input$0 = index.constant 0 : index\n" * count

    findings = lint_source(Path("dense.loom"), source)

    assert [(finding.line, finding.column) for finding in findings] == [
        (line, 1) for line in range(1, count + 1)
    ]


def test_long_number_names_and_decorators_preserve_classification() -> None:
    number = "one" * 2048
    decorated = "i32_" * 2048 + "one" + "_bytes" * 2048
    semantic = number + "_batch_dim"
    source = "\n".join(
        f"%{name} = index.constant 1 : index" for name in [number, decorated, semantic]
    )

    assert _finding_names(source) == [number, decorated]


@pytest.mark.parametrize("suffix", [".loom", ".loom-test"])
def test_cli_rejects_copied_names_with_a_role_based_remedy(
    suffix: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_path = tmp_path / ("copied" + suffix)
    source_path.write_text(
        "// copied output\n  %double$17$0 = index.constant 2 : index\n",
        encoding="utf-8",
    )

    assert main([str(source_path)]) == 1

    diagnostic = capsys.readouterr().err
    assert f"{source_path}:2:3: error:" in diagnostic
    assert "%double$17$0" in diagnostic
    assert "program role" in diagnostic
    assert f"[{GENERATED_NAME_RULE}]" in diagnostic


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
