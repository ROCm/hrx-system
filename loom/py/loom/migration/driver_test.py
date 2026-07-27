# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

from loom.diagnostics import DiagnosticSeverity
from loom.format.bytecode.writer import FORMAT_VERSION, MAGIC
from loom.migration.driver import (
    check_bytecode_version,
    discover_migration_files,
    migrate_loom_test_text,
    migrate_source_text,
)
from loom.migration.rules import MigrationRule, MigrationRuleApplication
from loom.migration.source import MigrationSourceDiagnostic, SourceEdit


def _edit_rule(
    rule_id: str,
    edit_fn,
) -> MigrationRule:
    return MigrationRule(
        rule_id=rule_id,
        rewrite=lambda document: MigrationRuleApplication(edits=(edit_fn(document),)),
    )


def test_source_rule_applies_precision_edit_and_validates_current_output() -> None:
    text = "func.def @f() {\n  func.return\n}\n"

    result = migrate_source_text(
        text,
        filename=Path("input.loom"),
        rules=(
            _edit_rule(
                "rename-function",
                lambda document: SourceEdit(
                    byte_start=document.text.index("@f") + 1,
                    byte_end=document.text.index("@f") + 2,
                    replacement_text="g",
                ),
            ),
        ),
    )

    assert result.ok
    assert result.changed
    assert result.text == "func.def @g() {\n  func.return\n}\n"


def test_source_rules_reject_overlapping_edits() -> None:
    result = migrate_source_text(
        "abcdef",
        rules=(
            _edit_rule("first", lambda _: SourceEdit(1, 4, "BCD")),
            _edit_rule("second", lambda _: SourceEdit(3, 5, "DE")),
        ),
        validate=False,
    )

    assert not result.ok
    assert result.diagnostics[0].severity == DiagnosticSeverity.ERROR
    assert result.diagnostics[0].rule_id == "second"
    assert "ordered and non-overlapping" in result.diagnostics[0].message


def test_source_migration_validates_rewritten_text_with_current_parser() -> None:
    text = "func.def @f() {\n  func.return\n}\n"

    result = migrate_source_text(
        text,
        filename=Path("broken.loom"),
        rules=(
            _edit_rule(
                "break-current-parse",
                lambda document: SourceEdit(
                    byte_start=document.text.index("func.return"),
                    byte_end=document.text.index("func.return") + len("func.return"),
                    replacement_text="not-an-op",
                ),
            ),
        ),
    )

    assert not result.ok
    assert result.diagnostics[0].rule_id == "loom.migrate.current_parse"
    assert result.diagnostics[0].source_range is not None
    assert result.diagnostics[0].source_range.filename == Path("broken.loom")


def test_loom_test_migrates_input_ir_without_touching_expected_output() -> None:
    text = (
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

    result = migrate_loom_test_text(
        text,
        filename=Path("case.loom-test"),
        rules=(
            _edit_rule(
                "rename-function",
                lambda document: SourceEdit(
                    byte_start=document.text.index("@f") + 1,
                    byte_end=document.text.index("@f") + 2,
                    replacement_text="g",
                ),
            ),
        ),
    )

    assert result.ok
    assert result.changed
    assert result.text == (
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


def test_loom_test_rule_diagnostic_maps_to_case_input_line() -> None:
    text = (
        "// RUN: roundtrip\n"
        "// descriptive harness comment\n"
        "\n"
        "func.def @bad() {\n"
        "  not-an-op\n"
        "}\n"
        "// ----\n"
        "func.def @bad() {\n"
        "  not-an-op\n"
        "}\n"
    )

    def diagnose_bad_op(document):
        byte_start = document.text.index("not-an-op")
        return MigrationRuleApplication(
            diagnostics=(
                MigrationSourceDiagnostic(
                    severity=DiagnosticSeverity.ERROR,
                    message="bad op",
                    source_range=document.byte_source_range(
                        byte_start,
                        byte_start + len("not-an-op"),
                    ),
                    rule_id="test.bad_op",
                ),
            )
        )

    result = migrate_loom_test_text(
        text,
        filename=Path("broken.loom-test"),
        rules=(MigrationRule("test.bad_op", diagnose_bad_op),),
    )

    assert not result.ok
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.rule_id == "test.bad_op"
    assert diagnostic.source_range is not None
    assert diagnostic.source_range.filename == Path("broken.loom-test")
    assert diagnostic.source_range.start_line == 5
    assert diagnostic.source_range.start_column == 3


def test_loom_test_fragment_input_is_left_to_loom_check() -> None:
    text = (
        "// RUN: roundtrip\n"
        "\n"
        "%a = test.constant 1 : i32\n"
        "%b = test.constant 2 : i32\n"
        "%c = test.addi %a, %b : i32\n"
    )

    result = migrate_loom_test_text(text, filename=Path("fragment.loom-test"))

    assert result.ok
    assert not result.changed


def test_loom_test_requires_input_is_left_to_loom_check() -> None:
    text = (
        "// RUN: roundtrip\n"
        "// REQUIRES: loom-check-test-unavailable\n"
        "\n"
        "this.is.invalid.ir\n"
    )

    result = migrate_loom_test_text(text, filename=Path("requires.loom-test"))

    assert result.ok
    assert not result.changed


def test_loom_test_annotated_negative_input_is_not_locally_rejected() -> None:
    text = (
        "// RUN: verify\n"
        "\n"
        "func.def @unknown_op() {\n"
        "  // ERROR@+1: PARSE/006\n"
        "  bogus.nonexistent\n"
        "}\n"
    )

    result = migrate_loom_test_text(text, filename=Path("negative.loom-test"))

    assert result.ok
    assert not result.changed


def test_loom_test_diagnostic_mapping_accounts_for_earlier_rewrites() -> None:
    text = (
        "func.def @first() {\n"
        "  func.return\n"
        "}\n"
        "// ====\n"
        "func.def @bad() {\n"
        "  not-an-op\n"
        "}\n"
    )

    def rewrite_or_diagnose(document):
        if "@first" not in document.text:
            byte_start = document.text.index("not-an-op")
            return MigrationRuleApplication(
                diagnostics=(
                    MigrationSourceDiagnostic(
                        severity=DiagnosticSeverity.ERROR,
                        message="bad op",
                        source_range=document.byte_source_range(
                            byte_start,
                            byte_start + len("not-an-op"),
                        ),
                        rule_id="test.bad_op",
                    ),
                )
            )
        return MigrationRuleApplication(edits=(SourceEdit(0, 0, "// migrated\n"),))

    result = migrate_loom_test_text(
        text,
        filename=Path("shifted.loom-test"),
        rules=(MigrationRule("rewrite-or-diagnose", rewrite_or_diagnose),),
    )

    assert not result.ok
    assert result.diagnostics[0].rule_id == "test.bad_op"
    assert result.diagnostics[0].source_range is not None
    assert result.diagnostics[0].source_range.start_line == 7


def test_current_bytecode_version_is_accepted() -> None:
    assert check_bytecode_version(MAGIC + bytes([FORMAT_VERSION])) == ()


def test_old_bytecode_version_reports_actual_and_current_versions() -> None:
    diagnostics = check_bytecode_version(MAGIC + bytes([FORMAT_VERSION - 1]))

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.rule_id == "loom.migrate.bytecode_version"
    assert diagnostic.actual_version == FORMAT_VERSION - 1
    assert diagnostic.current_version == FORMAT_VERSION
    assert "Regenerate" in diagnostic.fixup_hint


def test_discover_migration_files_sorts_supported_kinds(tmp_path: Path) -> None:
    (tmp_path / "z.loom").write_text("", encoding="utf-8")
    (tmp_path / "a.loombc").write_bytes(MAGIC + bytes([FORMAT_VERSION]))
    (tmp_path / "b.loom-test").write_text("", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "m.loom").write_text("", encoding="utf-8")

    assert discover_migration_files(tmp_path) == (
        tmp_path / "a.loombc",
        tmp_path / "b.loom-test",
        nested / "m.loom",
        tmp_path / "z.loom",
    )
