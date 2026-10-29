# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

from loom.assembly import COLON, AttrDict, Ref, TypeOf
from loom.dsl import ANY, ATTR_TYPE_DICT, AttrDef, LegacyFormat, Op, Operand, Result
from loom.migration.driver import (
    default_migration_rules,
    migrate_loom_test_text,
    migrate_source_text,
)
from loom.migration.rules import (
    BUFFER_ASSUME_MEMORY_SPACE_ATTR_DICT_RULE_ID,
    CHECK_EXPECT_CLOSE_APPROXIMATE_RULE_ID,
    MigrationRuleApplication,
    migration_rules_from_ops,
)
from loom.migration.source import SourceDocument


def test_default_rules_are_generated_from_op_legacy_formats() -> None:
    rule_ids = tuple(rule.rule_id for rule in default_migration_rules())

    assert BUFFER_ASSUME_MEMORY_SPACE_ATTR_DICT_RULE_ID in rule_ids
    assert CHECK_EXPECT_CLOSE_APPROXIMATE_RULE_ID in rule_ids


def test_check_expect_close_rewrites_to_typed_approximate() -> None:
    text = (
        "check.case @case {\n"
        "  %actual = check.literal value(1.0) : f32\n"
        "  %expected = check.literal value(1.0) : f32\n"
        "  check.expect.close actual(%actual) expected(%expected) "
        "atol(0.0) rtol(1.0e-5) nan(same) : f32\n"
        "  check.return\n"
        "}\n"
    )

    result = migrate_source_text(text, filename=Path("input.loom"))

    assert result.ok, result.diagnostics
    assert result.changed
    assert result.text == (
        "check.case @case {\n"
        "  %actual = check.literal value(1.0) : f32\n"
        "  %expected = check.literal value(1.0) : f32\n"
        "  check.expect.approximate actual(%actual) expected(%expected) "
        "atol(0.0) rtol(1.0e-5) nan(same) : f32, f32\n"
        "  check.return\n"
        "}\n"
    )


def test_check_expect_close_rewrite_tolerates_zero_length_compact_types() -> None:
    text = (
        "func.def @empty() -> (vector<0xi32>) {\n"
        "  %empty = vector.empty : vector<0xi32>\n"
        "  func.return %empty : vector<0xi32>\n"
        "}\n"
        "check.case @case {\n"
        "  %actual = check.literal value(1.0) : f32\n"
        "  check.expect.close actual(%actual) expected(%actual) "
        "atol(0.0) rtol(0.0) nan(same) : f32\n"
        "  check.return\n"
        "}\n"
    )

    result = migrate_source_text(text, filename=Path("input.loom"))

    assert result.ok, result.diagnostics
    assert result.changed
    assert result.text.count("vector<0xi32>") == 3
    assert "check.expect.approximate" in result.text


def test_check_expect_approximate_current_syntax_is_noop() -> None:
    text = (
        "check.case @case {\n"
        "  %actual = check.literal value(1.0) : f16\n"
        "  %expected = check.literal value(1.0) : f64\n"
        "  check.expect.approximate actual(%actual) expected(%expected) "
        "atol(0.0) rtol(1.0e-5) nan(same) : f16, f64\n"
        "  check.return\n"
        "}\n"
    )

    result = migrate_source_text(text, filename=Path("input.loom"))

    assert result.ok
    assert not result.changed
    assert result.text == text


def test_check_expect_close_rewrite_is_idempotent() -> None:
    text = (
        "check.case @case {\n"
        "  %actual = check.literal value(1.0) : bf16\n"
        "  %expected = check.literal value(1.0) : bf16\n"
        "  check.expect.close actual(%actual) expected(%expected) "
        "atol(0.01) rtol(0.0) nan(different) : bf16\n"
        "  check.return\n"
        "}\n"
    )

    first_result = migrate_source_text(text, filename=Path("input.loom"))
    second_result = migrate_source_text(
        first_result.text,
        filename=Path("input.loom"),
    )

    assert first_result.ok, first_result.diagnostics
    assert first_result.changed
    assert second_result.ok
    assert not second_result.changed


def test_check_expect_close_rewrites_loom_test_input_only() -> None:
    legacy = (
        "check.case @case {\n"
        "  %actual = check.literal value(1.0) : f32\n"
        "  check.expect.close actual(%actual) expected(%actual) "
        "atol(0.0) rtol(0.0) nan(same) : f32\n"
        "  check.return\n"
        "}\n"
    )
    text = "// RUN: roundtrip\n\n" + legacy + "// ----\n" + legacy

    result = migrate_loom_test_text(text, filename=Path("case.loom-test"))

    assert result.ok, result.diagnostics
    assert result.changed
    input_text, expected_text = result.text.split("// ----\n", maxsplit=1)
    assert "check.expect.approximate" in input_text
    assert " : f32, f32" in input_text
    assert expected_text == legacy


def test_buffer_assume_memory_space_attr_dict_rewrites_source() -> None:
    text = (
        "func.def @f(%buffer: buffer) -> (buffer) {\n"
        "  %global = buffer.assume.memory_space %buffer "
        "{memory_space = global} : buffer\n"
        "  func.return %global : buffer\n"
        "}\n"
    )

    result = migrate_source_text(text, filename=Path("input.loom"))

    assert result.ok
    assert result.changed
    assert result.text == (
        "func.def @f(%buffer: buffer) -> (buffer) {\n"
        "  %global = buffer.assume.memory_space<global> %buffer : buffer\n"
        "  func.return %global : buffer\n"
        "}\n"
    )


def test_buffer_assume_memory_space_current_syntax_is_noop() -> None:
    text = (
        "func.def @f(%buffer: buffer) -> (buffer) {\n"
        "  %global = buffer.assume.memory_space<global> %buffer : buffer\n"
        "  func.return %global : buffer\n"
        "}\n"
    )

    result = migrate_source_text(text, filename=Path("input.loom"))

    assert result.ok
    assert not result.changed
    assert result.text == text


def test_buffer_assume_memory_space_rule_does_not_tokenize_unrelated_source() -> None:
    text = "config.decl @tuner.workgroup_size : %value: index\n"

    result = migrate_source_text(
        text,
        filename=Path("unrelated.loom"),
        validate=False,
    )

    assert result.ok
    assert not result.changed


def test_buffer_assume_memory_space_rewrite_is_idempotent() -> None:
    text = (
        "func.def @f(%buffer: buffer) -> (buffer) {\n"
        "  %global = buffer.assume.memory_space %buffer "
        "{memory_space = global} : buffer\n"
        "  func.return %global : buffer\n"
        "}\n"
    )

    first_result = migrate_source_text(text, filename=Path("input.loom"))
    second_result = migrate_source_text(first_result.text, filename=Path("input.loom"))

    assert first_result.ok
    assert first_result.changed
    assert second_result.ok
    assert not second_result.changed


def test_buffer_assume_memory_space_malformed_legacy_syntax_reports_rule() -> None:
    text = (
        "func.def @f(%buffer: buffer) -> (buffer) {\n"
        "  %global = buffer.assume.memory_space %buffer {space = global} : buffer\n"
        "  func.return %global : buffer\n"
        "}\n"
    )

    result = migrate_source_text(text, filename=Path("broken.loom"))

    assert not result.ok
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.rule_id == BUFFER_ASSUME_MEMORY_SPACE_ATTR_DICT_RULE_ID
    assert diagnostic.source_range is not None
    assert diagnostic.source_range.filename == Path("broken.loom")
    assert "expected legacy" in diagnostic.message


def test_buffer_assume_memory_space_rewrites_loom_test_input_only() -> None:
    text = (
        "// RUN: roundtrip\n"
        "\n"
        "func.def @f(%buffer: buffer) -> (buffer) {\n"
        "  %global = buffer.assume.memory_space %buffer "
        "{memory_space = global} : buffer\n"
        "  func.return %global : buffer\n"
        "}\n"
        "// ----\n"
        "func.def @f(%buffer: buffer) -> (buffer) {\n"
        "  %global = buffer.assume.memory_space %buffer "
        "{memory_space = global} : buffer\n"
        "  func.return %global : buffer\n"
        "}\n"
    )

    result = migrate_loom_test_text(text, filename=Path("case.loom-test"))

    assert result.ok
    assert result.changed
    assert result.text == (
        "// RUN: roundtrip\n"
        "\n"
        "func.def @f(%buffer: buffer) -> (buffer) {\n"
        "  %global = buffer.assume.memory_space<global> %buffer : buffer\n"
        "  func.return %global : buffer\n"
        "}\n"
        "// ----\n"
        "func.def @f(%buffer: buffer) -> (buffer) {\n"
        "  %global = buffer.assume.memory_space %buffer "
        "{memory_space = global} : buffer\n"
        "  func.return %global : buffer\n"
        "}\n"
    )


def test_legacy_format_rewrite_hook_is_registered_by_name() -> None:
    op = Op(
        "test.semantic",
        attrs=[AttrDef("attrs", ATTR_TYPE_DICT)],
        legacy_formats=[
            LegacyFormat(
                "test.semantic.old",
                format=[AttrDict("attrs")],
                replaced_by="loom-source-format-2026-06-09",
                rewrite_hook="semantic_rewrite",
            )
        ],
    )
    seen_documents: list[str] = []

    def semantic_rewrite(document: SourceDocument) -> MigrationRuleApplication:
        seen_documents.append(document.text)
        return MigrationRuleApplication()

    rules = migration_rules_from_ops(
        (op,), rewrite_hooks={"semantic_rewrite": semantic_rewrite}
    )
    application = rules[0].rewrite(SourceDocument("test.semantic {kind = old}\n"))

    assert application == MigrationRuleApplication()
    assert seen_documents == ["test.semantic {kind = old}\n"]


def test_legacy_format_rewrite_hook_must_be_registered() -> None:
    op = Op(
        "test.semantic",
        attrs=[AttrDef("attrs", ATTR_TYPE_DICT)],
        legacy_formats=[
            LegacyFormat(
                "test.semantic.old",
                format=[AttrDict("attrs")],
                replaced_by="loom-source-format-2026-06-09",
                rewrite_hook="semantic_rewrite",
            )
        ],
    )

    with pytest.raises(ValueError, match="unknown rewrite hook"):
        migration_rules_from_ops((op,))


def test_unsupported_structural_format_requires_rewrite_hook() -> None:
    op = Op(
        "test.unsupported",
        operands=[Operand("input", ANY)],
        results=[Result("result", ANY)],
        attrs=[AttrDef("attrs", ATTR_TYPE_DICT)],
        format=[Ref("input"), COLON, TypeOf("result")],
        legacy_formats=[
            LegacyFormat(
                "test.unsupported.old",
                format=[AttrDict("attrs")],
                replaced_by="loom-source-format-2026-06-09",
            )
        ],
    )

    with pytest.raises(ValueError, match="provide a rewrite hook"):
        migration_rules_from_ops((op,))
