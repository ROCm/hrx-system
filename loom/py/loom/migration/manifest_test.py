# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.dsl import ANY, LegacyFormat, Op, Operand
from loom.migration.manifest import (
    CURRENT_BYTECODE_VERSION,
    CURRENT_MANIFEST,
    CURRENT_TEXT_BASELINE,
    DIAGNOSTIC_ERROR,
    DIAGNOSTIC_WARNING,
    REQUIRED_ACTIVE_RULE_TEST_KINDS,
    LoomRelease,
    MigrationManifest,
    MigrationRuleMetadata,
    TextBaseline,
    lint_current_manifest,
    lint_legacy_formats,
    migration_rule_metadata_from_ops,
)


def _manifest_with_rule(rule: MigrationRuleMetadata) -> MigrationManifest:
    return MigrationManifest(
        text_baselines=(
            TextBaseline("pre-release"),
            TextBaseline("loom-source-format-2026-06-09"),
            TextBaseline("loom-source-format-2026-07-01"),
        ),
        current_text_baseline="loom-source-format-2026-06-09",
        current_bytecode_version=14,
        rules=(rule,),
    )


def test_current_manifest_has_no_releases_before_first_release() -> None:
    assert CURRENT_MANIFEST.current_text_baseline == CURRENT_TEXT_BASELINE
    assert CURRENT_MANIFEST.current_bytecode_version == CURRENT_BYTECODE_VERSION
    assert CURRENT_MANIFEST.releases == ()
    assert lint_current_manifest().diagnostics == ()


def test_release_table_accepts_known_baselines_and_current_bytecode() -> None:
    manifest = MigrationManifest(
        text_baselines=(
            TextBaseline("pre-release"),
            TextBaseline("loom-source-format-2026-06-09"),
        ),
        current_text_baseline="loom-source-format-2026-06-09",
        current_bytecode_version=14,
        releases=(
            LoomRelease(
                version="2026.06.0",
                date="2026-06-30",
                text_baseline="loom-source-format-2026-06-09",
                bytecode_version=14,
            ),
        ),
    )

    assert manifest.lint().diagnostics == ()


def test_unknown_current_baseline_is_an_error() -> None:
    manifest = MigrationManifest(
        text_baselines=(TextBaseline("pre-release"),),
        current_text_baseline="loom-source-format-missing",
        current_bytecode_version=14,
    )

    report = manifest.lint()

    assert [diagnostic.severity for diagnostic in report.diagnostics] == [
        DIAGNOSTIC_ERROR
    ]
    assert report.diagnostics[0].baseline == "loom-source-format-missing"


def test_duplicate_baseline_is_an_error() -> None:
    manifest = MigrationManifest(
        text_baselines=(TextBaseline("same"), TextBaseline("same")),
        current_text_baseline="same",
        current_bytecode_version=14,
    )

    report = manifest.lint()

    assert [(d.severity, d.baseline) for d in report.diagnostics] == [
        (DIAGNOSTIC_ERROR, "same")
    ]


def test_release_with_unknown_baseline_is_an_error() -> None:
    manifest = MigrationManifest(
        text_baselines=(TextBaseline("pre-release"),),
        current_text_baseline="pre-release",
        current_bytecode_version=14,
        releases=(
            LoomRelease(
                version="2026.06.0",
                date="2026-06-30",
                text_baseline="loom-source-format-missing",
                bytecode_version=14,
            ),
        ),
    )

    report = manifest.lint()

    assert [(d.severity, d.release, d.baseline) for d in report.diagnostics] == [
        (DIAGNOSTIC_ERROR, "2026.06.0", "loom-source-format-missing")
    ]


def test_rule_window_order_is_validated() -> None:
    manifest = _manifest_with_rule(
        MigrationRuleMetadata(
            rule_id="demo-rule",
            introduced="loom-source-format-2026-07-01",
            replaced_by="loom-source-format-2026-06-09",
            expires_after="pre-release",
            test_kinds=REQUIRED_ACTIVE_RULE_TEST_KINDS,
        )
    )

    report = manifest.lint()

    assert [(d.severity, d.rule_id) for d in report.diagnostics] == [
        (DIAGNOSTIC_ERROR, "demo-rule"),
        (DIAGNOSTIC_ERROR, "demo-rule"),
        (DIAGNOSTIC_WARNING, "demo-rule"),
    ]


def test_active_rule_requires_all_test_kinds() -> None:
    manifest = _manifest_with_rule(
        MigrationRuleMetadata(
            rule_id="demo-rule",
            replaced_by="loom-source-format-2026-06-09",
            expires_after="loom-source-format-2026-07-01",
        )
    )

    report = manifest.lint()

    assert len(report.diagnostics) == 1
    assert report.diagnostics[0].severity == DIAGNOSTIC_ERROR
    assert report.diagnostics[0].rule_id == "demo-rule"
    assert "legacy_rewrite" in report.diagnostics[0].message


def test_active_rule_with_required_tests_is_clean() -> None:
    manifest = _manifest_with_rule(
        MigrationRuleMetadata(
            rule_id="demo-rule",
            replaced_by="loom-source-format-2026-06-09",
            test_kinds=REQUIRED_ACTIVE_RULE_TEST_KINDS,
        )
    )

    assert manifest.lint().diagnostics == ()


def test_expired_rule_reports_compatibility_window_warning() -> None:
    manifest = MigrationManifest(
        text_baselines=(
            TextBaseline("pre-release"),
            TextBaseline("loom-source-format-2026-06-09"),
            TextBaseline("loom-source-format-2026-07-01"),
        ),
        current_text_baseline="loom-source-format-2026-07-01",
        current_bytecode_version=14,
        rules=(
            MigrationRuleMetadata(
                rule_id="demo-rule",
                replaced_by="loom-source-format-2026-06-09",
                expires_after="loom-source-format-2026-06-09",
            ),
        ),
    )

    report = manifest.lint()

    assert [(d.severity, d.rule_id, d.baseline) for d in report.diagnostics] == [
        (DIAGNOSTIC_WARNING, "demo-rule", "loom-source-format-2026-06-09")
    ]


def test_expired_rule_can_be_promoted_to_error() -> None:
    manifest = MigrationManifest(
        text_baselines=(
            TextBaseline("pre-release"),
            TextBaseline("loom-source-format-2026-06-09"),
            TextBaseline("loom-source-format-2026-07-01"),
        ),
        current_text_baseline="loom-source-format-2026-07-01",
        current_bytecode_version=14,
        rules=(
            MigrationRuleMetadata(
                rule_id="demo-rule",
                replaced_by="loom-source-format-2026-06-09",
                expires_after="loom-source-format-2026-06-09",
            ),
        ),
    )

    report = manifest.lint(expired_rules_are_errors=True)

    assert [(d.severity, d.rule_id, d.baseline) for d in report.diagnostics] == [
        (DIAGNOSTIC_ERROR, "demo-rule", "loom-source-format-2026-06-09")
    ]


def test_legacy_format_metadata_can_be_extracted_from_ops() -> None:
    op = Op(
        "test.op",
        operands=[Operand("input", ANY)],
        legacy_formats=[
            LegacyFormat(
                "test.op.old",
                format=[],
                introduced="pre-release",
                replaced_by="loom-source-format-2026-06-09",
            )
        ],
    )

    assert migration_rule_metadata_from_ops([op]) == (
        MigrationRuleMetadata(
            rule_id="test.op.old",
            introduced="pre-release",
            replaced_by="loom-source-format-2026-06-09",
        ),
    )


def test_legacy_format_lint_accepts_known_baseline_window() -> None:
    manifest = MigrationManifest(
        text_baselines=(
            TextBaseline("pre-release"),
            TextBaseline("loom-source-format-2026-06-09"),
            TextBaseline("loom-source-format-2026-07-01"),
        ),
        current_text_baseline="loom-source-format-2026-06-09",
        current_bytecode_version=14,
    )
    op = Op(
        "test.op",
        operands=[Operand("input", ANY)],
        legacy_formats=[
            LegacyFormat(
                "test.op.old",
                format=[],
                replaced_by="loom-source-format-2026-06-09",
            )
        ],
    )

    assert lint_legacy_formats(manifest, [op]).diagnostics == ()


def test_legacy_format_lint_reports_unknown_baseline() -> None:
    manifest = MigrationManifest(
        text_baselines=(TextBaseline("pre-release"),),
        current_text_baseline="pre-release",
        current_bytecode_version=14,
    )
    op = Op(
        "test.op",
        legacy_formats=[
            LegacyFormat(
                "test.op.old",
                format=[],
                replaced_by="loom-source-format-missing",
                expires_after="loom-source-format-missing",
            )
        ],
    )

    report = lint_legacy_formats(manifest, [op])

    assert [(d.severity, d.rule_id, d.baseline) for d in report.diagnostics] == [
        (DIAGNOSTIC_ERROR, "test.op.old", "loom-source-format-missing"),
        (DIAGNOSTIC_ERROR, "test.op.old", "loom-source-format-missing"),
    ]


def test_legacy_format_lint_reports_duplicate_rule_ids_across_ops() -> None:
    manifest = MigrationManifest(
        text_baselines=(
            TextBaseline("pre-release"),
            TextBaseline("loom-source-format-2026-06-09"),
        ),
        current_text_baseline="loom-source-format-2026-06-09",
        current_bytecode_version=14,
    )
    first_op = Op(
        "test.first",
        legacy_formats=[
            LegacyFormat(
                "test.duplicate",
                format=[],
                replaced_by="loom-source-format-2026-06-09",
                expires_after="loom-source-format-2026-06-09",
            )
        ],
    )
    second_op = Op(
        "test.second",
        legacy_formats=[
            LegacyFormat(
                "test.duplicate",
                format=[],
                replaced_by="loom-source-format-2026-06-09",
                expires_after="loom-source-format-2026-06-09",
            )
        ],
    )

    report = lint_legacy_formats(manifest, [first_op, second_op])

    assert [(d.severity, d.rule_id) for d in report.diagnostics] == [
        (DIAGNOSTIC_ERROR, "test.duplicate")
    ]
