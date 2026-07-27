# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Release and compatibility-window metadata for Loom migrations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from loom.format.bytecode.writer import FORMAT_VERSION

DIAGNOSTIC_ERROR = "error"
DIAGNOSTIC_WARNING = "warning"

RULE_TEST_LEGACY_REWRITE = "legacy_rewrite"
RULE_TEST_CURRENT_NOOP = "current_noop"
RULE_TEST_IDEMPOTENCE = "idempotence"
RULE_TEST_CURRENT_PARSE = "current_parse"

REQUIRED_ACTIVE_RULE_TEST_KINDS = frozenset(
    (
        RULE_TEST_LEGACY_REWRITE,
        RULE_TEST_CURRENT_NOOP,
        RULE_TEST_IDEMPOTENCE,
        RULE_TEST_CURRENT_PARSE,
    )
)

CURRENT_TEXT_BASELINE = "loom-source-format-2026-06-09"
CURRENT_BYTECODE_VERSION = FORMAT_VERSION


@dataclass(frozen=True, slots=True)
class TextBaseline:
    """Ordered text-format compatibility coordinate."""

    id: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class LoomRelease:
    """User-facing release mapped to exact migration coordinates."""

    version: str
    date: str
    text_baseline: str
    bytecode_version: int


@dataclass(frozen=True, slots=True)
class MigrationRuleMetadata:
    """Compatibility window and test coverage for one legacy migration rule."""

    rule_id: str
    replaced_by: str
    expires_after: str = ""
    introduced: str = "pre-release"
    test_kinds: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "test_kinds", frozenset(self.test_kinds))


@dataclass(frozen=True, slots=True)
class MigrationDiagnostic:
    """Validation diagnostic for migration metadata."""

    severity: str
    message: str
    rule_id: str | None = None
    release: str | None = None
    baseline: str | None = None


@dataclass(frozen=True, slots=True)
class MigrationLintReport:
    """Structured report from migration manifest validation."""

    diagnostics: tuple[MigrationDiagnostic, ...]

    @property
    def errors(self) -> tuple[MigrationDiagnostic, ...]:
        return tuple(
            diagnostic
            for diagnostic in self.diagnostics
            if diagnostic.severity == DIAGNOSTIC_ERROR
        )

    @property
    def warnings(self) -> tuple[MigrationDiagnostic, ...]:
        return tuple(
            diagnostic
            for diagnostic in self.diagnostics
            if diagnostic.severity == DIAGNOSTIC_WARNING
        )

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class MigrationManifest:
    """Complete migration baseline, release, and rule metadata."""

    text_baselines: tuple[TextBaseline, ...]
    current_text_baseline: str
    current_bytecode_version: int
    releases: tuple[LoomRelease, ...] = ()
    rules: tuple[MigrationRuleMetadata, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "text_baselines", tuple(self.text_baselines))
        object.__setattr__(self, "releases", tuple(self.releases))
        object.__setattr__(self, "rules", tuple(self.rules))

    def lint(self, *, expired_rules_are_errors: bool = False) -> MigrationLintReport:
        return _lint_manifest(
            self,
            expired_rules_are_errors=expired_rules_are_errors,
            require_active_rule_tests=True,
        )


def lint_legacy_formats(
    manifest: MigrationManifest,
    ops: Iterable[Any],
    *,
    expired_rules_are_errors: bool = False,
) -> MigrationLintReport:
    """Validates op-authored legacy formats against manifest baselines."""
    rule_metadata = migration_rule_metadata_from_ops(ops)
    combined_manifest = MigrationManifest(
        text_baselines=manifest.text_baselines,
        current_text_baseline=manifest.current_text_baseline,
        current_bytecode_version=manifest.current_bytecode_version,
        releases=manifest.releases,
        rules=manifest.rules + rule_metadata,
    )
    return _lint_manifest(
        combined_manifest,
        expired_rules_are_errors=expired_rules_are_errors,
        require_active_rule_tests=False,
    )


def migration_rule_metadata_from_ops(
    ops: Iterable[Any],
) -> tuple[MigrationRuleMetadata, ...]:
    """Extracts migration rule metadata from op legacy-format declarations."""
    rules: list[MigrationRuleMetadata] = []
    for op in ops:
        rules.extend(
            (
                MigrationRuleMetadata(
                    rule_id=legacy_format.rule_id,
                    introduced=legacy_format.introduced,
                    replaced_by=legacy_format.replaced_by,
                    expires_after=legacy_format.expires_after,
                )
            )
            for legacy_format in getattr(op, "legacy_formats", ())
        )
    return tuple(rules)


def lint_current_manifest() -> MigrationLintReport:
    """Validates the checked-in migration manifest."""
    return CURRENT_MANIFEST.lint()


def _lint_manifest(
    manifest: MigrationManifest,
    *,
    expired_rules_are_errors: bool,
    require_active_rule_tests: bool,
) -> MigrationLintReport:
    diagnostics: list[MigrationDiagnostic] = []
    baseline_positions = _collect_baselines(manifest.text_baselines, diagnostics)

    if manifest.current_text_baseline not in baseline_positions:
        diagnostics.append(
            MigrationDiagnostic(
                severity=DIAGNOSTIC_ERROR,
                message=(
                    "current text baseline is not listed in the migration "
                    "baseline table"
                ),
                baseline=manifest.current_text_baseline,
            )
        )
    if manifest.current_bytecode_version < 0:
        diagnostics.append(
            MigrationDiagnostic(
                severity=DIAGNOSTIC_ERROR,
                message="current bytecode version must be non-negative",
            )
        )

    _lint_releases(manifest, baseline_positions, diagnostics)
    _lint_rules(
        manifest,
        baseline_positions,
        diagnostics,
        expired_rules_are_errors=expired_rules_are_errors,
        require_active_rule_tests=require_active_rule_tests,
    )
    return MigrationLintReport(tuple(diagnostics))


def _collect_baselines(
    baselines: tuple[TextBaseline, ...],
    diagnostics: list[MigrationDiagnostic],
) -> dict[str, int]:
    positions: dict[str, int] = {}
    for ordinal, baseline in enumerate(baselines):
        if not baseline.id:
            diagnostics.append(
                MigrationDiagnostic(
                    severity=DIAGNOSTIC_ERROR,
                    message="text baseline id must not be empty",
                )
            )
            continue
        if baseline.id in positions:
            diagnostics.append(
                MigrationDiagnostic(
                    severity=DIAGNOSTIC_ERROR,
                    message="duplicate text baseline id",
                    baseline=baseline.id,
                )
            )
            continue
        positions[baseline.id] = ordinal
    return positions


def _lint_releases(
    manifest: MigrationManifest,
    baseline_positions: dict[str, int],
    diagnostics: list[MigrationDiagnostic],
) -> None:
    seen_versions: set[str] = set()
    for release in manifest.releases:
        if not release.version:
            diagnostics.append(
                MigrationDiagnostic(
                    severity=DIAGNOSTIC_ERROR,
                    message="release version must not be empty",
                )
            )
        elif release.version in seen_versions:
            diagnostics.append(
                MigrationDiagnostic(
                    severity=DIAGNOSTIC_ERROR,
                    message="duplicate release version",
                    release=release.version,
                )
            )
        seen_versions.add(release.version)

        if release.text_baseline not in baseline_positions:
            diagnostics.append(
                MigrationDiagnostic(
                    severity=DIAGNOSTIC_ERROR,
                    message="release references unknown text baseline",
                    release=release.version,
                    baseline=release.text_baseline,
                )
            )
        if release.bytecode_version > manifest.current_bytecode_version:
            diagnostics.append(
                MigrationDiagnostic(
                    severity=DIAGNOSTIC_ERROR,
                    message="release bytecode version exceeds current bytecode version",
                    release=release.version,
                )
            )
        if release.bytecode_version < 0:
            diagnostics.append(
                MigrationDiagnostic(
                    severity=DIAGNOSTIC_ERROR,
                    message="release bytecode version must be non-negative",
                    release=release.version,
                )
            )


def _lint_rules(
    manifest: MigrationManifest,
    baseline_positions: dict[str, int],
    diagnostics: list[MigrationDiagnostic],
    *,
    expired_rules_are_errors: bool,
    require_active_rule_tests: bool,
) -> None:
    seen_rule_ids: set[str] = set()
    current_position = baseline_positions.get(manifest.current_text_baseline)
    for rule in manifest.rules:
        if not rule.rule_id:
            diagnostics.append(
                MigrationDiagnostic(
                    severity=DIAGNOSTIC_ERROR,
                    message="migration rule id must not be empty",
                )
            )
        elif rule.rule_id in seen_rule_ids:
            diagnostics.append(
                MigrationDiagnostic(
                    severity=DIAGNOSTIC_ERROR,
                    message="duplicate migration rule id",
                    rule_id=rule.rule_id,
                )
            )
        seen_rule_ids.add(rule.rule_id)

        introduced_position = _rule_baseline_position(
            rule.rule_id, "introduced", rule.introduced, baseline_positions, diagnostics
        )
        replaced_position = _rule_baseline_position(
            rule.rule_id,
            "replaced_by",
            rule.replaced_by,
            baseline_positions,
            diagnostics,
        )
        expires_position = None
        if rule.expires_after:
            expires_position = _rule_baseline_position(
                rule.rule_id,
                "expires_after",
                rule.expires_after,
                baseline_positions,
                diagnostics,
            )

        if (
            introduced_position is not None
            and replaced_position is not None
            and introduced_position > replaced_position
        ):
            diagnostics.append(
                MigrationDiagnostic(
                    severity=DIAGNOSTIC_ERROR,
                    message="migration rule introduced baseline is after replaced_by",
                    rule_id=rule.rule_id,
                    baseline=rule.introduced,
                )
            )
        if (
            replaced_position is not None
            and expires_position is not None
            and replaced_position > expires_position
        ):
            diagnostics.append(
                MigrationDiagnostic(
                    severity=DIAGNOSTIC_ERROR,
                    message=(
                        "migration rule replaced_by baseline is after expires_after"
                    ),
                    rule_id=rule.rule_id,
                    baseline=rule.expires_after,
                )
            )

        rule_is_expired = (
            current_position is not None
            and expires_position is not None
            and expires_position < current_position
        )
        if rule_is_expired:
            diagnostics.append(
                MigrationDiagnostic(
                    severity=(
                        DIAGNOSTIC_ERROR
                        if expired_rules_are_errors
                        else DIAGNOSTIC_WARNING
                    ),
                    message="migration rule has passed its compatibility window",
                    rule_id=rule.rule_id,
                    baseline=rule.expires_after,
                )
            )
        elif require_active_rule_tests and current_position is not None:
            missing_tests = REQUIRED_ACTIVE_RULE_TEST_KINDS.difference(rule.test_kinds)
            if missing_tests:
                diagnostics.append(
                    MigrationDiagnostic(
                        severity=DIAGNOSTIC_ERROR,
                        message=(
                            "active migration rule is missing required test kinds: "
                            + ", ".join(sorted(missing_tests))
                        ),
                        rule_id=rule.rule_id,
                    )
                )


def _rule_baseline_position(
    rule_id: str,
    field: str,
    baseline: str,
    baseline_positions: dict[str, int],
    diagnostics: list[MigrationDiagnostic],
) -> int | None:
    if baseline in baseline_positions:
        return baseline_positions[baseline]
    diagnostics.append(
        MigrationDiagnostic(
            severity=DIAGNOSTIC_ERROR,
            message=f"migration rule {field} references unknown text baseline",
            rule_id=rule_id,
            baseline=baseline,
        )
    )
    return None


CURRENT_MANIFEST = MigrationManifest(
    text_baselines=(
        TextBaseline("pre-release", "Development before migration tracking."),
        TextBaseline(CURRENT_TEXT_BASELINE, "Initial Loom source-format baseline."),
    ),
    current_text_baseline=CURRENT_TEXT_BASELINE,
    current_bytecode_version=CURRENT_BYTECODE_VERSION,
)
