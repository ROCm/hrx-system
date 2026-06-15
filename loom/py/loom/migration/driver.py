# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Programmatic driver for Loom source and bytecode migration."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import Any

from loom.diagnostics import DiagnosticSeverity, SourceRange
from loom.format.bytecode.writer import FORMAT_VERSION, MAGIC
from loom.format.text.parser import Parser
from loom.format.text.tokenizer import ParseError
from loom.importers.check.cases import CheckCase, rewrite_inline_case_inputs
from loom.migration.rules import (
    MigrationRule,
    migration_rules_from_ops,
)
from loom.migration.source import (
    MigrationSourceDiagnostic,
    SourceDocument,
    SourceEdit,
)

LOOM_SOURCE_SUFFIX = ".loom"
LOOM_TEST_SUFFIX = ".loom-test"
LOOM_BYTECODE_SUFFIX = ".loombc"


class MigrationFileKind(StrEnum):
    """File kind understood by the migration driver."""

    LOOM_SOURCE = "loom"
    LOOM_TEST = "loom-test"
    LOOM_BYTECODE = "loombc"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class MigrationFileDiagnostic:
    """Structured diagnostic emitted while migrating one file."""

    severity: DiagnosticSeverity
    message: str
    file: Path | None = None
    rule_id: str = ""
    source_range: SourceRange | None = None
    fixup_hint: str = ""
    actual_version: int | None = None
    current_version: int | None = None

    @property
    def is_error(self) -> bool:
        return self.severity == DiagnosticSeverity.ERROR

    def to_json_object(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "severity": self.severity.value,
            "message": self.message,
        }
        if self.file is not None:
            result["file"] = str(self.file)
        if self.rule_id:
            result["rule_id"] = self.rule_id
        if self.fixup_hint:
            result["fixup_hint"] = self.fixup_hint
        if self.actual_version is not None:
            result["actual_version"] = self.actual_version
        if self.current_version is not None:
            result["current_version"] = self.current_version
        if self.source_range is not None:
            result["source_range"] = _source_range_to_json(self.source_range)
        return result


@dataclass(frozen=True, slots=True)
class SourceMigrationResult:
    """Result of migrating one Loom source document."""

    text: str
    changed: bool
    diagnostics: tuple[MigrationFileDiagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(diagnostic.is_error for diagnostic in self.diagnostics)


@dataclass(frozen=True, slots=True)
class FileMigrationResult:
    """Result of migrating or checking one filesystem path."""

    path: Path
    kind: MigrationFileKind
    text: str | None = None
    changed: bool = False
    diagnostics: tuple[MigrationFileDiagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(diagnostic.is_error for diagnostic in self.diagnostics)


@dataclass(frozen=True, slots=True)
class MigrationRunResult:
    """Aggregate result for a single CLI-style migration run."""

    files: tuple[FileMigrationResult, ...]

    @property
    def diagnostics(self) -> tuple[MigrationFileDiagnostic, ...]:
        return tuple(
            diagnostic
            for file_result in self.files
            for diagnostic in file_result.diagnostics
        )

    @property
    def ok(self) -> bool:
        return not any(diagnostic.is_error for diagnostic in self.diagnostics)

    @property
    def changed(self) -> bool:
        return any(file_result.changed for file_result in self.files)

    def to_json_object(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "changed": self.changed,
            "files": [
                {
                    "path": str(file_result.path),
                    "kind": file_result.kind.value,
                    "changed": file_result.changed,
                    "ok": file_result.ok,
                    "diagnostics": [
                        diagnostic.to_json_object()
                        for diagnostic in file_result.diagnostics
                    ],
                }
                for file_result in self.files
            ],
            "diagnostics": [
                diagnostic.to_json_object() for diagnostic in self.diagnostics
            ],
        }


def discover_migration_files(root: Path) -> tuple[Path, ...]:
    """Returns migration-supported files below ``root`` in deterministic order."""
    if root.is_file():
        return (root,) if classify_file(root) != MigrationFileKind.UNSUPPORTED else ()
    return tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and classify_file(path) != MigrationFileKind.UNSUPPORTED
        )
    )


def classify_file(path: Path) -> MigrationFileKind:
    """Returns the migration kind for ``path`` based on its suffix."""
    if path.suffix == LOOM_SOURCE_SUFFIX:
        return MigrationFileKind.LOOM_SOURCE
    if path.suffix == LOOM_TEST_SUFFIX:
        return MigrationFileKind.LOOM_TEST
    if path.suffix == LOOM_BYTECODE_SUFFIX:
        return MigrationFileKind.LOOM_BYTECODE
    return MigrationFileKind.UNSUPPORTED


def migrate_files(
    paths: Sequence[Path],
    *,
    rules: Sequence[MigrationRule] | None = None,
) -> MigrationRunResult:
    """Migrates or checks each path without writing results to disk."""
    rules = _resolve_rules(rules)
    return MigrationRunResult(tuple(migrate_file(path, rules=rules) for path in paths))


def migrate_file(
    path: Path,
    *,
    rules: Sequence[MigrationRule] | None = None,
) -> FileMigrationResult:
    """Migrates or checks one filesystem path without writing results to disk."""
    rules = _resolve_rules(rules)
    kind = classify_file(path)
    if kind == MigrationFileKind.LOOM_SOURCE:
        text = path.read_text(encoding="utf-8")
        result = migrate_source_text(text, filename=path, rules=rules)
        return FileMigrationResult(
            path=path,
            kind=kind,
            text=result.text,
            changed=result.changed,
            diagnostics=result.diagnostics,
        )
    if kind == MigrationFileKind.LOOM_TEST:
        text = path.read_text(encoding="utf-8")
        result = migrate_loom_test_text(text, filename=path, rules=rules)
        return FileMigrationResult(
            path=path,
            kind=kind,
            text=result.text,
            changed=result.changed,
            diagnostics=result.diagnostics,
        )
    if kind == MigrationFileKind.LOOM_BYTECODE:
        diagnostics = check_bytecode_version(path.read_bytes(), path)
        return FileMigrationResult(path=path, kind=kind, diagnostics=diagnostics)
    return FileMigrationResult(
        path=path,
        kind=kind,
        diagnostics=(
            MigrationFileDiagnostic(
                severity=DiagnosticSeverity.ERROR,
                message="unsupported file extension for Loom migration",
                file=path,
                rule_id="loom.migrate.unsupported_file",
            ),
        ),
    )


def migrate_source_text(
    text: str,
    *,
    filename: Path | None = None,
    rules: Sequence[MigrationRule] | None = None,
    validate: bool = True,
) -> SourceMigrationResult:
    """Applies source-preserving migration rules to one Loom source string."""
    rules = _resolve_rules(rules)
    document = SourceDocument(text, filename)
    diagnostics: list[MigrationFileDiagnostic] = []
    edits: list[SourceEdit] = []

    for rule in rules:
        application = rule.rewrite(document)
        diagnostics.extend(
            _source_diagnostic_to_migration(diagnostic, filename)
            for diagnostic in application.diagnostics
        )
        edits.extend(_edits_with_rule_id(rule, application.edits))

    migrated_text = text
    sorted_edits = tuple(
        sorted(edits, key=lambda edit: (edit.byte_start, edit.byte_end))
    )
    if sorted_edits and not _has_errors(diagnostics):
        try:
            migrated_text = document.apply_edits(sorted_edits)
        except ValueError as exc:
            diagnostics.append(
                _edit_application_diagnostic(document, sorted_edits, str(exc))
            )

    changed = migrated_text != text
    if validate and changed and not _has_errors(diagnostics):
        diagnostics.extend(validate_current_source(migrated_text, filename=filename))

    return SourceMigrationResult(
        text=migrated_text,
        changed=changed,
        diagnostics=tuple(diagnostics),
    )


def migrate_loom_test_text(
    text: str,
    *,
    filename: Path | None = None,
    rules: Sequence[MigrationRule] | None = None,
    validate: bool = True,
) -> SourceMigrationResult:
    """Applies Loom source migration rules to .loom-test input IR sections."""
    rules = _resolve_rules(rules)
    case_results: dict[int, SourceMigrationResult] = {}

    def migrate_input(
        check_case: CheckCase,
        input_text: str,
    ) -> str:
        result = migrate_source_text(
            input_text,
            filename=filename,
            rules=rules,
            validate=False,
        )
        case_results[check_case.index] = result
        return result.text

    try:
        rewrite_result = rewrite_inline_case_inputs(
            filename or Path("<input.loom-test>"),
            text,
            migrate_input,
        )
    except ValueError as exc:
        return SourceMigrationResult(
            text=text,
            changed=False,
            diagnostics=(
                MigrationFileDiagnostic(
                    severity=DiagnosticSeverity.ERROR,
                    message=str(exc),
                    file=filename,
                    rule_id="loom.migrate.loom_test_parse",
                    source_range=SourceRange(filename=filename),
                    fixup_hint="Check that the .loom-test file has at least one case.",
                ),
            ),
        )

    diagnostics = _remap_loom_test_case_diagnostics(
        rewrite_result.source,
        rewrite_result.cases,
        case_results,
        filename,
    )
    return SourceMigrationResult(
        text=rewrite_result.source,
        changed=rewrite_result.source != text,
        diagnostics=diagnostics,
    )


def validate_current_source(
    text: str,
    *,
    filename: Path | None = None,
) -> tuple[MigrationFileDiagnostic, ...]:
    """Validates source text with the current strict Loom parser."""
    parser = Parser()
    parser.register_ops(default_migration_ops())
    parser.register_types(default_migration_types())
    try:
        parser.parse(text, str(filename) if filename is not None else "<input>")
    except ParseError as exc:
        document = SourceDocument(text, filename)
        return (
            MigrationFileDiagnostic(
                severity=DiagnosticSeverity.ERROR,
                message=_parse_error_summary(exc),
                file=filename,
                rule_id="loom.migrate.current_parse",
                source_range=document.source_range(exc.location, exc.location),
                fixup_hint="Migration output must parse as current Loom source.",
            ),
        )
    return ()


def check_bytecode_version(
    data: bytes,
    path: Path | None = None,
) -> tuple[MigrationFileDiagnostic, ...]:
    """Checks a .loombc header version without attempting compatibility reads."""
    if len(data) < len(MAGIC) + 1:
        return (
            MigrationFileDiagnostic(
                severity=DiagnosticSeverity.ERROR,
                message="Loom bytecode file is too small to contain a header",
                file=path,
                rule_id="loom.migrate.bytecode_header",
                source_range=SourceRange(filename=path),
                current_version=FORMAT_VERSION,
                fixup_hint="Regenerate this .loombc file with the current toolchain.",
            ),
        )
    actual_magic = data[: len(MAGIC)]
    if actual_magic != MAGIC:
        return (
            MigrationFileDiagnostic(
                severity=DiagnosticSeverity.ERROR,
                message=f"invalid Loom bytecode magic: expected {MAGIC!r}",
                file=path,
                rule_id="loom.migrate.bytecode_magic",
                source_range=SourceRange(filename=path),
                current_version=FORMAT_VERSION,
                fixup_hint="Regenerate this .loombc file with the current toolchain.",
            ),
        )
    actual_version = data[len(MAGIC)]
    if actual_version != FORMAT_VERSION:
        return (
            MigrationFileDiagnostic(
                severity=DiagnosticSeverity.ERROR,
                message=(
                    "unsupported Loom bytecode version; compatibility migration "
                    "is only available for text sources"
                ),
                file=path,
                rule_id="loom.migrate.bytecode_version",
                source_range=SourceRange(filename=path),
                actual_version=actual_version,
                current_version=FORMAT_VERSION,
                fixup_hint="Regenerate this .loombc file from its source inputs.",
            ),
        )
    return ()


def default_migration_ops() -> tuple[Any, ...]:
    """Returns the broad current op registry used by migration validation."""
    from loom.dialect.buffer import ALL_BUFFER_OPS
    from loom.dialect.cfg import ALL_CFG_OPS
    from loom.dialect.check import ALL_CHECK_OPS
    from loom.dialect.config import ALL_CONFIG_OPS
    from loom.dialect.encoding import ALL_ENCODING_OPS
    from loom.dialect.func import ALL_FUNC_OPS
    from loom.dialect.globals import ALL_GLOBAL_OPS
    from loom.dialect.index import ALL_INDEX_OPS
    from loom.dialect.kernel import ALL_KERNEL_OPS
    from loom.dialect.llvmir import ALL_LLVMIR_OPS
    from loom.dialect.low import ALL_LOW_OPS
    from loom.dialect.pass_ import ALL_PASS_OPS
    from loom.dialect.pool import ALL_POOL_OPS
    from loom.dialect.scalar import ALL_SCALAR_OPS
    from loom.dialect.scf import ALL_SCF_OPS
    from loom.dialect.target import ALL_TARGET_OPS
    from loom.dialect.test import ALL_TEST_OPS
    from loom.dialect.vector import ALL_VECTOR_OPS
    from loom.dialect.view import ALL_VIEW_OPS
    from loom.target.arch.amdgpu.dialect import ALL_AMDGPU_OPS
    from loom.target.arch.ireevm.dialect import ALL_IREEVM_OPS
    from loom.target.arch.spirv.dialect import ALL_SPIRV_OPS
    from loom.target.arch.wasm.dialect import ALL_WASM_OPS
    from loom.target.arch.x86.dialect import ALL_X86_OPS

    return (
        *ALL_TEST_OPS,
        *ALL_SCALAR_OPS,
        *ALL_FUNC_OPS,
        *ALL_ENCODING_OPS,
        *ALL_POOL_OPS,
        *ALL_GLOBAL_OPS,
        *ALL_SCF_OPS,
        *ALL_CFG_OPS,
        *ALL_CHECK_OPS,
        *ALL_BUFFER_OPS,
        *ALL_VIEW_OPS,
        *ALL_VECTOR_OPS,
        *ALL_INDEX_OPS,
        *ALL_KERNEL_OPS,
        *ALL_LLVMIR_OPS,
        *ALL_TARGET_OPS,
        *ALL_LOW_OPS,
        *ALL_PASS_OPS,
        *ALL_CONFIG_OPS,
        *ALL_AMDGPU_OPS,
        *ALL_X86_OPS,
        *ALL_SPIRV_OPS,
        *ALL_WASM_OPS,
        *ALL_IREEVM_OPS,
    )


def default_migration_types() -> tuple[Any, ...]:
    """Returns the broad current type registry used by migration validation."""
    from loom.builtin_types import ALL_BUILTIN_TYPES
    from loom.dialect.hal import ALL_HAL_TYPES
    from loom.dialect.kernel import ALL_KERNEL_TYPES
    from loom.target.arch.ireevm.dialect import ALL_IREEVM_TYPES

    return (
        *ALL_BUILTIN_TYPES,
        *ALL_HAL_TYPES,
        *ALL_KERNEL_TYPES,
        *ALL_IREEVM_TYPES,
    )


@cache
def default_migration_rules() -> tuple[MigrationRule, ...]:
    """Returns migration rules generated from current op legacy formats."""
    return migration_rules_from_ops(default_migration_ops())


def _resolve_rules(
    rules: Sequence[MigrationRule] | None,
) -> tuple[MigrationRule, ...]:
    return default_migration_rules() if rules is None else tuple(rules)


def _source_range_to_json(source_range: SourceRange) -> dict[str, Any]:
    result: dict[str, Any] = {
        "provenance": source_range.provenance.value,
        "start_byte": source_range.start,
        "end_byte": source_range.end,
        "start_line": source_range.start_line,
        "start_column": source_range.start_column,
        "end_line": source_range.end_line,
        "end_column": source_range.end_column,
    }
    if source_range.filename is not None:
        result["file"] = str(source_range.filename)
    return result


def _source_diagnostic_to_migration(
    diagnostic: MigrationSourceDiagnostic,
    filename: Path | None,
) -> MigrationFileDiagnostic:
    return MigrationFileDiagnostic(
        severity=diagnostic.severity,
        message=diagnostic.message,
        file=filename or diagnostic.source_range.filename,
        rule_id=diagnostic.rule_id,
        source_range=diagnostic.source_range,
        fixup_hint=diagnostic.fixup_hint,
    )


def _parse_error_summary(exc: ParseError) -> str:
    message = str(exc)
    _prefix, separator, summary = message.partition(": error: ")
    return summary if separator else message


def _remap_loom_test_case_diagnostics(
    rewritten_text: str,
    cases: Sequence[CheckCase],
    case_results: dict[int, SourceMigrationResult],
    filename: Path | None,
) -> tuple[MigrationFileDiagnostic, ...]:
    document = SourceDocument(rewritten_text, filename)
    diagnostics: list[MigrationFileDiagnostic] = []
    character_delta = 0
    for check_case in cases:
        if check_case.input_span is None:
            continue
        result = case_results.get(check_case.index)
        if result is None:
            continue
        input_start = check_case.input_span.start + character_delta
        original_length = check_case.input_span.end - check_case.input_span.start
        character_delta += len(result.text) - original_length
        diagnostics.extend(
            _remap_loom_test_diagnostic(
                document,
                input_start,
                diagnostic,
                filename,
            )
            for diagnostic in result.diagnostics
        )
    return tuple(diagnostics)


def _remap_loom_test_diagnostic(
    document: SourceDocument,
    input_start: int,
    diagnostic: MigrationFileDiagnostic,
    filename: Path | None,
) -> MigrationFileDiagnostic:
    source_range = diagnostic.source_range
    if source_range is None:
        return replace(diagnostic, file=filename)
    input_byte_start = len(document.text[:input_start].encode("utf-8"))
    remapped_range = document.byte_source_range(
        input_byte_start + source_range.start,
        input_byte_start + source_range.end,
    )
    return replace(diagnostic, file=filename, source_range=remapped_range)


def _edits_with_rule_id(
    rule: MigrationRule,
    edits: Iterable[SourceEdit],
) -> tuple[SourceEdit, ...]:
    return tuple(
        edit if edit.rule_id is not None else replace(edit, rule_id=rule.rule_id)
        for edit in edits
    )


def _edit_application_diagnostic(
    document: SourceDocument,
    edits: tuple[SourceEdit, ...],
    message: str,
) -> MigrationFileDiagnostic:
    source_range = SourceRange(filename=document.filename)
    rule_id = "loom.migrate.source_edits"
    for previous_edit, edit in pairwise(edits):
        if edit.byte_start < previous_edit.byte_end:
            source_range = document.byte_source_range(
                edit.byte_start,
                max(edit.byte_end, previous_edit.byte_end),
            )
            rule_id = edit.rule_id or previous_edit.rule_id or rule_id
            break
    return MigrationFileDiagnostic(
        severity=DiagnosticSeverity.ERROR,
        message=message,
        file=document.filename,
        rule_id=rule_id,
        source_range=source_range,
        fixup_hint="Compose overlapping rewrites into one source edit.",
    )


def _has_errors(diagnostics: Iterable[MigrationFileDiagnostic]) -> bool:
    return any(diagnostic.is_error for diagnostic in diagnostics)
