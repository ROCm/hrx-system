# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict version-zero Loom compile report documents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

COMPILE_REPORT_KIND = "loom.compile_report"
COMPILE_REPORT_SCHEMA_VERSION = 0

_REPORT_IDENTITY_FIELDS = (
    "mode",
    "artifact_kind",
    "artifact_format",
    "backend",
    "module",
    "function",
    "target_family",
    "target_key",
    "target_bundle",
    "target_snapshot",
    "target_export",
    "target_export_symbol",
    "target_config",
    "workload",
)

_ENTRY_IDENTITY_FIELDS = (
    "function",
    "source_function",
    "target_export",
    "target_export_symbol",
)

_ENTRY_CONTEXT_FIELDS = (
    "target_bundle",
    "target_snapshot",
    "target_config",
    "workload",
)

_ENVELOPE_CONTEXT_FIELDS = (
    "run_id",
    "candidate_id",
    "candidate_index",
    "case",
    "benchmark",
    "sample_id",
    "sample_index",
    "sample_compilation",
)


class CompileReportError(ValueError):
    """Raised when an input is not a current compile report document."""


class IncomparableCompileReportsError(ValueError):
    """Raised when two reports do not describe the same compilation."""


@dataclass(frozen=True)
class CompileReportEntryIdentity:
    """Stable source/export identity for one compiled entry."""

    function: str | None
    source_function: str | None
    target_export: str | None
    target_export_symbol: str | None

    def display_name(self) -> str:
        """Returns the most useful available entry name."""
        return (
            self.source_function
            or self.target_export
            or self.function
            or self.target_export_symbol
            or "<unnamed>"
        )

    def to_json_object(self) -> dict[str, object]:
        """Returns a compact deterministic identity for diagnostic views."""
        name = self.display_name()
        aliases = {
            key: value
            for key, value in (
                ("function", self.function),
                ("source_function", self.source_function),
                ("target_export", self.target_export),
                ("target_export_symbol", self.target_export_symbol),
            )
            if value is not None and value != name
        }
        identity: dict[str, object] = {"name": name}
        if aliases:
            identity["aliases"] = aliases
        return identity


@dataclass(frozen=True)
class CompileReportDocument:
    """Validated compile report plus explicit producer context."""

    source: str
    container_kind: str
    report: dict[str, object]
    entries: tuple[dict[str, object], ...]
    config_bindings: tuple[tuple[str, str], ...]
    envelope_context: tuple[tuple[str, str | int | bool], ...]

    @property
    def mode(self) -> str:
        return cast(str, self.report["mode"])

    @property
    def status_code(self) -> int:
        status = cast(dict[str, object], self.report["status"])
        return cast(int, status["code"])

    @property
    def status_name(self) -> str:
        status = cast(dict[str, object], self.report["status"])
        return cast(str, status["name"])


@dataclass(frozen=True)
class CompileReportEntryPair:
    """Matched entries from two comparable reports."""

    identity: CompileReportEntryIdentity
    baseline: dict[str, object]
    candidate: dict[str, object]


def load_compile_report(path: Path) -> CompileReportDocument:
    """Loads and validates a direct report or explicit benchmark envelope."""
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except OSError as exc:
        raise CompileReportError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CompileReportError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc
    return parse_compile_report(value, source=str(path))


def parse_compile_report(
    value: object, *, source: str = "<memory>"
) -> CompileReportDocument:
    """Validates a decoded JSON compile report document."""
    root = _require_object(value, source)
    container_kind = "direct"
    envelope_context: tuple[tuple[str, str | int | bool], ...] = ()
    if root.get("kind") == COMPILE_REPORT_KIND:
        report = root
    elif root.get("type") == "compile_report":
        container_kind = "benchmark_envelope"
        report = _require_object(root.get("compile_report"), f"{source}.compile_report")
        envelope_context = tuple(
            (field, cast(str | int | bool, root[field]))
            for field in _ENVELOPE_CONTEXT_FIELDS
            if field in root and isinstance(root[field], (str, int, bool))
        )
    else:
        raise CompileReportError(
            f"{source}: expected kind={COMPILE_REPORT_KIND!r} or an explicit "
            "type='compile_report' benchmark envelope"
        )

    kind = _require_string(report.get("kind"), f"{source}.kind")
    if kind != COMPILE_REPORT_KIND:
        raise CompileReportError(
            f"{source}.kind: expected {COMPILE_REPORT_KIND!r}, got {kind!r}"
        )
    schema_version = _require_integer(
        report.get("schema_version"), f"{source}.schema_version"
    )
    if schema_version != COMPILE_REPORT_SCHEMA_VERSION:
        raise CompileReportError(
            f"{source}.schema_version: expected "
            f"{COMPILE_REPORT_SCHEMA_VERSION}, got {schema_version}; regenerate "
            "the report with this compiler checkout"
        )
    mode = _require_string(report.get("mode"), f"{source}.mode")
    if mode not in ("summary", "details"):
        raise CompileReportError(
            f"{source}.mode: expected 'summary' or 'details', got {mode!r}"
        )

    status = _require_object(report.get("status"), f"{source}.status")
    _require_integer(status.get("code"), f"{source}.status.code")
    _require_string(status.get("name"), f"{source}.status.name")

    entries_object = _require_object(report.get("entries"), f"{source}.entries")
    entries = _validate_indexed_rows(entries_object, f"{source}.entries")
    entry_identities: set[CompileReportEntryIdentity] = set()
    for index, entry in enumerate(entries):
        identity = compile_report_entry_identity(
            entry, f"{source}.entries.rows[{index}]"
        )
        if identity in entry_identities:
            raise CompileReportError(
                f"{source}.entries.rows[{index}]: duplicate entry identity "
                f"{identity.display_name()!r}"
            )
        entry_identities.add(identity)

    config_bindings: tuple[tuple[str, str], ...] = ()
    if "config_bindings" in report:
        bindings_object = _require_object(
            report["config_bindings"], f"{source}.config_bindings"
        )
        binding_rows = _validate_indexed_rows(
            bindings_object, f"{source}.config_bindings"
        )
        binding_map: dict[str, str] = {}
        for index, row in enumerate(binding_rows):
            key = _require_string(
                row.get("key"), f"{source}.config_bindings.rows[{index}].key"
            )
            binding_value = _require_string(
                row.get("value"), f"{source}.config_bindings.rows[{index}].value"
            )
            if key in binding_map:
                raise CompileReportError(
                    f"{source}.config_bindings.rows[{index}]: duplicate key {key!r}"
                )
            binding_map[key] = binding_value
        config_bindings = tuple(sorted(binding_map.items()))

    return CompileReportDocument(
        source=source,
        container_kind=container_kind,
        report=report,
        entries=entries,
        config_bindings=config_bindings,
        envelope_context=envelope_context,
    )


def compile_report_entry_identity(
    entry: dict[str, object], source: str = "<entry>"
) -> CompileReportEntryIdentity:
    """Extracts one exact source/export identity."""
    values = tuple(
        _optional_string(entry.get(field), f"{source}.{field}")
        for field in _ENTRY_IDENTITY_FIELDS
    )
    identity = CompileReportEntryIdentity(*values)
    if all(value is None for value in values):
        raise CompileReportError(f"{source}: entry has no source or export identity")
    return identity


def match_compile_report_entries(
    baseline: CompileReportDocument,
    candidate: CompileReportDocument,
) -> tuple[CompileReportEntryPair, ...]:
    """Checks strict comparability and returns exact entry pairs."""
    differences: list[str] = []
    if baseline.status_code != 0:
        differences.append(
            f"baseline status is {baseline.status_name} ({baseline.status_code})"
        )
    if candidate.status_code != 0:
        differences.append(
            f"candidate status is {candidate.status_name} ({candidate.status_code})"
        )

    for field in _REPORT_IDENTITY_FIELDS:
        baseline_value = _canonical_value(baseline.report.get(field))
        candidate_value = _canonical_value(candidate.report.get(field))
        if baseline_value != candidate_value:
            differences.append(
                f"{field}: {_display_identity_value(baseline.report.get(field))} "
                f"!= {_display_identity_value(candidate.report.get(field))}"
            )
    if baseline.config_bindings != candidate.config_bindings:
        differences.append(
            "config_bindings: "
            f"{dict(baseline.config_bindings)!r} != "
            f"{dict(candidate.config_bindings)!r}"
        )

    baseline_entries = _entries_by_identity(baseline)
    candidate_entries = _entries_by_identity(candidate)
    baseline_identities = set(baseline_entries)
    candidate_identities = set(candidate_entries)
    differences.extend(
        f"candidate is missing entry {identity.display_name()!r}"
        for identity in sorted(
            baseline_identities - candidate_identities,
            key=_entry_identity_sort_key,
        )
    )
    differences.extend(
        f"baseline is missing entry {identity.display_name()!r}"
        for identity in sorted(
            candidate_identities - baseline_identities,
            key=_entry_identity_sort_key,
        )
    )

    pairs: list[CompileReportEntryPair] = []
    for identity in sorted(
        baseline_identities & candidate_identities, key=_entry_identity_sort_key
    ):
        baseline_entry = baseline_entries[identity]
        candidate_entry = candidate_entries[identity]
        for field in _ENTRY_CONTEXT_FIELDS:
            baseline_value = _canonical_value(baseline_entry.get(field))
            candidate_value = _canonical_value(candidate_entry.get(field))
            if baseline_value != candidate_value:
                differences.append(
                    f"entry {identity.display_name()!r} {field}: "
                    f"{_display_identity_value(baseline_entry.get(field))} != "
                    f"{_display_identity_value(candidate_entry.get(field))}"
                )
        pairs.append(
            CompileReportEntryPair(
                identity=identity,
                baseline=baseline_entry,
                candidate=candidate_entry,
            )
        )

    if differences:
        formatted = "\n  ".join(differences)
        raise IncomparableCompileReportsError(
            "compile reports are not comparable:\n  " + formatted
        )
    return tuple(pairs)


def report_identity_json(document: CompileReportDocument) -> dict[str, object]:
    """Returns the exact report-level identity used by comparisons."""
    identity = {
        field: document.report[field]
        for field in _REPORT_IDENTITY_FIELDS
        if document.report.get(field) is not None
    }
    identity["schema_version"] = COMPILE_REPORT_SCHEMA_VERSION
    if document.config_bindings:
        identity["config_bindings"] = [
            {"key": key, "value": value} for key, value in document.config_bindings
        ]
    return identity


def _validate_indexed_rows(
    value: dict[str, object], source: str
) -> tuple[dict[str, object], ...]:
    count = _require_integer(value.get("count"), f"{source}.count")
    rows_value = value.get("rows")
    if not isinstance(rows_value, list):
        raise CompileReportError(f"{source}.rows: expected an array")
    if count != len(rows_value):
        raise CompileReportError(
            f"{source}: count is {count}, but rows has {len(rows_value)} entries"
        )
    rows: list[dict[str, object]] = []
    for index, row_value in enumerate(rows_value):
        row = _require_object(row_value, f"{source}.rows[{index}]")
        row_index = _require_integer(row.get("index"), f"{source}.rows[{index}].index")
        if row_index != index:
            raise CompileReportError(
                f"{source}.rows[{index}].index: expected {index}, got {row_index}"
            )
        rows.append(row)
    return tuple(rows)


def _entries_by_identity(
    document: CompileReportDocument,
) -> dict[CompileReportEntryIdentity, dict[str, object]]:
    return {compile_report_entry_identity(entry): entry for entry in document.entries}


def _entry_identity_sort_key(
    identity: CompileReportEntryIdentity,
) -> tuple[str, str, str, str]:
    return tuple(
        value or ""
        for value in (
            identity.source_function,
            identity.target_export,
            identity.function,
            identity.target_export_symbol,
        )
    )


def _canonical_value(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _display_identity_value(value: object) -> str:
    return _canonical_value(value)


def _require_object(value: object, source: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CompileReportError(f"{source}: expected an object")
    return cast(dict[str, object], value)


def _require_string(value: object, source: str) -> str:
    if not isinstance(value, str):
        raise CompileReportError(f"{source}: expected a string")
    return value


def _optional_string(value: object, source: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, source)


def _require_integer(value: object, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompileReportError(f"{source}: expected an integer")
    return value
