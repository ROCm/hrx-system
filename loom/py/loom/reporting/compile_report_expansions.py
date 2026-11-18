# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Per-operation expansion alarms from retained lowering counts."""

from __future__ import annotations

from dataclasses import dataclass

from loom.reporting.compile_report import CompileReportDocument, CompileReportError
from loom.reporting.compile_report_suggestions import (
    CompileReportSuggestion,
    CompileReportSuggestionEvidence,
)

# A reporting heuristic, not a legality limit or an estimate of hardware cost.
_SMALL_EXPANSION_LIMIT = 5


@dataclass(frozen=True, slots=True)
class CompileReportExpansionInventory:
    """Stage-local warnings and explicitly missing per-operation evidence."""

    # Validated rows exceeding the reporting threshold, in capture order.
    warnings: tuple[dict[str, object], ...]
    # Report paths lacking the counts needed to rule out large expansions.
    missing_evidence: tuple[str, ...]


def parse_compile_report_expansions(
    report: dict[str, object], mode: str, source: str
) -> CompileReportExpansionInventory:
    """Indexes existing producer counts without reconstructing IR ancestry."""
    warnings: list[dict[str, object]] = []
    missing: list[str] = []

    def record(
        row: dict[str, object],
        path: str,
        stage: str,
        field: str,
        unit: str,
        recipe_field: str,
        source_op_count: int | None = None,
    ) -> None:
        if field not in row:
            missing.append(f"{path}.{field}")
            return
        count = _count(row[field], f"{source}.{path}.{field}")
        if count <= _SMALL_EXPANSION_LIMIT:
            return
        warning: dict[str, object] = {
            "severity": "warning",
            "stage": stage,
            "evidence_path": f"{path}.{field}",
            "maximum_op_count": count,
            "unit": unit,
            "source_op_count": source_op_count,
        }
        for field_name, output_name in (
            ("function", "function"),
            ("source_op", "source_op"),
            (recipe_field, "recipe"),
        ):
            value = row.get(field_name)
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise CompileReportError(
                        f"{source}.{path}.{field_name}: expected nonempty string"
                    )
                warning[output_name] = value
        warnings.append(warning)

    source_low = report.get("source_low")
    if source_low is not None:
        section = _object(source_low, f"{source}.source_low")
        if mode == "details" and "rows" in section:
            for position, row in enumerate(_rows(section, f"{source}.source_low")):
                record(
                    row,
                    f"source_low.rows[{position}]",
                    "source_to_low",
                    "emitted_low_op_count",
                    "Low operations",
                    "plan_key",
                    1,
                )
        elif "selection_summaries" in section:
            path = "source_low.selection_summaries"
            summaries = _object(section["selection_summaries"], f"{source}.{path}")
            for position, row in enumerate(_rows(summaries, f"{source}.{path}")):
                row_path = f"{path}.rows[{position}]"
                count = _count(
                    row.get("selected_op_count"),
                    f"{source}.{row_path}.selected_op_count",
                )
                if count == 0:
                    raise CompileReportError(
                        f"{source}.{row_path}.selected_op_count: "
                        "expected positive count"
                    )
                record(
                    row,
                    row_path,
                    "source_to_low",
                    "maximum_emitted_low_op_count",
                    "Low operations",
                    "plan_key",
                    count,
                )
        elif (
            section.get("selected_op_count")
            and "unkeyed_maximum_emitted_op_count" not in section
        ):
            missing.append("source_low.selection_summaries")
        if mode == "summary" and "unkeyed_maximum_emitted_op_count" in section:
            record(
                section,
                "source_low",
                "source_to_low",
                "unkeyed_maximum_emitted_op_count",
                "Low operations",
                "plan_key",
            )

    for stage, recipe_field in (
        ("math_legalization", "recipe"),
        ("target_legalization", "legalizer"),
    ):
        value = report.get(stage)
        if value is None:
            continue
        section = _object(value, f"{source}.{stage}")
        if mode == "details" and "rows" in section:
            for position, row in enumerate(_rows(section, f"{source}.{stage}")):
                if row.get("action") == "rewritten":
                    record(
                        row,
                        f"{stage}.rows[{position}]",
                        stage,
                        "created_op_count",
                        "source IR operations",
                        recipe_field,
                        1,
                    )
        elif "maximum_created_op_count" in section:
            record(
                section,
                stage,
                stage,
                "maximum_created_op_count",
                "source IR operations",
                recipe_field,
            )
        elif section.get("rewritten_op_count"):
            missing.append(f"{stage}.maximum_created_op_count")

    return CompileReportExpansionInventory(tuple(warnings), tuple(missing))


def build_expansion_show(document: CompileReportDocument) -> dict[str, object] | None:
    """Shows expansion alarms even when compilation failed after lowering."""
    inventory = document.expansion_inventory
    if not inventory.warnings and not inventory.missing_evidence:
        return None
    return {
        "small_expansion_limit": _SMALL_EXPANSION_LIMIT,
        "count_basis": "per_stage_not_final_instructions",
        "rows": sorted(
            inventory.warnings, key=lambda row: -int(row["maximum_op_count"])
        ),
        "missing_evidence": list(inventory.missing_evidence),
    }


def _describe(warning: dict[str, object]) -> str:
    identity = " ".join(
        str(warning[key]) for key in ("function", "source_op") if key in warning
    )
    description = (
        f"{identity or warning['stage']}: up to {warning['maximum_op_count']} "
        f"{warning['unit']} per source operation"
    )
    if warning["source_op_count"] is not None and int(warning["source_op_count"]) > 1:
        description += f" ({warning['source_op_count']} source operations summarized)"
    if "recipe" in warning:
        description += f" via {warning['recipe']}"
    return description


def append_expansion_show_text(lines: list[str], view: dict[str, object]) -> None:
    """Renders explicit warning labels in the ordinary report view."""
    lines.extend(
        (
            "",
            "Per-operation expansion warnings",
            (
                f"  threshold: more than {view['small_expansion_limit']} operations; "
                "stage-local counts, not final instructions or runtime"
            ),
        )
    )
    for warning in view["rows"]:
        lines.append(f"  WARNING: {_describe(warning)}")
        lines.append(f"    evidence: {warning['evidence_path']}")
    if view["missing_evidence"]:
        lines.append(
            "  Expansion evidence incomplete: regenerate with --compile-report=details."
        )
        lines.extend(f"    unavailable: {path}" for path in view["missing_evidence"])


def suggest_expansions(
    document: CompileReportDocument,
) -> tuple[CompileReportSuggestion, ...]:
    """Recommends investigating large realizations without relaxing semantics."""
    suggestions = []
    for warning in document.expansion_inventory.warnings:
        details = (
            " Capture --compile-report=details to identify the source operation."
            if "source_op" not in warning
            else ""
        )
        suggestions.append(
            CompileReportSuggestion(
                suggestion_id="lowering.reduce_expansion",
                entry_name=str(warning.get("function", "<report>")),
                variant=str(warning["stage"]),
                example_magnitude=int(warning["maximum_op_count"]),
                action=(
                    f"WARNING: {_describe(warning)}.{details} "
                    "Inspect this lowering for a compact native or vector "
                    "implementation that preserves the source semantics. Fast-math "
                    "flags alone do not prove a cheaper realization. Check final "
                    "instructions, registers and measured runtime; these stage-local "
                    "counts are not cumulative costs."
                ),
                evidence=(
                    CompileReportSuggestionEvidence(
                        str(warning["evidence_path"]), warning["maximum_op_count"]
                    ),
                ),
            )
        )
    suggestions.extend(
        CompileReportSuggestion(
            suggestion_id="lowering.inspect_expansion",
            entry_name="<report>",
            action=(
                "Per-operation expansion counts are unavailable for part of "
                "this report. Regenerate with --compile-report=details; "
                "aggregate totals or averages "
                "cannot rule out a single expensive operation."
            ),
            evidence=(CompileReportSuggestionEvidence(path, "unavailable"),),
        )
        for path in document.expansion_inventory.missing_evidence
    )
    return tuple(suggestions)


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CompileReportError(f"{path}: expected object")
    return value


def _count(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CompileReportError(f"{path}: expected nonnegative integer")
    return value


def _rows(section: dict[str, object], path: str) -> list[dict[str, object]]:
    rows = section.get("rows")
    if not isinstance(rows, list):
        raise CompileReportError(f"{path}.rows: expected array")
    if _count(section.get("count"), f"{path}.count") != len(rows):
        raise CompileReportError(f"{path}.count: does not match rows")
    for position, value in enumerate(rows):
        row = _object(value, f"{path}.rows[{position}]")
        if _count(row.get("index"), f"{path}.rows[{position}].index") != position:
            raise CompileReportError(
                f"{path}.rows[{position}].index: expected {position}"
            )
    return rows
