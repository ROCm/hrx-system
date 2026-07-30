# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-neutral contracts for compile report suggestions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from loom.reporting.compile_report import (
    COMPILE_REPORT_SCHEMA_VERSION,
    CompileReportDocument,
)

SUGGEST_KIND = "loom.compile_report.suggest"


class CompileReportSuggestionConfidence(StrEnum):
    """Evidence tier for one proposed experiment."""

    HIGH = "high"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True, slots=True)
class CompileReportSuggestionOptions:
    """Controls whether lower-confidence experiments may be surfaced."""

    include_experimental: bool = False


@dataclass(frozen=True, slots=True)
class CompileReportSuggestionEvidence:
    """One report or target-info fact supporting a suggestion."""

    path: str
    value: object


@dataclass(frozen=True, slots=True)
class CompileReportSuggestion:
    """One actionable experiment justified by available evidence."""

    suggestion_id: str
    entry_name: str
    action: str
    evidence: tuple[CompileReportSuggestionEvidence, ...]
    confidence: CompileReportSuggestionConfidence = (
        CompileReportSuggestionConfidence.HIGH
    )


@dataclass(frozen=True, slots=True)
class CompileReportSuggestionResult:
    """Findings or an explicit reason target interpretation is unavailable."""

    provider_name: str | None
    unavailable_reason: str | None
    suggestions: tuple[CompileReportSuggestion, ...] = ()


class CompileReportSuggestionProvider(Protocol):
    """Target-owned interpreter for one exact target family."""

    target_family: str
    provider_name: str

    def suggest(
        self,
        document: CompileReportDocument,
        options: CompileReportSuggestionOptions,
    ) -> CompileReportSuggestionResult:
        """Returns ordered suggestions for a validated compile report."""
        ...


def build_compile_report_suggestions(
    document: CompileReportDocument,
    result: CompileReportSuggestionResult,
) -> dict[str, object]:
    """Builds a compact deterministic suggestion view."""
    target = {}
    target_family = document.report.get("target_family")
    target_key = document.report.get("target_key")
    if target_family is not None:
        target["family"] = target_family
    if target_key is not None:
        target["key"] = target_key

    view: dict[str, object] = {
        "kind": SUGGEST_KIND,
        "schema_version": COMPILE_REPORT_SCHEMA_VERSION,
        "source": document.source,
        "target": target,
        "status": (
            "unavailable" if result.unavailable_reason is not None else "available"
        ),
        "finding_count": len(result.suggestions),
        "findings": [
            {
                "id": suggestion.suggestion_id,
                "entry": suggestion.entry_name,
                "confidence": suggestion.confidence.value,
                "action": suggestion.action,
                "evidence": {
                    evidence.path: evidence.value for evidence in suggestion.evidence
                },
            }
            for suggestion in result.suggestions
        ],
    }
    if result.provider_name is not None:
        view["provider"] = result.provider_name
    if result.unavailable_reason is not None:
        view["reason"] = result.unavailable_reason
    return view


def format_compile_report_suggestions_text(view: dict[str, object]) -> str:
    """Formats target-owned suggestions for humans and agent prompts."""
    target = _expect_dict(view["target"])
    target_name = "/".join(
        str(value)
        for value in (target.get("family"), target.get("key"))
        if value is not None
    )
    lines = [
        "Loom compile report suggestions",
        f"  source: {view['source']}",
        f"  target: {target_name or 'unavailable'}",
        f"  provider: {view.get('provider', 'unavailable')}",
        f"  status: {view['status']}",
    ]
    if "reason" in view:
        lines.append(f"  reason: {view['reason']}")
    findings = _expect_list(view["findings"])
    if not findings:
        lines.append("  findings: none")
    for finding_value in findings:
        finding = _expect_dict(finding_value)
        lines.extend(
            (
                "",
                f"[{finding['id']}] {finding['entry']}",
                f"  confidence: {finding['confidence']}",
                f"  action: {finding['action']}",
                "  evidence:",
            )
        )
        evidence = _expect_dict(finding["evidence"])
        lines.extend(f"    {path}: {value}" for path, value in evidence.items())
    return "\n".join(lines) + "\n"


def _expect_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("invalid internal compile report suggestion object")
    return value


def _expect_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("invalid internal compile report suggestion array")
    return value
