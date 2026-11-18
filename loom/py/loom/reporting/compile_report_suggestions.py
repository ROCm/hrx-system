# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-neutral contracts for compile report suggestions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from textwrap import fill
from typing import Protocol

from loom.reporting.compile_report import (
    COMPILE_REPORT_SCHEMA_VERSION,
    CompileReportDocument,
)
from loom.reporting.compile_report_pipelines import (
    append_pipeline_phase_copies_text,
    append_pipeline_transfer_text,
    build_pipeline_phase_copies,
    build_pipeline_show,
    single_slot_refills,
)

SUGGEST_KIND = "loom.compile_report.suggest"
SUGGEST_GUIDANCE = (
    "Static compiler findings describe the program. Finding order and confidence "
    "do not rank performance benefit. For a port, establish the oracle's native "
    "instruction mix, intended-machine performance, and data reuse policy. For a "
    "novel kernel, count compulsory bytes and useful operations and measure matched "
    "primitives on the intended partition. Bound the end-to-end opportunity. "
    "An unexplained large oracle gap stops local tuning."
)


class CompileReportSuggestionConfidence(StrEnum):
    """Confidence in the static finding, not its runtime cost or tuning value."""

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
    # Distinct intervention or measurement domain within this rule, if any.
    variant: str = ""
    # Comparable static magnitude within this variant. Ties keep provider order.
    example_magnitude: int = 0


@dataclass(frozen=True, slots=True)
class CompileReportSuggestionResult:
    """Findings or an explicit reason target interpretation is unavailable."""

    provider_name: str | None
    unavailable_reason: str | None
    suggestions: tuple[CompileReportSuggestion, ...] = ()
    # Missing target interpretation when shared source findings remain available.
    target_unavailable_reason: str | None = None


def suggest_single_slot_ingress(
    document: CompileReportDocument,
) -> tuple[CompileReportSuggestion, ...]:
    findings = []
    for path, phase, binding, fifo, visits, width in single_slot_refills(document):
        if visits * width < 4096:
            continue
        findings.append(
            CompileReportSuggestion(
                suggestion_id="pipeline.single_slot_ingress",
                entry_name=f"{phase} binding[{binding}] → FIFO[{fifo}]",
                confidence=CompileReportSuggestionConfidence.EXPERIMENTAL,
                example_magnitude=visits * width,
                action=(
                    f"Each producer reserves the same single slot {visits:,} times "
                    f"per phase call and reads {width:,} B each time. Check whether "
                    "the next fill waits for the consumer. Compare two slots at "
                    "fixed external bytes and measure the completed phase."
                ),
                evidence=tuple(
                    CompileReportSuggestionEvidence(f"{path}.{field}", value)
                    for field, value in (
                        ("reserved_fifo_capacity", 1),
                        ("reservations_per_worker", visits),
                        ("bytes_per_copy", width),
                    )
                ),
            )
        )
    return tuple(findings)


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
    *,
    expand: str | None = None,
) -> dict[str, object]:
    """Shows one cited example per investigation, or the requested site evidence."""
    target = {}
    target_family = document.report.get("target_family")
    target_key = document.report.get("target_key")
    if target_family is not None:
        target["family"] = target_family
    if target_key is not None:
        target["key"] = target_key

    if expand is None:
        groups: dict[
            tuple[str, CompileReportSuggestionConfidence, str],
            tuple[CompileReportSuggestion, int],
        ] = {}
        for suggestion in result.suggestions:
            key = suggestion.suggestion_id, suggestion.confidence, suggestion.variant
            example, count = groups.get(key, (suggestion, 0))
            if suggestion.example_magnitude > example.example_magnitude:
                example = suggestion
            groups[key] = example, count + 1
        findings = []
        for example, count in groups.values():
            finding = _finding(example)
            if example.variant:
                finding["variant"] = example.variant
            if count > 1:
                finding["match_count"] = count
            findings.append(finding)
        match_count = len(result.suggestions)
    else:
        findings = [
            _finding(suggestion)
            for suggestion in result.suggestions
            if expand == "all" or suggestion.suggestion_id == expand
        ]
        match_count = len(findings)

    view: dict[str, object] = {
        "kind": SUGGEST_KIND,
        "schema_version": COMPILE_REPORT_SCHEMA_VERSION,
        "source": document.source,
        "target": target,
        "status": (
            "unavailable" if result.unavailable_reason is not None else "available"
        ),
        "guidance": SUGGEST_GUIDANCE,
        "finding_count": len(findings),
        "match_count": match_count,
        "findings": findings,
    }
    if expand is not None:
        view["expanded"] = expand
    if result.provider_name is not None:
        view["provider"] = result.provider_name
    if result.unavailable_reason is not None:
        view["reason"] = result.unavailable_reason
    if result.target_unavailable_reason is not None:
        view["target_unavailable_reason"] = result.target_unavailable_reason
    pipelines = build_pipeline_show(document, summary_only=True)
    if pipelines is not None:
        view["pipelines"] = pipelines
    phase_copies = build_pipeline_phase_copies(document, summary_only=True)
    if phase_copies:
        view["pipeline_phase_copies"] = phase_copies
    return view


def _finding(suggestion: CompileReportSuggestion) -> dict[str, object]:
    return {
        "id": suggestion.suggestion_id,
        "entry": suggestion.entry_name,
        "confidence": suggestion.confidence.value,
        "action": suggestion.action,
        "evidence": {evidence.path: evidence.value for evidence in suggestion.evidence},
    }


def format_compile_report_suggestions_text(view: dict[str, object]) -> str:
    """Formats evidence-backed suggestions for humans and agent prompts."""
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
    if "target_unavailable_reason" in view:
        lines.append(f"  target interpretation: {view['target_unavailable_reason']}")
    lines.extend(
        (
            "",
            fill(
                str(view["guidance"]),
                width=100,
                initial_indent="  ",
                subsequent_indent="  ",
                break_long_words=False,
                break_on_hyphens=False,
            ),
        )
    )
    if "pipelines" in view:
        for plan in _expect_dict(view["pipelines"])["rows"]:
            append_pipeline_transfer_text(lines, plan)
    if "pipeline_phase_copies" in view:
        append_pipeline_phase_copies_text(
            lines, _expect_list(view["pipeline_phase_copies"])
        )
    findings = _expect_list(view["findings"])
    if view["match_count"] != view["finding_count"]:
        lines.append(
            f"\n{len(findings)} investigations cover {view['match_count']:,} matches. "
            "Use --expand ID for every match to one rule."
        )
    if "expanded" in view:
        lines.append(f"  expanded: {view['expanded']}")
    if not findings:
        lines.append("  findings: none")
    for finding_value in findings:
        finding = _expect_dict(finding_value)
        heading = f"[{finding['id']}]"
        if "variant" in finding:
            heading += f" {finding['variant']}"
        if "match_count" in finding:
            heading += f" ({finding['match_count']:,} matches) — example"
        lines.extend(
            (
                "",
                f"{heading} {finding['entry']}",
                f"  confidence: {finding['confidence']}",
                fill(
                    f"action: {finding['action']}",
                    width=100,
                    initial_indent="  ",
                    subsequent_indent="    ",
                    break_long_words=False,
                    break_on_hyphens=False,
                ),
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
