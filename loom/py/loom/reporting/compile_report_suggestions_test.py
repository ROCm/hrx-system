# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.reporting.compile_report import parse_compile_report
from loom.reporting.compile_report_suggestions import (
    CompileReportSuggestion,
    CompileReportSuggestionEvidence,
    CompileReportSuggestionResult,
    build_compile_report_suggestions,
    format_compile_report_suggestions_text,
)


def test_builds_compact_suggestion_view_with_cited_evidence() -> None:
    document = parse_compile_report(
        {
            "kind": "loom.compile_report",
            "schema_version": 0,
            "mode": "details",
            "status": {"code": 0, "name": "OK"},
            "target_family": "TEST",
            "target_key": "test0",
            "entries": {"count": 0, "rows": []},
        },
        source="report.json",
    )
    result = CompileReportSuggestionResult(
        provider_name="test",
        unavailable_reason=None,
        suggestions=(
            CompileReportSuggestion(
                suggestion_id="test.reduce_pressure",
                entry_name="kernel",
                action="Reduce pressure and recompile.",
                evidence=(
                    CompileReportSuggestionEvidence(
                        path="entries.rows[0].pressure",
                        value=128,
                    ),
                ),
            ),
        ),
    )

    view = build_compile_report_suggestions(document, result)

    assert view["status"] == "available"
    assert view["finding_count"] == 1
    assert view["findings"] == [
        {
            "id": "test.reduce_pressure",
            "entry": "kernel",
            "confidence": "high",
            "action": "Reduce pressure and recompile.",
            "evidence": {"entries.rows[0].pressure": 128},
        }
    ]
    text = format_compile_report_suggestions_text(view)
    assert "[test.reduce_pressure] kernel" in text
    assert "confidence: high" in text
    assert "entries.rows[0].pressure: 128" in text


def test_builds_explicit_unavailable_view() -> None:
    document = parse_compile_report(
        {
            "kind": "loom.compile_report",
            "schema_version": 0,
            "mode": "summary",
            "status": {"code": 0, "name": "OK"},
            "entries": {"count": 0, "rows": []},
        }
    )
    result = CompileReportSuggestionResult(
        provider_name=None,
        unavailable_reason="missing_target_family",
    )

    view = build_compile_report_suggestions(document, result)

    assert view["status"] == "unavailable"
    assert view["reason"] == "missing_target_family"
    assert view["findings"] == []
