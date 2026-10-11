# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.reporting.compile_report import CompileReportError, parse_compile_report
from loom.reporting.compile_report_scalarization import suggest_scalarization
from loom.reporting.compile_report_view import (
    build_compile_report_show,
    format_compile_report_show_text,
)


def _row(
    index: int,
    source_op: str,
    *,
    scalarized: bool,
    strategy: str,
    created: int,
) -> dict[str, object]:
    return {
        "index": index,
        "function": "gather_decode_dot",
        "source_op": source_op,
        "source_op_kind": 3611 + index,
        "target_bundle": "core_target",
        "target_config": "core_target",
        "legalizer": "vector" if strategy == "reference" else "aie2p",
        "legalizer_strategy": strategy,
        "mode": "eager",
        "policy": "prefer-native",
        "action": "rewritten",
        "scalarized": scalarized,
        "legalization_outcome": (
            "reference-fallback" if strategy == "reference" else "target-rewrite"
        ),
        "contract_outcome": "unhandled",
        "descriptor_key": None,
        "source_rejection_bits": 0,
        "source_rejection_detail": 0,
        "target_rejection_bits": 0,
        "missing_feature_bits": 0,
        "missing_fact_bits": 0,
        "created_op_count": created,
        "erased_op_count": 1,
    }


def _report(*, details: bool = True) -> dict[str, object]:
    legalization: dict[str, object] = {
        "legal_op_count": 0,
        "rewritten_op_count": 2,
        "scalarized_op_count": 1,
        "target_rewritten_op_count": 1,
        "reference_rewritten_op_count": 1,
        "deferred_op_count": 0,
        "invalid_ir_op_count": 0,
        "unsupported_op_count": 0,
        "unhandled_op_count": 0,
        "count": 2 if details else 0,
    }
    if details:
        legalization["rows"] = [
            _row(
                0,
                "vector.gather",
                scalarized=True,
                strategy="reference",
                created=81,
            ),
            _row(
                1,
                "vector.splat",
                scalarized=False,
                strategy="target",
                created=3,
            ),
        ]
    return {
        "kind": "loom.compile_report",
        "schema_version": 0,
        "mode": "details" if details else "summary",
        "status": {"code": 0, "name": "OK"},
        "entries": {
            "count": 1,
            "rows": [
                {
                    "index": 0,
                    "function": "gather_decode_dot",
                    "source_function": "gather_decode_dot",
                }
            ],
        },
        "target_legalization": legalization,
    }


def test_show_and_suggest_expose_only_explicit_scalarization() -> None:
    document = parse_compile_report(_report(), source="report.json")

    show = build_compile_report_show(document)
    scalarization = show["scalarization"]
    assert scalarization["scalarized_op_count"] == 1
    assert scalarization["rows"] == [
        {
            "index": 0,
            "function": "gather_decode_dot",
            "source_op": "vector.gather",
            "legalizer": "vector",
            "legalizer_strategy": "reference",
            "created_op_count": 81,
            "erased_op_count": 1,
            "net_op_growth": 80,
        }
    ]
    text = format_compile_report_show_text(show)
    assert "Scalar lane expansion" in text
    assert "vector.gather: 81 created, 1 erased (+80 net)" in text
    assert "rooted at vector/reference" in text

    (suggestion,) = suggest_scalarization(document)
    assert suggestion.suggestion_id == "vector.eliminate_scalarization"
    assert suggestion.entry_name == "gather_decode_dot"
    assert "rewrite rooted at vector.gather" in suggestion.action
    assert "created 81 operations" in suggestion.action
    evidence = {fact.path: fact.value for fact in suggestion.evidence}
    assert evidence["target_legalization.rows[0].scalarized"] is True
    assert evidence["target_legalization.rows[0].created_op_count"] == 81


def test_summary_makes_scalarization_visible_and_requests_details() -> None:
    document = parse_compile_report(_report(details=False))

    scalarization = build_compile_report_show(document)["scalarization"]
    assert scalarization == {
        "legalization_op_count": 2,
        "rewritten_op_count": 2,
        "scalarized_op_count": 1,
    }
    (suggestion,) = suggest_scalarization(document)
    assert suggestion.suggestion_id == "vector.inspect_scalarization"
    assert suggestion.entry_name == "gather_decode_dot"
    assert "--compile-report=details" in suggestion.action


def test_large_rewrite_without_scalarization_does_not_trigger_a_finding() -> None:
    report = _report()
    legalization = report["target_legalization"]
    legalization["scalarized_op_count"] = 0
    legalization["rows"][0]["scalarized"] = False
    document = parse_compile_report(report)

    assert "scalarization" not in build_compile_report_show(document)
    assert suggest_scalarization(document) == ()


def test_rejects_scalarization_count_that_disagrees_with_rows() -> None:
    report = _report()
    report["target_legalization"]["scalarized_op_count"] = 2

    with pytest.raises(CompileReportError, match="scalarized_op_count"):
        parse_compile_report(report)
