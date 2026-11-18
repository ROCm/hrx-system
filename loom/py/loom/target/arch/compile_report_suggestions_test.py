# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.reporting.compile_report import CompileReportError, parse_compile_report
from loom.target.arch.compile_report_suggestions import suggest_compile_report


def _math_row(
    index: int,
    *,
    function: str = "rsqrt_strict",
    source_op: str = "vector.rsqrtf",
    math_op: str = "rsqrtf",
    lane_domain: str = "vector",
    element_type: str = "f32",
    target_config: str = "x86.avx2.core",
    source_fastmath_flags: int = 0,
    native_required_fastmath_flags: int | None = 1 << 6,
) -> dict[str, object]:
    row: dict[str, object] = {
        "index": index,
        "function": function,
        "source_op": source_op,
        "source_op_kind": 101,
        "target_bundle": "x86-avx2",
        "target_config": target_config,
        "policy": "x86-math",
        "constraint_key": "math.recipe.rsqrt_sqrt_div",
        "math_op": math_op,
        "lane_domain": lane_domain,
        "element_type": element_type,
        "action": "rewritten",
        "recipe": "rsqrt-sqrt-div",
        "source_fastmath_flags": source_fastmath_flags,
        "recipe_fastmath_flags": 0,
        "created_op_count": 3,
        "erased_op_count": 1,
    }
    if native_required_fastmath_flags is not None:
        row["native_required_fastmath_flags"] = native_required_fastmath_flags
    return row


def _compile_report(
    rows: list[dict[str, object]],
    *,
    mode: str = "details",
    target_family: str = "x86",
    target_key: str | None = None,
) -> dict[str, object]:
    report: dict[str, object] = {
        "kind": "loom.compile_report",
        "schema_version": 0,
        "mode": mode,
        "status": {"code": 0, "name": "OK"},
        "target_family": target_family,
        "entries": {
            "count": 1,
            "rows": [{"index": 0, "function": "rsqrt_strict"}],
        },
        "math_legalization": {
            "rewritten_op_count": len(rows),
            "maximum_created_op_count": 3 if rows else 0,
            "native_alternative_op_count": sum(
                row.get("native_required_fastmath_flags") is not None for row in rows
            ),
            "rejected_op_count": 0,
            "missing_policy_op_count": 0,
            "missing_recipe_op_count": 0,
            "count": len(rows),
        },
    }
    if target_key is not None:
        report["target_key"] = target_key
    if mode == "details":
        report["math_legalization"]["rows"] = rows
    return report


def test_aggregates_pathological_repetition_into_one_finding() -> None:
    rows = [_math_row(index) for index in range(4096)]
    document = parse_compile_report(_compile_report(rows), source="report.json")

    result = suggest_compile_report(document)

    assert result.provider_name == "source+x86"
    assert len(document.math_relaxation_inventory.groups) == 1
    group = document.math_relaxation_inventory.groups[0]
    assert group.row_count == 4096
    assert len(group.motifs) == 1
    assert len(result.suggestions) == 1
    suggestion = result.suggestions[0]
    assert suggestion.suggestion_id == "math.use_native_fastmath"
    assert suggestion.entry_name == "rsqrt_strict"
    assert "4096 vector f32 `vector.rsqrtf` operations" in suggestion.action
    assert "`afn`" in suggestion.action
    assert "permits approximate math functions" in suggestion.action
    evidence = {item.path: item.value for item in suggestion.evidence}
    assert evidence["math_legalization.count"] == 4096
    assert evidence["math_legalization.rows[0].native_required_fastmath_flags"] == (
        1 << 6
    )


@pytest.mark.parametrize(
    ("target_family", "target_key", "provider_name"),
    [
        ("x86", None, "source+x86"),
        ("amdgpu", "gfx1100", "source+amdgpu"),
    ],
)
def test_uses_explicit_policy_fact_across_target_providers(
    target_family: str, target_key: str | None, provider_name: str
) -> None:
    row = _math_row(0, target_config="opaque.future.profile")

    result = suggest_compile_report(
        parse_compile_report(
            _compile_report([row], target_family=target_family, target_key=target_key)
        )
    )

    assert result.provider_name == provider_name
    assert len(result.suggestions) == 1
    evidence = {item.path: item.value for item in result.suggestions[0].evidence}
    assert evidence["math_legalization.rows[0].target_config"] == (
        "opaque.future.profile"
    )


def test_omits_recipes_without_an_explicit_native_alternative() -> None:
    row = _math_row(
        0,
        element_type="f64",
        native_required_fastmath_flags=None,
    )

    result = suggest_compile_report(parse_compile_report(_compile_report([row])))

    assert result.provider_name == "x86"
    assert result.suggestions == ()


def test_groups_the_complete_root_family_per_function() -> None:
    rows = [
        _math_row(0, source_op="scalar.rsqrtf", lane_domain="scalar"),
        _math_row(1),
        _math_row(
            2,
            source_op="scalar.rsqrtf",
            lane_domain="scalar",
            element_type="f64",
            target_config="x86.avx512.core",
        ),
        _math_row(3, element_type="f64", target_config="x86.avx512.core"),
        _math_row(
            4,
            source_op="scalar.rsqrtf",
            lane_domain="scalar",
            element_type="f16",
            target_config="x86.avx512_features.core",
        ),
        _math_row(
            5,
            element_type="f16",
            target_config="x86.avx512_features.core",
        ),
    ]

    result = suggest_compile_report(parse_compile_report(_compile_report(rows)))

    assert len(result.suggestions) == 1
    action = result.suggestions[0].action
    assert "scalar f32" in action
    assert "vector f32" in action
    assert "scalar f64" in action
    assert "vector f64" in action
    assert "scalar f16" in action
    assert "vector f16" in action


@pytest.mark.parametrize(
    ("flag", "name"),
    [
        (1 << 0, "reassoc"),
        (1 << 1, "nnan"),
        (1 << 2, "ninf"),
        (1 << 3, "nsz"),
        (1 << 4, "arcp"),
        (1 << 5, "contract"),
        (1 << 6, "afn"),
        (0x7F, "fast"),
    ],
)
def test_names_every_fastmath_permission(flag: int, name: str) -> None:
    result = suggest_compile_report(
        parse_compile_report(
            _compile_report([_math_row(0, native_required_fastmath_flags=flag)])
        )
    )

    assert f"`{name}`" in result.suggestions[0].action


def test_keeps_distinct_permission_experiments_separate() -> None:
    rows = [
        _math_row(0, native_required_fastmath_flags=1 << 4),
        _math_row(1, native_required_fastmath_flags=1 << 6),
    ]

    result = suggest_compile_report(parse_compile_report(_compile_report(rows)))

    assert len(result.suggestions) == 2
    assert "`arcp`" in result.suggestions[0].action
    assert "`afn`" not in result.suggestions[0].action
    assert "`afn`" in result.suggestions[1].action
    assert "`arcp`" not in result.suggestions[1].action
    assert result.suggestions[0].variant != result.suggestions[1].variant


def test_keeps_distinct_math_semantics_separate() -> None:
    rows = [
        _math_row(0, source_op="vector.geluf", math_op="geluf_tanh"),
        _math_row(1, source_op="vector.geluf", math_op="geluf_erf"),
    ]

    result = suggest_compile_report(parse_compile_report(_compile_report(rows)))

    action = result.suggestions[0].action
    assert "`vector.geluf` `geluf_tanh`" in action
    assert "`vector.geluf` `geluf_erf`" in action


@pytest.mark.parametrize(
    ("required_flags", "source_flags", "error"),
    [
        (0, 0, "expected nonzero flags"),
        (1 << 7, 0, "contains unknown fast-math flags"),
        (1 << 6, 1 << 6, "already satisfied by source"),
    ],
)
def test_rejects_inconsistent_native_relaxation_facts(
    required_flags: int, source_flags: int, error: str
) -> None:
    report = _compile_report(
        [
            _math_row(
                0,
                source_fastmath_flags=source_flags,
                native_required_fastmath_flags=required_flags,
            )
        ]
    )

    with pytest.raises(CompileReportError, match=error):
        parse_compile_report(report)


def test_rejects_native_alternative_count_that_does_not_match_rows() -> None:
    report = _compile_report([_math_row(0)])
    report["math_legalization"]["native_alternative_op_count"] = 2

    with pytest.raises(CompileReportError, match="does not match rows"):
        parse_compile_report(report)


def test_summary_report_requests_details_without_guessing_permission() -> None:
    report = _compile_report([], mode="summary")
    report["math_legalization"]["rewritten_op_count"] = 4096
    report["math_legalization"]["maximum_created_op_count"] = 3
    report["math_legalization"]["native_alternative_op_count"] = 4096

    result = suggest_compile_report(parse_compile_report(report))

    assert result.provider_name == "source+x86"
    (suggestion,) = result.suggestions
    assert suggestion.suggestion_id == "math.inspect_native_fastmath"
    assert "4096 source math operations" in suggestion.action
    assert "--compile-report=details" in suggestion.action
    assert "`afn`" not in suggestion.action
    assert suggestion.evidence[0].path == (
        "math_legalization.native_alternative_op_count"
    )
