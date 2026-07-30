# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.reporting.compile_report import parse_compile_report
from loom.target.arch.amdgpu.compile_report_suggestions import (
    AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER,
)


def _compile_report(
    *,
    target_key: str | None = "gfx1100",
    subgroup_size: int = 64,
) -> dict[str, object]:
    report = {
        "kind": "loom.compile_report",
        "schema_version": 0,
        "mode": "details",
        "status": {"code": 0, "name": "OK"},
        "target_family": "AMDGPU",
        "entries": {
            "count": 1,
            "rows": [
                {
                    "index": 0,
                    "function": "routed_linear",
                    "allocation_spill_count": 2,
                    "allocation_materialized_reload_bytes": 4096,
                    "private_memory_bytes": 128,
                    "target_resources": {
                        "subgroup_size": subgroup_size,
                        "residency": {
                            "current_tier": 5,
                            "next_better_tier": 6,
                            "unique_limiting_resource": {
                                "name": "amdgpu.vgpr",
                                "reduction_units_to_next_better_tier": 8,
                            },
                        },
                    },
                }
            ],
        },
    }
    if target_key is not None:
        report["target_key"] = target_key
    return report


def test_suggests_ordered_experiments_from_exact_target_evidence() -> None:
    document = parse_compile_report(_compile_report(), source="report.json")

    result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(document)

    assert result.unavailable_reason is None
    assert [suggestion.suggestion_id for suggestion in result.suggestions] == [
        "amdgpu.spill_traffic",
        "amdgpu.residency_cliff",
        "amdgpu.nondefault_wave_size",
    ]
    wave_evidence = result.suggestions[-1].evidence
    assert wave_evidence[-1].path == "target_info.wavefront.default_size"
    assert wave_evidence[-1].value == 32
    residency = result.suggestions[-2]
    assert "at least 8 units" in residency.action
    assert residency.evidence[0].path.endswith("residency.current_tier")


def test_resolves_overlay_target_to_its_processor_model() -> None:
    document = parse_compile_report(
        _compile_report(target_key="gfx1250-a0"), source="report.json"
    )

    result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(document)

    assert result.unavailable_reason is None
    wave_evidence = result.suggestions[-1].evidence
    assert wave_evidence[-1].path == "target_info.wavefront.default_size"
    assert wave_evidence[-1].value == 32


def test_resolves_qualified_target_without_losing_feature_identity() -> None:
    document = parse_compile_report(
        _compile_report(
            target_key="gfx942:sramecc+:xnack-",
            subgroup_size=32,
        ),
        source="report.json",
    )

    result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(document)

    assert result.unavailable_reason is None
    wave_evidence = result.suggestions[-1].evidence
    assert wave_evidence[-1].path == "target_info.wavefront.default_size"
    assert wave_evidence[-1].value == 64


def test_missing_unknown_and_invalid_target_keys_are_unavailable() -> None:
    missing = parse_compile_report(_compile_report(target_key=None))
    unknown = parse_compile_report(_compile_report(target_key="gfx9999"))
    invalid = parse_compile_report(_compile_report(target_key="gfx942:xnack+:xnack-"))

    missing_result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(missing)
    unknown_result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(unknown)
    invalid_result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(invalid)

    assert missing_result.unavailable_reason == "missing_target_key"
    assert missing_result.suggestions == ()
    assert unknown_result.unavailable_reason == "unknown_target_key"
    assert unknown_result.suggestions == ()
    assert invalid_result.unavailable_reason == "invalid_target_key"
    assert invalid_result.suggestions == ()


def test_private_memory_is_reported_once_without_spill_evidence() -> None:
    report = _compile_report()
    entry = report["entries"]["rows"][0]
    entry["allocation_spill_count"] = 0
    entry["allocation_materialized_reload_bytes"] = 0
    document = parse_compile_report(report)

    result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(document)

    assert [suggestion.suggestion_id for suggestion in result.suggestions] == [
        "amdgpu.private_memory",
        "amdgpu.residency_cliff",
        "amdgpu.nondefault_wave_size",
    ]
