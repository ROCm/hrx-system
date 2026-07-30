# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.reporting.compile_report import CompileReportError, parse_compile_report
from loom.reporting.compile_report_suggestions import (
    CompileReportSuggestionOptions,
)
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


def _add_bank_service_group(
    report: dict[str, object],
    *,
    model_evidence: str,
    exact_packet_count: int = 1,
    unknown_packet_count: int = 0,
    conflicted_packet_count: int = 1,
    extra_round_count: int = 8,
) -> None:
    report["source_low"] = {
        "memory": {
            "bank_service": {
                "modeled_packet_count": exact_packet_count + unknown_packet_count,
                "exact_packet_count": exact_packet_count,
                "unknown_packet_count": unknown_packet_count,
                "structural": {
                    "conflict_free_packet_count": 0,
                    "conflicted_packet_count": conflicted_packet_count,
                    "required_round_count": 16,
                    "uncontended_round_count": 8,
                    "extra_round_count": extra_round_count,
                    "maximum_request_multiplicity": 2,
                },
                "dynamic": {
                    "exact_packet_count": exact_packet_count,
                    "unknown_packet_count": unknown_packet_count,
                    "packet_count": exact_packet_count,
                    "required_round_count": 16,
                    "uncontended_round_count": 8,
                    "extra_round_count": extra_round_count,
                },
            },
            "bank_service_group_count": 1,
            "bank_service_groups": [
                {
                    "index": 0,
                    "function": "routed_linear",
                    "source_op": "vector.fragment.load",
                    "source_op_kind": 80,
                    "source_root": "scratch",
                    "memory_space": "workgroup",
                    "operation": "load",
                    "packet": "amdgpu.ds_read_b128",
                    "strategy": None,
                    "model": {
                        "key": ("amdgpu.lds.wave32.b128.quad-phases.read.count-each"),
                        "revision": "ROCm/rocm-libraries@model",
                        "evidence": model_evidence,
                        "request_policy": "count-each",
                        "wave_size": 32,
                        "bank_count": 32,
                        "bank_word_bytes": 4,
                        "packet_bank_words": 4,
                    },
                    "summary": {
                        "modeled_packet_count": (
                            exact_packet_count + unknown_packet_count
                        ),
                        "exact_packet_count": exact_packet_count,
                        "unknown_packet_count": unknown_packet_count,
                        "structural": {
                            "conflict_free_packet_count": 0,
                            "conflicted_packet_count": conflicted_packet_count,
                            "required_round_count": 16,
                            "uncontended_round_count": 8,
                            "extra_round_count": extra_round_count,
                            "maximum_request_multiplicity": 2,
                        },
                        "dynamic": {
                            "exact_packet_count": exact_packet_count,
                            "unknown_packet_count": unknown_packet_count,
                            "packet_count": exact_packet_count,
                            "required_round_count": 16,
                            "uncontended_round_count": 8,
                            "extra_round_count": extra_round_count,
                        },
                    },
                }
            ],
        }
    }


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


def test_unvalidated_bank_model_is_explicitly_opt_in() -> None:
    report = _compile_report(target_key="gfx1250-a0", subgroup_size=32)
    _add_bank_service_group(
        report,
        model_evidence="vendor-software-model-unvalidated",
    )
    document = parse_compile_report(report)

    default_result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(document)
    experimental_result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(
        document,
        CompileReportSuggestionOptions(include_experimental=True),
    )

    assert "amdgpu.lds_bank_service" not in {
        suggestion.suggestion_id for suggestion in default_result.suggestions
    }
    bank_suggestion = experimental_result.suggestions[-1]
    assert bank_suggestion.suggestion_id == "amdgpu.lds_bank_service"
    assert bank_suggestion.confidence == "experimental"
    assert "spill or occupancy regressions" in bank_suggestion.action
    assert "hardware timing" in bank_suggestion.action
    assert bank_suggestion.evidence[2].path.endswith(
        "summary.structural.extra_round_count"
    )
    assert bank_suggestion.evidence[2].value == 8


def test_calibrated_exact_bank_conflict_is_high_confidence() -> None:
    report = _compile_report(target_key="gfx1250-a0", subgroup_size=32)
    _add_bank_service_group(
        report,
        model_evidence="silicon-calibrated-vendor-model",
    )
    document = parse_compile_report(report)

    result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(document)

    bank_suggestion = result.suggestions[-1]
    assert bank_suggestion.suggestion_id == "amdgpu.lds_bank_service"
    assert bank_suggestion.confidence == "high"


def test_bank_suggestion_requires_complete_exact_conflict_evidence() -> None:
    report = _compile_report(target_key="gfx1250-a0", subgroup_size=32)
    _add_bank_service_group(
        report,
        model_evidence="silicon-calibrated-vendor-model",
        exact_packet_count=1,
        unknown_packet_count=1,
    )
    document = parse_compile_report(report)

    result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(document)

    assert "amdgpu.lds_bank_service" not in {
        suggestion.suggestion_id for suggestion in result.suggestions
    }


def test_bank_suggestion_rejects_unknown_model_evidence_class() -> None:
    report = _compile_report(target_key="gfx1250-a0", subgroup_size=32)
    _add_bank_service_group(
        report,
        model_evidence="unversioned-model",
    )
    document = parse_compile_report(report)

    with pytest.raises(CompileReportError, match="unsupported evidence class"):
        AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(document)
