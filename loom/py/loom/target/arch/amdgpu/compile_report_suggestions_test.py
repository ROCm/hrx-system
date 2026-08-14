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


def _add_fragment_packet_evidence(
    report: dict[str, object],
    *,
    expanded: bool,
    wave_proof: str | None = "exact",
    wave_coverage: str = "dense",
) -> None:
    entry = report["entries"]["rows"][0]
    entry["wait_plan"] = {"full_drain_count": 18}
    entry["static_instruction_mix"] = {"register_move_count": 64}
    if expanded:
        packet_count = 128
        scalar_packet_count = 128
        vector_packet_count = 0
        contiguous_vector_packet_count = 0
        emitted_low_op_count = 248
        packet_rows = [
            {
                "index": 0,
                "function": "routed_linear",
                "source_root": "weights",
                "source_root_argument_index": 1,
                "memory_space": "global",
                "operation": "load",
                "packet": "amdgpu.global_load_b16_d16_saddr",
                "strategy": "strided_d16_packed_b16_fragment_load",
                "fallback_reason": None,
                "storage": {"element_format": "f16"},
                "packet_count": 64,
                "scalar_packet_count": 64,
                "contiguous_vector_packet_count": 0,
            },
            {
                "index": 1,
                "function": "routed_linear",
                "source_root": "weights",
                "source_root_argument_index": 1,
                "memory_space": "global",
                "operation": "load",
                "packet": "amdgpu.global_load_b16_d16_hi_saddr",
                "strategy": "strided_d16_packed_b16_fragment_load",
                "fallback_reason": None,
                "storage": {"element_format": "f16"},
                "packet_count": 64,
                "scalar_packet_count": 64,
                "contiguous_vector_packet_count": 0,
            },
        ]
    else:
        packet_count = 8
        scalar_packet_count = 0
        vector_packet_count = 8
        contiguous_vector_packet_count = 8
        emitted_low_op_count = 16
        packet_rows = [
            {
                "index": 0,
                "function": "routed_linear",
                "source_root": "weights",
                "source_root_argument_index": 1,
                "memory_space": "global",
                "operation": "load",
                "packet": "amdgpu.global_load_b128_saddr",
                "strategy": "contiguous_b128_fragment_load",
                "fallback_reason": None,
                "storage": {"element_format": "f16"},
                "packet_count": 8,
                "scalar_packet_count": 0,
                "contiguous_vector_packet_count": 8,
            }
        ]
    report["source_low"] = {
        "selection_summaries": {
            "count": 1,
            "rows": [
                {
                    "index": 0,
                    "function": "routed_linear",
                    "source_op": "vector.fragment.load",
                    "source_op_kind": 80,
                    "selection": "plan",
                    "plan_key": (
                        "strided_d16_packed_b16_fragment_load"
                        if expanded
                        else "contiguous_b128_fragment_load"
                    ),
                    "descriptor_key": None,
                    "descriptor_semantic_tag": None,
                    "selected_op_count": 8,
                    "emitted_low_op_count": emitted_low_op_count,
                }
            ],
        },
        "memory": {
            "argument_count": 1,
            "arguments": [
                {
                    "index": 0,
                    "function": "routed_linear",
                    "source_root": "weights",
                    "source_root_argument_index": 1,
                    "memory_space": "global",
                    "packet_count": packet_count,
                    "scalar_packet_count": scalar_packet_count,
                    "vector_packet_count": vector_packet_count,
                    "contiguous_vector_packet_count": (contiguous_vector_packet_count),
                }
            ],
            "argument_packet_count": len(packet_rows),
            "argument_packets": packet_rows,
        },
    }
    if expanded and wave_proof is not None:
        subgroup_size = entry["target_resources"]["subgroup_size"]
        subgroup_groups = [
            _fragment_wave_group(
                index=index,
                packet=packet_row["packet"],
                proof=wave_proof,
                coverage=wave_coverage,
                subgroup_size=subgroup_size,
            )
            for index, packet_row in enumerate(packet_rows)
        ]
        memory = report["source_low"]["memory"]
        memory["subgroup_access"] = _fragment_wave_summary(
            packet_count,
            proof=wave_proof,
            coverage=wave_coverage,
        )
        memory["subgroup_access_group_count"] = len(subgroup_groups)
        memory["subgroup_access_groups"] = subgroup_groups


def _fragment_wave_group(
    *,
    index: int,
    packet: str,
    proof: str,
    coverage: str,
    subgroup_size: int,
) -> dict[str, object]:
    packet_count = 64
    packet_bytes = 2
    address = {
        "lane_address_proof": "compiled-fragment-lane-register-layout",
        "active_lane_proof": "subgroup-uniform-control-full-wave",
        "lane_mapping": "digit-terms",
        "subgroup_size": subgroup_size,
        "per_lane_packet_bytes": packet_bytes,
        "linear_lane_stride_bytes": 0,
        "lane_terms": [
            {
                "divisor": 1,
                "modulus": 16,
                "byte_stride": 2 if coverage == "dense" else 64,
            }
        ],
    }
    access: dict[str, object] = {
        "proof": proof,
        "unknown_reason": None,
        "address": address,
    }
    if proof == "exact":
        distinct_start_count = 16
        lane_stride_bytes = 2 if coverage == "dense" else 64
        unique_bytes = distinct_start_count * packet_bytes
        span_bytes = (distinct_start_count - 1) * lane_stride_bytes + packet_bytes
        access["geometry"] = {
            "interval_coverage": coverage,
            "subgroup_requested_bytes": subgroup_size * packet_bytes,
            "subgroup_unique_bytes": unique_bytes,
            "subgroup_span_bytes": span_bytes,
            "maximum_adjacent_lane_delta_bytes": (
                (distinct_start_count - 1) * lane_stride_bytes
            ),
            "maximum_uncovered_gap_bytes": max(0, lane_stride_bytes - packet_bytes),
            "distinct_lane_address_count": distinct_start_count,
            "contiguous_region_count": 1 if coverage == "dense" else 16,
        }
    else:
        access["unknown_reason"] = "active-lane-control-not-uniform"
    return {
        "index": index,
        "function": "routed_linear",
        "source_op": "vector.fragment.load",
        "source_op_kind": 80,
        "source_root": "weights",
        "source_root_argument_index": 1,
        "memory_space": "global",
        "operation": "load",
        "packet": packet,
        "strategy": "strided_d16_packed_b16_fragment_load",
        "access": access,
        "summary": _fragment_wave_summary(
            packet_count,
            proof=proof,
            coverage=coverage,
        ),
    }


def _fragment_wave_summary(
    packet_count: int,
    *,
    proof: str,
    coverage: str,
) -> dict[str, object]:
    exact_packet_count = packet_count if proof == "exact" else 0
    return {
        "modeled_packet_count": packet_count,
        "exact_packet_count": exact_packet_count,
        "unknown_packet_count": packet_count - exact_packet_count,
        "structural": {
            "dense_packet_count": (
                packet_count if proof == "exact" and coverage == "dense" else 0
            ),
            "gapped_packet_count": (
                packet_count if proof == "exact" and coverage == "gapped" else 0
            ),
            "overlapping_packet_count": packet_count if proof == "exact" else 0,
        },
        "dynamic": {
            "exact_packet_count": 0,
            "unknown_packet_count": packet_count,
            "packet_count": 0,
            "dense_packet_count": 0,
            "gapped_packet_count": 0,
            "overlapping_packet_count": 0,
        },
    }


def _add_operand_bank_materialization_evidence(
    report: dict[str, object],
    *,
    packet_count: int,
) -> None:
    report["entries"]["rows"][0]["move_causes"] = {
        "kind_count": 1,
        "packet_count": packet_count,
        "unit_count": packet_count,
        "causes": [
            {
                "cause": "operand_bank_materialization",
                "packet_count": packet_count,
                "unit_count": packet_count,
            }
        ],
    }


def _add_vmem_source_reuse_evidence(
    report: dict[str, object],
    *,
    action_count: int,
    source_reuse_full_drain_count: int,
    total_full_drain_count: int,
) -> None:
    entry = report["entries"]["rows"][0]
    entry["wait_plan"] = {"full_drain_count": total_full_drain_count}
    report["wait_reason_summary_rows"] = {
        "count": 1,
        "rows": [
            {
                "index": 0,
                "function": "routed_linear",
                "counter": "vmem_load",
                "reason": "amdgpu.memory_source_reuse",
                "summary": {
                    "action_count": action_count,
                    "full_drain_count": source_reuse_full_drain_count,
                    "max_full_drain_outstanding_before": 3,
                },
            }
        ],
    }


def _add_lds_ssa_use_evidence(
    report: dict[str, object],
    *,
    action_count: int,
    ssa_use_full_drain_count: int,
    ssa_use_partial_wait_count: int,
    total_full_drain_count: int,
    max_outstanding_before: int,
) -> None:
    entry = report["entries"]["rows"][0]
    entry["wait_plan"] = {"full_drain_count": total_full_drain_count}
    report["wait_reason_summary_rows"] = {
        "count": 1,
        "rows": [
            {
                "index": 0,
                "function": "routed_linear",
                "counter": "lds",
                "reason": "amdgpu.ssa_use",
                "summary": {
                    "action_count": action_count,
                    "full_drain_count": ssa_use_full_drain_count,
                    "partial_wait_count": ssa_use_partial_wait_count,
                    "max_outstanding_before": max_outstanding_before,
                },
            }
        ],
    }


def _add_single_subgroup_communication_evidence(
    report: dict[str, object],
    *,
    flat_workgroup_size: int,
    barrier_count: int,
) -> None:
    entry = report["entries"]["rows"][0]
    entry["workload"] = {
        "workgroup_size": {
            "x": flat_workgroup_size,
            "y": 1,
            "z": 1,
            "flat": flat_workgroup_size,
        }
    }
    entry["local_memory_bytes"] = 528
    entry["static_instruction_mix"] = {
        "barrier_count": barrier_count,
        "local_memory_count": 16,
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


def test_suggests_dominant_vmem_source_reuse_serialization() -> None:
    report = _compile_report()
    _add_vmem_source_reuse_evidence(
        report,
        action_count=30,
        source_reuse_full_drain_count=30,
        total_full_drain_count=107,
    )
    document = parse_compile_report(report)

    result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(document)

    suggestion = result.suggestions[0]
    assert suggestion.suggestion_id == "amdgpu.vmem_source_reuse_serialization"
    assert "source-state turnover" in suggestion.action
    evidence = {item.path: item.value for item in suggestion.evidence}
    assert evidence["wait_reason_summary_rows.rows[0].summary.full_drain_count"] == 30
    assert evidence["entries.rows[0].wait_plan.full_drain_count"] == 107


@pytest.mark.parametrize(
    ("source_reuse_full_drain_count", "total_full_drain_count"),
    [
        (7, 100),
        (30, 121),
    ],
)
def test_ignores_sparse_vmem_source_reuse_serialization(
    source_reuse_full_drain_count: int,
    total_full_drain_count: int,
) -> None:
    report = _compile_report()
    _add_vmem_source_reuse_evidence(
        report,
        action_count=source_reuse_full_drain_count,
        source_reuse_full_drain_count=source_reuse_full_drain_count,
        total_full_drain_count=total_full_drain_count,
    )
    document = parse_compile_report(report)

    result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(document)

    assert "amdgpu.vmem_source_reuse_serialization" not in {
        suggestion.suggestion_id for suggestion in result.suggestions
    }


def test_suggests_dominant_lds_ssa_use_serialization() -> None:
    report = _compile_report()
    _add_lds_ssa_use_evidence(
        report,
        action_count=27,
        ssa_use_full_drain_count=27,
        ssa_use_partial_wait_count=0,
        total_full_drain_count=125,
        max_outstanding_before=1,
    )
    document = parse_compile_report(report)

    result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(document)

    suggestion = result.suggestions[0]
    assert suggestion.suggestion_id == "amdgpu.lds_ssa_use_serialization"
    assert "independent LDS or DS producers" in suggestion.action
    evidence = {item.path: item.value for item in suggestion.evidence}
    assert evidence["wait_reason_summary_rows.rows[0].summary.full_drain_count"] == 27
    assert evidence["wait_reason_summary_rows.rows[0].summary.partial_wait_count"] == 0
    assert evidence["entries.rows[0].wait_plan.full_drain_count"] == 125


def test_ignores_pipelined_lds_ssa_uses() -> None:
    report = _compile_report()
    _add_lds_ssa_use_evidence(
        report,
        action_count=27,
        ssa_use_full_drain_count=3,
        ssa_use_partial_wait_count=24,
        total_full_drain_count=101,
        max_outstanding_before=4,
    )
    document = parse_compile_report(report)

    result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(document)

    assert "amdgpu.lds_ssa_use_serialization" not in {
        suggestion.suggestion_id for suggestion in result.suggestions
    }


def test_suggests_single_subgroup_workgroup_communication() -> None:
    report = _compile_report()
    _add_single_subgroup_communication_evidence(
        report,
        flat_workgroup_size=64,
        barrier_count=4,
    )
    document = parse_compile_report(report)

    result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(document)

    suggestion = next(
        suggestion
        for suggestion in result.suggestions
        if suggestion.suggestion_id == "amdgpu.single_subgroup_workgroup_communication"
    )
    assert "fits within one subgroup" in suggestion.action
    evidence = {item.path: item.value for item in suggestion.evidence}
    assert evidence["entries.rows[0].workload.workgroup_size.flat"] == 64
    assert evidence["entries.rows[0].target_resources.subgroup_size"] == 64
    assert evidence["entries.rows[0].static_instruction_mix.barrier_count"] == 4
    assert evidence["entries.rows[0].static_instruction_mix.local_memory_count"] == 16
    assert evidence["entries.rows[0].local_memory_bytes"] == 528


@pytest.mark.parametrize(
    ("flat_workgroup_size", "barrier_count"),
    [
        (128, 4),
        (64, 0),
    ],
)
def test_ignores_multi_subgroup_or_barrier_free_communication(
    flat_workgroup_size: int,
    barrier_count: int,
) -> None:
    report = _compile_report()
    _add_single_subgroup_communication_evidence(
        report,
        flat_workgroup_size=flat_workgroup_size,
        barrier_count=barrier_count,
    )
    document = parse_compile_report(report)

    result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(document)

    assert "amdgpu.single_subgroup_workgroup_communication" not in {
        suggestion.suggestion_id for suggestion in result.suggestions
    }


@pytest.mark.parametrize(
    ("target_key", "subgroup_size"),
    [
        ("gfx942", 64),
        ("gfx1250-a0", 32),
    ],
)
def test_fragment_packet_expansion_cites_source_packets_and_pressure(
    target_key: str,
    subgroup_size: int,
) -> None:
    report = _compile_report(
        target_key=target_key,
        subgroup_size=subgroup_size,
    )
    _add_fragment_packet_evidence(report, expanded=True)
    _add_operand_bank_materialization_evidence(report, packet_count=12)
    document = parse_compile_report(report)

    result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(
        document,
        CompileReportSuggestionOptions(include_experimental=True),
    )

    suggestion = next(
        suggestion
        for suggestion in result.suggestions
        if suggestion.suggestion_id == "amdgpu.fragment_packet_expansion"
    )
    assert suggestion.confidence == "experimental"
    assert "Inspect allocator placement before changing fragment storage" in (
        suggestion.action
    )
    assert "vector.fragment.load/weights/global" in suggestion.action
    assert "128 scalar load packets" in suggestion.action
    assert "strided_d16_packed_b16_fragment_load" in suggestion.action
    assert "12 target-created operand-bank materialization packets" in (
        suggestion.action
    )
    assert "Check whether those repairs coincide" in suggestion.action
    assert f"{subgroup_size} lanes x 2 B/lane" in suggestion.action
    assert "dense, overlapping coverage" in suggestion.action
    assert "32 B unique in a 32 B span" in suggestion.action
    assert "maximum gap 0 B" in suggestion.action
    assert "maximum adjacent-lane delta 30 B" in suggestion.action
    assert "Packet width alone is not an objective" in suggestion.action
    assert "reject a wider variant that worsens dispersion" in suggestion.action
    evidence = {item.path: item.value for item in suggestion.evidence}
    assert evidence["source_low.memory.argument_packets[0].scalar_packet_count"] == 64
    assert evidence["source_low.memory.argument_packets[1].scalar_packet_count"] == 64
    assert (
        evidence["source_low.memory.argument_packets[0].contiguous_vector_packet_count"]
        == 0
    )
    assert evidence["entries.rows[0].wait_plan.full_drain_count"] == 18
    assert evidence["entries.rows[0].static_instruction_mix.register_move_count"] == 64
    assert evidence["entries.rows[0].move_causes.causes[0].packet_count"] == 12
    assert evidence["entries.rows[0].move_causes.causes[0].unit_count"] == 12
    assert (
        evidence[
            "source_low.memory.subgroup_access_groups[0].access.address."
            "per_lane_packet_bytes"
        ]
        == 2
    )
    assert (
        evidence[
            "source_low.memory.subgroup_access_groups[0].access.geometry."
            "interval_coverage"
        ]
        == "dense"
    )
    assert (
        evidence[
            "source_low.memory.subgroup_access_groups[0].access.geometry."
            "subgroup_span_bytes"
        ]
        == 32
    )


def test_fragment_packet_expansion_reports_mixed_packet_shapes_honestly() -> None:
    report = _compile_report(target_key="gfx942", subgroup_size=64)
    _add_fragment_packet_evidence(report, expanded=True)
    memory = report["source_low"]["memory"]
    memory["arguments"][0]["scalar_packet_count"] = 96
    memory["arguments"][0]["vector_packet_count"] = 32
    memory["argument_packets"][0]["scalar_packet_count"] = 32
    document = parse_compile_report(report)

    result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(
        document,
        CompileReportSuggestionOptions(include_experimental=True),
    )

    suggestion = next(
        suggestion
        for suggestion in result.suggestions
        if suggestion.suggestion_id == "amdgpu.fragment_packet_expansion"
    )
    assert "96 of 128 load packets as scalar packets" in suggestion.action
    assert "128 scalar load packets" not in suggestion.action


def test_fragment_packet_expansion_without_move_causes_stays_layout_scoped() -> None:
    report = _compile_report()
    _add_fragment_packet_evidence(report, expanded=True)
    document = parse_compile_report(report)

    result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(
        document,
        CompileReportSuggestionOptions(include_experimental=True),
    )

    suggestion = next(
        suggestion
        for suggestion in result.suggestions
        if suggestion.suggestion_id == "amdgpu.fragment_packet_expansion"
    )
    assert "Run a bounded operand-layout and packing experiment" in (suggestion.action)
    assert "allocator placement" not in suggestion.action


def test_contiguous_fragment_packets_do_not_suggest_expansion() -> None:
    report = _compile_report()
    _add_fragment_packet_evidence(report, expanded=False)
    document = parse_compile_report(report)

    result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(
        document,
        CompileReportSuggestionOptions(include_experimental=True),
    )

    assert "amdgpu.fragment_packet_expansion" not in {
        suggestion.suggestion_id for suggestion in result.suggestions
    }


def test_fragment_packet_expansion_is_explicitly_opt_in() -> None:
    report = _compile_report()
    _add_fragment_packet_evidence(report, expanded=True)
    document = parse_compile_report(report)

    result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(document)

    assert "amdgpu.fragment_packet_expansion" not in {
        suggestion.suggestion_id for suggestion in result.suggestions
    }


@pytest.mark.parametrize("wave_proof", [None, "unknown"])
def test_fragment_packet_expansion_requires_exact_wave_geometry(
    wave_proof: str | None,
) -> None:
    report = _compile_report()
    _add_fragment_packet_evidence(
        report,
        expanded=True,
        wave_proof=wave_proof,
    )
    document = parse_compile_report(report)

    result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(
        document,
        CompileReportSuggestionOptions(include_experimental=True),
    )

    assert "amdgpu.fragment_packet_expansion" not in {
        suggestion.suggestion_id for suggestion in result.suggestions
    }


def test_gapped_fragment_packet_shape_does_not_recommend_widening() -> None:
    report = _compile_report()
    _add_fragment_packet_evidence(
        report,
        expanded=True,
        wave_coverage="gapped",
    )
    document = parse_compile_report(report)

    result = AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER.suggest(
        document,
        CompileReportSuggestionOptions(include_experimental=True),
    )

    suggestion = next(
        suggestion
        for suggestion in result.suggestions
        if suggestion.suggestion_id == "amdgpu.fragment_packet_expansion"
    )
    assert "gapped, overlapping coverage" in suggestion.action
    assert "32 B unique in a 962 B span" in suggestion.action
    assert "maximum gap 62 B" in suggestion.action
    assert "maximum adjacent-lane delta 960 B" in suggestion.action
    assert "reject a wider variant that worsens dispersion" in suggestion.action


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
