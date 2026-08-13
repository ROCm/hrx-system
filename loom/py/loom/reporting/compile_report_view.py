# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-neutral views and identity-aware diffs for Loom compile reports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from loom.reporting.compile_report import (
    COMPILE_REPORT_SCHEMA_VERSION,
    CompileReportComparisonMode,
    CompileReportDocument,
    CompileReportEntryIdentity,
    compile_report_entry_identity,
    match_compile_report_entries,
    report_common_identity_json,
    report_identity_json,
    report_target_identity_json,
)
from loom.reporting.compile_report_bank_service import (
    append_bank_service_diff_text,
    append_bank_service_show_text,
    build_bank_service_diff,
    build_bank_service_show,
)
from loom.reporting.compile_report_capabilities import (
    append_target_capability_diff_text,
    build_target_capability_diff,
)
from loom.reporting.compile_report_execution_economics import (
    append_execution_economics_diff_text,
    append_execution_economics_show_text,
    build_execution_economics_diff,
    build_execution_economics_show,
    execution_economics_diff_has_changes,
)
from loom.reporting.compile_report_move_causes import (
    append_move_cause_diff_text,
    append_move_cause_show_text,
    build_move_cause_diff,
    build_move_cause_show,
    move_cause_diff_has_changes,
)
from loom.reporting.compile_report_subgroup_access import (
    append_subgroup_access_diff_text,
    append_subgroup_access_show_text,
    build_subgroup_access_diff,
    build_subgroup_access_show,
)
from loom.reporting.compile_report_workload import (
    append_workload_diff_text,
    append_workload_show_text,
    build_workload_diff,
    build_workload_show,
    workload_diff_has_changes,
)

SHOW_KIND = "loom.compile_report.show"
DIFF_KIND = "loom.compile_report.diff"

_MISSING = object()


class EvidenceClass(Enum):
    """Provenance class for a displayed report value."""

    ARTIFACT_FACT = "artifact_fact"
    COMPILER_ANALYSIS = "compiler_analysis"


@dataclass(frozen=True)
class MetricSpec:
    """One stable scalar selected from a version-zero entry."""

    key: str
    label: str
    path: str
    evidence: EvidenceClass
    unit: str | None = None


def _artifact(key: str, label: str, path: str, unit: str | None = None) -> MetricSpec:
    return MetricSpec(key, label, path, EvidenceClass.ARTIFACT_FACT, unit)


def _analysis(key: str, label: str, path: str, unit: str | None = None) -> MetricSpec:
    return MetricSpec(key, label, path, EvidenceClass.COMPILER_ANALYSIS, unit)


_METRIC_SPECS = (
    _artifact("instruction_count", "instructions", "instruction_count"),
    _artifact(
        "body_instruction_count",
        "body instructions",
        "body_instruction_count",
    ),
    _artifact(
        "entry_instruction_count",
        "entry-envelope instructions",
        "entry_instruction_count",
    ),
    _artifact(
        "coissued_instruction_count",
        "native coissued instructions",
        "coissued_instruction_count",
    ),
    _artifact(
        "coissued_component_count",
        "coissued semantic components",
        "coissued_component_count",
    ),
    _artifact("code_byte_count", "code bytes", "code_byte_count", "bytes"),
    _artifact(
        "code_storage_byte_count",
        "code storage bytes",
        "code_storage_byte_count",
        "bytes",
    ),
    _artifact("local_memory_bytes", "local memory", "local_memory_bytes", "bytes"),
    _artifact(
        "private_memory_bytes",
        "private memory",
        "private_memory_bytes",
        "bytes",
    ),
    _artifact(
        "scalar_alu_count",
        "scalar ALU instructions",
        "static_instruction_mix.scalar_alu_count",
    ),
    _artifact(
        "unclassified_instruction_count",
        "unclassified instructions",
        "static_instruction_mix.unknown_count",
    ),
    _artifact(
        "vector_alu_count",
        "vector ALU instructions",
        "static_instruction_mix.vector_alu_count",
    ),
    _artifact(
        "matrix_count",
        "matrix instructions",
        "static_instruction_mix.matrix_count",
    ),
    _artifact("mfma_count", "MFMA instructions", "static_instruction_mix.mfma_count"),
    _artifact(
        "smfmac_count",
        "SMFMAC instructions",
        "static_instruction_mix.smfmac_count",
    ),
    _artifact("wmma_count", "WMMA instructions", "static_instruction_mix.wmma_count"),
    _artifact(
        "swmmac_count",
        "SWMMAC instructions",
        "static_instruction_mix.swmmac_count",
    ),
    _artifact("dot_count", "dot instructions", "static_instruction_mix.dot_count"),
    _artifact(
        "global_load_count",
        "global loads",
        "static_instruction_mix.global_load_count",
    ),
    _artifact(
        "global_store_count",
        "global stores",
        "static_instruction_mix.global_store_count",
    ),
    _artifact(
        "buffer_load_count",
        "buffer loads",
        "static_instruction_mix.buffer_load_count",
    ),
    _artifact(
        "buffer_store_count",
        "buffer stores",
        "static_instruction_mix.buffer_store_count",
    ),
    _artifact(
        "local_memory_count",
        "local-memory instructions",
        "static_instruction_mix.local_memory_count",
    ),
    _artifact(
        "conversion_count",
        "conversion instructions",
        "static_instruction_mix.conversion_count",
    ),
    _artifact(
        "register_move_count",
        "register moves",
        "static_instruction_mix.register_move_count",
    ),
    _artifact("barrier_count", "barriers", "static_instruction_mix.barrier_count"),
    _artifact("branch_count", "branches", "static_instruction_mix.branch_count"),
    _analysis(
        "scalar_register_count",
        "final scalar registers",
        "target_resources.scalar.final.register_count",
    ),
    _analysis(
        "vector_register_count",
        "final vector registers",
        "target_resources.vector.final.register_count",
    ),
    _analysis(
        "scalar_pressure_peak",
        "scheduled scalar pressure",
        "target_resources.scalar.scheduled_pressure.peak_live_units",
    ),
    _analysis(
        "vector_pressure_peak",
        "scheduled vector pressure",
        "target_resources.vector.scheduled_pressure.peak_live_units",
    ),
    _analysis(
        "resident_subgroups_per_simd",
        "resident subgroups per SIMD",
        "target_resources.resident_subgroups_per_simd",
    ),
    _analysis(
        "occupancy_percent",
        "modeled occupancy",
        "target_resources.occupancy_percent",
        "percent",
    ),
    _analysis(
        "limiting_resource",
        "occupancy limit",
        "target_resources.limiting_resource",
    ),
    _analysis(
        "residency_current_tier",
        "residency tier",
        "target_resources.residency.current_tier",
    ),
    _analysis(
        "residency_next_better_tier",
        "next better residency tier",
        "target_resources.residency.next_better_tier",
    ),
    _analysis(
        "residency_limiting_resource",
        "residency limiting resource",
        "target_resources.residency.unique_limiting_resource.name",
    ),
    _analysis(
        "residency_reduction_to_next_better_tier",
        "units to next better tier",
        "target_resources.residency.unique_limiting_resource."
        "reduction_units_to_next_better_tier",
    ),
    _analysis(
        "residency_additional_units_to_next_worse_tier",
        "units to next worse tier",
        "target_resources.residency.unique_limiting_resource."
        "next_worse.additional_units",
    ),
    _analysis("schedule_node_count", "schedule nodes", "schedule_node_count"),
    _analysis(
        "schedule_dependency_count",
        "schedule dependencies",
        "schedule_dependency_count",
    ),
    _analysis(
        "schedule_hazard_gap_count",
        "schedule hazard gaps",
        "schedule_hazard_gap_count",
    ),
    _analysis(
        "allocation_spill_count",
        "planned spills",
        "allocation_spill_count",
    ),
    _analysis(
        "materialized_copy_count",
        "materialized copies",
        "allocation_materialized_copy_count",
    ),
    _analysis(
        "materialized_spill_storage_bytes",
        "spill storage",
        "allocation_materialized_spill_storage_bytes",
        "bytes",
    ),
    _analysis(
        "materialized_reload_bytes",
        "reload traffic",
        "allocation_materialized_reload_bytes",
        "bytes",
    ),
    _analysis("wait_action_count", "wait actions", "wait_plan.action_count"),
    _analysis(
        "explicit_wait_action_count",
        "explicit wait actions",
        "wait_plan.explicit_action_count",
    ),
    _analysis(
        "planned_wait_action_count",
        "target-planned wait actions",
        "wait_plan.planned_action_count",
    ),
    _analysis("full_drain_count", "full drains", "wait_plan.full_drain_count"),
    _analysis("partial_wait_count", "partial waits", "wait_plan.partial_wait_count"),
    _analysis(
        "wait_drained_packet_count",
        "packets drained by waits",
        "wait_plan.drained_count",
    ),
    _analysis(
        "wait_maximum_drained_packet_count",
        "maximum packets drained by one wait",
        "wait_plan.max_drained_count",
    ),
    _analysis(
        "wait_maximum_outstanding_packet_count",
        "maximum packets outstanding before a wait",
        "wait_plan.max_outstanding_before",
    ),
    _analysis(
        "wait_maximum_full_drain_outstanding_packet_count",
        "maximum packets outstanding before a full drain",
        "wait_plan.max_full_drain_outstanding_before",
    ),
    _analysis(
        "target_insertion_static_packet_count",
        "target-inserted static packets",
        "target_insertions.static_packet_count",
    ),
    _analysis(
        "target_insertion_exact_dynamic_packet_count",
        "target insertions with exact dynamic counts",
        "target_insertions.exact_dynamic_packet_count",
    ),
    _analysis(
        "target_insertion_unknown_dynamic_packet_count",
        "target insertions with unknown dynamic counts",
        "target_insertions.unknown_dynamic_packet_count",
    ),
    _analysis(
        "target_insertion_dynamic_packet_count",
        "target-inserted dynamic packets",
        "target_insertions.dynamic_packet_count",
    ),
)


def build_compile_report_show(
    document: CompileReportDocument,
) -> dict[str, object]:
    """Builds a deterministic target-neutral report view."""
    report_workload = build_workload_show(
        document.report.get("workload"),
        document.report.get("target_resources"),
        document.source,
    )
    view: dict[str, object] = {
        "kind": SHOW_KIND,
        "schema_version": COMPILE_REPORT_SCHEMA_VERSION,
        "missing_evidence": "omitted_metrics_are_unavailable",
        "source": document.source,
        "container_kind": document.container_kind,
        "status": {
            "code": document.status_code,
            "name": document.status_name,
        },
        "identity": report_identity_json(document),
        "workload": report_workload,
        "entries": [
            _show_entry_json(
                entry,
                compile_report_entry_identity(entry),
                report_workload,
                document.report.get("workload"),
                document.report.get("target_resources"),
                f"{document.source}.entries.rows[{entry['index']}]",
            )
            for entry in document.entries
        ],
    }
    bank_service = build_bank_service_show(document)
    if bank_service is not None:
        view["bank_service"] = bank_service
    subgroup_access = build_subgroup_access_show(document)
    if subgroup_access is not None:
        view["subgroup_access"] = subgroup_access
    if document.envelope_context:
        view["envelope_context"] = dict(document.envelope_context)
    return view


def build_compile_report_diff(
    baseline: CompileReportDocument,
    candidate: CompileReportDocument,
    comparison_mode: CompileReportComparisonMode = CompileReportComparisonMode.EXACT,
    *,
    force: bool = False,
) -> dict[str, object]:
    """Builds a deterministic diff under the selected identity contract."""
    match = match_compile_report_entries(
        baseline,
        candidate,
        comparison_mode,
        force=force,
    )
    baseline_report_workload = build_workload_show(
        baseline.report.get("workload"),
        baseline.report.get("target_resources"),
        baseline.source,
    )
    candidate_report_workload = build_workload_show(
        candidate.report.get("workload"),
        candidate.report.get("target_resources"),
        candidate.source,
    )
    report_workload = build_workload_diff(
        baseline_report_workload, candidate_report_workload
    )
    entries = []
    unchanged_entry_count = 0
    for pair in match.pairs:
        baseline_entry_source = (
            f"{baseline.source}.entries.rows[{pair.baseline['index']}]"
        )
        candidate_entry_source = (
            f"{candidate.source}.entries.rows[{pair.candidate['index']}]"
        )
        artifact_facts = _diff_metrics(
            pair.baseline,
            pair.candidate,
            EvidenceClass.ARTIFACT_FACT,
        )
        compiler_analysis = _diff_metrics(
            pair.baseline,
            pair.candidate,
            EvidenceClass.COMPILER_ANALYSIS,
        )
        baseline_entry_workload = build_workload_show(
            pair.baseline.get("workload", baseline.report.get("workload")),
            pair.baseline.get(
                "target_resources", baseline.report.get("target_resources")
            ),
            baseline_entry_source,
        )
        candidate_entry_workload = build_workload_show(
            pair.candidate.get("workload", candidate.report.get("workload")),
            pair.candidate.get(
                "target_resources", candidate.report.get("target_resources")
            ),
            candidate_entry_source,
        )
        entry_workload = None
        if (
            baseline_entry_workload != baseline_report_workload
            or candidate_entry_workload != candidate_report_workload
        ):
            entry_workload = build_workload_diff(
                baseline_entry_workload, candidate_entry_workload
            )
        execution_economics = build_execution_economics_diff(
            build_execution_economics_show(
                pair.baseline, baseline_entry_workload, baseline_entry_source
            ),
            build_execution_economics_show(
                pair.candidate, candidate_entry_workload, candidate_entry_source
            ),
        )
        move_causes = build_move_cause_diff(
            pair.baseline,
            pair.candidate,
            baseline_entry_source,
            candidate_entry_source,
        )
        if (
            not _diff_group_has_changes(artifact_facts)
            and not _diff_group_has_changes(compiler_analysis)
            and not workload_diff_has_changes(entry_workload)
            and not execution_economics_diff_has_changes(execution_economics)
            and not move_cause_diff_has_changes(move_causes)
        ):
            unchanged_entry_count += 1
            continue
        entry_view: dict[str, object] = {
            "artifact_facts": artifact_facts,
            "execution_economics": execution_economics,
            "compiler_analysis": compiler_analysis,
        }
        if workload_diff_has_changes(entry_workload):
            entry_view["workload"] = entry_workload
        if move_causes is not None:
            entry_view["move_causes"] = move_causes
        if force:
            entry_view["identities"] = {
                "baseline": pair.baseline_identity.to_json_object(),
                "candidate": pair.candidate_identity.to_json_object(),
            }
        else:
            entry_view["identity"] = pair.baseline_identity.to_json_object()
        entries.append(entry_view)
    view: dict[str, object] = {
        "kind": DIFF_KIND,
        "schema_version": COMPILE_REPORT_SCHEMA_VERSION,
        "missing_evidence": "omitted_metrics_are_unavailable",
        "baseline_source": baseline.source,
        "candidate_source": candidate.source,
        "comparison_mode": comparison_mode.value,
        "changed_entry_count": len(entries),
        "unchanged_entry_count": unchanged_entry_count,
        "entries": entries,
    }
    if force:
        pair = match.pairs[0]
        view["forced"] = True
        view["identities"] = {
            "baseline": report_identity_json(baseline),
            "candidate": report_identity_json(candidate),
        }
        view["entry_identities"] = {
            "baseline": pair.baseline_identity.to_json_object(),
            "candidate": pair.candidate_identity.to_json_object(),
        }
        view["identity_mismatches"] = [
            mismatch.to_json_object() for mismatch in match.identity_mismatches
        ]
    else:
        view["identity"] = report_common_identity_json(baseline, comparison_mode)
    if workload_diff_has_changes(report_workload):
        view["workload"] = report_workload
    if comparison_mode is CompileReportComparisonMode.TARGET and not force:
        view["targets"] = {
            "baseline": report_target_identity_json(baseline),
            "candidate": report_target_identity_json(candidate),
        }
    bank_service = build_bank_service_diff(baseline, candidate)
    if bank_service is not None:
        view["bank_service"] = bank_service
    subgroup_access = build_subgroup_access_diff(
        baseline,
        candidate,
        entry_function_pairs=(
            tuple(
                (
                    pair.baseline_identity.function,
                    pair.candidate_identity.function,
                )
                for pair in match.pairs
            )
            if force
            else ()
        ),
    )
    if subgroup_access is not None:
        view["subgroup_access"] = subgroup_access
    if comparison_mode is CompileReportComparisonMode.TARGET and not force:
        target_capabilities = build_target_capability_diff(baseline, candidate)
        if target_capabilities is not None:
            view["target_capabilities"] = target_capabilities
    return view


def format_compile_report_show_text(view: dict[str, object]) -> str:
    """Formats a show view for humans and agent prompts."""
    identity = _expect_dict(view["identity"])
    status = _expect_dict(view["status"])
    lines = [
        "Loom compile report",
        f"  source: {view['source']}",
        (
            f"  schema: loom.compile_report v{identity['schema_version']} "
            f"({identity['mode']}, {view['container_kind']})"
        ),
        f"  status: {status['name']} ({status['code']})",
        f"  artifact: {_format_artifact(identity)}",
        f"  target: {_format_target(identity)}",
        f"  specialization: {_format_specialization(identity)}",
    ]
    envelope_context = _expect_dict(view.get("envelope_context", {}))
    if envelope_context:
        lines.append(f"  benchmark: {_format_context(envelope_context)}")
    config_bindings = _expect_list(identity.get("config_bindings", []))
    if config_bindings:
        lines.append("  config:")
        for binding_value in config_bindings:
            binding = _expect_dict(binding_value)
            lines.append(f"    {binding['key']} = {binding['value']}")
    lines.append("")
    append_workload_show_text(lines, _expect_dict(view["workload"]))

    entries = _expect_list(view["entries"])
    if not entries:
        lines.append("  entries: none")
    for entry_value in entries:
        entry = _expect_dict(entry_value)
        entry_identity = _expect_dict(entry["identity"])
        lines.append("")
        lines.append(f"Entry {_entry_display_name(entry_identity)}")
        entry_workload = entry.get("workload")
        if isinstance(entry_workload, dict):
            append_workload_show_text(lines, entry_workload, indent="  ")
        _append_metric_group(
            lines,
            "Artifact facts",
            _expect_dict(entry["artifact_facts"]),
            EvidenceClass.ARTIFACT_FACT,
        )
        append_execution_economics_show_text(
            lines, _expect_dict(entry["execution_economics"])
        )
        _append_metric_group(
            lines,
            "Compiler analysis",
            _expect_dict(entry["compiler_analysis"]),
            EvidenceClass.COMPILER_ANALYSIS,
        )
        move_causes = entry.get("move_causes")
        if isinstance(move_causes, dict):
            append_move_cause_show_text(lines, move_causes)
    subgroup_access = view.get("subgroup_access")
    if isinstance(subgroup_access, dict):
        append_subgroup_access_show_text(lines, subgroup_access)
    bank_service = view.get("bank_service")
    if isinstance(bank_service, dict):
        append_bank_service_show_text(lines, bank_service)
    return "\n".join(lines) + "\n"


def format_compile_report_diff_text(view: dict[str, object]) -> str:
    """Formats a diff for humans and agent prompts."""
    lines = [
        "Loom compile report diff",
        f"  baseline: {view['baseline_source']}",
        f"  candidate: {view['candidate_source']}",
    ]
    if view.get("forced") is True:
        identities = _expect_dict(view["identities"])
        baseline_identity = _expect_dict(identities["baseline"])
        candidate_identity = _expect_dict(identities["candidate"])
        lines.extend(
            (
                "  comparison: forced single-entry observation",
                "  warning: identity contract bypassed; deltas are not causal",
                f"  baseline target: {_format_target(baseline_identity)}",
                (
                    "  baseline specialization: "
                    f"{_format_specialization(baseline_identity)}"
                ),
                f"  candidate target: {_format_target(candidate_identity)}",
                (
                    "  candidate specialization: "
                    f"{_format_specialization(candidate_identity)}"
                ),
            )
        )
        mismatches = _expect_list(view["identity_mismatches"])
        lines.append(f"  identity mismatches: {len(mismatches)}")
        workload_mismatch_count = 0
        for mismatch_value in mismatches:
            mismatch = _expect_dict(mismatch_value)
            if str(mismatch["path"]).endswith(".workload"):
                workload_mismatch_count += 1
                continue
            lines.append(
                f"    {mismatch['path']}: {mismatch['baseline']!r} != "
                f"{mismatch['candidate']!r}"
            )
        if workload_mismatch_count:
            lines.append(
                f"    workload mismatches: {workload_mismatch_count}; expanded below"
            )
    elif view.get("comparison_mode") == CompileReportComparisonMode.TARGET.value:
        targets = _expect_dict(view["targets"])
        baseline_target = _expect_dict(targets["baseline"])
        candidate_target = _expect_dict(targets["candidate"])
        lines.extend(
            (
                "  comparison: target specialization",
                f"  baseline target: {_format_target(baseline_target)}",
                (
                    "  baseline specialization: "
                    f"{_format_specialization(baseline_target)}"
                ),
                f"  candidate target: {_format_target(candidate_target)}",
                (
                    "  candidate specialization: "
                    f"{_format_specialization(candidate_target)}"
                ),
            )
        )
    else:
        identity = _expect_dict(view["identity"])
        lines.extend(
            (
                f"  target: {_format_target(identity)}",
                f"  specialization: {_format_specialization(identity)}",
            )
        )
    report_workload = view.get("workload")
    if isinstance(report_workload, dict):
        lines.append("")
        append_workload_diff_text(lines, report_workload)
    lines.append(
        f"  entries: {view['changed_entry_count']} changed, "
        f"{view['unchanged_entry_count']} unchanged"
    )
    if (
        view.get("comparison_mode") == CompileReportComparisonMode.TARGET.value
        and view["changed_entry_count"] == 0
        and view["unchanged_entry_count"] != 0
    ):
        lines.append("  reported entry evidence: unchanged")
    for entry_value in _expect_list(view["entries"]):
        entry = _expect_dict(entry_value)
        if view.get("forced") is True:
            entry_identities = _expect_dict(entry["identities"])
            baseline_entry_identity = _expect_dict(entry_identities["baseline"])
            candidate_entry_identity = _expect_dict(entry_identities["candidate"])
            baseline_name = _entry_display_name(baseline_entry_identity)
            candidate_name = _entry_display_name(candidate_entry_identity)
            entry_name = (
                baseline_name
                if baseline_name == candidate_name
                else f"{baseline_name} -> {candidate_name}"
            )
        else:
            entry_identity = _expect_dict(entry["identity"])
            entry_name = _entry_display_name(entry_identity)
        lines.append("")
        lines.append(f"Entry {entry_name}")
        entry_workload = entry.get("workload")
        if isinstance(entry_workload, dict):
            append_workload_diff_text(lines, entry_workload, indent="  ")
        _append_diff_group(
            lines,
            "Artifact facts",
            _expect_dict(entry["artifact_facts"]),
            EvidenceClass.ARTIFACT_FACT,
        )
        append_execution_economics_diff_text(
            lines, _expect_dict(entry["execution_economics"])
        )
        _append_diff_group(
            lines,
            "Compiler analysis",
            _expect_dict(entry["compiler_analysis"]),
            EvidenceClass.COMPILER_ANALYSIS,
        )
        move_causes = entry.get("move_causes")
        if isinstance(move_causes, dict):
            append_move_cause_diff_text(lines, move_causes)
    subgroup_access = view.get("subgroup_access")
    if isinstance(subgroup_access, dict):
        append_subgroup_access_diff_text(lines, subgroup_access)
    bank_service = view.get("bank_service")
    if isinstance(bank_service, dict):
        append_bank_service_diff_text(lines, bank_service)
    target_capabilities = view.get("target_capabilities")
    if isinstance(target_capabilities, dict):
        append_target_capability_diff_text(lines, target_capabilities)
    return "\n".join(lines) + "\n"


def _show_entry_json(
    entry: dict[str, object],
    identity: CompileReportEntryIdentity,
    report_workload: dict[str, object],
    report_workload_value: object,
    report_target_resources_value: object,
    source: str,
) -> dict[str, object]:
    entry_workload = build_workload_show(
        entry.get("workload", report_workload_value),
        entry.get("target_resources", report_target_resources_value),
        source,
    )
    view: dict[str, object] = {
        "identity": identity.to_json_object(),
        "artifact_facts": _show_metrics(entry, EvidenceClass.ARTIFACT_FACT),
        "execution_economics": build_execution_economics_show(
            entry, entry_workload, source
        ),
        "compiler_analysis": _show_metrics(entry, EvidenceClass.COMPILER_ANALYSIS),
    }
    if entry_workload != report_workload:
        view["workload"] = entry_workload
    move_causes = build_move_cause_show(entry, source)
    if move_causes is not None:
        view["move_causes"] = move_causes
    return view


def _show_metrics(
    entry: dict[str, object], evidence: EvidenceClass
) -> dict[str, object]:
    metrics = {}
    for spec in _METRIC_SPECS:
        if spec.evidence is not evidence:
            continue
        value = _lookup(entry, spec.path)
        if value is not _MISSING and value is not None:
            metrics[spec.key] = value
    return metrics


def _diff_metrics(
    baseline: dict[str, object],
    candidate: dict[str, object],
    evidence: EvidenceClass,
) -> dict[str, object]:
    changed: dict[str, object] = {}
    incomplete: dict[str, object] = {}
    unchanged_count = 0
    unavailable_count = 0
    for spec in _METRIC_SPECS:
        if spec.evidence is not evidence:
            continue
        baseline_value = _lookup(baseline, spec.path)
        candidate_value = _lookup(candidate, spec.path)
        if baseline_value is None:
            baseline_value = _MISSING
        if candidate_value is None:
            candidate_value = _MISSING
        if baseline_value is _MISSING and candidate_value is _MISSING:
            unavailable_count += 1
            continue
        if baseline_value is _MISSING or candidate_value is _MISSING:
            incomplete[spec.key] = {
                "baseline": None if baseline_value is _MISSING else baseline_value,
                "candidate": None if candidate_value is _MISSING else candidate_value,
            }
            continue
        if baseline_value == candidate_value:
            unchanged_count += 1
            continue
        metric: dict[str, object] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
        }
        if _is_number(baseline_value) and _is_number(candidate_value):
            delta = candidate_value - baseline_value
            metric["delta"] = delta
            if baseline_value != 0:
                metric["change_percent"] = delta * 100.0 / baseline_value
        changed[spec.key] = metric
    return {
        "changed": changed,
        "incomplete": incomplete,
        "unchanged_count": unchanged_count,
        "unavailable_count": unavailable_count,
    }


def _diff_group_has_changes(group: dict[str, object]) -> bool:
    return bool(group["changed"] or group["incomplete"])


def _lookup(root: object, path: str) -> object:
    value = root
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            return _MISSING
        value = value[component]
    return value


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _append_metric_group(
    lines: list[str],
    heading: str,
    metric_values: dict[str, object],
    evidence: EvidenceClass,
) -> None:
    lines.append(f"  {heading}")
    for spec in _METRIC_SPECS:
        if spec.evidence is not evidence:
            continue
        value = _format_value(metric_values.get(spec.key), spec.unit)
        lines.append(f"    {spec.label}: {value}")


def _append_diff_group(
    lines: list[str],
    heading: str,
    metric_values: dict[str, object],
    evidence: EvidenceClass,
) -> None:
    changed = _expect_dict(metric_values["changed"])
    incomplete = _expect_dict(metric_values["incomplete"])
    lines.append(f"  {heading}")
    if not changed and not incomplete:
        lines.append("    no metric changes")
    for spec in _METRIC_SPECS:
        if spec.evidence is not evidence:
            continue
        metric_value = changed.get(spec.key, incomplete.get(spec.key))
        if metric_value is None:
            continue
        metric = _expect_dict(metric_value)
        baseline = _format_value(metric["baseline"], spec.unit)
        candidate = _format_value(metric["candidate"], spec.unit)
        suffix = ""
        if "delta" in metric:
            suffix = f", delta {_format_signed_value(metric['delta'], spec.unit)}"
        if "change_percent" in metric:
            suffix += f" ({metric['change_percent']:+.2f}%)"
        lines.append(f"    {spec.label}: {baseline} -> {candidate}{suffix}")
    lines.append(
        f"    unchanged: {metric_values['unchanged_count']}; "
        f"unavailable in both: {metric_values['unavailable_count']}"
    )


def _format_value(value: object, unit: object) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        formatted = f"{value:,}"
    elif isinstance(value, float):
        formatted = f"{value:.4g}"
    else:
        return str(value)
    if unit == "bytes":
        return f"{formatted} B"
    if unit == "percent":
        return f"{formatted}%"
    return formatted


def _format_signed_value(value: object, unit: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        formatted = f"{value:+,}"
    elif isinstance(value, float):
        formatted = f"{value:+.4g}"
    else:
        return str(value)
    if unit == "bytes":
        return f"{formatted} B"
    if unit == "percent":
        return f"{formatted} percentage points"
    return formatted


def _format_target(identity: dict[str, object]) -> str:
    family = identity.get("target_family") or "unavailable"
    key = identity.get("target_key") or "unavailable"
    backend = identity.get("backend")
    suffix = f" via {backend}" if backend else ""
    return f"{family}/{key}{suffix}"


def _format_artifact(identity: dict[str, object]) -> str:
    kind = identity.get("artifact_kind") or "unavailable"
    artifact_format = identity.get("artifact_format")
    suffix = f" ({artifact_format})" if artifact_format else ""
    return f"{kind}{suffix}"


def _format_specialization(identity: dict[str, object]) -> str:
    values = (
        ("bundle", identity.get("target_bundle")),
        ("snapshot", identity.get("target_snapshot")),
        ("config", identity.get("target_config")),
    )
    return ", ".join(f"{label}={value or 'unavailable'}" for label, value in values)


def _format_context(context: dict[str, object]) -> str:
    return ", ".join(f"{key}={value}" for key, value in context.items())


def _entry_display_name(identity: dict[str, object]) -> str:
    return str(identity["name"])


def _expect_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("invalid internal compile report view object")
    return value


def _expect_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("invalid internal compile report view array")
    return value
