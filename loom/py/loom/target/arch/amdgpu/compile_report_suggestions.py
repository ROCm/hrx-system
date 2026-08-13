# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU compile report suggestions grounded in target-info rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from loom.reporting.compile_report import (
    CompileReportDocument,
    CompileReportError,
    compile_report_entry_identity,
)
from loom.reporting.compile_report_move_causes import (
    CompileReportMoveCause,
    parse_compile_report_move_causes,
)
from loom.reporting.compile_report_subgroup_access import (
    build_subgroup_access_show,
)
from loom.reporting.compile_report_suggestions import (
    CompileReportSuggestion,
    CompileReportSuggestionConfidence,
    CompileReportSuggestionEvidence,
    CompileReportSuggestionOptions,
    CompileReportSuggestionResult,
)
from loom.target.arch.amdgpu.lds_bank_service import (
    AMDGPU_LDS_BANK_SERVICE_EVIDENCE_PUBLIC_VENDOR_DOCUMENTATION,
    AMDGPU_LDS_BANK_SERVICE_EVIDENCE_SILICON_CALIBRATED_VENDOR_MODEL,
    AMDGPU_LDS_BANK_SERVICE_EVIDENCE_VENDOR_SOFTWARE_MODEL_UNVALIDATED,
)
from loom.target.arch.amdgpu.target_identity import (
    AmdgpuArtifactTargetKeyError,
    parse_amdgpu_artifact_target_key,
)


class AmdgpuCompileReportSuggestionProvider:
    """Interprets AMDGPU evidence after resolving its structured target."""

    target_family = "AMDGPU"
    provider_name = "amdgpu"

    def suggest(
        self,
        document: CompileReportDocument,
        options: CompileReportSuggestionOptions | None = None,
    ) -> CompileReportSuggestionResult:
        if options is None:
            options = CompileReportSuggestionOptions()
        target_key = document.report.get("target_key")
        if target_key is None:
            return CompileReportSuggestionResult(
                provider_name=self.provider_name,
                unavailable_reason="missing_target_key",
            )
        try:
            target_identity = parse_amdgpu_artifact_target_key(target_key)
        except AmdgpuArtifactTargetKeyError:
            return CompileReportSuggestionResult(
                provider_name=self.provider_name,
                unavailable_reason="invalid_target_key",
            )
        if target_identity is None:
            return CompileReportSuggestionResult(
                provider_name=self.provider_name,
                unavailable_reason="unknown_target_key",
            )

        suggestions = list(_suggest_wait_serialization(document))
        for entry in document.entries:
            entry_index = entry["index"]
            entry_name = compile_report_entry_identity(entry).display_name()
            path_prefix = f"entries.rows[{entry_index}]"
            spill_suggestion = _suggest_spill_traffic(entry, entry_name, path_prefix)
            if spill_suggestion is not None:
                suggestions.append(spill_suggestion)
            else:
                private_memory_suggestion = _suggest_private_memory(
                    entry, entry_name, path_prefix
                )
                if private_memory_suggestion is not None:
                    suggestions.append(private_memory_suggestion)
            residency_suggestion = _suggest_residency_cliff(
                entry, entry_name, path_prefix
            )
            if residency_suggestion is not None:
                suggestions.append(residency_suggestion)
            communication_suggestion = _suggest_single_subgroup_communication(
                entry, entry_name, path_prefix
            )
            if communication_suggestion is not None:
                suggestions.append(communication_suggestion)
            wave_suggestion = _suggest_nondefault_wave_size(
                entry,
                entry_name,
                path_prefix,
                target_identity.processor.wavefront.default_size,
            )
            if wave_suggestion is not None:
                suggestions.append(wave_suggestion)
        suggestions.extend(_suggest_fragment_packet_expansion(document, options))
        suggestions.extend(_suggest_lds_bank_service(document, options))
        return CompileReportSuggestionResult(
            provider_name=self.provider_name,
            unavailable_reason=None,
            suggestions=tuple(suggestions),
        )


AMDGPU_COMPILE_REPORT_SUGGESTION_PROVIDER = AmdgpuCompileReportSuggestionProvider()


def _suggest_wait_serialization(
    document: CompileReportDocument,
) -> tuple[CompileReportSuggestion, ...]:
    rows_value = document.report.get("wait_reason_summary_rows")
    if rows_value is None:
        return ()
    rows = _report_indexed_rows(
        _report_object(rows_value, "wait_reason_summary_rows"),
        "wait_reason_summary_rows",
    )
    suggestions = []
    for position, row in enumerate(rows):
        counter = row.get("counter")
        reason = row.get("reason")
        is_vmem_source_reuse = (
            counter == "vmem_load" and reason == "amdgpu.memory_source_reuse"
        )
        is_lds_ssa_use = counter == "lds" and reason == "amdgpu.ssa_use"
        if not is_vmem_source_reuse and not is_lds_ssa_use:
            continue
        row_path = f"wait_reason_summary_rows.rows[{position}]"
        summary = _report_object(row.get("summary"), f"{row_path}.summary")
        action_count = _report_integer(
            summary.get("action_count"), f"{row_path}.summary.action_count"
        )
        full_drain_count = _report_integer(
            summary.get("full_drain_count"),
            f"{row_path}.summary.full_drain_count",
        )
        # Isolated waits are ordinary scheduling fallout. Surface only repeated
        # full drains that dominate their reason and materially affect the entry.
        if full_drain_count < 8 or full_drain_count * 2 < action_count:
            continue
        function_name = _optional_report_string(
            row.get("function"), f"{row_path}.function"
        )
        entry = _entry_for_function(document, function_name)
        if entry is None:
            continue
        entry_index = entry["index"]
        wait_plan = _object_at(entry, "wait_plan")
        if wait_plan is None:
            continue
        wait_plan_path = f"entries.rows[{entry_index}].wait_plan"
        total_full_drain_count = _report_integer(
            wait_plan.get("full_drain_count"),
            f"{wait_plan_path}.full_drain_count",
        )
        if total_full_drain_count == 0:
            continue
        if is_vmem_source_reuse:
            if full_drain_count * 4 < total_full_drain_count:
                continue
            max_full_drain_outstanding = _report_integer(
                summary.get("max_full_drain_outstanding_before"),
                f"{row_path}.summary.max_full_drain_outstanding_before",
            )
            suggestion_id = "amdgpu.vmem_source_reuse_serialization"
            action = (
                "Reduce global-load source-state turnover by consolidating "
                "or staging loads, preserving independent address state, "
                "or scheduling more work before source-register reuse; "
                "then require memory-source full drains to fall before "
                "benchmarking."
            )
            reason_evidence = (
                CompileReportSuggestionEvidence(
                    path=(f"{row_path}.summary.max_full_drain_outstanding_before"),
                    value=max_full_drain_outstanding,
                ),
            )
        else:
            if full_drain_count * 5 < total_full_drain_count:
                continue
            partial_wait_count = _report_integer(
                summary.get("partial_wait_count"),
                f"{row_path}.summary.partial_wait_count",
            )
            max_outstanding = _report_integer(
                summary.get("max_outstanding_before"),
                f"{row_path}.summary.max_outstanding_before",
            )
            suggestion_id = "amdgpu.lds_ssa_use_serialization"
            action = (
                "Issue independent LDS or DS producers before consuming their "
                "SSA results so waits can remain partial; then require "
                "LDS SSA-use full drains to fall before benchmarking."
            )
            reason_evidence = (
                CompileReportSuggestionEvidence(
                    path=f"{row_path}.summary.partial_wait_count",
                    value=partial_wait_count,
                ),
                CompileReportSuggestionEvidence(
                    path=f"{row_path}.summary.max_outstanding_before",
                    value=max_outstanding,
                ),
            )
        suggestions.append(
            CompileReportSuggestion(
                suggestion_id=suggestion_id,
                entry_name=compile_report_entry_identity(entry).display_name(),
                action=action,
                evidence=(
                    CompileReportSuggestionEvidence(
                        path=f"{row_path}.summary.action_count",
                        value=action_count,
                    ),
                    CompileReportSuggestionEvidence(
                        path=f"{row_path}.summary.full_drain_count",
                        value=full_drain_count,
                    ),
                    *reason_evidence,
                    CompileReportSuggestionEvidence(
                        path=(f"{wait_plan_path}.full_drain_count"),
                        value=total_full_drain_count,
                    ),
                ),
            )
        )
    return tuple(suggestions)


def _suggest_spill_traffic(
    entry: dict[str, object],
    entry_name: str,
    path_prefix: str,
) -> CompileReportSuggestion | None:
    metrics = (
        ("allocation_spill_count", "allocation_spill_count"),
        (
            "allocation_materialized_spill_storage_bytes",
            "allocation_materialized_spill_storage_bytes",
        ),
        (
            "allocation_materialized_reload_bytes",
            "allocation_materialized_reload_bytes",
        ),
    )
    evidence = tuple(
        CompileReportSuggestionEvidence(
            path=f"{path_prefix}.{path}",
            value=value,
        )
        for field, path in metrics
        if (value := _integer(entry.get(field))) is not None and value > 0
    )
    if not evidence:
        return None
    return CompileReportSuggestion(
        suggestion_id="amdgpu.spill_traffic",
        entry_name=entry_name,
        action=(
            "Shorten live ranges or reduce fragment state, then require spill "
            "and reload evidence to fall before benchmarking."
        ),
        evidence=evidence,
    )


def _suggest_private_memory(
    entry: dict[str, object],
    entry_name: str,
    path_prefix: str,
) -> CompileReportSuggestion | None:
    private_memory_bytes = _integer(entry.get("private_memory_bytes"))
    if private_memory_bytes is None or private_memory_bytes == 0:
        return None
    return CompileReportSuggestion(
        suggestion_id="amdgpu.private_memory",
        entry_name=entry_name,
        action=(
            "Reduce per-workitem private storage and inspect the emitted scratch "
            "traffic before benchmarking."
        ),
        evidence=(
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.private_memory_bytes",
                value=private_memory_bytes,
            ),
        ),
    )


def _suggest_residency_cliff(
    entry: dict[str, object],
    entry_name: str,
    path_prefix: str,
) -> CompileReportSuggestion | None:
    residency = _object_at(entry, "target_resources", "residency")
    limiting_resource = _object_at(
        entry,
        "target_resources",
        "residency",
        "unique_limiting_resource",
    )
    if residency is None or limiting_resource is None:
        return None
    current_tier = _integer(residency.get("current_tier"))
    next_better_tier = _integer(residency.get("next_better_tier"))
    reduction = _integer(limiting_resource.get("reduction_units_to_next_better_tier"))
    resource_name = limiting_resource.get("name")
    if (
        current_tier is None
        or next_better_tier is None
        or next_better_tier <= current_tier
        or reduction is None
        or reduction == 0
        or not isinstance(resource_name, str)
    ):
        return None
    return CompileReportSuggestion(
        suggestion_id="amdgpu.residency_cliff",
        entry_name=entry_name,
        action=(
            f"Reduce {resource_name} use by at least {reduction} units, then "
            f"recompile and benchmark the predicted tier {next_better_tier}."
        ),
        evidence=(
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.target_resources.residency.current_tier",
                value=current_tier,
            ),
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.target_resources.residency.next_better_tier",
                value=next_better_tier,
            ),
            CompileReportSuggestionEvidence(
                path=(
                    f"{path_prefix}.target_resources.residency."
                    "unique_limiting_resource.name"
                ),
                value=resource_name,
            ),
            CompileReportSuggestionEvidence(
                path=(
                    f"{path_prefix}.target_resources.residency."
                    "unique_limiting_resource."
                    "reduction_units_to_next_better_tier"
                ),
                value=reduction,
            ),
        ),
    )


def _suggest_nondefault_wave_size(
    entry: dict[str, object],
    entry_name: str,
    path_prefix: str,
    default_wave_size: int,
) -> CompileReportSuggestion | None:
    target_resources = _object_at(entry, "target_resources")
    if target_resources is None:
        return None
    subgroup_size = _integer(target_resources.get("subgroup_size"))
    if (
        subgroup_size is None
        or subgroup_size == 0
        or subgroup_size == default_wave_size
    ):
        return None
    return CompileReportSuggestion(
        suggestion_id="amdgpu.nondefault_wave_size",
        entry_name=entry_name,
        action=(
            f"Benchmark an equivalent wave{default_wave_size} specialization and "
            f"retain wave{subgroup_size} only when measured faster."
        ),
        evidence=(
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.target_resources.subgroup_size",
                value=subgroup_size,
            ),
            CompileReportSuggestionEvidence(
                path="target_info.wavefront.default_size",
                value=default_wave_size,
            ),
        ),
    )


def _suggest_single_subgroup_communication(
    entry: dict[str, object],
    entry_name: str,
    path_prefix: str,
) -> CompileReportSuggestion | None:
    workgroup_size = _object_at(entry, "workload", "workgroup_size")
    target_resources = _object_at(entry, "target_resources")
    instruction_mix = _object_at(entry, "static_instruction_mix")
    if workgroup_size is None or target_resources is None or instruction_mix is None:
        return None
    flat_workgroup_size = _integer(workgroup_size.get("flat"))
    subgroup_size = _integer(target_resources.get("subgroup_size"))
    barrier_count = _integer(instruction_mix.get("barrier_count"))
    if (
        flat_workgroup_size is None
        or flat_workgroup_size == 0
        or subgroup_size is None
        or subgroup_size == 0
        or flat_workgroup_size > subgroup_size
        or barrier_count is None
        or barrier_count == 0
    ):
        return None

    evidence = [
        CompileReportSuggestionEvidence(
            path=f"{path_prefix}.workload.workgroup_size.flat",
            value=flat_workgroup_size,
        ),
        CompileReportSuggestionEvidence(
            path=f"{path_prefix}.target_resources.subgroup_size",
            value=subgroup_size,
        ),
        CompileReportSuggestionEvidence(
            path=f"{path_prefix}.static_instruction_mix.barrier_count",
            value=barrier_count,
        ),
    ]
    local_memory_instruction_count = _integer(instruction_mix.get("local_memory_count"))
    if local_memory_instruction_count is not None:
        evidence.append(
            CompileReportSuggestionEvidence(
                path=(f"{path_prefix}.static_instruction_mix.local_memory_count"),
                value=local_memory_instruction_count,
            )
        )
    local_memory_bytes = _integer(entry.get("local_memory_bytes"))
    if local_memory_bytes is not None:
        evidence.append(
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.local_memory_bytes",
                value=local_memory_bytes,
            )
        )
    return CompileReportSuggestion(
        suggestion_id="amdgpu.single_subgroup_workgroup_communication",
        entry_name=entry_name,
        action=(
            "The workgroup fits within one subgroup but still emits workgroup "
            "barriers. Inspect whether workgroup exchange or reduction can use "
            "subgroup operations, or whether the barriers are redundant; then "
            "require barrier and local-memory traffic to fall before "
            "benchmarking."
        ),
        evidence=tuple(evidence),
    )


@dataclass(frozen=True, slots=True)
class _FragmentWaveShape:
    """Exact cross-lane geometry for one fragment packet variant."""

    subgroup_size: int
    per_lane_packet_bytes: int
    interval_coverage: str
    subgroup_requested_bytes: int
    subgroup_unique_bytes: int
    subgroup_span_bytes: int
    maximum_uncovered_gap_bytes: int
    maximum_adjacent_lane_delta_bytes: int


@dataclass(frozen=True, slots=True)
class _FragmentWaveAccess:
    """One report group carrying an exact fragment wave shape."""

    path_prefix: str
    modeled_packet_count: int
    shape: _FragmentWaveShape


_FRAGMENT_WAVE_IDENTITY_FIELDS = (
    "function",
    "source_op",
    "source_op_kind",
    "source_root",
    "source_root_argument_index",
    "memory_space",
    "operation",
    "packet",
    "strategy",
)


def _suggest_fragment_packet_expansion(
    document: CompileReportDocument,
    options: CompileReportSuggestionOptions,
) -> tuple[CompileReportSuggestion, ...]:
    if not options.include_experimental:
        return ()
    source_low_value = document.report.get("source_low")
    if source_low_value is None:
        return ()
    source_low = _report_object(source_low_value, "source_low")
    selection_summaries_value = source_low.get("selection_summaries")
    memory_value = source_low.get("memory")
    if selection_summaries_value is None or memory_value is None:
        return ()
    selection_summaries = _report_object(
        selection_summaries_value,
        "source_low.selection_summaries",
    )
    selection_rows = _report_indexed_rows(
        selection_summaries,
        "source_low.selection_summaries",
    )
    memory = _report_object(memory_value, "source_low.memory")
    subgroup_access = build_subgroup_access_show(document)
    if subgroup_access is None:
        return ()
    subgroup_groups = cast(list[dict[str, object]], subgroup_access["groups"])
    subgroup_groups_by_identity: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for group in subgroup_groups:
        identity = cast(dict[str, object], group["identity"])
        key = tuple(identity[field] for field in _FRAGMENT_WAVE_IDENTITY_FIELDS)
        subgroup_groups_by_identity.setdefault(key, []).append(group)
    argument_packet_rows = _report_indexed_rows(
        memory,
        "source_low.memory",
        rows_key="argument_packets",
        count_key="argument_packet_count",
    )
    strategy_rows = _report_indexed_rows(
        memory,
        "source_low.memory",
        rows_key="strategies",
        count_key="strategy_count",
    )

    suggestions = []
    for selection_position, selection in enumerate(selection_rows):
        source_operation = _optional_report_string(
            selection.get("source_op"),
            (f"source_low.selection_summaries.rows[{selection_position}].source_op"),
        )
        if source_operation not in (
            "vector.fragment.load",
            "vector.fragment.store",
        ):
            continue
        source_operation_kind = _report_integer(
            selection.get("source_op_kind"),
            (
                "source_low.selection_summaries."
                f"rows[{selection_position}].source_op_kind"
            ),
        )
        strategy = _optional_report_string(
            selection.get("plan_key"),
            (f"source_low.selection_summaries.rows[{selection_position}].plan_key"),
        )
        if strategy is None:
            continue
        function_name = _optional_report_string(
            selection.get("function"),
            (f"source_low.selection_summaries.rows[{selection_position}].function"),
        )
        selected_operation_count = _report_integer(
            selection.get("selected_op_count"),
            (
                "source_low.selection_summaries."
                f"rows[{selection_position}].selected_op_count"
            ),
        )
        emitted_low_operation_count = _report_integer(
            selection.get("emitted_low_op_count"),
            (
                "source_low.selection_summaries."
                f"rows[{selection_position}].emitted_low_op_count"
            ),
        )
        if selected_operation_count == 0:
            continue

        matched_packet_rows = []
        packet_row_path = "source_low.memory.argument_packets"
        for packet_position, packet_row in enumerate(argument_packet_rows):
            packet_strategy = _optional_report_string(
                packet_row.get("strategy"),
                (f"source_low.memory.argument_packets[{packet_position}].strategy"),
            )
            packet_function = _optional_report_string(
                packet_row.get("function"),
                (f"source_low.memory.argument_packets[{packet_position}].function"),
            )
            if packet_strategy != strategy or packet_function != function_name:
                continue
            matched_packet_rows.append((packet_position, packet_row))
        if not matched_packet_rows:
            packet_row_path = "source_low.memory.strategies"
            for packet_position, packet_row in enumerate(strategy_rows):
                packet_strategy = _optional_report_string(
                    packet_row.get("strategy"),
                    f"{packet_row_path}[{packet_position}].strategy",
                )
                packet_function = _optional_report_string(
                    packet_row.get("function"),
                    f"{packet_row_path}[{packet_position}].function",
                )
                if packet_strategy == strategy and packet_function == function_name:
                    matched_packet_rows.append((packet_position, packet_row))

        packet_groups: dict[
            tuple[str | None, int | None, str | None, str | None],
            list[tuple[int, dict[str, object]]],
        ] = {}
        for packet_position, packet_row in matched_packet_rows:
            memory_space = _optional_report_string(
                packet_row.get("memory_space"),
                f"{packet_row_path}[{packet_position}].memory_space",
            )
            operation = _optional_report_string(
                packet_row.get("operation"),
                f"{packet_row_path}[{packet_position}].operation",
            )
            source_root = _optional_report_string(
                packet_row.get("source_root"),
                f"{packet_row_path}[{packet_position}].source_root",
            )
            source_root_argument_index_value = packet_row.get(
                "source_root_argument_index"
            )
            source_root_argument_index = (
                None
                if source_root_argument_index_value is None
                else _report_integer(
                    source_root_argument_index_value,
                    (
                        f"{packet_row_path}[{packet_position}]."
                        "source_root_argument_index"
                    ),
                )
            )
            group_key = (
                source_root,
                source_root_argument_index,
                memory_space,
                operation,
            )
            packet_groups.setdefault(group_key, []).append(
                (packet_position, packet_row)
            )

        for (
            source_root,
            source_root_argument_index,
            memory_space,
            operation,
        ), packet_group in packet_groups.items():
            packet_count = 0
            scalar_packet_count = 0
            contiguous_vector_packet_count = 0
            packet_names = []
            storage_formats = []
            packet_evidence = []
            packet_rows_with_scalar_packets = []
            for packet_position, packet_row in packet_group:
                packet_path = f"{packet_row_path}[{packet_position}]"
                packet_name = _report_string(
                    packet_row.get("packet"),
                    f"{packet_path}.packet",
                )
                packet_names.append(packet_name)
                storage = _report_object(
                    packet_row.get("storage"),
                    f"{packet_path}.storage",
                )
                storage_format = _optional_report_string(
                    storage.get("element_format"),
                    f"{packet_path}.storage.element_format",
                )
                if storage_format is not None:
                    storage_formats.append(storage_format)
                row_packet_count = _report_integer(
                    packet_row.get("packet_count"),
                    f"{packet_path}.packet_count",
                )
                row_scalar_packet_count = _report_integer(
                    packet_row.get("scalar_packet_count"),
                    f"{packet_path}.scalar_packet_count",
                )
                row_contiguous_vector_packet_count = _report_integer(
                    packet_row.get("contiguous_vector_packet_count"),
                    f"{packet_path}.contiguous_vector_packet_count",
                )
                packet_count += row_packet_count
                scalar_packet_count += row_scalar_packet_count
                contiguous_vector_packet_count += row_contiguous_vector_packet_count
                if row_scalar_packet_count != 0:
                    packet_rows_with_scalar_packets.append(
                        (packet_position, packet_name, row_packet_count)
                    )
                packet_evidence.extend(
                    (
                        CompileReportSuggestionEvidence(
                            path=f"{packet_path}.packet",
                            value=packet_name,
                        ),
                        CompileReportSuggestionEvidence(
                            path=f"{packet_path}.packet_count",
                            value=row_packet_count,
                        ),
                        CompileReportSuggestionEvidence(
                            path=f"{packet_path}.scalar_packet_count",
                            value=row_scalar_packet_count,
                        ),
                        CompileReportSuggestionEvidence(
                            path=(f"{packet_path}.contiguous_vector_packet_count"),
                            value=row_contiguous_vector_packet_count,
                        ),
                    )
                )
            if (
                packet_count < 16
                or scalar_packet_count * 4 < packet_count * 3
                or contiguous_vector_packet_count != 0
            ):
                continue
            wave_accesses = _match_exact_fragment_wave_accesses(
                subgroup_groups_by_identity,
                function_name=function_name,
                source_operation=source_operation,
                source_operation_kind=source_operation_kind,
                source_root=source_root,
                source_root_argument_index=source_root_argument_index,
                memory_space=memory_space,
                operation=operation,
                strategy=strategy,
                packet_rows=packet_rows_with_scalar_packets,
                packet_row_path=packet_row_path,
            )
            if wave_accesses is None:
                continue
            packet_evidence.extend(_fragment_wave_evidence(wave_accesses))

            selection_path = (
                f"source_low.selection_summaries.rows[{selection_position}]"
            )
            evidence = [
                CompileReportSuggestionEvidence(
                    path=f"{selection_path}.plan_key",
                    value=strategy,
                ),
                CompileReportSuggestionEvidence(
                    path=f"{selection_path}.selected_op_count",
                    value=selected_operation_count,
                ),
                CompileReportSuggestionEvidence(
                    path=f"{selection_path}.emitted_low_op_count",
                    value=emitted_low_operation_count,
                ),
                *packet_evidence,
            ]
            entry = _entry_for_function(document, function_name)
            operand_bank_materialization: CompileReportMoveCause | None = None
            if entry is not None:
                entry_index = _report_integer(
                    entry.get("index"),
                    "entries.rows[].index",
                )
                entry_path = f"entries.rows[{entry_index}]"
                wait_plan = _object_at(entry, "wait_plan")
                if wait_plan is not None:
                    full_drain_count = _integer(wait_plan.get("full_drain_count"))
                    if full_drain_count is not None:
                        evidence.append(
                            CompileReportSuggestionEvidence(
                                path=(f"{entry_path}.wait_plan.full_drain_count"),
                                value=full_drain_count,
                            )
                        )
                instruction_mix = _object_at(entry, "static_instruction_mix")
                if instruction_mix is not None:
                    register_move_count = _integer(
                        instruction_mix.get("register_move_count")
                    )
                    if register_move_count is not None:
                        evidence.append(
                            CompileReportSuggestionEvidence(
                                path=(
                                    f"{entry_path}.static_instruction_mix."
                                    "register_move_count"
                                ),
                                value=register_move_count,
                            )
                        )
                move_causes = parse_compile_report_move_causes(entry, entry_path)
                if move_causes is not None and move_causes.causes is not None:
                    operand_bank_materialization = next(
                        (
                            cause
                            for cause in move_causes.causes
                            if cause.cause == "operand_bank_materialization"
                        ),
                        None,
                    )
                    if operand_bank_materialization is not None:
                        cause_path = (
                            f"{entry_path}.move_causes.causes"
                            f"[{operand_bank_materialization.position}]"
                        )
                        evidence.extend(
                            (
                                CompileReportSuggestionEvidence(
                                    path=f"{cause_path}.packet_count",
                                    value=operand_bank_materialization.packet_count,
                                ),
                                CompileReportSuggestionEvidence(
                                    path=f"{cause_path}.unit_count",
                                    value=operand_bank_materialization.unit_count,
                                ),
                            )
                        )

            location = "/".join(
                value
                for value in (source_operation, source_root, memory_space)
                if value is not None
            )
            packet_summary = ", ".join(dict.fromkeys(packet_names))
            storage_summary = "/".join(dict.fromkeys(storage_formats))
            operation_name = operation or "memory"
            if scalar_packet_count == packet_count:
                scalar_packet_summary = (
                    f"{packet_count} scalar {operation_name} packets"
                )
            else:
                scalar_packet_summary = (
                    f"{scalar_packet_count} of {packet_count} {operation_name} "
                    "packets as scalar packets"
                )
            wave_summary = _format_fragment_wave_accesses(wave_accesses)
            geometry_guardrail = (
                "Packet width alone is not an objective. Compare candidates "
                "by total packet count and exact cross-lane coverage, span, "
                "gap, and adjacent-lane delta; reject a wider variant that "
                "worsens dispersion unless hardware timing pays for it."
            )
            if operand_bank_materialization is not None:
                action = (
                    "Inspect allocator placement before changing fragment "
                    f"storage: the {strategy} plan for {location} "
                    f"{storage_summary or 'storage'} emits "
                    f"{scalar_packet_summary} "
                    f"({packet_summary}) and no contiguous vector packets, "
                    "while the entry also contains "
                    f"{operand_bank_materialization.packet_count} "
                    "target-created operand-bank materialization packets. "
                    f"{wave_summary} {geometry_guardrail} "
                    "Check whether those repairs coincide with these fragment "
                    "loads; if so, test disjoint address/destination placement "
                    "or a shorter overlapping live window and require both "
                    "repairs and full drains to fall before changing the "
                    "memory hierarchy."
                )
            else:
                action = (
                    f"Run a bounded operand-layout and packing experiment for "
                    f"{location} "
                    f"{storage_summary or 'storage'}: the {strategy} plan "
                    f"emits {scalar_packet_summary} ({packet_summary}) and "
                    f"no contiguous vector packets. {wave_summary} "
                    f"{geometry_guardrail} Require full drains and register "
                    "moves to fall before benchmarking."
                )
            suggestions.append(
                CompileReportSuggestion(
                    suggestion_id="amdgpu.fragment_packet_expansion",
                    entry_name=_entry_name_for_function(document, function_name),
                    confidence=CompileReportSuggestionConfidence.EXPERIMENTAL,
                    action=action,
                    evidence=tuple(evidence),
                )
            )
    return tuple(suggestions)


def _match_exact_fragment_wave_accesses(
    subgroup_groups_by_identity: dict[tuple[object, ...], list[dict[str, object]]],
    *,
    function_name: str | None,
    source_operation: str,
    source_operation_kind: int,
    source_root: str | None,
    source_root_argument_index: int | None,
    memory_space: str | None,
    operation: str | None,
    strategy: str,
    packet_rows: list[tuple[int, str, int]],
    packet_row_path: str,
) -> tuple[_FragmentWaveAccess, ...] | None:
    """Matches packet rows to exact wave facts without reconstructing geometry."""
    accesses = []
    used_group_indexes = set()
    for packet_position, packet_name, packet_count in packet_rows:
        group_key = (
            function_name,
            source_operation,
            source_operation_kind,
            source_root,
            source_root_argument_index,
            memory_space,
            operation,
            packet_name,
            strategy,
        )
        matched_groups = subgroup_groups_by_identity.get(group_key)
        if not matched_groups:
            return None

        modeled_packet_count = 0
        for group in matched_groups:
            report_index = cast(int, group["report_index"])
            if report_index in used_group_indexes:
                raise CompileReportError(
                    "one subgroup access group matched multiple fragment packet rows"
                )
            used_group_indexes.add(report_index)
            path_prefix = f"source_low.memory.subgroup_access_groups[{report_index}]"
            summary = cast(dict[str, object], group["summary"])
            group_packet_count = cast(int, summary["modeled_packet_count"])
            modeled_packet_count += group_packet_count
            access = cast(dict[str, object], group["access"])
            if access["proof"] != "exact":
                return None
            address = cast(dict[str, object], access["address"])
            geometry = cast(dict[str, object], access["geometry"])
            accesses.append(
                _FragmentWaveAccess(
                    path_prefix=path_prefix,
                    modeled_packet_count=group_packet_count,
                    shape=_FragmentWaveShape(
                        subgroup_size=cast(int, address["subgroup_size"]),
                        per_lane_packet_bytes=cast(
                            int, address["per_lane_packet_bytes"]
                        ),
                        interval_coverage=cast(str, geometry["interval_coverage"]),
                        subgroup_requested_bytes=cast(
                            int, geometry["subgroup_requested_bytes"]
                        ),
                        subgroup_unique_bytes=cast(
                            int, geometry["subgroup_unique_bytes"]
                        ),
                        subgroup_span_bytes=cast(int, geometry["subgroup_span_bytes"]),
                        maximum_uncovered_gap_bytes=cast(
                            int, geometry["maximum_uncovered_gap_bytes"]
                        ),
                        maximum_adjacent_lane_delta_bytes=cast(
                            int, geometry["maximum_adjacent_lane_delta_bytes"]
                        ),
                    ),
                )
            )
        if modeled_packet_count != packet_count:
            raise CompileReportError(
                f"{packet_row_path}[{packet_position}].packet_count: subgroup "
                f"access groups model {modeled_packet_count}, got {packet_count}"
            )
    return tuple(accesses)


def _fragment_wave_evidence(
    accesses: tuple[_FragmentWaveAccess, ...],
) -> tuple[CompileReportSuggestionEvidence, ...]:
    evidence = []
    for access in accesses:
        path_prefix = access.path_prefix
        shape = access.shape
        values = (
            ("access.proof", "exact"),
            ("summary.modeled_packet_count", access.modeled_packet_count),
            ("access.address.subgroup_size", shape.subgroup_size),
            ("access.address.per_lane_packet_bytes", shape.per_lane_packet_bytes),
            ("access.geometry.interval_coverage", shape.interval_coverage),
            (
                "access.geometry.subgroup_requested_bytes",
                shape.subgroup_requested_bytes,
            ),
            ("access.geometry.subgroup_unique_bytes", shape.subgroup_unique_bytes),
            ("access.geometry.subgroup_span_bytes", shape.subgroup_span_bytes),
            (
                "access.geometry.maximum_uncovered_gap_bytes",
                shape.maximum_uncovered_gap_bytes,
            ),
            (
                "access.geometry.maximum_adjacent_lane_delta_bytes",
                shape.maximum_adjacent_lane_delta_bytes,
            ),
        )
        evidence.extend(
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.{suffix}",
                value=value,
            )
            for suffix, value in values
        )
    return tuple(evidence)


def _format_fragment_wave_accesses(
    accesses: tuple[_FragmentWaveAccess, ...],
) -> str:
    shapes = sorted(
        {access.shape for access in accesses},
        key=lambda shape: (
            shape.subgroup_size,
            shape.per_lane_packet_bytes,
            shape.interval_coverage,
            shape.subgroup_span_bytes,
            shape.maximum_uncovered_gap_bytes,
        ),
    )
    if len(shapes) == 1:
        shape = shapes[0]
        overlap = shape.subgroup_requested_bytes > shape.subgroup_unique_bytes
        coverage = shape.interval_coverage
        if overlap:
            coverage += ", overlapping"
        return (
            f"The exact wave shape is {shape.subgroup_size} lanes x "
            f"{shape.per_lane_packet_bytes:,} B/lane with {coverage} coverage: "
            f"{shape.subgroup_requested_bytes:,} B requested, "
            f"{shape.subgroup_unique_bytes:,} B unique in a "
            f"{shape.subgroup_span_bytes:,} B span, maximum gap "
            f"{shape.maximum_uncovered_gap_bytes:,} B, and maximum "
            "adjacent-lane delta "
            f"{shape.maximum_adjacent_lane_delta_bytes:,} B."
        )

    subgroup_sizes = sorted({shape.subgroup_size for shape in shapes})
    packet_widths = sorted({shape.per_lane_packet_bytes for shape in shapes})
    coverages = sorted(
        {
            shape.interval_coverage
            + (
                "+overlapping"
                if shape.subgroup_requested_bytes > shape.subgroup_unique_bytes
                else ""
            )
            for shape in shapes
        }
    )
    unique_bytes = [shape.subgroup_unique_bytes for shape in shapes]
    span_bytes = [shape.subgroup_span_bytes for shape in shapes]
    return (
        f"The exact wave evidence contains {len(shapes)} shapes across "
        f"subgroup sizes {_format_integer_values(subgroup_sizes)}, packet "
        f"widths {_format_integer_values(packet_widths)} B/lane, and coverage "
        f"{', '.join(coverages)}: unique footprints "
        f"{_format_integer_range(unique_bytes)} B, spans "
        f"{_format_integer_range(span_bytes)} B, maximum gap up to "
        f"{max(shape.maximum_uncovered_gap_bytes for shape in shapes):,} B, "
        "and maximum adjacent-lane delta up to "
        f"{max(shape.maximum_adjacent_lane_delta_bytes for shape in shapes):,} B."
    )


def _format_integer_values(values: list[int]) -> str:
    return "/".join(f"{value:,}" for value in values)


def _format_integer_range(values: list[int]) -> str:
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return f"{minimum:,}"
    return f"{minimum:,}-{maximum:,}"


def _suggest_lds_bank_service(
    document: CompileReportDocument,
    options: CompileReportSuggestionOptions,
) -> tuple[CompileReportSuggestion, ...]:
    source_low = document.report.get("source_low")
    if source_low is None:
        return ()
    source_low_object = _report_object(source_low, "source_low")
    memory = source_low_object.get("memory")
    if memory is None:
        return ()
    memory_object = _report_object(memory, "source_low.memory")
    group_values = memory_object.get("bank_service_groups")
    if group_values is None:
        return ()
    if not isinstance(group_values, list):
        raise CompileReportError(
            "source_low.memory.bank_service_groups: expected array"
        )
    group_count = _report_integer(
        memory_object.get("bank_service_group_count"),
        "source_low.memory.bank_service_group_count",
    )
    if group_count != len(group_values):
        raise CompileReportError(
            "source_low.memory.bank_service_group_count: "
            f"expected {len(group_values)}, got {group_count}"
        )

    suggestions = []
    for group_position, group_value in enumerate(group_values):
        path_prefix = f"source_low.memory.bank_service_groups[{group_position}]"
        group = _report_object(group_value, path_prefix)
        report_index = _report_integer(group.get("index"), f"{path_prefix}.index")
        if report_index != group_position:
            raise CompileReportError(
                f"{path_prefix}.index: expected {group_position}, got {report_index}"
            )
        model = _report_object(group.get("model"), f"{path_prefix}.model")
        summary = _report_object(group.get("summary"), f"{path_prefix}.summary")
        structural = _report_object(
            summary.get("structural"),
            f"{path_prefix}.summary.structural",
        )

        exact_packet_count = _report_integer(
            summary.get("exact_packet_count"),
            f"{path_prefix}.summary.exact_packet_count",
        )
        unknown_packet_count = _report_integer(
            summary.get("unknown_packet_count"),
            f"{path_prefix}.summary.unknown_packet_count",
        )
        conflicted_packet_count = _report_integer(
            structural.get("conflicted_packet_count"),
            f"{path_prefix}.summary.structural.conflicted_packet_count",
        )
        extra_round_count = _report_integer(
            structural.get("extra_round_count"),
            f"{path_prefix}.summary.structural.extra_round_count",
        )
        maximum_request_multiplicity = _report_integer(
            structural.get("maximum_request_multiplicity"),
            (f"{path_prefix}.summary.structural.maximum_request_multiplicity"),
        )
        if (
            exact_packet_count == 0
            or unknown_packet_count != 0
            or conflicted_packet_count == 0
            or extra_round_count == 0
        ):
            continue

        model_evidence = _report_string(
            model.get("evidence"),
            f"{path_prefix}.model.evidence",
        )
        if model_evidence in (
            AMDGPU_LDS_BANK_SERVICE_EVIDENCE_PUBLIC_VENDOR_DOCUMENTATION,
            AMDGPU_LDS_BANK_SERVICE_EVIDENCE_SILICON_CALIBRATED_VENDOR_MODEL,
        ):
            confidence = CompileReportSuggestionConfidence.HIGH
        elif (
            model_evidence
            == AMDGPU_LDS_BANK_SERVICE_EVIDENCE_VENDOR_SOFTWARE_MODEL_UNVALIDATED
        ):
            if not options.include_experimental:
                continue
            confidence = CompileReportSuggestionConfidence.EXPERIMENTAL
        else:
            raise CompileReportError(
                f"{path_prefix}.model.evidence: unsupported evidence class "
                f"{model_evidence!r}"
            )

        function_name = _optional_report_string(
            group.get("function"),
            f"{path_prefix}.function",
        )
        source_op = _optional_report_string(
            group.get("source_op"),
            f"{path_prefix}.source_op",
        )
        source_root = _optional_report_string(
            group.get("source_root"),
            f"{path_prefix}.source_root",
        )
        source_root_argument_index_value = group.get("source_root_argument_index")
        if source_root_argument_index_value is not None:
            source_root_argument_index = _report_integer(
                source_root_argument_index_value,
                f"{path_prefix}.source_root_argument_index",
            )
            if source_root is None:
                source_root = f"arg{source_root_argument_index}"
        packet = _optional_report_string(
            group.get("packet"),
            f"{path_prefix}.packet",
        )
        entry_name = _entry_name_for_function(document, function_name)
        location = "/".join(
            value for value in (source_op, source_root, packet) if value is not None
        )
        suggestions.append(
            CompileReportSuggestion(
                suggestion_id="amdgpu.lds_bank_service",
                entry_name=entry_name,
                confidence=confidence,
                action=(
                    f"Search layout variants for {location or 'this LDS access'} "
                    "using pitch or padding, lane mapping, fragment layout, or "
                    "packet width to reduce exact structural extra rounds. "
                    "Recompile each candidate, reject spill or occupancy "
                    "regressions, and select only from hardware timing."
                ),
                evidence=(
                    CompileReportSuggestionEvidence(
                        path=f"{path_prefix}.summary.exact_packet_count",
                        value=exact_packet_count,
                    ),
                    CompileReportSuggestionEvidence(
                        path=(
                            f"{path_prefix}.summary.structural.conflicted_packet_count"
                        ),
                        value=conflicted_packet_count,
                    ),
                    CompileReportSuggestionEvidence(
                        path=(f"{path_prefix}.summary.structural.extra_round_count"),
                        value=extra_round_count,
                    ),
                    CompileReportSuggestionEvidence(
                        path=(
                            f"{path_prefix}.summary.structural."
                            "maximum_request_multiplicity"
                        ),
                        value=maximum_request_multiplicity,
                    ),
                    CompileReportSuggestionEvidence(
                        path=f"{path_prefix}.model.evidence",
                        value=model_evidence,
                    ),
                    CompileReportSuggestionEvidence(
                        path=f"{path_prefix}.model.revision",
                        value=_report_string(
                            model.get("revision"),
                            f"{path_prefix}.model.revision",
                        ),
                    ),
                ),
            )
        )
    return tuple(suggestions)


def _entry_name_for_function(
    document: CompileReportDocument,
    function_name: str | None,
) -> str:
    if function_name is not None:
        for entry in document.entries:
            if function_name in (
                entry.get("function"),
                entry.get("source_function"),
                entry.get("target_export"),
                entry.get("target_export_symbol"),
            ):
                return compile_report_entry_identity(entry).display_name()
        return function_name
    if len(document.entries) == 1:
        return compile_report_entry_identity(document.entries[0]).display_name()
    return "<report>"


def _entry_for_function(
    document: CompileReportDocument,
    function_name: str | None,
) -> dict[str, object] | None:
    if function_name is None:
        return document.entries[0] if len(document.entries) == 1 else None
    for entry in document.entries:
        if function_name in (
            entry.get("function"),
            entry.get("source_function"),
            entry.get("target_export"),
            entry.get("target_export_symbol"),
        ):
            return entry
    return None


def _report_indexed_rows(
    parent: dict[str, object],
    path: str,
    *,
    rows_key: str = "rows",
    count_key: str = "count",
) -> tuple[dict[str, object], ...]:
    rows_value = parent.get(rows_key)
    if rows_value is None:
        return ()
    if not isinstance(rows_value, list):
        raise CompileReportError(f"{path}.{rows_key}: expected array")
    count = _report_integer(parent.get(count_key), f"{path}.{count_key}")
    if count != len(rows_value):
        raise CompileReportError(
            f"{path}.{count_key}: expected {len(rows_value)}, got {count}"
        )
    rows = []
    for position, row_value in enumerate(rows_value):
        row_path = f"{path}.{rows_key}[{position}]"
        row = _report_object(row_value, row_path)
        index = _report_integer(row.get("index"), f"{row_path}.index")
        if index != position:
            raise CompileReportError(
                f"{row_path}.index: expected {position}, got {index}"
            )
        rows.append(row)
    return tuple(rows)


def _report_object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CompileReportError(f"{path}: expected object")
    return value


def _report_integer(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CompileReportError(f"{path}: expected integer")
    return value


def _report_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise CompileReportError(f"{path}: expected string")
    return value


def _optional_report_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _report_string(value, path)


def _object_at(root: dict[str, object], *components: str) -> dict[str, object] | None:
    value: object = root
    for component in components:
        if not isinstance(value, dict):
            return None
        value = value.get(component)
    return value if isinstance(value, dict) else None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
