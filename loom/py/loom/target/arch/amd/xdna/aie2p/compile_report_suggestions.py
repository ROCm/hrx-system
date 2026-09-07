# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AIE2P compile report suggestions grounded in physical plan evidence."""

from __future__ import annotations

from loom.reporting.compile_report import (
    CompileReportDocument,
    CompileReportError,
    compile_report_entry_identity,
)
from loom.reporting.compile_report_suggestions import (
    CompileReportSuggestion,
    CompileReportSuggestionConfidence,
    CompileReportSuggestionEvidence,
    CompileReportSuggestionOptions,
    CompileReportSuggestionResult,
)

_MINIMUM_VLIW_BUNDLE_COUNT = 64
_CODE_HEADROOM_ACTION = (
    "Contract repeated address/control sequences, reduce specialization "
    "unrolling, or split a deliberately banked role before adding more work "
    "to this tile; then require program-memory headroom to rise."
)
_BANK_PRESSURE_ACTION = (
    "Redistribute worker state and channel ring slots across local banks, or "
    "reduce only the rings that do not carry latency-hiding value; then "
    "require peak bank occupancy to fall without reducing measured overlap."
)


class Aie2pCompileReportSuggestionProvider:
    """Interprets exact AIE2P leaf and spatial realization evidence."""

    target_family = "amd.xdna.aie2p"
    provider_name = "amd.xdna.aie2p"

    def suggest(
        self,
        document: CompileReportDocument,
        options: CompileReportSuggestionOptions | None = None,
    ) -> CompileReportSuggestionResult:
        if options is None:
            options = CompileReportSuggestionOptions()

        suggestions = []
        for entry_index, entry in enumerate(document.entries):
            entry_name = compile_report_entry_identity(entry).display_name()
            path_prefix = f"entries.rows[{entry_index}]"
            spill = _suggest_spill_traffic(entry, entry_name, path_prefix)
            if spill is not None:
                suggestions.append(spill)
            vliw = _suggest_vliw_coissue_density(entry, entry_name, path_prefix)
            if vliw is not None:
                suggestions.append(vliw)

        pipeline_plan_values = document.report.get("pipeline_plans", [])
        if not isinstance(pipeline_plan_values, list):
            raise CompileReportError("pipeline_plans: expected array")
        for plan_index, pipeline_plan_value in enumerate(pipeline_plan_values):
            plan_path = f"pipeline_plans[{plan_index}]"
            pipeline_plan = _report_object(pipeline_plan_value, plan_path)
            root_name = _report_string(pipeline_plan.get("root"), f"{plan_path}.root")
            workers = _optional_indexed_rows(pipeline_plan, plan_path, "workers")
            if workers:
                for position, worker in enumerate(workers):
                    path_prefix = f"{plan_path}.workers.rows[{position}]"
                    suggestions.extend(_suggest_worker_pressure(worker, path_prefix))
            else:
                suggestions.extend(_suggest_summary_pressure(pipeline_plan, plan_path))
            channels = _optional_indexed_rows(pipeline_plan, plan_path, "channels")
            for position, channel in enumerate(channels):
                path_prefix = f"{plan_path}.channels.rows[{position}]"
                entry_name = _channel_entry_name(channel, root_name)
                transform = _suggest_storage_transform(channel, entry_name, path_prefix)
                if transform is not None:
                    suggestions.append(transform)
                if options.include_experimental:
                    dma = _suggest_dma_record_coalescing(
                        channel, entry_name, path_prefix
                    )
                    if dma is not None:
                        suggestions.append(dma)

        return CompileReportSuggestionResult(
            provider_name=self.provider_name,
            unavailable_reason=None,
            suggestions=tuple(suggestions),
        )


AIE2P_COMPILE_REPORT_SUGGESTION_PROVIDER = Aie2pCompileReportSuggestionProvider()


def _suggest_spill_traffic(
    entry: dict[str, object], entry_name: str, path_prefix: str
) -> CompileReportSuggestion | None:
    metrics = (
        "allocation_spill_count",
        "allocation_materialized_spill_storage_bytes",
        "allocation_materialized_spill_store_bytes",
        "allocation_materialized_reload_bytes",
    )
    evidence = tuple(
        CompileReportSuggestionEvidence(
            path=f"{path_prefix}.{field}",
            value=value,
        )
        for field in metrics
        if (value := _optional_report_integer(entry.get(field))) is not None
        and value > 0
    )
    if not evidence:
        return None
    return CompileReportSuggestion(
        suggestion_id="aie2p.spill_traffic",
        entry_name=entry_name,
        action=(
            "Shorten live ranges, interleave fewer accumulators, or choose a "
            "narrower native shape; then require all spill storage, stores, and "
            "reloads to disappear before silicon benchmarking."
        ),
        evidence=evidence,
    )


def _suggest_vliw_coissue_density(
    entry: dict[str, object], entry_name: str, path_prefix: str
) -> CompileReportSuggestion | None:
    body_bundle_count = _optional_report_integer(entry.get("body_instruction_count"))
    coissued_bundle_count = _optional_report_integer(
        entry.get("coissued_instruction_count")
    )
    coissued_component_count = _optional_report_integer(
        entry.get("coissued_component_count")
    )
    if (
        body_bundle_count is None
        or coissued_bundle_count is None
        or coissued_component_count is None
    ):
        return None
    if coissued_bundle_count > body_bundle_count:
        raise CompileReportError(
            f"{path_prefix}.coissued_instruction_count: exceeds body bundle count"
        )
    if coissued_component_count < 2 * coissued_bundle_count:
        raise CompileReportError(
            f"{path_prefix}.coissued_component_count: fewer than two components "
            "per coissued bundle"
        )
    if (
        body_bundle_count < _MINIMUM_VLIW_BUNDLE_COUNT
        or coissued_bundle_count * 4 >= body_bundle_count
    ):
        return None
    return CompileReportSuggestion(
        suggestion_id="aie2p.vliw_coissue_density",
        entry_name=entry_name,
        action=(
            "Expose more independent ALU, load/store, and control work by "
            "interleaving accumulators or unrolling one bounded record step. "
            "Recompile and require fewer physical bundles at equal semantics "
            "without increasing register pressure before benchmarking."
        ),
        evidence=(
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.body_instruction_count",
                value=body_bundle_count,
            ),
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.coissued_instruction_count",
                value=coissued_bundle_count,
            ),
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.coissued_component_count",
                value=coissued_component_count,
            ),
        ),
    )


def _suggest_summary_pressure(
    pipeline_plan: dict[str, object],
    path_prefix: str,
) -> tuple[CompileReportSuggestion, ...]:
    entry_name = _report_string(pipeline_plan.get("root"), f"{path_prefix}.root")
    suggestions = []

    code_byte_count = _report_integer(
        pipeline_plan.get("maximum_worker_code_byte_count"),
        f"{path_prefix}.maximum_worker_code_byte_count",
    )
    code_headroom = _report_integer(
        pipeline_plan.get("minimum_worker_code_headroom_byte_count"),
        f"{path_prefix}.minimum_worker_code_headroom_byte_count",
    )
    code_capacity = code_byte_count + code_headroom
    code_suggestion = _capacity_pressure_suggestion(
        suggestion_id="aie2p.code_headroom",
        entry_name=entry_name,
        action=_CODE_HEADROOM_ACTION,
        usage=code_byte_count,
        capacity=code_capacity,
        evidence=(
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.maximum_worker_code_byte_count",
                value=code_byte_count,
            ),
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.minimum_worker_code_headroom_byte_count",
                value=code_headroom,
            ),
        ),
    )
    if code_suggestion is not None:
        suggestions.append(code_suggestion)

    bank_storage = _report_integer(
        pipeline_plan.get("maximum_bank_storage_byte_count"),
        f"{path_prefix}.maximum_bank_storage_byte_count",
    )
    bank_capacity = _report_integer(
        pipeline_plan.get("bank_storage_capacity_byte_count"),
        f"{path_prefix}.bank_storage_capacity_byte_count",
    )
    bank_suggestion = _capacity_pressure_suggestion(
        suggestion_id="aie2p.bank_pressure",
        entry_name=entry_name,
        action=_BANK_PRESSURE_ACTION,
        usage=bank_storage,
        capacity=bank_capacity,
        evidence=(
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.maximum_bank_storage_byte_count",
                value=bank_storage,
            ),
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.bank_storage_capacity_byte_count",
                value=bank_capacity,
            ),
        ),
        usage_path=f"{path_prefix}.maximum_bank_storage_byte_count",
    )
    if bank_suggestion is not None:
        suggestions.append(bank_suggestion)
    return tuple(suggestions)


def _suggest_worker_pressure(
    worker: dict[str, object], path_prefix: str
) -> tuple[CompileReportSuggestion, ...]:
    entry_name = _report_string(worker.get("entry"), f"{path_prefix}.entry")
    suggestions = []

    code_byte_count = _report_integer(
        worker.get("code_byte_count"), f"{path_prefix}.code_byte_count"
    )
    code_capacity = _report_integer(
        worker.get("code_capacity_byte_count"),
        f"{path_prefix}.code_capacity_byte_count",
    )
    code_suggestion = _capacity_pressure_suggestion(
        suggestion_id="aie2p.code_headroom",
        entry_name=entry_name,
        action=_CODE_HEADROOM_ACTION,
        usage=code_byte_count,
        capacity=code_capacity,
        evidence=(
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.code_byte_count",
                value=code_byte_count,
            ),
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.code_capacity_byte_count",
                value=code_capacity,
            ),
        ),
        usage_path=f"{path_prefix}.code_byte_count",
    )
    if code_suggestion is not None:
        suggestions.append(code_suggestion)

    bank_storage = _report_integer(
        worker.get("maximum_bank_storage_byte_count"),
        f"{path_prefix}.maximum_bank_storage_byte_count",
    )
    bank_capacity = _report_integer(
        worker.get("bank_storage_capacity_byte_count"),
        f"{path_prefix}.bank_storage_capacity_byte_count",
    )
    bank_suggestion = _capacity_pressure_suggestion(
        suggestion_id="aie2p.bank_pressure",
        entry_name=entry_name,
        action=_BANK_PRESSURE_ACTION,
        usage=bank_storage,
        capacity=bank_capacity,
        evidence=(
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.maximum_bank_storage_byte_count",
                value=bank_storage,
            ),
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.bank_storage_capacity_byte_count",
                value=bank_capacity,
            ),
        ),
        usage_path=f"{path_prefix}.maximum_bank_storage_byte_count",
    )
    if bank_suggestion is not None:
        suggestions.append(bank_suggestion)
    return tuple(suggestions)


def _capacity_pressure_suggestion(
    *,
    suggestion_id: str,
    entry_name: str,
    action: str,
    usage: int,
    capacity: int,
    evidence: tuple[CompileReportSuggestionEvidence, ...],
    usage_path: str | None = None,
) -> CompileReportSuggestion | None:
    if usage_path is not None:
        _require_usage_within_capacity(usage, capacity, usage_path)
    if capacity == 0 or usage * 4 < capacity * 3:
        return None
    return CompileReportSuggestion(
        suggestion_id=suggestion_id,
        entry_name=entry_name,
        action=action,
        evidence=evidence,
    )


def _suggest_storage_transform(
    channel: dict[str, object], entry_name: str, path_prefix: str
) -> CompileReportSuggestion | None:
    storage_value = channel.get("storage")
    if storage_value is None:
        return None
    storage = _report_object(storage_value, f"{path_prefix}.storage")
    transform = _report_string(
        storage.get("transform"), f"{path_prefix}.storage.transform"
    )
    extra_byte_count = _report_integer(
        storage.get("transform_extra_byte_count"),
        f"{path_prefix}.storage.transform_extra_byte_count",
    )
    if transform == "none" and extra_byte_count == 0:
        return None
    schema = _report_string(storage.get("schema"), f"{path_prefix}.storage.schema")
    return CompileReportSuggestion(
        suggestion_id="aie2p.storage_transform",
        entry_name=entry_name,
        action=(
            "Benchmark direct canonical-record consumption against this storage "
            "transform and retain the transform only when its measured compute "
            "gain repays the extra bytes and staging work."
        ),
        evidence=(
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.storage.schema", value=schema
            ),
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.storage.transform", value=transform
            ),
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.storage.transform_extra_byte_count",
                value=extra_byte_count,
            ),
        ),
    )


def _suggest_dma_record_coalescing(
    channel: dict[str, object], entry_name: str, path_prefix: str
) -> CompileReportSuggestion | None:
    transfer_value = channel.get("external_transfer")
    if transfer_value is None:
        return None
    transfer = _report_object(transfer_value, f"{path_prefix}.external_transfer")
    task_byte_count = _report_integer(
        transfer.get("task_byte_count"),
        f"{path_prefix}.external_transfer.task_byte_count",
    )
    repeat_count = _report_integer(
        transfer.get("task_repeat_count"),
        f"{path_prefix}.external_transfer.task_repeat_count",
    )
    if task_byte_count == 0:
        raise CompileReportError(
            f"{path_prefix}.external_transfer.task_byte_count: expected positive"
        )
    if task_byte_count > 512 or repeat_count < 8:
        return None
    return CompileReportSuggestion(
        suggestion_id="aie2p.dma_record_coalescing",
        entry_name=entry_name,
        confidence=CompileReportSuggestionConfidence.EXPERIMENTAL,
        action=(
            "Benchmark packing two or four adjacent logical records into each "
            "DMA task while preserving the same bounded ring and resident fold. "
            "Keep the wider task only when silicon timing improves without "
            "reducing compute/DMA overlap."
        ),
        evidence=(
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.external_transfer.task_byte_count",
                value=task_byte_count,
            ),
            CompileReportSuggestionEvidence(
                path=f"{path_prefix}.external_transfer.task_repeat_count",
                value=repeat_count,
            ),
        ),
    )


def _optional_indexed_rows(
    pipeline_plan: dict[str, object], path_prefix: str, collection_name: str
) -> tuple[dict[str, object], ...]:
    collection_value = pipeline_plan.get(collection_name)
    if collection_value is None:
        return ()
    collection_path = f"{path_prefix}.{collection_name}"
    collection = _report_object(collection_value, collection_path)
    rows_value = collection.get("rows")
    if rows_value is None:
        return ()
    if not isinstance(rows_value, list):
        raise CompileReportError(f"{collection_path}.rows: expected array")
    count = _report_integer(collection.get("count"), f"{collection_path}.count")
    if count != len(rows_value):
        raise CompileReportError(
            f"{collection_path}.count: expected {len(rows_value)}, got {count}"
        )
    rows = []
    index_name = "worker_index" if collection_name == "workers" else "channel_index"
    for position, row_value in enumerate(rows_value):
        row_path = f"{collection_path}.rows[{position}]"
        row = _report_object(row_value, row_path)
        index = _report_integer(row.get(index_name), f"{row_path}.{index_name}")
        if index != position:
            raise CompileReportError(
                f"{row_path}.{index_name}: expected {position}, got {index}"
            )
        rows.append(row)
    return tuple(rows)


def _channel_entry_name(channel: dict[str, object], root_name: str) -> str:
    receiver = channel.get("receiver")
    if isinstance(receiver, dict) and receiver.get("owner") == "worker":
        owner_index = _optional_report_integer(receiver.get("owner_index"))
        if owner_index is not None:
            return f"{root_name}:worker[{owner_index}]"
    channel_index = _optional_report_integer(channel.get("channel_index"))
    return (
        f"{root_name}:channel[{channel_index}]"
        if channel_index is not None
        else root_name
    )


def _require_usage_within_capacity(usage: int, capacity: int, path: str) -> None:
    if usage > capacity:
        raise CompileReportError(f"{path}: exceeds capacity {capacity}")


def _report_object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CompileReportError(f"{path}: expected object")
    return value


def _report_integer(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CompileReportError(f"{path}: expected nonnegative integer")
    return value


def _optional_report_integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _report_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise CompileReportError(f"{path}: expected string")
    return value
