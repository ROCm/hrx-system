# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Applied source loop schedules and experiments with their resource costs."""

from __future__ import annotations

from typing import cast

from loom.reporting.compile_report import (
    CompileReportDocument,
    CompileReportError,
    compile_report_entry_identity,
)
from loom.reporting.compile_report_suggestions import (
    CompileReportSuggestion,
    CompileReportSuggestionEvidence,
)

_PATH = "source_low.loop_pipelines"


def build_loop_pipeline_show(
    document: CompileReportDocument,
) -> dict[str, object] | None:
    """Validates public report input and groups schedules by applied policy."""
    source_low = document.report.get("source_low")
    if source_low is None:
        return None
    value = _object(source_low, f"{document.source}.source_low").get("loop_pipelines")
    if value is None:
        return None
    path = f"{document.source}.{_PATH}"
    section = _object(value, path)
    rows = _array(section.get("rows"), f"{path}.rows")
    if _count(section.get("count"), f"{path}.count") != len(rows):
        raise CompileReportError(f"{path}.count: does not match policy rows")
    policies: dict[tuple[str, int], dict[str, object]] = {}
    for position, value in enumerate(rows):
        row_path = f"{path}.rows[{position}]"
        row = _object(value, row_path)
        function = _string(row.get("function"), f"{row_path}.function")
        loop = _count(row.get("loop"), f"{row_path}.loop")
        identity = (function, loop)
        if identity in policies:
            raise CompileReportError(f"{row_path}: duplicate function/loop identity")
        depth = _count(row.get("depth"), f"{row_path}.depth")
        if depth == 0:
            raise CompileReportError(f"{row_path}.depth: expected positive depth")
        queue_records = _count(row.get("queue_records"), f"{row_path}.queue_records")
        width = _count(row.get("values_per_record"), f"{row_path}.values_per_record")
        reads = _count(row.get("read_count"), f"{row_path}.read_count")
        outcome = "serial" if depth == 1 else "pipelined"
        if row.get("schedule") != "read_ahead" or row.get("outcome") != outcome:
            raise CompileReportError(f"{row_path}: unsupported schedule or outcome")
        if queue_records != depth - 1:
            raise CompileReportError(f"{row_path}.queue_records: expected depth - 1")
        if (depth == 1 and (width or reads)) or (
            depth > 1 and (not width or not reads)
        ):
            raise CompileReportError(f"{row_path}: queue/read counts contradict depth")
        policies[identity] = {
            "function": function,
            "loop": loop,
            "schedule": "read_ahead",
            "outcome": outcome,
            "depth": depth,
            "queue_records": queue_records,
            "values_per_record": width,
            "read_count": reads,
        }
    if "stages" in section:
        for policy in policies.values():
            policy["stages"] = []
        for position, value in enumerate(_array(section["stages"], f"{path}.stages")):
            stage_path = f"{path}.stages[{position}]"
            row = _object(value, stage_path)
            identity = (
                _string(row.get("function"), f"{stage_path}.function"),
                _count(row.get("loop"), f"{stage_path}.loop"),
            )
            policy = policies.get(identity)
            if policy is None:
                raise CompileReportError(f"{stage_path}: no matching applied policy")
            stages = cast(list[dict[str, object]], policy["stages"])
            operation_position = _count(row.get("position"), f"{stage_path}.position")
            previous_position = cast(int, stages[-1]["position"]) if stages else -1
            if operation_position not in (previous_position, previous_position + 1):
                raise CompileReportError(
                    f"{stage_path}.position: expected {previous_position} or "
                    f"{previous_position + 1}"
                )
            name = _string(row.get("op"), f"{stage_path}.op")
            partition = row.get("partition")
            if partition not in (None, "guarded"):
                raise CompileReportError(f"{stage_path}.partition: unsupported value")
            stage = row.get("stage")
            lookahead = _count(
                row.get("iteration_lookahead"), f"{stage_path}.iteration_lookahead"
            )
            if (
                policy["depth"] == 1
                or stage not in ("producer", "consumer")
                or lookahead != (policy["queue_records"] if stage == "producer" else 0)
            ):
                raise CompileReportError(
                    f"{stage_path}: stage contradicts applied depth"
                )
            if operation_position == previous_position:
                previous = stages[-1]
                if (
                    previous["op"] != name
                    or previous.get("partition") != partition
                    or previous["stage"] != "producer"
                    or stage != "consumer"
                ):
                    raise CompileReportError(
                        f"{stage_path}: repeated position is not a matching stage pair"
                    )
            normalized_stage: dict[str, object] = {
                "position": operation_position,
                "op": name,
                "stage": stage,
                "iteration_lookahead": lookahead,
            }
            if partition is not None:
                normalized_stage["partition"] = partition
            stages.append(normalized_stage)
        for policy in policies.values():
            if policy["depth"] != 1 and not policy["stages"]:
                raise CompileReportError(
                    f"{path}.stages: missing pipelined operation schedule"
                )
            stages = cast(list[dict[str, object]], policy["stages"])
            guarded_positions = {
                cast(int, stage["position"])
                for stage in stages
                if stage.get("partition") == "guarded"
            }
            for guarded_position in guarded_positions:
                copies = [
                    stage for stage in stages if stage["position"] == guarded_position
                ]
                if (
                    len(copies) != 2
                    or any(stage.get("partition") != "guarded" for stage in copies)
                    or tuple(stage["stage"] for stage in copies)
                    != (
                        "producer",
                        "consumer",
                    )
                ):
                    raise CompileReportError(
                        f"{path}.stages: guarded position {guarded_position} "
                        "must contain producer and consumer copies"
                    )
    return {"count": len(policies), "rows": list(policies.values())}


def append_loop_pipeline_show_text(lines: list[str], view: dict[str, object]) -> None:
    """Appends the applied depth, queue shape, and optional source stage cut."""
    lines.extend(("", "Source loop pipelines"))
    for row in cast(list[dict[str, object]], view["rows"]):
        lines.append(
            f"  {row['function']} loop {row['loop']}: {row['outcome']} "
            f"depth={row['depth']} queue_records={row['queue_records']} "
            f"values_per_record={row['values_per_record']} reads={row['read_count']}"
        )
        for stage in cast(list[dict[str, object]], row.get("stages", [])):
            partition = (
                f" partition={stage['partition']}" if "partition" in stage else ""
            )
            lines.append(
                f"    {stage['position']}: {stage['op']}{partition} "
                f"{stage['stage']} "
                f"iteration_lookahead={stage['iteration_lookahead']}"
            )


def suggest_loop_pipelines(
    document: CompileReportDocument,
) -> tuple[CompileReportSuggestion, ...]:
    """Proposes depth experiments without inferring cost deltas from one report."""
    view = build_loop_pipeline_show(document)
    if view is None or document.status_code != 0:
        return ()
    entries_by_function: dict[str, list[dict[str, object]]] = {}
    for entry in document.entries:
        identity = compile_report_entry_identity(entry)
        for name in {identity.function, identity.source_function} - {None}:
            entries_by_function.setdefault(cast(str, name), []).append(entry)
    suggestions = []
    for position, row in enumerate(cast(list[dict[str, object]], view["rows"])):
        depth = cast(int, row["depth"])
        if depth == 1:
            continue
        path = f"{_PATH}.rows[{position}]"
        for entry in entries_by_function.get(cast(str, row["function"]), []):
            evidence = [
                CompileReportSuggestionEvidence(f"{path}.{key}", row[key])
                for key in (
                    "function",
                    "loop",
                    "depth",
                    "queue_records",
                    "values_per_record",
                    "read_count",
                )
            ]
            entry_path = f"entries.rows[{entry['index']}]"
            for metric in (
                "target_resources.scalar.final.register_count",
                "target_resources.vector.final.register_count",
                "target_resources.occupancy_percent",
                "allocation_spill_count",
                "private_memory_bytes",
                "code_byte_count",
            ):
                value: object = entry
                for component in metric.split("."):
                    if value is None:
                        break
                    value = _object(
                        value, f"{document.source}.{entry_path}.{metric}"
                    ).get(component)
                if value is not None:
                    count = _count(value, f"{document.source}.{entry_path}.{metric}")
                    evidence.append(
                        CompileReportSuggestionEvidence(f"{entry_path}.{metric}", count)
                    )
            queued_values = cast(int, row["queue_records"]) * cast(
                int, row["values_per_record"]
            )
            suggestions.append(
                CompileReportSuggestion(
                    suggestion_id="scf.compare_pipeline_depth",
                    entry_name=compile_report_entry_identity(entry).display_name(),
                    action=(
                        f"Loop {row['loop']} uses read-ahead depth {depth}, retaining "
                        f"{queued_values} queued SSA values. Compare a smaller "
                        "pipeline depth, including depth one as a serial control, "
                        "with the unroll factor and workload held fixed. Compare "
                        "final registers, spills, occupancy, code size, compile "
                        "time, and measured runtime. Queue size measures SSA state; "
                        "the matched comparison establishes "
                        "resource and performance deltas."
                    ),
                    evidence=tuple(evidence),
                )
            )
    return tuple(suggestions)


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CompileReportError(f"{path}: expected object")
    return value


def _array(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise CompileReportError(f"{path}: expected array")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CompileReportError(f"{path}: expected nonempty string")
    return value


def _count(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CompileReportError(f"{path}: expected nonnegative integer")
    return value
