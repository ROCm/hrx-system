#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Joins ID4 plan JSON with IREE dispatch-event profile rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DispatchProfileJoinError(ValueError):
    """Raised when plan/profile inputs cannot be joined exactly."""


@dataclass
class KernelAggregate:
    count: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    total_duration_ns: int = 0
    min_duration_ns: int | None = None
    max_duration_ns: int = 0
    durations_ns: list[int] | None = None


GENERATION_STAGE_ISSUE_ORDER = (
    "qwen",
    "sampler_noise",
)

GENERATION_DENOISE_STAGE_ISSUE_ORDER = (
    "dit_conditioned",
    "dit_unconditioned",
    "sampler_denoise",
)

GENERATION_FINAL_STAGE_ISSUE_ORDER = ("decode",)


def _percentile(sorted_values: list[int], percentile: int) -> int | None:
    if not sorted_values:
        return None
    index = (len(sorted_values) * percentile) // 100
    if index >= len(sorted_values):
        index = len(sorted_values) - 1
    return sorted_values[index]


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DispatchProfileJoinError(f"{field_name} must be an object")
    return value


def _require_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise DispatchProfileJoinError(f"{field_name} must be a list")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise DispatchProfileJoinError(f"{field_name} must be a non-empty string")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise DispatchProfileJoinError(f"{field_name} must be an integer")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise DispatchProfileJoinError(f"{field_name} must be a boolean")
    return value


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise DispatchProfileJoinError(f"{description} not found: {path}")
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise DispatchProfileJoinError(f"invalid {description}: {path}: {exc}") from exc
    return _require_dict(payload, description)


def _load_jsonl_objects(path: Path, description: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DispatchProfileJoinError(f"{description} not found: {path}")
    records = []
    try:
        with path.open(encoding="utf-8") as file:
            for line_ordinal, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DispatchProfileJoinError(
                        f"invalid {description}: {path}:{line_ordinal}: {exc}"
                    ) from exc
                records.append(_require_dict(payload, f"{description} row"))
    except OSError as exc:
        raise DispatchProfileJoinError(f"failed to read {description}: {path}") from exc
    return records


def _u32_triplet(value: Any, field_name: str) -> tuple[int, int, int]:
    values = _require_list(value, field_name)
    if len(values) != 3:
        raise DispatchProfileJoinError(f"{field_name} must contain 3 integers")
    triplet = tuple(_require_int(v, f"{field_name}[]") for v in values)
    if any(v < 0 or v > 0xFFFFFFFF for v in triplet):
        raise DispatchProfileJoinError(f"{field_name} values must be uint32")
    return triplet


def _launch_geometry(
    dispatch_ordinal: int,
    plan_dispatch: dict[str, Any],
    profile_event: dict[str, Any],
) -> tuple[tuple[int, int, int], tuple[int, int, int], str]:
    event_count = _u32_triplet(
        profile_event.get("workgroup_count"),
        f"profile.dispatch_events[{dispatch_ordinal}].workgroup_count",
    )
    event_size = _u32_triplet(
        profile_event.get("workgroup_size"),
        f"profile.dispatch_events[{dispatch_ordinal}].workgroup_size",
    )

    plan_has_count = "workgroup_count" in plan_dispatch
    plan_has_size = "workgroup_size" in plan_dispatch
    if plan_has_count != plan_has_size:
        raise DispatchProfileJoinError(
            f"program.dispatches[{dispatch_ordinal}] must contain both "
            "workgroup_count and workgroup_size, or neither"
        )
    if not plan_has_count:
        return event_count, event_size, "profile"

    plan_count = _u32_triplet(
        plan_dispatch.get("workgroup_count"),
        f"program.dispatches[{dispatch_ordinal}].workgroup_count",
    )
    if plan_count != event_count:
        raise DispatchProfileJoinError(
            f"dispatch {dispatch_ordinal} workgroup_count mismatch: "
            f"plan {plan_count}, profile {event_count}"
        )
    plan_size = _u32_triplet(
        plan_dispatch.get("workgroup_size"),
        f"program.dispatches[{dispatch_ordinal}].workgroup_size",
    )
    if plan_size != event_size:
        raise DispatchProfileJoinError(
            f"dispatch {dispatch_ordinal} workgroup_size mismatch: "
            f"plan {plan_size}, profile {event_size}"
        )
    return plan_count, plan_size, "plan+profile"


def _dispatch_events(profile_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for record in profile_records:
        if record.get("type") == "dispatch_event":
            events.append(record)
    if not events:
        raise DispatchProfileJoinError("profile contains no dispatch_event records")
    return events


def _generation_stages(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stages = _require_dict(plan.get("stages"), "stages")
    return {
        key: _require_dict(stage_plan, f"stages.{key}")
        for key, stage_plan in stages.items()
    }


def _generation_issue_sequence(plan: dict[str, Any]) -> list[tuple[str, int | None]]:
    summary = _require_dict(plan.get("summary"), "summary")
    denoise_step_count = _require_int(
        summary.get("denoise_step_count"), "summary.denoise_step_count"
    )
    if denoise_step_count < 0:
        raise DispatchProfileJoinError(
            "summary.denoise_step_count must be non-negative"
        )

    stages = _generation_stages(plan)
    sequence: list[tuple[str, int | None]] = []
    for stage_key in GENERATION_STAGE_ISSUE_ORDER:
        if stage_key not in stages:
            raise DispatchProfileJoinError(f"generation stage is missing: {stage_key}")
        sequence.append((stage_key, None))
    for denoise_step_index in range(denoise_step_count):
        for stage_key in GENERATION_DENOISE_STAGE_ISSUE_ORDER:
            if stage_key not in stages:
                raise DispatchProfileJoinError(
                    f"generation stage is missing: {stage_key}"
                )
            sequence.append((stage_key, denoise_step_index))
    for stage_key in GENERATION_FINAL_STAGE_ISSUE_ORDER:
        if stage_key not in stages:
            raise DispatchProfileJoinError(f"generation stage is missing: {stage_key}")
        sequence.append((stage_key, None))
    return sequence


def _program_dispatches(plan: dict[str, Any]) -> list[dict[str, Any]]:
    program = plan.get("program")
    if program is None:
        raise DispatchProfileJoinError("plan does not contain a source program")
    program = _require_dict(program, "program")
    dispatches = _require_list(program.get("dispatches"), "program.dispatches")
    expected_count = _require_int(
        program.get("dispatch_count"), "program.dispatch_count"
    )
    if expected_count != len(dispatches):
        raise DispatchProfileJoinError(
            "program.dispatch_count does not match dispatch row count"
        )
    return [_require_dict(row, "program.dispatches[]") for row in dispatches]


def _region(plan: dict[str, Any]) -> dict[str, Any]:
    regions = _require_list(plan.get("regions"), "regions")
    if len(regions) != 1:
        raise DispatchProfileJoinError(
            f"profile joining requires exactly one region, found {len(regions)}"
        )
    return _require_dict(regions[0], "regions[0]")


def _join_row(
    stage_name: str,
    region_row: dict[str, Any],
    dispatch_ordinal: int,
    plan_dispatch: dict[str, Any],
    profile_event: dict[str, Any],
) -> dict[str, Any]:
    plan_dispatch_ordinal = _require_int(
        plan_dispatch.get("dispatch_ordinal"),
        f"program.dispatches[{dispatch_ordinal}].dispatch_ordinal",
    )
    if plan_dispatch_ordinal != dispatch_ordinal:
        raise DispatchProfileJoinError(
            f"program.dispatches[{dispatch_ordinal}].dispatch_ordinal "
            f"is {plan_dispatch_ordinal}"
        )
    plan_function = _require_string(
        plan_dispatch.get("function_name"),
        f"program.dispatches[{dispatch_ordinal}].function_name",
    )
    profile_function = _require_string(
        profile_event.get("key"),
        f"profile.dispatch_events[{dispatch_ordinal}].key",
    )
    if plan_function != profile_function:
        raise DispatchProfileJoinError(
            f"dispatch {dispatch_ordinal} function mismatch: "
            f"plan {plan_function}, profile {profile_function}"
        )
    workgroup_count, workgroup_size, launch_geometry_source = _launch_geometry(
        dispatch_ordinal, plan_dispatch, profile_event
    )

    valid = _require_bool(profile_event.get("valid"), "profile.valid")
    duration_ns = profile_event.get("duration_ns")
    if valid:
        duration_ns = _require_int(duration_ns, "profile.duration_ns")
        if duration_ns < 0:
            raise DispatchProfileJoinError("profile.duration_ns must be non-negative")
    else:
        duration_ns = None

    region_id = _require_int(region_row.get("id"), "region.id")
    return {
        "stage": stage_name,
        "region_id": region_id,
        "region_name": _require_string(region_row.get("name"), "region.name"),
        "dispatch_ordinal": plan_dispatch_ordinal,
        "profile_event_id": _require_int(profile_event.get("event_id"), "event_id"),
        "submission_id": _require_int(
            profile_event.get("submission_id"), "submission_id"
        ),
        "command_buffer_id": _require_int(
            profile_event.get("command_buffer_id"), "command_buffer_id"
        ),
        "command_index": _require_int(
            profile_event.get("command_index"), "command_index"
        ),
        "operation_ordinal": _require_int(
            plan_dispatch.get("operation_ordinal"), "operation_ordinal"
        ),
        "region_operation_ordinal": _require_int(
            plan_dispatch.get("region_operation_ordinal"),
            "region_operation_ordinal",
        ),
        "name": _require_string(plan_dispatch.get("name"), "dispatch.name"),
        "module_path": _require_string(plan_dispatch.get("module_path"), "module_path"),
        "function_name": plan_function,
        "workgroup_count": list(workgroup_count),
        "workgroup_size": list(workgroup_size),
        "launch_geometry_source": launch_geometry_source,
        "config_bindings": _require_list(
            plan_dispatch.get("config_bindings"), "config_bindings"
        ),
        "bindings": _require_list(plan_dispatch.get("bindings"), "bindings"),
        "valid": valid,
        "duration_ns": duration_ns,
    }


def _accumulate_kernel_aggregate(
    aggregate: KernelAggregate, row: dict[str, Any]
) -> None:
    aggregate.count += 1
    if row["valid"]:
        aggregate.valid_count += 1
        duration_ns = _require_int(row["duration_ns"], "duration_ns")
        aggregate.total_duration_ns += duration_ns
        if aggregate.min_duration_ns is None:
            aggregate.min_duration_ns = duration_ns
        aggregate.min_duration_ns = min(aggregate.min_duration_ns, duration_ns)
        aggregate.max_duration_ns = max(aggregate.max_duration_ns, duration_ns)
        if aggregate.durations_ns is None:
            aggregate.durations_ns = []
        aggregate.durations_ns.append(duration_ns)
    else:
        aggregate.invalid_count += 1


def _kernel_aggregate_duration_fields(aggregate: KernelAggregate) -> dict[str, Any]:
    durations_ns = aggregate.durations_ns or []
    durations_ns.sort()
    return {
        "count": aggregate.count,
        "valid_count": aggregate.valid_count,
        "invalid_count": aggregate.invalid_count,
        "total_duration_ns": aggregate.total_duration_ns,
        "min_duration_ns": aggregate.min_duration_ns,
        "p50_duration_ns": _percentile(durations_ns, 50),
        "p90_duration_ns": _percentile(durations_ns, 90),
        "p99_duration_ns": _percentile(durations_ns, 99),
        "max_duration_ns": aggregate.max_duration_ns,
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: dict[tuple[str, str], KernelAggregate] = defaultdict(KernelAggregate)
    for row in rows:
        key = (
            _require_string(row["module_path"], "module_path"),
            row["function_name"],
        )
        _accumulate_kernel_aggregate(aggregates[key], row)
    records = []
    for (module_path, function_name), aggregate in aggregates.items():
        records.append(
            {
                "module_path": module_path,
                "function_name": function_name,
                **_kernel_aggregate_duration_fields(aggregate),
            }
        )
    records.sort(key=lambda row: (-row["total_duration_ns"], row["function_name"]))
    return records


def _join_stage_dispatches(
    plan: dict[str, Any],
    profile_events: list[dict[str, Any]],
    event_offset: int = 0,
) -> dict[str, Any]:
    stage_name = _require_string(plan.get("stage"), "stage")
    region_row = _region(plan)
    plan_dispatches = _program_dispatches(plan)
    if event_offset < 0:
        raise DispatchProfileJoinError("profile event offset must be non-negative")
    if event_offset + len(plan_dispatches) > len(profile_events):
        raise DispatchProfileJoinError(
            f"plan/profile dispatch count mismatch: "
            f"{len(plan_dispatches)} plan rows starting at event offset "
            f"{event_offset}, {len(profile_events)} profile rows"
        )

    rows = []
    for dispatch_ordinal, (plan_dispatch, profile_event) in enumerate(
        zip(plan_dispatches, profile_events[event_offset:])
    ):
        rows.append(
            _join_row(
                stage_name,
                region_row,
                dispatch_ordinal,
                plan_dispatch,
                profile_event,
            )
        )
    return {
        "stage": stage_name,
        "region": {
            "id": _require_int(region_row.get("id"), "region.id"),
            "name": _require_string(region_row.get("name"), "region.name"),
        },
        "event_offset": event_offset,
        "dispatches": rows,
        "next_event_offset": event_offset + len(rows),
    }


def _summary_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_duration_count = sum(1 for row in rows if row["valid"])
    total_duration_ns = sum(
        _require_int(row["duration_ns"], "duration_ns") for row in rows if row["valid"]
    )
    return {
        "dispatch_count": len(rows),
        "valid_duration_count": valid_duration_count,
        "invalid_duration_count": len(rows) - valid_duration_count,
        "total_duration_ns": total_duration_ns,
    }


def join_dispatch_profile(
    plan: dict[str, Any],
    profile_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Returns a deterministic joined profile report for one stage plan."""

    profile_events = _dispatch_events(profile_records)
    stage_report = _join_stage_dispatches(plan, profile_events)
    if stage_report["next_event_offset"] != len(profile_events):
        raise DispatchProfileJoinError(
            f"plan/profile dispatch count mismatch: "
            f"{stage_report['next_event_offset']} consumed profile rows, "
            f"{len(profile_events)} profile rows"
        )
    rows = stage_report["dispatches"]
    summary = _summary_for_rows(rows)
    return {
        "schema_version": 1,
        "stage": stage_report["stage"],
        "region": stage_report["region"],
        "summary": summary,
        "by_kernel": _aggregate_rows(rows),
        "dispatches": rows,
    }


def _aggregate_generation_stages(
    stage_invocations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    aggregates: dict[str, KernelAggregate] = defaultdict(KernelAggregate)
    invocation_counts: dict[str, int] = defaultdict(int)
    for invocation in stage_invocations:
        stage_key = _require_string(invocation["stage_key"], "stage_key")
        invocation_counts[stage_key] += 1
        for row in invocation["dispatches"]:
            _accumulate_kernel_aggregate(aggregates[stage_key], row)
    records = []
    for stage_key, aggregate in aggregates.items():
        records.append(
            {
                "stage_key": stage_key,
                "invocation_count": invocation_counts[stage_key],
                **_kernel_aggregate_duration_fields(aggregate),
            }
        )
    records.sort(key=lambda row: (-row["total_duration_ns"], row["stage_key"]))
    return records


def _aggregate_generation_kernels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: dict[tuple[str, str, str], KernelAggregate] = defaultdict(
        KernelAggregate
    )
    for row in rows:
        key = (
            _require_string(row["stage_key"], "stage_key"),
            _require_string(row["module_path"], "module_path"),
            _require_string(row["function_name"], "function_name"),
        )
        _accumulate_kernel_aggregate(aggregates[key], row)
    records = []
    for (stage_key, module_path, function_name), aggregate in aggregates.items():
        records.append(
            {
                "stage_key": stage_key,
                "module_path": module_path,
                "function_name": function_name,
                **_kernel_aggregate_duration_fields(aggregate),
            }
        )
    records.sort(
        key=lambda row: (
            -row["total_duration_ns"],
            row["stage_key"],
            row["function_name"],
        )
    )
    return records


def join_generation_dispatch_profile(
    plan: dict[str, Any],
    profile_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Returns a deterministic joined profile report for a generation plan."""

    if plan.get("kind") != "ideogram4_generation":
        raise DispatchProfileJoinError("plan kind must be ideogram4_generation")
    profile_events = _dispatch_events(profile_records)
    stages = _generation_stages(plan)
    sequence = _generation_issue_sequence(plan)

    event_offset = 0
    flattened_rows = []
    stage_invocations = []
    for invocation_ordinal, (stage_key, denoise_step_index) in enumerate(sequence):
        stage_report = _join_stage_dispatches(
            stages[stage_key], profile_events, event_offset
        )
        event_offset = _require_int(
            stage_report["next_event_offset"], "next_event_offset"
        )
        invocation_rows = []
        for row in stage_report["dispatches"]:
            generation_row = dict(row)
            generation_row["stage_key"] = stage_key
            generation_row["stage_invocation_ordinal"] = invocation_ordinal
            generation_row["generation_dispatch_ordinal"] = len(flattened_rows)
            generation_row["denoise_step_index"] = denoise_step_index
            invocation_rows.append(generation_row)
            flattened_rows.append(generation_row)
        stage_summary = _summary_for_rows(invocation_rows)
        stage_invocations.append(
            {
                "stage_key": stage_key,
                "stage": stage_report["stage"],
                "stage_invocation_ordinal": invocation_ordinal,
                "denoise_step_index": denoise_step_index,
                "event_offset": stage_report["event_offset"],
                "summary": stage_summary,
                "region": stage_report["region"],
                "by_kernel": _aggregate_rows(invocation_rows),
                "dispatches": invocation_rows,
            }
        )

    if event_offset != len(profile_events):
        raise DispatchProfileJoinError(
            f"generation issue profile consumed {event_offset} dispatch_event rows, "
            f"but profile contains {len(profile_events)} rows"
        )

    return {
        "schema_version": 1,
        "kind": "ideogram4_generation_dispatch_profile",
        "summary": {
            **_summary_for_rows(flattened_rows),
            "stage_invocation_count": len(stage_invocations),
            "denoise_step_count": _require_int(
                _require_dict(plan.get("summary"), "summary").get("denoise_step_count"),
                "summary.denoise_step_count",
            ),
        },
        "by_stage": _aggregate_generation_stages(stage_invocations),
        "by_kernel": _aggregate_generation_kernels(flattened_rows),
        "stage_invocations": stage_invocations,
        "dispatches": flattened_rows,
    }


def join_profile(
    plan: dict[str, Any], profile_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Returns a deterministic joined profile report for a supported plan."""

    if plan.get("kind") == "ideogram4_generation":
        return join_generation_dispatch_profile(plan, profile_records)
    return join_dispatch_profile(plan, profile_records)


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path, help="ID4 plan JSON path")
    parser.add_argument(
        "--profile",
        required=True,
        type=Path,
        help="iree-profile dispatch --dispatch_events JSONL path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output report JSON path. Defaults to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_arguments(argv)
    try:
        plan = _load_json_object(args.plan, "plan")
        profile_records = _load_jsonl_objects(args.profile, "profile")
        report = join_profile(plan, profile_records)
        if args.output is None:
            json.dump(report, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as file:
                json.dump(report, file, indent=2, sort_keys=True)
                file.write("\n")
    except DispatchProfileJoinError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
