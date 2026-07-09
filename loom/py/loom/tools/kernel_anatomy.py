# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Summarizes target kernel artifacts into comparable anatomy reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loom.tools.amdgpu_asm import (
    AmdgpuDisassemblyBlock,
    summarize_amdgpu_disassembly,
    summarize_amdgpu_disassembly_blocks,
)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class NamedPath:
    name: str
    path: Path


@dataclass(frozen=True)
class ComparisonSpec:
    baseline: str
    candidate: str


@dataclass(frozen=True)
class SymbolWeightSpec:
    disassembly_name: str
    symbol_regex: str
    weight: float | int


@dataclass(frozen=True)
class WeightedSymbolGroupSpec:
    disassembly_name: str
    group_name: str
    symbol_regex: str
    weight: float | int


def _parse_named_path(value: str) -> NamedPath:
    if "=" not in value:
        path = Path(value)
        return NamedPath(name=path.stem, path=path)
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("named paths must have a non-empty name")
    return NamedPath(name=name, path=Path(path))


def _parse_comparison_spec(value: str) -> ComparisonSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("comparisons must use BASELINE=CANDIDATE")
    baseline, candidate = value.split("=", 1)
    if not baseline or not candidate:
        raise argparse.ArgumentTypeError(
            "comparison baseline and candidate must be non-empty"
        )
    return ComparisonSpec(baseline=baseline, candidate=candidate)


def _parse_symbol_weight_spec(value: str) -> SymbolWeightSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "symbol weights must use DISASSEMBLY_NAME=SYMBOL_REGEX=WEIGHT"
        )
    disassembly_name, rest = value.split("=", 1)
    if "=" not in rest:
        raise argparse.ArgumentTypeError(
            "symbol weights must use DISASSEMBLY_NAME=SYMBOL_REGEX=WEIGHT"
        )
    symbol_regex, weight_text = rest.rsplit("=", 1)
    if not disassembly_name or not symbol_regex:
        raise argparse.ArgumentTypeError(
            "symbol weight disassembly name and regex must be non-empty"
        )
    try:
        weight = int(weight_text, 0)
    except ValueError:
        try:
            weight = float(weight_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("symbol weight must be numeric") from exc
    if weight < 0:
        raise argparse.ArgumentTypeError("symbol weight must be non-negative")
    return SymbolWeightSpec(
        disassembly_name=disassembly_name,
        symbol_regex=symbol_regex,
        weight=weight,
    )


def _parse_weighted_symbol_group_spec(value: str) -> WeightedSymbolGroupSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "weighted symbol groups must use "
            "DISASSEMBLY_NAME/GROUP_NAME=SYMBOL_REGEX=WEIGHT"
        )
    disassembly_group, rest = value.split("=", 1)
    if "/" not in disassembly_group:
        raise argparse.ArgumentTypeError(
            "weighted symbol groups must use "
            "DISASSEMBLY_NAME/GROUP_NAME=SYMBOL_REGEX=WEIGHT"
        )
    disassembly_name, group_name = disassembly_group.split("/", 1)
    if "=" not in rest:
        raise argparse.ArgumentTypeError(
            "weighted symbol groups must use "
            "DISASSEMBLY_NAME/GROUP_NAME=SYMBOL_REGEX=WEIGHT"
        )
    symbol_regex, weight_text = rest.rsplit("=", 1)
    if not disassembly_name or not group_name or not symbol_regex:
        raise argparse.ArgumentTypeError(
            "weighted symbol group disassembly name, group name, and regex "
            "must be non-empty"
        )
    if "/" in group_name:
        raise argparse.ArgumentTypeError("weighted symbol group names cannot contain /")
    try:
        weight = int(weight_text, 0)
    except ValueError:
        try:
            weight = float(weight_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "weighted symbol weight must be numeric"
            ) from exc
    if weight < 0:
        raise argparse.ArgumentTypeError("weighted symbol weight must be non-negative")
    return WeightedSymbolGroupSpec(
        disassembly_name=disassembly_name,
        group_name=group_name,
        symbol_regex=symbol_regex,
        weight=weight,
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_compile_report(data: Any) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError("compile report root must be a JSON object")
    report = data.get("compile_report", data)
    if not isinstance(report, Mapping):
        raise ValueError("compile_report must be a JSON object")
    return report


def _first_entry(
    report: Mapping[str, Any], preferred_function: str | None = None
) -> Mapping[str, Any]:
    entries = report.get("entries")
    if not isinstance(entries, Mapping):
        return report
    rows = entries.get("rows")
    if not isinstance(rows, list) or not rows:
        return report
    if preferred_function:
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if row.get("function") == preferred_function:
                return row
            if row.get("source_function") == preferred_function:
                return row
    entry = rows[0]
    if not isinstance(entry, Mapping):
        return report
    return entry


def _summarize_loom_compile_report(
    report: Mapping[str, Any],
    path: Path | None = None,
    preferred_function: str | None = None,
) -> dict[str, Any]:
    """Extracts kernel-economics fields from a Loom compile-report object."""

    entry = _first_entry(report, preferred_function)
    report_economics = report.get("economics")
    if not isinstance(report_economics, Mapping):
        report_economics = {}
    target_resources = entry.get("target_resources")
    if not isinstance(target_resources, Mapping):
        target_resources = {}
    static_instruction_mix = entry.get("static_instruction_mix")
    if not isinstance(static_instruction_mix, Mapping):
        static_instruction_mix = {}
    dynamic_instruction_mix = entry.get("dynamic_instruction_mix")
    if not isinstance(dynamic_instruction_mix, Mapping):
        dynamic_instruction_mix = {}
    economics = entry.get("economics")
    if not isinstance(economics, Mapping):
        economics = {}
    source_low = _summarize_source_low_economics(report_economics) or (
        _summarize_source_low_economics(economics)
    )
    workload = entry.get("workload")
    if not isinstance(workload, Mapping):
        workload = {}
    wait_plan = entry.get("wait_plan")
    if not isinstance(wait_plan, Mapping):
        wait_plan = {}
    summary = {
        "function": entry.get("function") or report.get("function"),
        "target_key": report.get("target_key"),
        "executable_format": report.get("executable_format"),
        "workload": workload,
        "instruction_count": entry.get("instruction_count"),
        "code_byte_count": entry.get("code_byte_count"),
        "private_memory_bytes": entry.get("private_memory_bytes"),
        "local_memory_bytes": entry.get("local_memory_bytes"),
        "static_instruction_mix": static_instruction_mix,
        "dynamic_instruction_mix": dynamic_instruction_mix,
        "economics": economics,
        "source_low": source_low,
        "target_resources": target_resources,
        "wait_plan": wait_plan,
        "wait_reasons": _summarize_wait_reason_rows(report, entry),
    }
    if path is not None:
        summary["path"] = path.as_posix()
    return summary


_WAIT_REASON_SUMMARY_METRICS = (
    "action_count",
    "explicit_action_count",
    "planned_action_count",
    "full_drain_count",
    "partial_wait_count",
    "drained_count",
    "max_drained_count",
    "max_outstanding_before",
    "max_full_drain_outstanding_before",
)


def _summarize_wait_reason_rows(
    report: Mapping[str, Any], entry: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = _table_rows(report.get("wait_reason_summary_rows"))
    if not rows:
        rows = _table_rows(entry.get("wait_reason_summary_rows"))
    summaries: list[dict[str, Any]] = []
    for row_value in rows:
        row = _as_mapping(row_value)
        summary = _as_mapping(row.get("summary"))
        summarized_row = {
            "counter": row.get("counter"),
            "reason": row.get("reason"),
        }
        for metric in _WAIT_REASON_SUMMARY_METRICS:
            value = summary.get(metric)
            if _as_number(value) is not None:
                summarized_row[metric] = value
        summaries.append(summarized_row)
    return sorted(
        summaries,
        key=lambda row: (
            _as_number(row.get("action_count")) or 0,
            _as_number(row.get("drained_count")) or 0,
        ),
        reverse=True,
    )


_SOURCE_LOW_TOTAL_METRICS = (
    "packet_count",
    "load_packet_count",
    "store_packet_count",
    "scalar_packet_count",
    "vector_packet_count",
    "source_lane_count",
    "source_byte_count",
    "read_byte_count",
    "write_byte_count",
    "issued_read_byte_count",
    "issued_write_byte_count",
    "exact_dynamic_packet_count",
    "unknown_dynamic_packet_count",
    "dynamic_packet_count",
    "dynamic_source_byte_count",
    "dynamic_read_byte_count",
    "dynamic_write_byte_count",
    "dynamic_issued_read_byte_count",
    "dynamic_issued_write_byte_count",
    "contiguous_vector_packet_count",
    "strided_vector_packet_count",
    "unknown_stride_vector_packet_count",
)

_SOURCE_LOW_NESTED_METRICS = (
    "read_bytes",
    "write_bytes",
    "total_bytes",
    "byte_count",
    "begin_min_bytes",
    "end_max_bytes",
)


def _summarize_source_low_economics(
    economics: Mapping[str, Any],
) -> dict[str, Any]:
    memory = _as_mapping(economics.get("memory"))
    source_low = _as_mapping(memory.get("source_low")) or _as_mapping(
        economics.get("source_low")
    )
    if not source_low:
        return {}
    summary = _copy_numeric_fields(source_low, _SOURCE_LOW_TOTAL_METRICS)
    for nested_name in (
        "dispatch_source",
        "dispatch_issued",
        "interval_envelope",
        "read_interval_envelope",
        "write_interval_envelope",
    ):
        nested_summary = _copy_numeric_fields(
            _as_mapping(source_low.get(nested_name)), _SOURCE_LOW_NESTED_METRICS
        )
        if nested_summary:
            summary[nested_name] = nested_summary
    argument_packets = [
        _summarize_source_low_argument_packet(row)
        for row in _table_rows(source_low.get("argument_packets"))
    ]
    if argument_packets:
        summary["argument_packets"] = sorted(
            argument_packets,
            key=lambda row: (
                _as_number(row.get("dynamic_packet_count")) or 0,
                _as_number(row.get("dynamic_source_byte_count")) or 0,
                _as_number(_as_mapping(row.get("dispatch_source")).get("total_bytes"))
                or 0,
            ),
            reverse=True,
        )
    return summary


def _summarize_source_low_argument_packet(row_value: Any) -> dict[str, Any]:
    row = _as_mapping(row_value)
    summary = {
        "root_argument_name": row.get("root_argument_name")
        or row.get("argument")
        or row.get("root")
        or row.get("name"),
        "memory_space": row.get("memory_space"),
        "operation": row.get("operation"),
        "packet": row.get("packet") or row.get("packet_key"),
    }
    summary.update(_copy_numeric_fields(row, _SOURCE_LOW_TOTAL_METRICS))
    for nested_name in ("dispatch_source", "dispatch_issued", "interval_envelope"):
        nested_summary = _copy_numeric_fields(
            _as_mapping(row.get(nested_name)), _SOURCE_LOW_NESTED_METRICS
        )
        if nested_summary:
            summary[nested_name] = nested_summary
    return summary


def _copy_numeric_fields(
    source: Mapping[str, Any], fields: Sequence[str]
) -> dict[str, Any]:
    return {
        field: value
        for field in fields
        if _as_number(value := source.get(field)) is not None
    }


def _table_rows(table_value: Any) -> list[Any]:
    table = _as_mapping(table_value)
    rows = table.get("rows")
    return rows if isinstance(rows, list) else []


def summarize_loom_compile_report(path: Path) -> dict[str, Any]:
    """Extracts kernel-economics fields from a Loom compile-report artifact."""

    report = _extract_compile_report(_read_json(path))
    return _summarize_loom_compile_report(report, path)


def summarize_loom_benchmark_jsonl(path: Path) -> dict[str, Any]:
    """Extracts timing and compile summaries from benchmark JSONL events."""

    benchmark_rows: list[dict[str, Any]] = []
    repetition_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    compile_rows: list[dict[str, Any]] = []
    device_rows: list[dict[str, Any]] = []
    benchmark_entries: dict[tuple[Any, Any, Any], str] = {}
    for line_number, line in enumerate(_read_text(path).splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
        if not isinstance(event, Mapping):
            continue
        row_kind = event.get("row")
        if row_kind == "device":
            device_rows.append(_summarize_loom_benchmark_device_row(event))
            continue
        if row_kind == "plan":
            actual_entry = event.get("actual_entry")
            if isinstance(actual_entry, str) and actual_entry:
                benchmark_entries[
                    (
                        event.get("candidate_id"),
                        event.get("benchmark"),
                        event.get("case"),
                    )
                ] = actual_entry
            continue
        if row_kind == "compile":
            compile_rows.append(_summarize_loom_benchmark_compile_row(event))
            continue
        if row_kind == "comparison":
            comparison_rows.append(_summarize_loom_benchmark_comparison_row(event))
            continue
        if row_kind == "benchmark.repetition":
            benchmark_result = event.get("benchmark_result")
            if isinstance(benchmark_result, Mapping):
                preferred_entry = _lookup_loom_benchmark_entry(
                    benchmark_entries, event, benchmark_result
                )
                repetition_rows.append(
                    _summarize_loom_benchmark_repetition_row(
                        event, benchmark_result, preferred_entry
                    )
                )
            continue
        if row_kind != "benchmark":
            continue
        benchmark_result = event.get("benchmark_result")
        if not isinstance(benchmark_result, Mapping):
            continue
        preferred_entry = _lookup_loom_benchmark_entry(
            benchmark_entries, event, benchmark_result
        )
        benchmark_rows.append(
            _summarize_loom_benchmark_result(event, benchmark_result, preferred_entry)
        )
    return {
        "path": path.as_posix(),
        "benchmark_count": len(benchmark_rows),
        "comparison_count": len(comparison_rows),
        "compile_count": len(compile_rows),
        "device_count": len(device_rows),
        "repetition_count": len(repetition_rows),
        "benchmarks": benchmark_rows,
        "comparisons": comparison_rows,
        "compiles": compile_rows,
        "devices": device_rows,
        "repetitions": repetition_rows,
        "resource_findings": _build_loom_benchmark_resource_findings(
            device_rows, compile_rows
        ),
    }


def _summarize_loom_benchmark_result(
    event: Mapping[str, Any],
    benchmark_result: Mapping[str, Any],
    preferred_entry: str | None = None,
) -> dict[str, Any]:
    measurement = benchmark_result.get("measurement")
    if not isinstance(measurement, Mapping):
        measurement = {}
    operation_timing = measurement.get("operation_timing_ns")
    if not isinstance(operation_timing, Mapping):
        operation_timing = {}
    timing_interpretation = measurement.get("timing_interpretation")
    if not isinstance(timing_interpretation, Mapping):
        timing_interpretation = {}
    correctness = benchmark_result.get("correctness")
    if not isinstance(correctness, Mapping):
        correctness = {}
    compile_report = benchmark_result.get("compile_report")
    compile_summary = None
    if isinstance(compile_report, Mapping):
        compile_summary = _summarize_loom_compile_report(
            compile_report, preferred_function=preferred_entry
        )
    return {
        "benchmark": benchmark_result.get("benchmark"),
        "case": benchmark_result.get("case"),
        "state": benchmark_result.get("state"),
        "candidate_id": event.get("candidate_id"),
        "candidate_index": event.get("candidate_index"),
        "sample_compilation": benchmark_result.get("sample_compilation"),
        "timing_ns": operation_timing,
        "timing_interpretation": timing_interpretation,
        "correctness": correctness,
        "compile_report": compile_summary,
    }


def _summarize_loom_benchmark_repetition_row(
    event: Mapping[str, Any],
    benchmark_result: Mapping[str, Any],
    preferred_entry: str | None = None,
) -> dict[str, Any]:
    summary = _summarize_loom_benchmark_result(event, benchmark_result, preferred_entry)
    summary.update(
        {
            "baseline_candidate_id": event.get("baseline_candidate_id"),
            "comparison_group": event.get("comparison_group"),
            "method": event.get("method"),
            "order_index": event.get("order_index"),
            "repetition_index": event.get("repetition_index"),
            "schedule_token": event.get("schedule_token"),
        }
    )
    return summary


def _lookup_loom_benchmark_entry(
    benchmark_entries: Mapping[tuple[Any, Any, Any], str],
    event: Mapping[str, Any],
    benchmark_result: Mapping[str, Any],
) -> str | None:
    direct_entry = event.get("entry") or benchmark_result.get("entry")
    if isinstance(direct_entry, str) and direct_entry:
        return direct_entry
    return benchmark_entries.get(
        (
            event.get("candidate_id"),
            benchmark_result.get("benchmark"),
            benchmark_result.get("case"),
        )
    )


def _summarize_loom_benchmark_comparison_row(
    event: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": event.get("run_id"),
        "comparison_group": event.get("comparison_group"),
        "method": event.get("method"),
        "baseline_candidate_id": event.get("baseline_candidate_id"),
        "candidate_id": event.get("candidate_id"),
        "baseline_repetition_count": event.get("baseline_repetition_count"),
        "candidate_repetition_count": event.get("candidate_repetition_count"),
        "baseline_p50_ns": event.get("baseline_p50_ns"),
        "candidate_p50_ns": event.get("candidate_p50_ns"),
        "baseline_p90_ns": event.get("baseline_p90_ns"),
        "candidate_p90_ns": event.get("candidate_p90_ns"),
        "baseline_p50_spread_ppm": event.get("baseline_p50_spread_ppm"),
        "candidate_p50_spread_ppm": event.get("candidate_p50_spread_ppm"),
        "baseline_p90_spread_ppm": event.get("baseline_p90_spread_ppm"),
        "candidate_p90_spread_ppm": event.get("candidate_p90_spread_ppm"),
        "ratio_p50": event.get("ratio_p50"),
        "ratio_p90": event.get("ratio_p90"),
        "speedup_p50": event.get("speedup_p50"),
        "speedup_p90": event.get("speedup_p90"),
    }


def _summarize_loom_benchmark_device_row(event: Mapping[str, Any]) -> dict[str, Any]:
    device_spec = _as_mapping(event.get("device_spec"))
    dispatch = _as_mapping(device_spec.get("dispatch"))
    execution = _as_mapping(dispatch.get("execution"))
    physical_devices = device_spec.get("physical_devices")
    physical_device = {}
    if isinstance(physical_devices, list) and physical_devices:
        physical_device = _as_mapping(physical_devices[0])
    return {
        "run_id": event.get("run_id"),
        "device_uri": event.get("device_uri"),
        "driver": event.get("driver"),
        "provider": event.get("provider"),
        "target_family": event.get("target_family"),
        "display_name": physical_device.get("display_name"),
        "backend_path": physical_device.get("backend_path"),
        "maximum_workgroup_local_memory_size": execution.get(
            "maximum_workgroup_local_memory_size"
        ),
        "maximum_workgroup_local_memory_size_optin": execution.get(
            "maximum_workgroup_local_memory_size_optin"
        ),
    }


def _summarize_loom_benchmark_compile_row(event: Mapping[str, Any]) -> dict[str, Any]:
    compile_report = event.get("compile_report")
    compile_summary = None
    if isinstance(compile_report, Mapping):
        entry = event.get("entry")
        compile_summary = _summarize_loom_compile_report(
            compile_report,
            preferred_function=entry if isinstance(entry, str) else None,
        )
    return {
        "run_id": event.get("run_id"),
        "candidate_id": event.get("candidate_id"),
        "candidate_index": event.get("candidate_index"),
        "benchmark": event.get("benchmark"),
        "case": event.get("case"),
        "entry": event.get("entry"),
        "state": event.get("state"),
        "diagnostic_error_count": event.get("diagnostic_error_count"),
        "diagnostic_warning_count": event.get("diagnostic_warning_count"),
        "diagnostic_remark_count": event.get("diagnostic_remark_count"),
        "hal_executable_path": event.get("hal_executable_path"),
        "target_artifact_path": event.get("target_artifact_path"),
        "target_listing_path": event.get("target_listing_path"),
        "compile_report": compile_summary,
    }


def _build_loom_benchmark_resource_findings(
    device_rows: Sequence[Mapping[str, Any]],
    compile_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    devices_by_run_id = {
        device.get("run_id"): device for device in device_rows if device.get("run_id")
    }
    findings: list[dict[str, Any]] = []
    for compile_row in compile_rows:
        compile_report = _as_mapping(compile_row.get("compile_report"))
        local_memory_bytes = _as_number(compile_report.get("local_memory_bytes"))
        if local_memory_bytes is None:
            continue
        device = devices_by_run_id.get(compile_row.get("run_id"))
        if device is None and len(device_rows) == 1:
            device = device_rows[0]
        if device is None:
            continue
        local_memory_limit = _as_number(
            device.get("maximum_workgroup_local_memory_size")
        )
        if local_memory_limit is None or local_memory_bytes <= local_memory_limit:
            continue
        findings.append(
            {
                "severity": "error",
                "resource": "workgroup_local_memory",
                "run_id": compile_row.get("run_id"),
                "candidate_id": compile_row.get("candidate_id"),
                "benchmark": compile_row.get("benchmark"),
                "case": compile_row.get("case"),
                "entry": compile_row.get("entry"),
                "required_bytes": local_memory_bytes,
                "limit_bytes": local_memory_limit,
                "overage_bytes": local_memory_bytes - local_memory_limit,
                "device_uri": device.get("device_uri"),
                "device": device.get("display_name") or device.get("backend_path"),
            }
        )
    return findings


_ROCBLAS_DEVICE_PATTERN = re.compile(
    r"Device ID (?P<ordinal>[0-9]+) : (?P<name>.+?) (?P<arch>gfx[0-9a-z]+)"
)
_ROCBLAS_HIPBLASLT_PATTERN = re.compile(
    r"hipBLASLt version:\s*(?P<version>\S+)(?:\s+commit-hash:\s*(?P<commit>\S+))?"
)


def summarize_rocblas_log_path(named_path: NamedPath) -> dict[str, Any]:
    """Extracts selected solution and timing fields from rocBLAS logs."""

    text = _read_text(named_path.path)
    lines = text.splitlines()
    timing_header: list[str] | None = None
    timing_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    kernel_parameters: dict[str, str] = {}
    summary: dict[str, Any] = {
        "name": named_path.name,
        "paths": [named_path.path.as_posix()],
        "devices": [],
        "kernel_parameters": kernel_parameters,
        "timing_rows": timing_rows,
        "trace_rows": trace_rows,
    }
    in_kernel_parameters = False
    for line in lines:
        stripped = line.strip()
        if trace_row := _parse_rocblas_trace_row(stripped):
            trace_rows.append(trace_row)
        elif stripped.startswith("rocBLAS version:"):
            summary["rocblas_version"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("rocBLAS-commit-hash:"):
            summary["rocblas_commit_hash"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Tensile-commit-hash:"):
            summary["tensile_commit_hash"] = stripped.split(":", 1)[1].strip()
        elif match := _ROCBLAS_HIPBLASLT_PATTERN.match(stripped):
            summary["hipblaslt_version"] = match.group("version")
            summary["hipblaslt_commit_hash"] = match.group("commit")
        elif match := _ROCBLAS_DEVICE_PATTERN.match(stripped):
            summary["devices"].append(
                {
                    "ordinal": int(match.group("ordinal")),
                    "name": match.group("name").strip(),
                    "arch": match.group("arch"),
                }
            )
        elif stripped.startswith("Library logic solution index of winning solution:"):
            summary["solution_index"] = _parse_numeric_scalar(stripped.split(":", 1)[1])
        elif stripped.startswith("Running kernel:"):
            summary["running_kernel"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Kernel name:"):
            summary["kernel_name"] = stripped.split(":", 1)[1].strip()
        elif stripped == "Kernel parameters:":
            in_kernel_parameters = True
        elif stripped.startswith("transA,transB,"):
            in_kernel_parameters = False
            timing_header = next(csv.reader([stripped]))
        elif timing_header is not None and stripped:
            timing_row = _parse_rocblas_timing_row(timing_header, stripped)
            if timing_row is not None:
                timing_rows.append(timing_row)
        elif in_kernel_parameters and ":" in stripped:
            key, value = stripped.split(":", 1)
            kernel_parameters[key.strip()] = value.strip()
    summary["selected_timing_row"] = _select_rocblas_timing_row(timing_rows)
    symbol_parameters = _parse_tensile_kernel_symbol(
        summary.get("running_kernel") or summary.get("kernel_name")
    )
    if symbol_parameters:
        summary["symbol_parameters"] = symbol_parameters
    return summary


_ROCBLAS_TRACE_ROW_PATTERN = re.compile(r"^-\s*\{(?P<payload>.*)\}\s*$")


def _parse_rocblas_trace_row(row: str) -> dict[str, Any] | None:
    match = _ROCBLAS_TRACE_ROW_PATTERN.match(row)
    if match is None or "rocblas_function" not in row:
        return None
    fields = next(csv.reader([match.group("payload")], skipinitialspace=True))
    parsed_row: dict[str, Any] = {}
    for field in fields:
        if ":" not in field:
            continue
        key, value = field.split(":", 1)
        parsed_row[key.strip()] = _parse_rocblas_trace_value(value.strip())
    _annotate_rocblas_trace_row(parsed_row)
    return parsed_row


def _annotate_rocblas_trace_row(row: dict[str, Any]) -> None:
    operation_key = _rocblas_trace_operation_key(row)
    shape_key = _rocblas_trace_shape_key(row)
    row["operation_key"] = operation_key
    row["shape_key"] = shape_key
    arguments = _rocblas_trace_bench_arguments(row)
    if arguments:
        row["rocblas_bench_arguments"] = arguments
        row["rocblas_bench_command"] = shlex.join(["rocblas-bench", *arguments])


def _rocblas_trace_operation_key(row: Mapping[str, Any]) -> str:
    function = row.get("rocblas_function")
    return str(function) if function is not None else "?"


def _rocblas_trace_shape_key(row: Mapping[str, Any]) -> str:
    m = row.get("M", "?")
    n = row.get("N", "?")
    k = row.get("K", "?")
    beta = row.get("beta", "?")
    return f"M{m}_N{n}_K{k}_beta{beta}"


_ROCBLAS_BENCH_FUNCTIONS = {
    "rocblas_gemm_ex": "gemm_ex",
    "rocblas_gemm_strided_batched_ex": "gemm_strided_batched_ex",
    "rocblas_gemm_batched_ex": "gemm_batched_ex",
}

_ROCBLAS_BENCH_FLAG_BY_TRACE_KEY = (
    ("transA", "--transposeA"),
    ("transB", "--transposeB"),
    ("M", "--sizem"),
    ("N", "--sizen"),
    ("K", "--sizek"),
    ("a_type", "--a_type"),
    ("b_type", "--b_type"),
    ("c_type", "--c_type"),
    ("d_type", "--d_type"),
    ("compute_type", "--compute_type"),
    ("alpha", "--alpha"),
    ("beta", "--beta"),
    ("lda", "--lda"),
    ("ldb", "--ldb"),
    ("ldc", "--ldc"),
    ("ldd", "--ldd"),
    ("batch_count", "--batch_count"),
    ("algo", "--algo"),
    ("solution_index", "--solution_index"),
    ("flags", "--flags"),
    ("cold_iters", "--cold_iters"),
    ("iters", "--iters"),
    ("device", "--device"),
)


def _rocblas_trace_bench_arguments(row: Mapping[str, Any]) -> list[str]:
    bench_function = _rocblas_trace_bench_function(row.get("rocblas_function"))
    if bench_function is None:
        return []
    arguments = ["--function", bench_function]
    for key, flag in _ROCBLAS_BENCH_FLAG_BY_TRACE_KEY:
        value = row.get(key)
        if value is None:
            continue
        arguments.extend([flag, _format_rocblas_bench_argument_value(key, value)])
    atomics_mode = row.get("atomics_mode")
    if atomics_mode == "atomics_allowed":
        arguments.append("--atomics_allowed")
    elif atomics_mode == "atomics_not_allowed":
        arguments.append("--atomics_not_allowed")
    return arguments


def _rocblas_trace_bench_function(function: Any) -> str | None:
    if not isinstance(function, str) or not function:
        return None
    if function in _ROCBLAS_BENCH_FUNCTIONS:
        return _ROCBLAS_BENCH_FUNCTIONS[function]
    if function.startswith("rocblas_"):
        return function.removeprefix("rocblas_")
    return function


def _format_rocblas_bench_argument_value(key: str, value: Any) -> str:
    if key == "flags" and value == "none":
        return "0"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _parse_rocblas_trace_value(value: str) -> Any:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return _parse_numeric_scalar(value)


def _merge_rocblas_log_summaries(
    target: dict[str, Any], source: Mapping[str, Any]
) -> dict[str, Any]:
    target_paths = target.setdefault("paths", [])
    source_paths = source.get("paths", [])
    if isinstance(target_paths, list) and isinstance(source_paths, list):
        target_paths.extend(source_paths)
    for key, value in source.items():
        if key in {"devices", "kernel_parameters", "paths", "timing_rows"}:
            continue
        if key not in target or target[key] in (None, "", [], {}):
            target[key] = value
    target_devices = target.setdefault("devices", [])
    source_devices = source.get("devices", [])
    if isinstance(target_devices, list) and isinstance(source_devices, list):
        for device in source_devices:
            if device not in target_devices:
                target_devices.append(device)
    target_parameters = target.setdefault("kernel_parameters", {})
    source_parameters = source.get("kernel_parameters", {})
    if isinstance(target_parameters, dict) and isinstance(source_parameters, Mapping):
        for key, value in source_parameters.items():
            target_parameters.setdefault(key, value)
    target_timing_rows = target.setdefault("timing_rows", [])
    source_timing_rows = source.get("timing_rows", [])
    if isinstance(target_timing_rows, list) and isinstance(source_timing_rows, list):
        target_timing_rows.extend(source_timing_rows)
        target["selected_timing_row"] = _select_rocblas_timing_row(target_timing_rows)
    target_trace_rows = target.setdefault("trace_rows", [])
    source_trace_rows = source.get("trace_rows", [])
    if isinstance(target_trace_rows, list) and isinstance(source_trace_rows, list):
        target_trace_rows.extend(source_trace_rows)
    return target


def _select_rocblas_timing_row(
    timing_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if not timing_rows:
        return None
    return max(
        timing_rows,
        key=lambda row: _as_number(row.get("hot_iters")) or 0,
    )


def _parse_rocblas_timing_row(header: Sequence[str], row: str) -> dict[str, Any] | None:
    values = next(csv.reader([row], skipinitialspace=True))
    if len(values) != len(header):
        return None
    if header[:2] == ["transA", "transB"] and values[0] not in {"N", "T", "C"}:
        return None
    if header[:2] == ["transA", "transB"] and values[1] not in {"N", "T", "C"}:
        return None
    return {
        key: _parse_numeric_scalar(value)
        for key, value in zip(header, values, strict=False)
    }


def _parse_tensile_kernel_symbol(symbol_value: Any) -> dict[str, Any]:
    if not isinstance(symbol_value, str) or not symbol_value:
        return {}
    parameters: dict[str, Any] = {}
    _parse_tensile_symbol_triplet(
        parameters,
        symbol_value,
        r"(?:^|_)MT(?P<x>[0-9]+)x(?P<y>[0-9]+)x(?P<z>[0-9]+)(?:_|$)",
        "macro_tile",
    )
    _parse_tensile_symbol_quad(
        parameters,
        symbol_value,
        r"(?:^|_)MI(?P<x>[0-9]+)x(?P<y>[0-9]+)x(?P<z>[0-9]+)x(?P<w>[0-9]+)(?:_|$)",
        "matrix_instruction",
    )
    _parse_tensile_symbol_triplet(
        parameters,
        symbol_value,
        r"(?:^|_)WG(?P<x>[0-9]+)_(?P<y>[0-9]+)_(?P<z>[0-9]+)(?:_|$)",
        "workgroup_size",
    )
    _parse_tensile_symbol_pair(
        parameters,
        symbol_value,
        r"(?:^|_)TT(?P<x>[0-9]+)_(?P<y>[0-9]+)(?:_|$)",
        "thread_tile",
    )
    _parse_tensile_symbol_scalar(
        parameters, symbol_value, r"(?:^|_)ISA(?P<value>[0-9]+)(?:_|$)", "isa"
    )
    _parse_tensile_symbol_scalar(
        parameters,
        symbol_value,
        r"(?:^|_)WS(?P<value>[0-9]+)(?:_|$)",
        "wave_size",
    )
    _parse_tensile_symbol_scalar(
        parameters,
        symbol_value,
        r"(?:^|_)VW(?P<value>[0-9]+)(?:_|$)",
        "vector_width",
    )
    _parse_tensile_symbol_scalar(
        parameters,
        symbol_value,
        r"(?:^|_)GLVWA(?P<value>[0-9]+)(?:_|$)",
        "global_load_vector_width_a",
    )
    _parse_tensile_symbol_scalar(
        parameters,
        symbol_value,
        r"(?:^|_)GLVWB(?P<value>[0-9]+)(?:_|$)",
        "global_load_vector_width_b",
    )
    _parse_tensile_symbol_scalar(
        parameters,
        symbol_value,
        r"(?:^|_)LPA(?P<value>[0-9]+)(?:_|$)",
        "local_split_a",
    )
    _parse_tensile_symbol_scalar(
        parameters,
        symbol_value,
        r"(?:^|_)LPB(?P<value>[0-9]+)(?:_|$)",
        "local_split_b",
    )
    _parse_tensile_symbol_scalar(
        parameters,
        symbol_value,
        r"(?:^|_)LRVW(?P<value>[0-9]+)(?:_|$)",
        "local_read_vector_width",
    )
    _parse_tensile_symbol_scalar(
        parameters,
        symbol_value,
        r"(?:^|_)WGM(?P<value>[0-9]+)(?:_|$)",
        "workgroup_mapping",
    )
    macro_tile = _as_mapping(parameters.get("macro_tile"))
    if macro_tile:
        parameters["depth_u"] = macro_tile.get("z")
    workgroup_size = _as_mapping(parameters.get("workgroup_size"))
    if workgroup_size:
        parameters["flat_workgroup_size"] = (
            workgroup_size.get("x") * workgroup_size.get("y") * workgroup_size.get("z")
        )
    return parameters


def _parse_tensile_symbol_scalar(
    parameters: dict[str, Any], symbol: str, pattern: str, key: str
) -> None:
    match = re.search(pattern, symbol)
    if match is not None:
        parameters[key] = int(match.group("value"))


def _parse_tensile_symbol_pair(
    parameters: dict[str, Any], symbol: str, pattern: str, key: str
) -> None:
    match = re.search(pattern, symbol)
    if match is not None:
        parameters[key] = {"x": int(match.group("x")), "y": int(match.group("y"))}


def _parse_tensile_symbol_triplet(
    parameters: dict[str, Any], symbol: str, pattern: str, key: str
) -> None:
    match = re.search(pattern, symbol)
    if match is not None:
        parameters[key] = {
            "x": int(match.group("x")),
            "y": int(match.group("y")),
            "z": int(match.group("z")),
        }


def _parse_tensile_symbol_quad(
    parameters: dict[str, Any], symbol: str, pattern: str, key: str
) -> None:
    match = re.search(pattern, symbol)
    if match is not None:
        parameters[key] = {
            "x": int(match.group("x")),
            "y": int(match.group("y")),
            "z": int(match.group("z")),
            "w": int(match.group("w")),
        }


def _parse_numeric_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return stripped
    try:
        return int(stripped, 0)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def summarize_iree_dispatch_profile_path(
    named_path: NamedPath, top_kernel_count: int
) -> dict[str, Any]:
    """Extracts top kernel timing rows from an IREE dispatch profile JSON."""

    data = _read_json(named_path.path)
    if not isinstance(data, Mapping):
        raise ValueError("IREE dispatch profile root must be a JSON object")
    by_kernel = data.get("by_kernel")
    if not isinstance(by_kernel, list):
        raise ValueError("IREE dispatch profile must contain a by_kernel list")
    kernels = [
        _summarize_iree_dispatch_profile_kernel(row)
        for row in by_kernel
        if isinstance(row, Mapping)
    ]
    sorted_kernels = sorted(
        kernels,
        key=lambda row: _as_number(row.get("total_duration_ns")) or 0,
        reverse=True,
    )
    return {
        "name": named_path.name,
        "path": named_path.path.as_posix(),
        "kernel_count": len(kernels),
        "kernels": sorted_kernels[: max(top_kernel_count, 0)],
    }


def _summarize_iree_dispatch_profile_kernel(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "function_name": row.get("function_name"),
        "module_path": row.get("module_path"),
        "count": row.get("count"),
        "valid_count": row.get("valid_count"),
        "invalid_count": row.get("invalid_count"),
        "total_duration_ns": row.get("total_duration_ns"),
        "p50_duration_ns": row.get("p50_duration_ns"),
        "p90_duration_ns": row.get("p90_duration_ns"),
        "p99_duration_ns": row.get("p99_duration_ns"),
        "min_duration_ns": row.get("min_duration_ns"),
        "max_duration_ns": row.get("max_duration_ns"),
    }


def summarize_rocprof_kernel_trace_path(
    named_path: NamedPath, top_kernel_count: int
) -> dict[str, Any]:
    """Extracts per-kernel timing and resource rows from rocprof CSV output."""

    return summarize_rocprof_kernel_trace_paths(
        named_path.name, [named_path.path], top_kernel_count
    )


def summarize_rocprof_kernel_trace_paths(
    name: str, paths: Sequence[Path], top_kernel_count: int
) -> dict[str, Any]:
    """Extracts merged per-kernel timing rows from rocprof CSV outputs."""

    dispatches = [
        _summarize_rocprof_kernel_trace_row(row)
        for path in paths
        for row in _read_csv_dict_rows(path)
        if row.get("Kind") == "KERNEL_DISPATCH"
    ]
    kernels = _summarize_rocprof_dispatches_by_kernel(dispatches)
    return {
        "name": name,
        "path": paths[0].as_posix() if len(paths) == 1 else None,
        "paths": [path.as_posix() for path in paths],
        "dispatch_count": len(dispatches),
        "kernel_count": len(kernels),
        "kernels": kernels[: max(top_kernel_count, 0)],
    }


def _read_csv_dict_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(_read_text(path).splitlines()))


def _summarize_rocprof_kernel_trace_row(row: Mapping[str, str]) -> dict[str, Any]:
    start_timestamp = _parse_numeric_scalar(row.get("Start_Timestamp"))
    end_timestamp = _parse_numeric_scalar(row.get("End_Timestamp"))
    duration_ns = None
    if isinstance(start_timestamp, int) and isinstance(end_timestamp, int):
        duration_ns = end_timestamp - start_timestamp
    return {
        "correlation_id": _parse_numeric_scalar(row.get("Correlation_Id")),
        "dispatch_id": _parse_numeric_scalar(row.get("Dispatch_Id")),
        "kernel_id": _parse_numeric_scalar(row.get("Kernel_Id")),
        "kernel_name": row.get("Kernel_Name"),
        "duration_ns": duration_ns,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "local_memory_bytes": _parse_numeric_scalar(row.get("LDS_Block_Size")),
        "scratch_bytes": _parse_numeric_scalar(row.get("Scratch_Size")),
        "vgpr_count": _parse_numeric_scalar(row.get("VGPR_Count")),
        "accum_vgpr_count": _parse_numeric_scalar(row.get("Accum_VGPR_Count")),
        "sgpr_count": _parse_numeric_scalar(row.get("SGPR_Count")),
        "workgroup_size": _parse_extent_product(
            row.get("Workgroup_Size_X"),
            row.get("Workgroup_Size_Y"),
            row.get("Workgroup_Size_Z"),
        ),
        "grid_size": _parse_extent_product(
            row.get("Grid_Size_X"), row.get("Grid_Size_Y"), row.get("Grid_Size_Z")
        ),
    }


def _parse_extent_product(*values: Any) -> int | None:
    product = 1
    for value in values:
        numeric_value = _parse_numeric_scalar(value)
        if not isinstance(numeric_value, int):
            return None
        product *= numeric_value
    return product


def _summarize_rocprof_dispatches_by_kernel(
    dispatches: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for dispatch in dispatches:
        kernel_name = dispatch.get("kernel_name")
        if not isinstance(kernel_name, str) or not kernel_name:
            continue
        grouped.setdefault(kernel_name, []).append(dispatch)
    summaries = [
        _summarize_rocprof_kernel_group(kernel_name, rows)
        for kernel_name, rows in grouped.items()
    ]
    return sorted(
        summaries,
        key=lambda row: _as_number(row.get("total_duration_ns")) or 0,
        reverse=True,
    )


def _summarize_rocprof_kernel_group(
    kernel_name: str, dispatches: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    durations = [
        duration
        for dispatch in dispatches
        if isinstance((duration := dispatch.get("duration_ns")), (int, float))
    ]
    return {
        "kernel_name": kernel_name,
        "count": len(dispatches),
        "total_duration_ns": sum(durations),
        "p50_duration_ns": _percentile(durations, 0.50),
        "p90_duration_ns": _percentile(durations, 0.90),
        "min_duration_ns": min(durations) if durations else None,
        "max_duration_ns": max(durations) if durations else None,
        "local_memory_bytes": _max_numeric_field(dispatches, "local_memory_bytes"),
        "scratch_bytes": _max_numeric_field(dispatches, "scratch_bytes"),
        "vgpr_count": _max_numeric_field(dispatches, "vgpr_count"),
        "accum_vgpr_count": _max_numeric_field(dispatches, "accum_vgpr_count"),
        "sgpr_count": _max_numeric_field(dispatches, "sgpr_count"),
        "workgroup_size": _max_numeric_field(dispatches, "workgroup_size"),
        "grid_size": _max_numeric_field(dispatches, "grid_size"),
    }


def summarize_rocprof_counter_collection_path(
    named_path: NamedPath, top_kernel_count: int
) -> dict[str, Any]:
    """Extracts per-kernel hardware counter summaries from rocprof CSV output."""

    return summarize_rocprof_counter_collection_paths(
        named_path.name, [named_path.path], top_kernel_count
    )


def summarize_rocprof_counter_collection_paths(
    name: str, paths: Sequence[Path], top_kernel_count: int
) -> dict[str, Any]:
    """Extracts merged hardware counter summaries from rocprof CSV outputs."""

    rows = []
    for path in paths:
        for row in _read_csv_dict_rows(path):
            counter_row = _summarize_rocprof_counter_collection_row(row)
            counter_row["path"] = path.as_posix()
            rows.append(counter_row)
    kernels = _summarize_rocprof_counter_rows_by_kernel(rows)
    return {
        "name": name,
        "path": paths[0].as_posix() if len(paths) == 1 else None,
        "paths": [path.as_posix() for path in paths],
        "row_count": len(rows),
        "dispatch_count": len(
            {(row.get("path"), row.get("correlation_id")) for row in rows}
        ),
        "counter_count": len(
            {
                row.get("counter_name")
                for row in rows
                if isinstance(row.get("counter_name"), str)
            }
        ),
        "kernel_count": len(kernels),
        "kernels": kernels[: max(top_kernel_count, 0)],
    }


def _summarize_rocprof_counter_collection_row(
    row: Mapping[str, str],
) -> dict[str, Any]:
    start_timestamp = _parse_numeric_scalar(row.get("Start_Timestamp"))
    end_timestamp = _parse_numeric_scalar(row.get("End_Timestamp"))
    duration_ns = None
    if isinstance(start_timestamp, int) and isinstance(end_timestamp, int):
        duration_ns = end_timestamp - start_timestamp
    return {
        "correlation_id": _parse_numeric_scalar(row.get("Correlation_Id")),
        "dispatch_id": _parse_numeric_scalar(row.get("Dispatch_Id")),
        "kernel_id": _parse_numeric_scalar(row.get("Kernel_Id")),
        "kernel_name": row.get("Kernel_Name"),
        "duration_ns": duration_ns,
        "local_memory_bytes": _parse_numeric_scalar(row.get("LDS_Block_Size")),
        "scratch_bytes": _parse_numeric_scalar(row.get("Scratch_Size")),
        "vgpr_count": _parse_numeric_scalar(row.get("VGPR_Count")),
        "accum_vgpr_count": _parse_numeric_scalar(row.get("Accum_VGPR_Count")),
        "sgpr_count": _parse_numeric_scalar(row.get("SGPR_Count")),
        "workgroup_size": _parse_numeric_scalar(row.get("Workgroup_Size")),
        "grid_size": _parse_numeric_scalar(row.get("Grid_Size")),
        "counter_name": row.get("Counter_Name"),
        "counter_value": _parse_numeric_scalar(row.get("Counter_Value")),
    }


def _summarize_rocprof_counter_rows_by_kernel(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        kernel_name = row.get("kernel_name")
        if not isinstance(kernel_name, str) or not kernel_name:
            continue
        grouped.setdefault(kernel_name, []).append(row)
    summaries = [
        _summarize_rocprof_counter_kernel_group(kernel_name, kernel_rows)
        for kernel_name, kernel_rows in grouped.items()
    ]
    return sorted(
        summaries,
        key=lambda row: _as_number(row.get("total_duration_ns")) or 0,
        reverse=True,
    )


def _summarize_rocprof_counter_kernel_group(
    kernel_name: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    dispatches: dict[Any, Mapping[str, Any]] = {}
    counters: dict[str, list[float | int]] = {}
    for row in rows:
        dispatch_key = (row.get("path"), row.get("correlation_id"))
        if dispatch_key[1] is not None:
            dispatches.setdefault(dispatch_key, row)
        counter_name = row.get("counter_name")
        counter_value = row.get("counter_value")
        if isinstance(counter_name, str) and isinstance(counter_value, (int, float)):
            counters.setdefault(counter_name, []).append(counter_value)
    dispatch_rows = list(dispatches.values())
    durations = [
        duration
        for dispatch in dispatch_rows
        if isinstance((duration := dispatch.get("duration_ns")), (int, float))
    ]
    return {
        "kernel_name": kernel_name,
        "count": len(dispatch_rows),
        "total_duration_ns": sum(durations),
        "p50_duration_ns": _percentile(durations, 0.50),
        "p90_duration_ns": _percentile(durations, 0.90),
        "min_duration_ns": min(durations) if durations else None,
        "max_duration_ns": max(durations) if durations else None,
        "local_memory_bytes": _max_numeric_field(dispatch_rows, "local_memory_bytes"),
        "scratch_bytes": _max_numeric_field(dispatch_rows, "scratch_bytes"),
        "vgpr_count": _max_numeric_field(dispatch_rows, "vgpr_count"),
        "accum_vgpr_count": _max_numeric_field(dispatch_rows, "accum_vgpr_count"),
        "sgpr_count": _max_numeric_field(dispatch_rows, "sgpr_count"),
        "workgroup_size": _max_numeric_field(dispatch_rows, "workgroup_size"),
        "grid_size": _max_numeric_field(dispatch_rows, "grid_size"),
        "counters": {
            counter_name: _summarize_numeric_values(values)
            for counter_name, values in sorted(counters.items())
        },
    }


def _max_numeric_field(
    rows: Sequence[Mapping[str, Any]], field: str
) -> float | int | None:
    values = [
        value for row in rows if isinstance((value := row.get(field)), (int, float))
    ]
    return max(values) if values else None


def _summarize_numeric_values(values: Sequence[float | int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "sum": sum(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _percentile(values: Sequence[float | int], percentile: float) -> float | int | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, math.ceil(percentile * len(values)) - 1))
    return sorted_values[index]


_KERNEL_BLOCK_PATTERN = re.compile(r"^\s+- \.args:\s*$")
_KERNEL_FIELD_PATTERN = re.compile(r"^    \.(\w+):\s*(.*)$")
_INTEGER_FIELDS = frozenset(
    {
        "group_segment_fixed_size",
        "kernarg_segment_align",
        "kernarg_segment_size",
        "max_flat_workgroup_size",
        "private_segment_fixed_size",
        "sgpr_count",
        "sgpr_spill_count",
        "vgpr_count",
        "vgpr_spill_count",
        "wavefront_size",
    }
)


def summarize_amdhsa_metadata_path(
    named_path: NamedPath,
    metadata_regex: re.Pattern[str] | None,
    top_kernel_count: int,
) -> dict[str, Any]:
    """Extracts kernel resource metadata from `llvm-readobj --notes` text."""

    kernels = tuple(
        kernel
        for kernel in _iter_amdhsa_metadata_kernels(_read_text(named_path.path))
        if _matches_metadata_regex(kernel, metadata_regex)
    )
    return {
        "name": named_path.name,
        "path": named_path.path.as_posix(),
        "kernel_count": len(kernels),
        "kernels": list(kernels[: max(top_kernel_count, 0)]),
    }


def summarize_disassembly_path(
    named_path: NamedPath,
    symbol_regex: re.Pattern[str] | None,
    symbol_weight_specs: Sequence[SymbolWeightSpec],
    weighted_symbol_group_specs: Sequence[WeightedSymbolGroupSpec],
    top_symbol_count: int,
    ordered_symbol_count: int,
) -> dict[str, Any]:
    """Builds a JSON-serializable disassembly summary for one artifact."""

    text = _read_text(named_path.path)
    whole_file = summarize_amdgpu_disassembly(text)
    blocks = summarize_amdgpu_disassembly_blocks(text)
    if symbol_regex is not None:
        blocks = tuple(block for block in blocks if symbol_regex.search(block.symbol))
    top_blocks = _top_blocks(blocks, top_symbol_count)
    summary = {
        "name": named_path.name,
        "path": named_path.path.as_posix(),
        "whole_file": whole_file.metadata(),
        "symbol_count": len(blocks),
        "top_symbols": [block.metadata() for block in top_blocks],
        "ordered_symbols": [
            block.metadata() for block in _ordered_blocks(blocks, ordered_symbol_count)
        ],
    }
    weighted_symbols = _summarize_weighted_symbols(blocks, symbol_weight_specs)
    if weighted_symbols is not None:
        summary["weighted_symbols"] = weighted_symbols
    weighted_symbol_groups = _summarize_weighted_symbol_groups(
        blocks, weighted_symbol_group_specs
    )
    if weighted_symbol_groups:
        summary["weighted_symbol_groups"] = weighted_symbol_groups
    return summary


def build_kernel_anatomy_report(
    disassembly_paths: Sequence[NamedPath],
    compile_report_paths: Sequence[NamedPath],
    benchmark_jsonl_paths: Sequence[NamedPath] = (),
    iree_dispatch_profile_paths: Sequence[NamedPath] = (),
    rocprof_kernel_trace_paths: Sequence[NamedPath] = (),
    rocprof_counter_collection_paths: Sequence[NamedPath] = (),
    rocblas_log_paths: Sequence[NamedPath] = (),
    symbol_regex: str | None = None,
    amdhsa_metadata_paths: Sequence[NamedPath] = (),
    amdhsa_metadata_regex: str | None = None,
    symbol_weight_specs: Sequence[SymbolWeightSpec] = (),
    weighted_symbol_group_specs: Sequence[WeightedSymbolGroupSpec] = (),
    top_symbol_count: int = 16,
    ordered_symbol_count: int = 0,
) -> dict[str, Any]:
    """Builds a stable anatomy report for dumped target artifacts."""

    compiled_symbol_regex = re.compile(symbol_regex) if symbol_regex else None
    compiled_metadata_regex = (
        re.compile(amdhsa_metadata_regex) if amdhsa_metadata_regex else None
    )
    compile_reports = {
        named_path.name: summarize_loom_compile_report(named_path.path)
        for named_path in compile_report_paths
    }
    benchmark_jsonl = {
        named_path.name: summarize_loom_benchmark_jsonl(named_path.path)
        for named_path in benchmark_jsonl_paths
    }
    iree_dispatch_profiles = {
        named_path.name: summarize_iree_dispatch_profile_path(
            named_path, top_symbol_count
        )
        for named_path in iree_dispatch_profile_paths
    }
    rocprof_kernel_traces = {
        name: summarize_rocprof_kernel_trace_paths(name, paths, top_symbol_count)
        for name, paths in _group_named_paths(rocprof_kernel_trace_paths).items()
    }
    rocprof_counter_collections = {
        name: summarize_rocprof_counter_collection_paths(name, paths, top_symbol_count)
        for name, paths in _group_named_paths(rocprof_counter_collection_paths).items()
    }
    rocblas_logs: dict[str, Any] = {}
    for named_path in rocblas_log_paths:
        summary = summarize_rocblas_log_path(named_path)
        existing_summary = rocblas_logs.get(named_path.name)
        if isinstance(existing_summary, dict):
            _merge_rocblas_log_summaries(existing_summary, summary)
        else:
            rocblas_logs[named_path.name] = summary
    amdhsa_metadata = {
        named_path.name: summarize_amdhsa_metadata_path(
            named_path, compiled_metadata_regex, top_symbol_count
        )
        for named_path in amdhsa_metadata_paths
    }
    disassemblies = {
        named_path.name: summarize_disassembly_path(
            named_path,
            compiled_symbol_regex,
            tuple(
                spec
                for spec in symbol_weight_specs
                if spec.disassembly_name == named_path.name
            ),
            tuple(
                spec
                for spec in weighted_symbol_group_specs
                if spec.disassembly_name == named_path.name
            ),
            top_symbol_count,
            ordered_symbol_count,
        )
        for named_path in disassembly_paths
    }
    return {
        "schema": "loom.kernel_anatomy",
        "schema_version": SCHEMA_VERSION,
        "amdhsa_metadata": amdhsa_metadata,
        "disassemblies": disassemblies,
        "iree_dispatch_profiles": iree_dispatch_profiles,
        "loom_benchmarks": benchmark_jsonl,
        "loom_compile_reports": compile_reports,
        "rocblas_logs": rocblas_logs,
        "rocprof_counter_collections": rocprof_counter_collections,
        "rocprof_kernel_traces": rocprof_kernel_traces,
    }


def _group_named_paths(named_paths: Sequence[NamedPath]) -> dict[str, list[Path]]:
    grouped_paths: dict[str, list[Path]] = {}
    for named_path in named_paths:
        grouped_paths.setdefault(named_path.name, []).append(named_path.path)
    return grouped_paths


def build_kernel_anatomy_comparisons(
    report: Mapping[str, Any], comparison_specs: Sequence[ComparisonSpec]
) -> dict[str, Any]:
    """Builds ranked numeric metric deltas between named artifact groups."""

    metric_groups = _collect_comparison_metric_groups(report)
    comparisons: dict[str, Any] = {}
    for spec in comparison_specs:
        name = f"{spec.baseline}={spec.candidate}"
        baseline_metrics = metric_groups.get(spec.baseline, {})
        candidate_metrics = metric_groups.get(spec.candidate, {})
        common_metrics = sorted(set(baseline_metrics) & set(candidate_metrics))
        deltas = [
            _build_metric_delta(
                metric,
                baseline_metrics[metric],
                candidate_metrics[metric],
            )
            for metric in common_metrics
        ]
        comparisons[name] = {
            "baseline": spec.baseline,
            "candidate": spec.candidate,
            "baseline_metric_count": len(baseline_metrics),
            "candidate_metric_count": len(candidate_metrics),
            "shared_metric_count": len(common_metrics),
            "deltas": sorted(deltas, key=_metric_delta_rank, reverse=True),
            "scorecard": _build_comparison_scorecard(deltas),
            "missing_baseline_metrics": sorted(
                set(candidate_metrics) - set(baseline_metrics)
            ),
            "missing_candidate_metrics": sorted(
                set(baseline_metrics) - set(candidate_metrics)
            ),
        }
    return comparisons


def build_kernel_anatomy_comparison_scorecard(
    comparisons: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Builds one ranked scorecard across all named metric comparisons."""

    scorecard: list[dict[str, Any]] = []
    for comparison_name, comparison_value in comparisons.items():
        comparison = _as_mapping(comparison_value)
        entries = comparison.get("scorecard", [])
        if not isinstance(entries, list):
            continue
        for entry_value in entries:
            entry = dict(_as_mapping(entry_value))
            if not entry:
                continue
            entry["comparison"] = comparison_name
            entry["baseline_group"] = comparison.get("baseline")
            entry["candidate_group"] = comparison.get("candidate")
            scorecard.append(entry)
    return sorted(scorecard, key=_scorecard_entry_rank, reverse=True)


def _collect_comparison_metric_groups(
    report: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    _collect_compile_report_metric_groups(groups, report)
    _collect_benchmark_metric_groups(groups, report)
    _collect_iree_dispatch_profile_metric_groups(groups, report)
    _collect_rocprof_kernel_trace_metric_groups(groups, report)
    _collect_rocprof_counter_collection_metric_groups(groups, report)
    _collect_rocblas_metric_groups(groups, report)
    _collect_disassembly_metric_groups(groups, report)
    _collect_metadata_metric_groups(groups, report)
    _record_per_matrix_metric_groups(groups)
    return groups


def _as_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (int, float)) else None


def _record_metric(
    groups: dict[str, dict[str, dict[str, Any]]],
    group_name: str,
    metric: str,
    value: Any,
    source: str,
) -> None:
    numeric_value = _as_number(value)
    if numeric_value is None:
        return
    group = groups.setdefault(group_name, {})
    if metric in group and group[metric]["value"] != numeric_value:
        group[f"{source}.{metric}"] = {
            "value": numeric_value,
            "source": source,
        }
        return
    group[metric] = {
        "value": numeric_value,
        "source": source,
    }


_PER_MATRIX_NUMERATOR_METRICS = (
    "barrier_count",
    "branch_count",
    "buffer_load_count",
    "buffer_store_count",
    "code_byte_count",
    "conversion_count",
    "device_memory_load_count",
    "device_memory_store_count",
    "flat_load_count",
    "flat_store_count",
    "global_load_count",
    "global_store_count",
    "instruction_count",
    "local_memory_access_bytes",
    "local_memory_instruction_count",
    "read_bytes",
    "register_move_count",
    "vector_alu_count",
    "wait_action_count",
    "wait_drained_count",
    "wait_full_drain_count",
    "wait_partial_wait_count",
    "write_bytes",
)


def _record_per_matrix_metric_groups(
    groups: dict[str, dict[str, dict[str, Any]]],
) -> None:
    for group_name, group in list(groups.items()):
        matrix_count = _metric_value(group, "matrix_instruction_count")
        if matrix_count is None or matrix_count <= 0:
            continue
        for numerator_metric in _PER_MATRIX_NUMERATOR_METRICS:
            numerator = _metric_value(group, numerator_metric)
            if numerator is None:
                continue
            _record_metric(
                groups,
                group_name,
                f"{numerator_metric}_per_matrix_instruction",
                numerator / matrix_count,
                "derived",
            )


def _metric_value(
    group: Mapping[str, Mapping[str, Any]], metric: str
) -> float | int | None:
    return _as_number(_as_mapping(group.get(metric)).get("value"))


def _collect_compile_report_metric_groups(
    groups: dict[str, dict[str, dict[str, Any]]], report: Mapping[str, Any]
) -> None:
    compile_reports = _as_mapping(report.get("loom_compile_reports"))
    for name, compile_report_value in compile_reports.items():
        compile_report = _as_mapping(compile_report_value)
        _record_compile_summary_metric_group(
            groups, name, compile_report, "compile_report"
        )


def _record_compile_summary_metric_group(
    groups: dict[str, dict[str, dict[str, Any]]],
    name: str,
    compile_report: Mapping[str, Any],
    source_prefix: str,
) -> None:
    _record_metric(
        groups,
        name,
        "instruction_count",
        compile_report.get("instruction_count"),
        source_prefix,
    )
    _record_metric(
        groups,
        name,
        "code_byte_count",
        compile_report.get("code_byte_count"),
        source_prefix,
    )
    _record_metric(
        groups,
        name,
        "local_memory_bytes",
        compile_report.get("local_memory_bytes"),
        source_prefix,
    )
    _record_metric(
        groups,
        name,
        "private_memory_bytes",
        compile_report.get("private_memory_bytes"),
        source_prefix,
    )
    static_mix = _as_mapping(compile_report.get("static_instruction_mix"))
    _record_metric(
        groups,
        name,
        "wmma_count",
        static_mix.get("wmma_count"),
        source_prefix,
    )
    _record_metric(
        groups,
        name,
        "mfma_count",
        static_mix.get("mfma_count"),
        source_prefix,
    )
    _record_metric(
        groups,
        name,
        "matrix_instruction_count",
        static_mix.get("matrix_count"),
        source_prefix,
    )
    _record_metric(
        groups,
        name,
        "vector_alu_count",
        static_mix.get("vector_alu_count"),
        source_prefix,
    )
    dynamic_mix = _as_mapping(compile_report.get("dynamic_instruction_mix"))
    _record_metric(
        groups,
        name,
        "dynamic_local_memory_count",
        dynamic_mix.get("local_memory_count"),
        source_prefix,
    )
    _record_metric(
        groups,
        name,
        "local_memory_instruction_count",
        static_mix.get("local_memory_count"),
        source_prefix,
    )
    _record_metric(
        groups,
        name,
        "dynamic_local_memory_instruction_count",
        dynamic_mix.get("local_memory_count"),
        source_prefix,
    )
    _record_metric(
        groups,
        name,
        "dynamic_local_memory_read_bytes",
        dynamic_mix.get("local_read_byte_count"),
        source_prefix,
    )
    _record_metric(
        groups,
        name,
        "dynamic_local_memory_write_bytes",
        dynamic_mix.get("local_write_byte_count"),
        source_prefix,
    )
    dynamic_local_read_bytes = _as_number(dynamic_mix.get("local_read_byte_count"))
    dynamic_local_write_bytes = _as_number(dynamic_mix.get("local_write_byte_count"))
    if dynamic_local_read_bytes is not None and dynamic_local_write_bytes is not None:
        _record_metric(
            groups,
            name,
            "dynamic_local_memory_access_bytes",
            dynamic_local_read_bytes + dynamic_local_write_bytes,
            source_prefix,
        )
    static_local_read_bytes = _as_number(static_mix.get("local_read_byte_count"))
    static_local_write_bytes = _as_number(static_mix.get("local_write_byte_count"))
    if static_local_read_bytes is not None and static_local_write_bytes is not None:
        _record_metric(
            groups,
            name,
            "local_memory_access_bytes",
            static_local_read_bytes + static_local_write_bytes,
            source_prefix,
        )
    resources = _as_mapping(compile_report.get("target_resources"))
    vector = _as_mapping(resources.get("vector"))
    vector_final = _as_mapping(vector.get("final"))
    _record_metric(
        groups,
        name,
        "vgpr_count",
        vector_final.get("register_count"),
        source_prefix,
    )
    _record_metric(
        groups,
        name,
        "occupancy_percent",
        resources.get("occupancy_percent"),
        source_prefix,
    )
    wait_plan = _as_mapping(compile_report.get("wait_plan"))
    for metric in _WAIT_REASON_SUMMARY_METRICS:
        _record_metric(
            groups,
            name,
            f"wait_{metric}",
            wait_plan.get(metric),
            f"{source_prefix}_wait_plan",
        )
    _record_source_low_metrics(
        groups,
        name,
        _as_mapping(compile_report.get("source_low")),
        f"{source_prefix}_source_low",
    )
    _record_wait_reason_metrics(
        groups,
        name,
        compile_report.get("wait_reasons"),
        f"{source_prefix}_wait_reason",
    )
    _record_compile_instruction_mix_metrics(
        groups,
        f"{name}/dynamic",
        dynamic_mix,
        f"{source_prefix}_dynamic",
    )


def _record_wait_reason_metrics(
    groups: dict[str, dict[str, dict[str, Any]]],
    group_name: str,
    wait_reasons_value: Any,
    source: str,
) -> None:
    if not isinstance(wait_reasons_value, list):
        return
    for wait_reason_value in wait_reasons_value:
        wait_reason = _as_mapping(wait_reason_value)
        counter = wait_reason.get("counter")
        reason = wait_reason.get("reason")
        if not isinstance(counter, str) or not isinstance(reason, str):
            continue
        reason_group = f"{group_name}/wait/{counter}/{reason}"
        for metric in _WAIT_REASON_SUMMARY_METRICS:
            _record_metric(
                groups,
                reason_group,
                metric,
                wait_reason.get(metric),
                source,
            )


def _record_source_low_metrics(
    groups: dict[str, dict[str, dict[str, Any]]],
    group_name: str,
    source_low: Mapping[str, Any],
    source: str,
) -> None:
    for metric in _SOURCE_LOW_TOTAL_METRICS:
        _record_metric(
            groups,
            group_name,
            f"source_low_{metric}",
            source_low.get(metric),
            source,
        )
    for nested_name in (
        "dispatch_source",
        "dispatch_issued",
        "interval_envelope",
        "read_interval_envelope",
        "write_interval_envelope",
    ):
        nested = _as_mapping(source_low.get(nested_name))
        for metric in _SOURCE_LOW_NESTED_METRICS:
            _record_metric(
                groups,
                group_name,
                f"source_low_{nested_name}_{metric}",
                nested.get(metric),
                source,
            )
    argument_packets = source_low.get("argument_packets")
    if not isinstance(argument_packets, list):
        return
    for argument_packet_value in argument_packets:
        argument_packet = _as_mapping(argument_packet_value)
        root_name = argument_packet.get("root_argument_name")
        operation = argument_packet.get("operation")
        if not isinstance(root_name, str) or not isinstance(operation, str):
            continue
        argument_group = f"{group_name}/source/{root_name}/{operation}"
        for metric in _SOURCE_LOW_TOTAL_METRICS:
            _record_metric(
                groups,
                argument_group,
                metric,
                argument_packet.get(metric),
                source,
            )
        for nested_name in ("dispatch_source", "dispatch_issued", "interval_envelope"):
            nested = _as_mapping(argument_packet.get(nested_name))
            for metric in _SOURCE_LOW_NESTED_METRICS:
                _record_metric(
                    groups,
                    argument_group,
                    f"{nested_name}_{metric}",
                    nested.get(metric),
                    source,
                )


def _record_compile_instruction_mix_metrics(
    groups: dict[str, dict[str, dict[str, Any]]],
    group_name: str,
    mix: Mapping[str, Any],
    source: str,
) -> None:
    _record_metric(
        groups,
        group_name,
        "matrix_instruction_count",
        mix.get("matrix_count"),
        source,
    )
    _record_metric(groups, group_name, "wmma_count", mix.get("wmma_count"), source)
    _record_metric(groups, group_name, "mfma_count", mix.get("mfma_count"), source)
    _record_metric(
        groups,
        group_name,
        "vector_alu_count",
        mix.get("vector_alu_count"),
        source,
    )
    for metric in (
        "global_load_count",
        "global_store_count",
        "buffer_load_count",
        "buffer_store_count",
        "flat_load_count",
        "flat_store_count",
        "barrier_count",
        "branch_count",
        "conversion_count",
        "register_move_count",
    ):
        _record_metric(groups, group_name, metric, mix.get(metric), source)
    device_memory_load_count = sum(
        _as_number(mix.get(metric)) or 0
        for metric in ("global_load_count", "buffer_load_count", "flat_load_count")
    )
    _record_metric(
        groups,
        group_name,
        "device_memory_load_count",
        device_memory_load_count,
        source,
    )
    device_memory_store_count = sum(
        _as_number(mix.get(metric)) or 0
        for metric in ("global_store_count", "buffer_store_count", "flat_store_count")
    )
    _record_metric(
        groups,
        group_name,
        "device_memory_store_count",
        device_memory_store_count,
        source,
    )
    _record_metric(
        groups,
        group_name,
        "local_memory_instruction_count",
        mix.get("local_memory_count"),
        source,
    )
    _record_metric(
        groups,
        group_name,
        "read_bytes",
        mix.get("memory_read_byte_count"),
        source,
    )
    _record_metric(
        groups,
        group_name,
        "write_bytes",
        mix.get("memory_write_byte_count"),
        source,
    )
    local_memory_read_bytes = _as_number(mix.get("local_read_byte_count"))
    local_memory_write_bytes = _as_number(mix.get("local_write_byte_count"))
    if local_memory_read_bytes is not None and local_memory_write_bytes is not None:
        _record_metric(
            groups,
            group_name,
            "local_memory_access_bytes",
            local_memory_read_bytes + local_memory_write_bytes,
            source,
        )


def _collect_benchmark_metric_groups(
    groups: dict[str, dict[str, dict[str, Any]]], report: Mapping[str, Any]
) -> None:
    benchmark_reports = _as_mapping(report.get("loom_benchmarks"))
    for name, benchmark_report_value in benchmark_reports.items():
        benchmark_report = _as_mapping(benchmark_report_value)
        benchmarks = benchmark_report.get("benchmarks", [])
        if not isinstance(benchmarks, list) or len(benchmarks) != 1:
            continue
        benchmark = _as_mapping(benchmarks[0])
        timing = _as_mapping(benchmark.get("timing_ns"))
        _record_metric(groups, name, "p50_ns", timing.get("p50"), "benchmark")
        _record_metric(
            groups,
            name,
            "operation_time_ns",
            timing.get("p50"),
            "benchmark",
        )
        _record_metric(groups, name, "p90_ns", timing.get("p90"), "benchmark")
        _record_metric(groups, name, "sample_count", timing.get("count"), "benchmark")
        correctness = _as_mapping(benchmark.get("correctness"))
        _record_metric(
            groups,
            name,
            "failed_sample_count",
            correctness.get("failed_sample_count"),
            "benchmark",
        )
        compile_report = _as_mapping(benchmark.get("compile_report"))
        if compile_report:
            _record_compile_summary_metric_group(
                groups, name, compile_report, "benchmark_compile_report"
            )


def _collect_iree_dispatch_profile_metric_groups(
    groups: dict[str, dict[str, dict[str, Any]]], report: Mapping[str, Any]
) -> None:
    iree_dispatch_profiles = _as_mapping(report.get("iree_dispatch_profiles"))
    for name, profile_value in iree_dispatch_profiles.items():
        profile = _as_mapping(profile_value)
        kernels = profile.get("kernels", [])
        if not isinstance(kernels, list):
            continue
        for kernel_value in kernels:
            kernel = _as_mapping(kernel_value)
            function_name = kernel.get("function_name")
            if not isinstance(function_name, str):
                continue
            group_name = f"{name}/{function_name}"
            _record_metric(
                groups,
                group_name,
                "operation_time_ns",
                kernel.get("p50_duration_ns"),
                "iree_dispatch_profile",
            )
            for metric in (
                "count",
                "valid_count",
                "invalid_count",
                "total_duration_ns",
                "p50_duration_ns",
                "p90_duration_ns",
                "p99_duration_ns",
                "min_duration_ns",
                "max_duration_ns",
            ):
                _record_metric(
                    groups,
                    group_name,
                    metric,
                    kernel.get(metric),
                    "iree_dispatch_profile",
                )


def _collect_rocprof_kernel_trace_metric_groups(
    groups: dict[str, dict[str, dict[str, Any]]], report: Mapping[str, Any]
) -> None:
    rocprof_kernel_traces = _as_mapping(report.get("rocprof_kernel_traces"))
    for name, trace_value in rocprof_kernel_traces.items():
        trace = _as_mapping(trace_value)
        kernels = trace.get("kernels", [])
        if not isinstance(kernels, list):
            continue
        for kernel_value in kernels:
            kernel = _as_mapping(kernel_value)
            kernel_name = kernel.get("kernel_name")
            if not isinstance(kernel_name, str):
                continue
            group_name = f"{name}/{kernel_name}"
            _record_metric(
                groups,
                group_name,
                "operation_time_ns",
                kernel.get("p50_duration_ns"),
                "rocprof_kernel_trace",
            )
            for metric in (
                "count",
                "total_duration_ns",
                "p50_duration_ns",
                "p90_duration_ns",
                "min_duration_ns",
                "max_duration_ns",
                "local_memory_bytes",
                "scratch_bytes",
                "vgpr_count",
                "accum_vgpr_count",
                "sgpr_count",
                "workgroup_size",
                "grid_size",
            ):
                _record_metric(
                    groups,
                    group_name,
                    metric,
                    kernel.get(metric),
                    "rocprof_kernel_trace",
                )


def _collect_rocprof_counter_collection_metric_groups(
    groups: dict[str, dict[str, dict[str, Any]]], report: Mapping[str, Any]
) -> None:
    counter_collections = _as_mapping(report.get("rocprof_counter_collections"))
    for name, collection_value in counter_collections.items():
        collection = _as_mapping(collection_value)
        kernels = collection.get("kernels", [])
        if not isinstance(kernels, list):
            continue
        for kernel_value in kernels:
            kernel = _as_mapping(kernel_value)
            kernel_name = kernel.get("kernel_name")
            if not isinstance(kernel_name, str):
                continue
            group_name = f"{name}/{kernel_name}"
            _record_metric(
                groups,
                group_name,
                "operation_time_ns",
                kernel.get("p50_duration_ns"),
                "rocprof_counter_collection",
            )
            for metric in (
                "count",
                "total_duration_ns",
                "p50_duration_ns",
                "p90_duration_ns",
                "min_duration_ns",
                "max_duration_ns",
                "local_memory_bytes",
                "scratch_bytes",
                "vgpr_count",
                "accum_vgpr_count",
                "sgpr_count",
                "workgroup_size",
                "grid_size",
            ):
                _record_metric(
                    groups,
                    group_name,
                    metric,
                    kernel.get(metric),
                    "rocprof_counter_collection",
                )
            counters = _as_mapping(kernel.get("counters"))
            for counter_name, counter_value in counters.items():
                counter = _as_mapping(counter_value)
                for statistic in ("sum", "mean", "p50", "p90", "min", "max"):
                    _record_metric(
                        groups,
                        group_name,
                        f"{counter_name}_{statistic}",
                        counter.get(statistic),
                        "rocprof_counter_collection",
                    )


def _collect_rocblas_metric_groups(
    groups: dict[str, dict[str, dict[str, Any]]], report: Mapping[str, Any]
) -> None:
    rocblas_logs = _as_mapping(report.get("rocblas_logs"))
    for name, rocblas_log_value in rocblas_logs.items():
        rocblas_log = _as_mapping(rocblas_log_value)
        _record_metric(
            groups,
            name,
            "solution_index",
            rocblas_log.get("solution_index"),
            "rocblas_log",
        )
        _record_tensile_symbol_parameter_metrics(
            groups,
            name,
            _as_mapping(rocblas_log.get("symbol_parameters")),
            "rocblas_symbol",
        )
        timing = _as_mapping(rocblas_log.get("selected_timing_row"))
        if timing:
            _record_metric(
                groups,
                name,
                "operation_time_ns",
                _microseconds_to_nanoseconds(timing.get("us")),
                "rocblas_log",
            )
            _record_metric(
                groups,
                name,
                "gflops",
                timing.get("rocblas-Gflops"),
                "rocblas_log",
            )
            for metric in ("M", "N", "K", "lda", "ldb", "ldc", "ldd"):
                _record_metric(groups, name, metric, timing.get(metric), "rocblas_log")
        timing_rows = rocblas_log.get("timing_rows", [])
        if isinstance(timing_rows, list):
            for timing_value in timing_rows:
                timing_row = _as_mapping(timing_value)
                group_name = _rocblas_timing_group_name(name, timing_row)
                _record_metric(
                    groups,
                    group_name,
                    "operation_time_ns",
                    _microseconds_to_nanoseconds(timing_row.get("us")),
                    "rocblas_log_timing",
                )
                _record_metric(
                    groups,
                    group_name,
                    "gflops",
                    timing_row.get("rocblas-Gflops"),
                    "rocblas_log_timing",
                )
                for metric in (
                    "M",
                    "N",
                    "K",
                    "lda",
                    "ldb",
                    "ldc",
                    "ldd",
                    "batch_count",
                    "cold_iters",
                    "hot_iters",
                ):
                    _record_metric(
                        groups,
                        group_name,
                        metric,
                        timing_row.get(metric),
                        "rocblas_log_timing",
                    )
        trace_rows = rocblas_log.get("trace_rows", [])
        if not isinstance(trace_rows, list):
            continue
        for trace_value in trace_rows:
            trace_row = _as_mapping(trace_value)
            group_name = _rocblas_trace_group_name(name, trace_row)
            for metric in (
                "M",
                "N",
                "K",
                "lda",
                "ldb",
                "ldc",
                "ldd",
                "batch_count",
                "solution_index",
                "call_count",
            ):
                _record_metric(
                    groups,
                    group_name,
                    metric,
                    trace_row.get(metric),
                    "rocblas_trace",
                )


def _record_tensile_symbol_parameter_metrics(
    groups: dict[str, dict[str, dict[str, Any]]],
    group_name: str,
    parameters: Mapping[str, Any],
    source: str,
) -> None:
    for parameter_name in ("macro_tile", "matrix_instruction", "workgroup_size"):
        parameter = _as_mapping(parameters.get(parameter_name))
        for axis in ("x", "y", "z", "w"):
            _record_metric(
                groups,
                group_name,
                f"{parameter_name}_{axis}",
                parameter.get(axis),
                source,
            )
    thread_tile = _as_mapping(parameters.get("thread_tile"))
    _record_metric(groups, group_name, "thread_tile_x", thread_tile.get("x"), source)
    _record_metric(groups, group_name, "thread_tile_y", thread_tile.get("y"), source)
    for parameter_name in (
        "depth_u",
        "flat_workgroup_size",
        "global_load_vector_width_a",
        "global_load_vector_width_b",
        "isa",
        "local_read_vector_width",
        "local_split_a",
        "local_split_b",
        "vector_width",
        "wave_size",
        "workgroup_mapping",
    ):
        _record_metric(
            groups,
            group_name,
            parameter_name,
            parameters.get(parameter_name),
            source,
        )


def _rocblas_trace_group_name(name: str, trace_row: Mapping[str, Any]) -> str:
    m = trace_row.get("M", "?")
    n = trace_row.get("N", "?")
    k = trace_row.get("K", "?")
    beta = trace_row.get("beta", "?")
    return f"{name}/gemm_M{m}_N{n}_K{k}_beta{beta}"


def _rocblas_timing_group_name(name: str, timing_row: Mapping[str, Any]) -> str:
    m = timing_row.get("M", "?")
    n = timing_row.get("N", "?")
    k = timing_row.get("K", "?")
    beta = timing_row.get("beta", "?")
    return f"{name}/gemm_M{m}_N{n}_K{k}_beta{beta}"


def _microseconds_to_nanoseconds(value: Any) -> float | int | None:
    numeric_value = _as_number(value)
    return numeric_value * 1000 if numeric_value is not None else None


def _collect_disassembly_metric_groups(
    groups: dict[str, dict[str, dict[str, Any]]], report: Mapping[str, Any]
) -> None:
    disassemblies = _as_mapping(report.get("disassemblies"))
    for name, disassembly_value in disassemblies.items():
        disassembly = _as_mapping(disassembly_value)
        whole_file = _as_mapping(disassembly.get("whole_file"))
        _record_metric(
            groups,
            name,
            "instruction_count",
            whole_file.get("instruction_count"),
            "disassembly",
        )
        families = _as_mapping(whole_file.get("family_counts"))
        memory = _as_mapping(whole_file.get("memory_byte_counts"))
        _record_disassembly_family_metrics(
            groups, name, families, memory, "disassembly"
        )
        weighted_symbols = _as_mapping(disassembly.get("weighted_symbols"))
        weighted_summary = _as_mapping(weighted_symbols.get("summary"))
        if weighted_summary:
            weighted_group = f"{name}/weighted_symbols"
            _record_metric(
                groups,
                weighted_group,
                "instruction_count",
                weighted_summary.get("instruction_count"),
                "weighted_disassembly",
            )
            _record_disassembly_family_metrics(
                groups,
                weighted_group,
                _as_mapping(weighted_summary.get("family_counts")),
                _as_mapping(weighted_summary.get("memory_byte_counts")),
                "weighted_disassembly",
            )
        weighted_symbol_groups = _as_mapping(disassembly.get("weighted_symbol_groups"))
        for group_name, weighted_group_value in weighted_symbol_groups.items():
            weighted_group = _as_mapping(weighted_group_value)
            weighted_summary = _as_mapping(weighted_group.get("summary"))
            if not weighted_summary:
                continue
            comparison_group = f"{name}/weighted/{group_name}"
            _record_metric(
                groups,
                comparison_group,
                "instruction_count",
                weighted_summary.get("instruction_count"),
                "weighted_disassembly",
            )
            _record_disassembly_family_metrics(
                groups,
                comparison_group,
                _as_mapping(weighted_summary.get("family_counts")),
                _as_mapping(weighted_summary.get("memory_byte_counts")),
                "weighted_disassembly",
            )


def _record_disassembly_family_metrics(
    groups: dict[str, dict[str, dict[str, Any]]],
    group_name: str,
    families: Mapping[str, Any],
    memory: Mapping[str, Any],
    source: str,
) -> None:
    _record_metric(
        groups,
        group_name,
        "wmma_count",
        families.get("v_wmma"),
        source,
    )
    _record_metric(
        groups,
        group_name,
        "mfma_count",
        families.get("v_mfma"),
        source,
    )
    matrix_count = (_as_number(families.get("v_wmma")) or 0) + (
        _as_number(families.get("v_mfma")) or 0
    )
    _record_metric(groups, group_name, "matrix_instruction_count", matrix_count, source)
    _record_metric(
        groups, group_name, "vector_alu_count", families.get("v_alu"), source
    )
    for family in (
        "global_load",
        "global_store",
        "buffer_load",
        "buffer_store",
        "flat_load",
        "flat_store",
        "ds_read",
        "ds_write",
        "s_waitcnt",
        "s_barrier",
    ):
        _record_metric(
            groups, group_name, f"{family}_count", families.get(family), source
        )
    device_memory_load_count = sum(
        _as_number(families.get(family)) or 0
        for family in ("global_load", "buffer_load", "flat_load")
    )
    _record_metric(
        groups,
        group_name,
        "device_memory_load_count",
        device_memory_load_count,
        source,
    )
    device_memory_store_count = sum(
        _as_number(families.get(family)) or 0
        for family in ("global_store", "buffer_store", "flat_store")
    )
    _record_metric(
        groups,
        group_name,
        "device_memory_store_count",
        device_memory_store_count,
        source,
    )
    local_memory_instruction_count = (_as_number(families.get("ds_read")) or 0) + (
        _as_number(families.get("ds_write")) or 0
    )
    _record_metric(
        groups,
        group_name,
        "local_memory_instruction_count",
        local_memory_instruction_count,
        source,
    )
    _record_metric(
        groups,
        group_name,
        "read_bytes",
        memory.get("read_bytes"),
        source,
    )
    _record_metric(groups, group_name, "write_bytes", memory.get("write_bytes"), source)
    local_memory_read_bytes = _as_number(memory.get("ds_read_bytes"))
    local_memory_write_bytes = _as_number(memory.get("ds_write_bytes"))
    if local_memory_read_bytes is not None and local_memory_write_bytes is not None:
        _record_metric(
            groups,
            group_name,
            "local_memory_access_bytes",
            local_memory_read_bytes + local_memory_write_bytes,
            source,
        )


def _collect_metadata_metric_groups(
    groups: dict[str, dict[str, dict[str, Any]]], report: Mapping[str, Any]
) -> None:
    amdhsa_metadata = _as_mapping(report.get("amdhsa_metadata"))
    for name, metadata_value in amdhsa_metadata.items():
        metadata = _as_mapping(metadata_value)
        kernels = metadata.get("kernels", [])
        if not isinstance(kernels, list) or not kernels:
            continue
        kernel = _as_mapping(kernels[0])
        _record_metric(
            groups,
            name,
            "local_memory_bytes",
            kernel.get("group_segment_fixed_size"),
            "amdhsa_metadata",
        )
        _record_metric(
            groups,
            name,
            "private_memory_bytes",
            kernel.get("private_segment_fixed_size"),
            "amdhsa_metadata",
        )
        _record_metric(
            groups, name, "vgpr_count", kernel.get("vgpr_count"), "amdhsa_metadata"
        )
        _record_metric(
            groups, name, "sgpr_count", kernel.get("sgpr_count"), "amdhsa_metadata"
        )
        _record_metric(
            groups,
            name,
            "vgpr_spill_count",
            kernel.get("vgpr_spill_count"),
            "amdhsa_metadata",
        )
        _record_metric(
            groups,
            name,
            "sgpr_spill_count",
            kernel.get("sgpr_spill_count"),
            "amdhsa_metadata",
        )
        _record_metric(
            groups,
            name,
            "wavefront_size",
            kernel.get("wavefront_size"),
            "amdhsa_metadata",
        )
        _record_metric(
            groups,
            name,
            "workgroup_size",
            kernel.get("max_flat_workgroup_size"),
            "amdhsa_metadata",
        )


def _build_metric_delta(
    metric: str, baseline_metric: Mapping[str, Any], candidate_metric: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_value = baseline_metric["value"]
    candidate_value = candidate_metric["value"]
    delta = candidate_value - baseline_value
    ratio = None
    if baseline_value != 0:
        ratio = candidate_value / baseline_value
    return {
        "metric": metric,
        "baseline": baseline_value,
        "candidate": candidate_value,
        "delta": delta,
        "ratio": ratio,
        "baseline_source": baseline_metric["source"],
        "candidate_source": candidate_metric["source"],
    }


_HIGHER_IS_BETTER_METRICS = frozenset(
    {
        "gflops",
        "occupancy_percent",
    }
)

_EXACT_PARITY_METRICS = frozenset(
    {
        "M",
        "N",
        "K",
        "batch_count",
        "grid_size",
        "depth_u",
        "flat_workgroup_size",
        "global_load_vector_width_a",
        "global_load_vector_width_b",
        "isa",
        "lda",
        "ldb",
        "ldc",
        "ldd",
        "local_read_vector_width",
        "local_split_a",
        "local_split_b",
        "macro_tile_x",
        "macro_tile_y",
        "macro_tile_z",
        "matrix_instruction_w",
        "matrix_instruction_x",
        "matrix_instruction_y",
        "matrix_instruction_z",
        "matrix_instruction_count",
        "mfma_count",
        "solution_index",
        "thread_tile_x",
        "thread_tile_y",
        "vector_width",
        "wavefront_size",
        "wave_size",
        "wmma_count",
        "workgroup_size",
        "workgroup_size_x",
        "workgroup_size_y",
        "workgroup_size_z",
    }
)

_IGNORED_SCORECARD_METRICS = frozenset(
    {
        "call_count",
        "sample_count",
    }
)


def _build_comparison_scorecard(
    deltas: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    scorecard = [
        entry
        for delta in deltas
        if (entry := _build_scorecard_entry(delta)) is not None
    ]
    return sorted(scorecard, key=_scorecard_entry_rank, reverse=True)


def _build_scorecard_entry(delta: Mapping[str, Any]) -> dict[str, Any] | None:
    metric = delta.get("metric")
    if not isinstance(metric, str) or metric in _IGNORED_SCORECARD_METRICS:
        return None
    baseline_value = _as_number(delta.get("baseline"))
    candidate_value = _as_number(delta.get("candidate"))
    if baseline_value is None or candidate_value is None:
        return None
    if _numeric_equal(baseline_value, candidate_value):
        return None
    ratio = delta.get("ratio")
    numeric_ratio = ratio if isinstance(ratio, (int, float)) else None
    goal = _scorecard_metric_goal(metric)
    if goal == "higher":
        if candidate_value >= baseline_value:
            return None
        category = "throughput"
        severity = _safe_inverse(numeric_ratio)
        finding = "candidate_lower"
    elif goal == "exact":
        category = "parity"
        severity = _metric_delta_rank(delta)
        if not math.isfinite(severity):
            severity = None
        finding = "mismatch"
    else:
        if candidate_value <= baseline_value:
            return None
        category = _scorecard_metric_category(metric)
        severity = numeric_ratio
        finding = "candidate_higher"
    return {
        "metric": metric,
        "category": category,
        "finding": finding,
        "goal": goal,
        "baseline": baseline_value,
        "candidate": candidate_value,
        "delta": delta.get("delta"),
        "ratio": numeric_ratio,
        "severity": severity,
        "baseline_source": delta.get("baseline_source"),
        "candidate_source": delta.get("candidate_source"),
    }


def _numeric_equal(lhs: float | int, rhs: float | int) -> bool:
    return math.isclose(float(lhs), float(rhs), rel_tol=1e-9, abs_tol=1e-12)


def _safe_inverse(value: float | int | None) -> float | None:
    if value is None or value == 0:
        return None
    return 1 / value


def _scorecard_metric_goal(metric: str) -> str:
    if metric in _HIGHER_IS_BETTER_METRICS:
        return "higher"
    if metric in _EXACT_PARITY_METRICS:
        return "exact"
    return "lower"


def _scorecard_metric_category(metric: str) -> str:
    if metric in {
        "operation_time_ns",
        "p50_duration_ns",
        "p90_duration_ns",
        "p99_duration_ns",
        "min_duration_ns",
        "max_duration_ns",
        "total_duration_ns",
    }:
        return "time"
    if metric.startswith("wait_") or metric in _WAIT_REASON_SUMMARY_METRICS:
        return "wait"
    if metric.startswith("source_low_") or metric in _SOURCE_LOW_TOTAL_METRICS:
        return "source_memory"
    if metric.startswith(
        (
            "macro_tile_",
            "matrix_instruction_",
            "thread_tile_",
            "workgroup_size_",
        )
    ) or metric in {
        "depth_u",
        "flat_workgroup_size",
        "local_read_vector_width",
        "local_split_a",
        "local_split_b",
        "vector_width",
        "wave_size",
    }:
        return "tiling"
    if "local_memory" in metric or metric.startswith("ds_"):
        return "local_memory"
    if "device_memory" in metric or metric.startswith(("global_", "buffer_", "flat_")):
        return "device_memory"
    if "byte" in metric or metric.endswith("_bytes"):
        return "memory"
    if "vgpr" in metric or "sgpr" in metric or metric == "private_memory_bytes":
        return "resources"
    if "count" in metric or metric.endswith("_instructions"):
        return "instructions"
    return "numeric"


def _scorecard_entry_rank(entry: Mapping[str, Any]) -> float:
    severity = entry.get("severity")
    if isinstance(severity, (int, float)) and severity > 0:
        return severity
    return 1.0


def _metric_delta_rank(delta: Mapping[str, Any]) -> float:
    ratio = delta.get("ratio")
    if isinstance(ratio, (int, float)) and ratio > 0:
        return ratio if ratio >= 1 else 1 / ratio
    return float("inf") if delta.get("delta") else 0.0


def _iter_amdhsa_metadata_kernels(text: str) -> Sequence[dict[str, Any]]:
    kernels: list[dict[str, Any]] = []
    current_kernel: dict[str, Any] | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        if _KERNEL_BLOCK_PATTERN.match(line):
            if current_kernel is not None:
                kernels.append(current_kernel)
            current_kernel = {"start_line": line_number}
            continue
        if current_kernel is None:
            continue
        match = _KERNEL_FIELD_PATTERN.match(line)
        if match is None:
            continue
        key = match.group(1)
        value = match.group(2).strip()
        if key in _INTEGER_FIELDS:
            try:
                current_kernel[key] = int(value, 0)
            except ValueError:
                current_kernel[key] = value
        elif key in {"name", "symbol"}:
            current_kernel[key] = value
    if current_kernel is not None:
        kernels.append(current_kernel)
    return tuple(kernels)


def _matches_metadata_regex(
    kernel: Mapping[str, Any], metadata_regex: re.Pattern[str] | None
) -> bool:
    if metadata_regex is None:
        return True
    name = kernel.get("name")
    symbol = kernel.get("symbol")
    name_matches = isinstance(name, str) and metadata_regex.search(name) is not None
    symbol_matches = (
        isinstance(symbol, str) and metadata_regex.search(symbol) is not None
    )
    return name_matches or symbol_matches


def _top_blocks(
    blocks: Sequence[AmdgpuDisassemblyBlock], top_symbol_count: int
) -> tuple[AmdgpuDisassemblyBlock, ...]:
    if top_symbol_count <= 0:
        return ()
    return tuple(
        sorted(
            blocks,
            key=lambda block: (
                _matrix_count(block),
                block.summary.instruction_count,
                block.symbol,
            ),
            reverse=True,
        )[:top_symbol_count]
    )


def _ordered_blocks(
    blocks: Sequence[AmdgpuDisassemblyBlock], ordered_symbol_count: int
) -> tuple[AmdgpuDisassemblyBlock, ...]:
    if ordered_symbol_count <= 0:
        return ()
    return tuple(
        sorted(
            blocks,
            key=lambda block: (
                block.address is None,
                block.address if block.address is not None else block.start_line,
            ),
        )[:ordered_symbol_count]
    )


def _summarize_weighted_symbols(
    blocks: Sequence[AmdgpuDisassemblyBlock],
    symbol_weight_specs: Sequence[SymbolWeightSpec],
) -> dict[str, Any] | None:
    if not symbol_weight_specs:
        return None
    instruction_count = 0.0
    family_counts: dict[str, float] = {}
    mnemonic_counts: dict[str, float] = {}
    matrix_counts: dict[str, float] = {}
    memory_byte_counts: dict[str, float] = {}
    rules: list[dict[str, Any]] = []
    matched_symbol_count = 0
    for spec in symbol_weight_specs:
        symbol_regex = re.compile(spec.symbol_regex)
        matched_blocks = tuple(
            block for block in blocks if symbol_regex.search(block.symbol)
        )
        matched_symbol_count += len(matched_blocks)
        rules.append(
            {
                "symbol_regex": spec.symbol_regex,
                "weight": spec.weight,
                "matched_symbol_count": len(matched_blocks),
                "matched_symbols": [block.symbol for block in matched_blocks],
            }
        )
        for block in matched_blocks:
            instruction_count += block.summary.instruction_count * spec.weight
            _add_weighted_counts(
                family_counts, block.summary.family_counts, spec.weight
            )
            _add_weighted_counts(
                mnemonic_counts, block.summary.mnemonic_counts, spec.weight
            )
            _add_weighted_counts(
                matrix_counts, block.summary.matrix_mnemonic_counts, spec.weight
            )
            _add_weighted_counts(
                memory_byte_counts, block.summary.memory_byte_counts, spec.weight
            )
    return {
        "rule_count": len(symbol_weight_specs),
        "matched_symbol_count": matched_symbol_count,
        "rules": rules,
        "summary": {
            "instruction_count": _normalize_weighted_number(instruction_count),
            "family_counts": _normalize_weighted_counts(family_counts),
            "mnemonic_counts": _normalize_weighted_counts(mnemonic_counts),
            "matrix_mnemonic_counts": _normalize_weighted_counts(matrix_counts),
            "memory_byte_counts": _normalize_weighted_counts(memory_byte_counts),
        },
    }


def _summarize_weighted_symbol_groups(
    blocks: Sequence[AmdgpuDisassemblyBlock],
    weighted_symbol_group_specs: Sequence[WeightedSymbolGroupSpec],
) -> dict[str, Any]:
    grouped_specs: dict[str, list[SymbolWeightSpec]] = {}
    for spec in weighted_symbol_group_specs:
        grouped_specs.setdefault(spec.group_name, []).append(
            SymbolWeightSpec(
                disassembly_name=spec.disassembly_name,
                symbol_regex=spec.symbol_regex,
                weight=spec.weight,
            )
        )
    summaries: dict[str, Any] = {}
    for group_name, group_specs in sorted(grouped_specs.items()):
        summary = _summarize_weighted_symbols(blocks, group_specs)
        if summary is not None:
            summaries[group_name] = summary
    return summaries


def _add_weighted_counts(
    target: dict[str, float], source: Mapping[str, int], weight: float | int
) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + value * weight


def _normalize_weighted_counts(source: Mapping[str, float]) -> dict[str, float | int]:
    return {
        key: _normalize_weighted_number(value)
        for key, value in sorted(source.items())
        if value
    }


def _normalize_weighted_number(value: float) -> float | int:
    rounded_value = round(value)
    if math.isclose(value, rounded_value):
        return int(rounded_value)
    return value


def _matrix_count(block: AmdgpuDisassemblyBlock) -> int:
    return sum(block.summary.matrix_mnemonic_counts.values())


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _append_compile_report_lines(lines: list[str], report: Mapping[str, Any]) -> None:
    compile_reports = report.get("loom_compile_reports", {})
    if not isinstance(compile_reports, Mapping) or not compile_reports:
        return
    lines.append("")
    lines.append("Loom compile reports:")
    for name, compile_report_value in compile_reports.items():
        compile_report = _as_mapping(compile_report_value)
        resources = _as_mapping(compile_report.get("target_resources"))
        vector = _as_mapping(resources.get("vector"))
        vector_final = _as_mapping(vector.get("final"))
        mix = _as_mapping(compile_report.get("static_instruction_mix"))
        lines.append(
            "  "
            f"{name}: instructions={compile_report.get('instruction_count')} "
            f"code_bytes={compile_report.get('code_byte_count')} "
            f"local_bytes={compile_report.get('local_memory_bytes')} "
            f"vgpr={vector_final.get('register_count')} "
            f"occupancy={resources.get('occupancy_percent')}% "
            f"wmma={mix.get('wmma_count')} "
            f"mfma={mix.get('mfma_count')} "
            f"valu={mix.get('vector_alu_count')}"
        )
        wait_plan = _as_mapping(compile_report.get("wait_plan"))
        if wait_plan:
            lines.append(
                "    "
                f"waits: actions={wait_plan.get('action_count')} "
                f"full={wait_plan.get('full_drain_count')} "
                f"partial={wait_plan.get('partial_wait_count')} "
                f"drained={wait_plan.get('drained_count')} "
                f"max_outstanding={wait_plan.get('max_outstanding_before')}"
            )
        _append_wait_reason_lines(lines, compile_report.get("wait_reasons"))
        _append_source_low_lines(lines, compile_report.get("source_low"))


def _append_wait_reason_lines(lines: list[str], wait_reasons_value: Any) -> None:
    if not isinstance(wait_reasons_value, list) or not wait_reasons_value:
        return
    for wait_reason_value in wait_reasons_value[:6]:
        wait_reason = _as_mapping(wait_reason_value)
        lines.append(
            "    "
            f"wait {wait_reason.get('counter')}/{wait_reason.get('reason')}: "
            f"actions={wait_reason.get('action_count')} "
            f"full={wait_reason.get('full_drain_count')} "
            f"partial={wait_reason.get('partial_wait_count')} "
            f"drained={wait_reason.get('drained_count')} "
            f"max_outstanding={wait_reason.get('max_outstanding_before')}"
        )


def _append_source_low_lines(lines: list[str], source_low_value: Any) -> None:
    source_low = _as_mapping(source_low_value)
    if not source_low:
        return
    dispatch_source = _as_mapping(source_low.get("dispatch_source"))
    lines.append(
        "    "
        f"source-low: packets={source_low.get('packet_count')} "
        f"dynamic_packets={source_low.get('dynamic_packet_count')} "
        f"read_bytes={source_low.get('dynamic_read_byte_count')} "
        f"write_bytes={source_low.get('dynamic_write_byte_count')} "
        f"dispatch_total_bytes={dispatch_source.get('total_bytes')}"
    )
    argument_packets = source_low.get("argument_packets")
    if not isinstance(argument_packets, list):
        return
    for argument_packet_value in argument_packets[:6]:
        argument_packet = _as_mapping(argument_packet_value)
        argument_dispatch_source = _as_mapping(argument_packet.get("dispatch_source"))
        lines.append(
            "      "
            f"{argument_packet.get('root_argument_name')}/"
            f"{argument_packet.get('operation')}: "
            f"packet={argument_packet.get('packet')} "
            f"packets={argument_packet.get('packet_count')} "
            f"dynamic_packets={argument_packet.get('dynamic_packet_count')} "
            f"dispatch_total_bytes={argument_dispatch_source.get('total_bytes')}"
        )


def _format_p50_ms(timing: Mapping[str, Any]) -> str:
    p50 = timing.get("p50")
    return f"{p50 / 1_000_000:.6g}" if isinstance(p50, (int, float)) else "?"


def _append_benchmark_report_lines(lines: list[str], report: Mapping[str, Any]) -> None:
    benchmark_reports = report.get("loom_benchmarks", {})
    if not isinstance(benchmark_reports, Mapping) or not benchmark_reports:
        return
    lines.append("")
    lines.append("Loom benchmark JSONL:")
    for name, benchmark_report_value in benchmark_reports.items():
        benchmark_report = _as_mapping(benchmark_report_value)
        lines.append(
            f"  {name}: benchmarks={benchmark_report.get('benchmark_count')} "
            f"repetitions={benchmark_report.get('repetition_count')} "
            f"comparisons={benchmark_report.get('comparison_count')} "
            f"compiles={benchmark_report.get('compile_count')} "
            f"devices={benchmark_report.get('device_count')}"
        )
        resource_findings = benchmark_report.get("resource_findings")
        if isinstance(resource_findings, list):
            for finding_value in resource_findings:
                finding = _as_mapping(finding_value)
                lines.append(
                    "    "
                    f"resource {finding.get('severity')}: "
                    f"{finding.get('resource')} "
                    f"required={finding.get('required_bytes')} "
                    f"limit={finding.get('limit_bytes')} "
                    f"overage={finding.get('overage_bytes')} "
                    f"device={finding.get('device')} "
                    f"benchmark={finding.get('benchmark')} "
                    f"entry={finding.get('entry')}"
                )
        benchmarks = benchmark_report.get("benchmarks", [])
        if not isinstance(benchmarks, list):
            continue
        for benchmark_value in benchmarks[:8]:
            benchmark = _as_mapping(benchmark_value)
            timing = _as_mapping(benchmark.get("timing_ns"))
            correctness = _as_mapping(benchmark.get("correctness"))
            compile_report = _as_mapping(benchmark.get("compile_report"))
            resources = _as_mapping(compile_report.get("target_resources"))
            vector = _as_mapping(resources.get("vector"))
            vector_final = _as_mapping(vector.get("final"))
            static_mix = _as_mapping(compile_report.get("static_instruction_mix"))
            dynamic_mix = _as_mapping(compile_report.get("dynamic_instruction_mix"))
            lines.append(
                "    "
                f"{benchmark.get('benchmark')}: "
                f"state={benchmark.get('state')} "
                f"p50_ms={_format_p50_ms(timing)} "
                f"failed_samples={correctness.get('failed_sample_count')}/"
                f"{correctness.get('sample_count')} "
                f"instructions={compile_report.get('instruction_count')} "
                f"code_bytes={compile_report.get('code_byte_count')} "
                f"local_bytes={compile_report.get('local_memory_bytes')} "
                f"vgpr={vector_final.get('register_count')} "
                f"occupancy={resources.get('occupancy_percent')}% "
                f"wmma={static_mix.get('wmma_count')} "
                f"valu={static_mix.get('vector_alu_count')} "
                f"dynamic_local={dynamic_mix.get('local_memory_count')}"
            )
        comparisons = benchmark_report.get("comparisons", [])
        if isinstance(comparisons, list) and comparisons:
            lines.append("    benchmark comparisons:")
            for comparison_value in comparisons[:8]:
                comparison = _as_mapping(comparison_value)
                lines.append(
                    "      "
                    f"{comparison.get('comparison_group')}: "
                    f"{comparison.get('method')} "
                    f"{comparison.get('baseline_candidate_id')}->"
                    f"{comparison.get('candidate_id')} "
                    f"base_p50_ms="
                    f"{_format_ns_as_ms(comparison.get('baseline_p50_ns'))} "
                    f"cand_p50_ms="
                    f"{_format_ns_as_ms(comparison.get('candidate_p50_ns'))} "
                    f"ratio_p50={_format_ratio(comparison.get('ratio_p50'))} "
                    f"speedup_p50={_format_ratio(comparison.get('speedup_p50'))} "
                    f"base_reps={comparison.get('baseline_repetition_count')} "
                    f"cand_reps={comparison.get('candidate_repetition_count')} "
                    f"base_spread_ppm="
                    f"{comparison.get('baseline_p50_spread_ppm')} "
                    f"cand_spread_ppm="
                    f"{comparison.get('candidate_p50_spread_ppm')}"
                )
        repetitions = benchmark_report.get("repetitions", [])
        if isinstance(repetitions, list) and repetitions:
            lines.append("    benchmark repetitions:")
            for repetition_value in repetitions[:8]:
                repetition = _as_mapping(repetition_value)
                timing = _as_mapping(repetition.get("timing_ns"))
                lines.append(
                    "      "
                    f"{repetition.get('schedule_token')}"
                    f"{repetition.get('repetition_index')}: "
                    f"{repetition.get('benchmark')} "
                    f"state={repetition.get('state')} "
                    f"p50_ms={_format_p50_ms(timing)} "
                    f"p90_ms={_format_ns_as_ms(timing.get('p90'))}"
                )


def _append_iree_dispatch_profile_lines(
    lines: list[str], report: Mapping[str, Any]
) -> None:
    iree_dispatch_profiles = report.get("iree_dispatch_profiles", {})
    if not isinstance(iree_dispatch_profiles, Mapping) or not iree_dispatch_profiles:
        return
    lines.append("")
    lines.append("IREE dispatch profiles:")
    for name, profile_value in iree_dispatch_profiles.items():
        profile = _as_mapping(profile_value)
        lines.append(f"  {name}: kernels={profile.get('kernel_count')}")
        kernels = profile.get("kernels", [])
        if not isinstance(kernels, list):
            continue
        for kernel_value in kernels:
            kernel = _as_mapping(kernel_value)
            lines.append(
                "    "
                f"{kernel.get('function_name')}: "
                f"count={kernel.get('count')} "
                f"total_ms={_format_ns_as_ms(kernel.get('total_duration_ns'))} "
                f"p50_ms={_format_ns_as_ms(kernel.get('p50_duration_ns'))} "
                f"p90_ms={_format_ns_as_ms(kernel.get('p90_duration_ns'))}"
            )


def _format_ns_as_ms(value: Any) -> str:
    numeric_value = _as_number(value)
    return f"{numeric_value / 1_000_000:.6g}" if numeric_value is not None else "?"


def _append_rocprof_kernel_trace_lines(
    lines: list[str], report: Mapping[str, Any]
) -> None:
    rocprof_kernel_traces = report.get("rocprof_kernel_traces", {})
    if not isinstance(rocprof_kernel_traces, Mapping) or not rocprof_kernel_traces:
        return
    lines.append("")
    lines.append("rocprof kernel traces:")
    for name, trace_value in rocprof_kernel_traces.items():
        trace = _as_mapping(trace_value)
        lines.append(
            "  "
            f"{name}: dispatches={trace.get('dispatch_count')} "
            f"kernels={trace.get('kernel_count')}"
        )
        kernels = trace.get("kernels", [])
        if not isinstance(kernels, list):
            continue
        for kernel_value in kernels:
            kernel = _as_mapping(kernel_value)
            lines.append(
                "    "
                f"{kernel.get('kernel_name')}: "
                f"count={kernel.get('count')} "
                f"total_ms={_format_ns_as_ms(kernel.get('total_duration_ns'))} "
                f"p50_ms={_format_ns_as_ms(kernel.get('p50_duration_ns'))} "
                f"p90_ms={_format_ns_as_ms(kernel.get('p90_duration_ns'))} "
                f"lds={kernel.get('local_memory_bytes')} "
                f"vgpr={kernel.get('vgpr_count')} "
                f"sgpr={kernel.get('sgpr_count')} "
                f"workgroup={kernel.get('workgroup_size')} "
                f"grid={kernel.get('grid_size')}"
            )


_ROCPROF_COUNTER_DISPLAY_ORDER = (
    "SQ_INSTS_LDS",
    "SQ_WAIT_INST_LDS",
    "LDSBankConflict",
    "ALUStalledByLDS",
    "SQ_INSTS_VALU",
    "VALUInsts",
    "MeanOccupancyPerCU",
    "OccupancyPercent",
)


def _append_rocprof_counter_collection_lines(
    lines: list[str], report: Mapping[str, Any]
) -> None:
    counter_collections = report.get("rocprof_counter_collections", {})
    if not isinstance(counter_collections, Mapping) or not counter_collections:
        return
    lines.append("")
    lines.append("rocprof counter collections:")
    for name, collection_value in counter_collections.items():
        collection = _as_mapping(collection_value)
        lines.append(
            "  "
            f"{name}: rows={collection.get('row_count')} "
            f"dispatches={collection.get('dispatch_count')} "
            f"counters={collection.get('counter_count')} "
            f"kernels={collection.get('kernel_count')}"
        )
        kernels = collection.get("kernels", [])
        if not isinstance(kernels, list):
            continue
        for kernel_value in kernels:
            kernel = _as_mapping(kernel_value)
            lines.append(
                "    "
                f"{kernel.get('kernel_name')}: "
                f"count={kernel.get('count')} "
                f"total_ms={_format_ns_as_ms(kernel.get('total_duration_ns'))} "
                f"p50_ms={_format_ns_as_ms(kernel.get('p50_duration_ns'))} "
                f"{_format_rocprof_counter_line(kernel.get('counters'))}"
            )


def _format_rocprof_counter_line(counters_value: Any) -> str:
    counters = _as_mapping(counters_value)
    ordered_names = [
        name for name in _ROCPROF_COUNTER_DISPLAY_ORDER if name in counters
    ] + sorted(set(counters) - set(_ROCPROF_COUNTER_DISPLAY_ORDER))
    fragments: list[str] = []
    for counter_name in ordered_names[:8]:
        counter = _as_mapping(counters.get(counter_name))
        mean = counter.get("mean")
        maximum = counter.get("max")
        fragments.append(
            f"{counter_name}_mean={_format_scalar(mean)} "
            f"{counter_name}_max={_format_scalar(maximum)}"
        )
    return " ".join(fragments)


def _format_scalar(value: Any) -> str:
    numeric_value = _as_number(value)
    if isinstance(numeric_value, int):
        return str(numeric_value)
    if isinstance(numeric_value, float):
        return f"{numeric_value:.6g}"
    return "?"


def _append_rocblas_log_lines(lines: list[str], report: Mapping[str, Any]) -> None:
    rocblas_logs = report.get("rocblas_logs", {})
    if not isinstance(rocblas_logs, Mapping) or not rocblas_logs:
        return
    lines.append("")
    lines.append("rocBLAS logs:")
    for name, rocblas_log_value in rocblas_logs.items():
        rocblas_log = _as_mapping(rocblas_log_value)
        devices = rocblas_log.get("devices", [])
        arch = "?"
        if isinstance(devices, list) and devices:
            arch = _as_mapping(devices[0]).get("arch", "?")
        lines.append(
            "  "
            f"{name}: solution={rocblas_log.get('solution_index')} "
            f"arch={arch} "
            f"kernel={rocblas_log.get('running_kernel')}"
        )
        _append_tensile_symbol_parameter_lines(
            lines, _as_mapping(rocblas_log.get("symbol_parameters"))
        )
        selected_timing = _as_mapping(rocblas_log.get("selected_timing_row"))
        if selected_timing:
            lines.append(
                "    "
                f"selected M={selected_timing.get('M')} "
                f"N={selected_timing.get('N')} "
                f"K={selected_timing.get('K')} "
                f"time_ms={_format_rocblas_time_ms(selected_timing)} "
                f"gflops={selected_timing.get('rocblas-Gflops')}"
            )
        timing_rows = rocblas_log.get("timing_rows", [])
        if isinstance(timing_rows, list):
            for timing_value in timing_rows[:4]:
                timing = _as_mapping(timing_value)
                lines.append(
                    "    "
                    f"M={timing.get('M')} N={timing.get('N')} "
                    f"K={timing.get('K')} "
                    f"time_ms={_format_rocblas_time_ms(timing)} "
                    f"gflops={timing.get('rocblas-Gflops')}"
                )
        kernel_parameters = _as_mapping(rocblas_log.get("kernel_parameters"))
        lines.extend(
            f"    {key}: {kernel_parameters[key]}"
            for key in ("MatrixInstruction", "workGroupSize", "macroTile", "depthU")
            if key in kernel_parameters
        )
        trace_rows = rocblas_log.get("trace_rows", [])
        if isinstance(trace_rows, list) and trace_rows:
            lines.append("    trace rows:")
            for trace_value in _top_rocblas_trace_rows(trace_rows)[:8]:
                trace_row = _as_mapping(trace_value)
                lines.append(
                    "      "
                    f"{trace_row.get('rocblas_function')}: "
                    f"M={trace_row.get('M')} "
                    f"N={trace_row.get('N')} "
                    f"K={trace_row.get('K')} "
                    f"beta={trace_row.get('beta')} "
                    f"solution={trace_row.get('solution_index')} "
                    f"calls={trace_row.get('call_count')}"
                )
                replay_command = trace_row.get("rocblas_bench_command")
                if isinstance(replay_command, str) and replay_command:
                    lines.append(f"        replay: {replay_command}")


def _top_rocblas_trace_rows(trace_rows: Sequence[Any]) -> list[Mapping[str, Any]]:
    return sorted(
        (_as_mapping(trace_value) for trace_value in trace_rows),
        key=lambda trace_row: _as_number(trace_row.get("call_count")) or 0,
        reverse=True,
    )


def _append_tensile_symbol_parameter_lines(
    lines: list[str], parameters: Mapping[str, Any]
) -> None:
    if not parameters:
        return
    lines.append(
        "    "
        f"symbol tile: "
        f"MT={_format_extent(_as_mapping(parameters.get('macro_tile')))} "
        f"MI={_format_extent(_as_mapping(parameters.get('matrix_instruction')))} "
        f"WG={_format_extent(_as_mapping(parameters.get('workgroup_size')))} "
        f"TT={_format_extent(_as_mapping(parameters.get('thread_tile')))} "
        f"WS={parameters.get('wave_size')} "
        f"VW={parameters.get('vector_width')} "
        f"GLVWA={parameters.get('global_load_vector_width_a')} "
        f"GLVWB={parameters.get('global_load_vector_width_b')} "
        f"LRVW={parameters.get('local_read_vector_width')}"
    )


def _format_extent(extent: Mapping[str, Any]) -> str:
    axes = [
        value
        for axis in ("x", "y", "z", "w")
        if (value := extent.get(axis)) is not None
    ]
    return "x".join(str(axis) for axis in axes) if axes else "?"


def _format_rocblas_time_ms(timing: Mapping[str, Any]) -> str:
    time_us = timing.get("us")
    time_ms = time_us / 1000 if isinstance(time_us, (int, float)) else None
    return f"{time_ms:.6g}" if time_ms is not None else "?"


def _format_ratio(ratio: Any) -> str:
    return f"{ratio:.6g}x" if isinstance(ratio, (int, float)) else "?"


def _append_comparison_scorecard_lines(
    lines: list[str], report: Mapping[str, Any]
) -> None:
    scorecard = report.get("comparison_scorecard")
    if scorecard is None:
        comparisons = report.get("comparisons", {})
        if isinstance(comparisons, Mapping):
            scorecard = build_kernel_anatomy_comparison_scorecard(comparisons)
    if not isinstance(scorecard, list) or not scorecard:
        return
    lines.append("")
    lines.append("Comparison scorecard:")
    for entry_value in scorecard[:16]:
        entry = _as_mapping(entry_value)
        lines.append(
            "  "
            f"{entry.get('comparison')} :: "
            f"{entry.get('metric')} [{entry.get('category')}]: "
            f"{entry.get('finding')} "
            f"severity={_format_ratio(entry.get('severity'))} "
            f"baseline={entry.get('baseline')} "
            f"candidate={entry.get('candidate')} "
            f"ratio={_format_ratio(entry.get('ratio'))}"
        )


def _append_comparison_lines(lines: list[str], report: Mapping[str, Any]) -> None:
    comparisons = report.get("comparisons", {})
    if not isinstance(comparisons, Mapping) or not comparisons:
        return
    lines.append("")
    lines.append("Comparisons:")
    for name, comparison_value in comparisons.items():
        comparison = _as_mapping(comparison_value)
        lines.append(
            "  "
            f"{name}: shared={comparison.get('shared_metric_count')} "
            f"baseline_metrics={comparison.get('baseline_metric_count')} "
            f"candidate_metrics={comparison.get('candidate_metric_count')}"
        )
        scorecard = comparison.get("scorecard", [])
        if isinstance(scorecard, list) and scorecard:
            lines.append("    scorecard:")
            for entry_value in scorecard[:12]:
                entry = _as_mapping(entry_value)
                lines.append(
                    "      "
                    f"{entry.get('metric')} [{entry.get('category')}]: "
                    f"{entry.get('finding')} "
                    f"severity={_format_ratio(entry.get('severity'))} "
                    f"baseline={entry.get('baseline')} "
                    f"candidate={entry.get('candidate')} "
                    f"ratio={_format_ratio(entry.get('ratio'))} "
                    f"sources={entry.get('baseline_source')}/"
                    f"{entry.get('candidate_source')}"
                )
        deltas = comparison.get("deltas", [])
        if not isinstance(deltas, list):
            continue
        for delta_value in deltas[:12]:
            delta = _as_mapping(delta_value)
            lines.append(
                "    "
                f"{delta.get('metric')}: "
                f"baseline={delta.get('baseline')} "
                f"candidate={delta.get('candidate')} "
                f"delta={delta.get('delta')} "
                f"ratio={_format_ratio(delta.get('ratio'))} "
                f"sources={delta.get('baseline_source')}/"
                f"{delta.get('candidate_source')}"
            )


def _append_disassembly_symbol_lines(
    lines: list[str], symbols: Any, indent: str
) -> None:
    if not isinstance(symbols, list):
        return
    for symbol_value in symbols[:8]:
        symbol = _as_mapping(symbol_value)
        summary = _as_mapping(symbol.get("summary"))
        symbol_families = _as_mapping(summary.get("family_counts"))
        lines.append(
            f"{indent}{symbol.get('symbol')}: "
            f"instructions={summary.get('instruction_count')} "
            f"wmma={symbol_families.get('v_wmma', 0)} "
            f"mfma={symbol_families.get('v_mfma', 0)} "
            f"global_load={symbol_families.get('global_load', 0)} "
            f"global_store={symbol_families.get('global_store', 0)} "
            f"buffer_load={symbol_families.get('buffer_load', 0)} "
            f"buffer_store={symbol_families.get('buffer_store', 0)} "
            f"ds_read={symbol_families.get('ds_read', 0)} "
            f"ds_write={symbol_families.get('ds_write', 0)}"
        )


def _append_ordered_symbol_lines(lines: list[str], ordered_symbols: Any) -> None:
    if not isinstance(ordered_symbols, list) or not ordered_symbols:
        return
    lines.append("    ordered symbols:")
    for symbol_value in ordered_symbols:
        symbol = _as_mapping(symbol_value)
        summary = _as_mapping(symbol.get("summary"))
        symbol_families = _as_mapping(summary.get("family_counts"))
        address = symbol.get("address")
        address_text = f"0x{address:x}" if isinstance(address, int) else "?"
        lines.append(
            "      "
            f"{address_text} {symbol.get('symbol')}: "
            f"instructions={summary.get('instruction_count')} "
            f"wmma={symbol_families.get('v_wmma', 0)} "
            f"mfma={symbol_families.get('v_mfma', 0)} "
            f"buffer_load={symbol_families.get('buffer_load', 0)} "
            f"buffer_store={symbol_families.get('buffer_store', 0)} "
            f"ds_read={symbol_families.get('ds_read', 0)} "
            f"ds_write={symbol_families.get('ds_write', 0)} "
            f"branch={symbol_families.get('s_branch', 0)}"
        )


def _append_weighted_symbol_lines(
    lines: list[str], weighted_symbols_value: Any
) -> None:
    weighted_symbols = _as_mapping(weighted_symbols_value)
    if not weighted_symbols:
        return
    _append_weighted_symbol_summary_lines(
        lines, "weighted symbols", weighted_symbols, "    "
    )


def _append_weighted_symbol_group_lines(
    lines: list[str], weighted_symbol_groups_value: Any
) -> None:
    weighted_symbol_groups = _as_mapping(weighted_symbol_groups_value)
    if not weighted_symbol_groups:
        return
    lines.append("    weighted symbol groups:")
    for group_name, weighted_symbols_value in weighted_symbol_groups.items():
        weighted_symbols = _as_mapping(weighted_symbols_value)
        if not weighted_symbols:
            continue
        _append_weighted_symbol_summary_lines(
            lines, str(group_name), weighted_symbols, "      "
        )


def _append_weighted_symbol_summary_lines(
    lines: list[str],
    label: str,
    weighted_symbols: Mapping[str, Any],
    indent: str,
) -> None:
    summary = _as_mapping(weighted_symbols.get("summary"))
    families = _as_mapping(summary.get("family_counts"))
    memory = _as_mapping(summary.get("memory_byte_counts"))
    group_text = "" if label == "weighted symbols" else f"group={label} "
    lines.append(
        f"{indent}"
        f"weighted symbols: rules={weighted_symbols.get('rule_count')} "
        f"{group_text}"
        f"matches={weighted_symbols.get('matched_symbol_count')} "
        f"instructions={summary.get('instruction_count')} "
        f"wmma={families.get('v_wmma', 0)} "
        f"mfma={families.get('v_mfma', 0)} "
        f"buffer_load={families.get('buffer_load', 0)} "
        f"buffer_store={families.get('buffer_store', 0)} "
        f"ds_read={families.get('ds_read', 0)} "
        f"ds_write={families.get('ds_write', 0)} "
        f"read_bytes={memory.get('read_bytes', 0)} "
        f"write_bytes={memory.get('write_bytes', 0)}"
    )
    rules = weighted_symbols.get("rules", [])
    if not isinstance(rules, list):
        return
    for rule_value in rules[:8]:
        rule = _as_mapping(rule_value)
        matched_symbols = rule.get("matched_symbols", [])
        if not isinstance(matched_symbols, list):
            matched_symbols = []
        lines.append(
            f"{indent}  "
            f"{rule.get('symbol_regex')}: "
            f"weight={rule.get('weight')} "
            f"matches={rule.get('matched_symbol_count')} "
            f"symbols={','.join(str(symbol) for symbol in matched_symbols[:4])}"
        )


def _append_disassembly_report_lines(
    lines: list[str], report: Mapping[str, Any]
) -> None:
    disassemblies = report.get("disassemblies", {})
    if not isinstance(disassemblies, Mapping) or not disassemblies:
        return
    lines.append("")
    lines.append("Disassemblies:")
    for name, disassembly_value in disassemblies.items():
        disassembly = _as_mapping(disassembly_value)
        whole_file = _as_mapping(disassembly.get("whole_file"))
        families = _as_mapping(whole_file.get("family_counts"))
        memory = _as_mapping(whole_file.get("memory_byte_counts"))
        lines.append(
            "  "
            f"{name}: symbols={disassembly.get('symbol_count')} "
            f"instructions={whole_file.get('instruction_count')} "
            f"wmma={families.get('v_wmma', 0)} "
            f"mfma={families.get('v_mfma', 0)} "
            f"global_load={families.get('global_load', 0)} "
            f"global_store={families.get('global_store', 0)} "
            f"buffer_load={families.get('buffer_load', 0)} "
            f"buffer_store={families.get('buffer_store', 0)} "
            f"ds_read={families.get('ds_read', 0)} "
            f"ds_write={families.get('ds_write', 0)} "
            f"wait={families.get('s_waitcnt', 0)} "
            f"barrier={families.get('s_barrier', 0)} "
            f"read_bytes={memory.get('read_bytes', 0)} "
            f"write_bytes={memory.get('write_bytes', 0)}"
        )
        _append_disassembly_symbol_lines(lines, disassembly.get("top_symbols"), "    ")
        _append_ordered_symbol_lines(lines, disassembly.get("ordered_symbols"))
        _append_weighted_symbol_lines(lines, disassembly.get("weighted_symbols"))
        _append_weighted_symbol_group_lines(
            lines, disassembly.get("weighted_symbol_groups")
        )


def _append_amdhsa_metadata_lines(lines: list[str], report: Mapping[str, Any]) -> None:
    amdhsa_metadata = report.get("amdhsa_metadata", {})
    if not isinstance(amdhsa_metadata, Mapping) or not amdhsa_metadata:
        return
    lines.append("")
    lines.append("AMDHSA metadata:")
    for name, metadata_value in amdhsa_metadata.items():
        metadata = _as_mapping(metadata_value)
        lines.append(f"  {name}: kernels={metadata.get('kernel_count')}")
        kernels = metadata.get("kernels", [])
        if not isinstance(kernels, list):
            continue
        for kernel_value in kernels[:8]:
            kernel = _as_mapping(kernel_value)
            symbol = kernel.get("symbol") or kernel.get("name")
            lines.append(
                "    "
                f"{symbol}: "
                f"lds={kernel.get('group_segment_fixed_size')} "
                f"private={kernel.get('private_segment_fixed_size')} "
                f"vgpr={kernel.get('vgpr_count')} "
                f"sgpr={kernel.get('sgpr_count')} "
                f"vgpr_spills={kernel.get('vgpr_spill_count')} "
                f"sgpr_spills={kernel.get('sgpr_spill_count')} "
                f"wavefront={kernel.get('wavefront_size')} "
                f"workgroup={kernel.get('max_flat_workgroup_size')} "
                f"kernarg={kernel.get('kernarg_segment_size')}"
            )


def format_text_report(report: Mapping[str, Any]) -> str:
    """Formats the anatomy report as concise human-readable text."""

    lines = ["Kernel anatomy report"]
    _append_compile_report_lines(lines, report)
    _append_benchmark_report_lines(lines, report)
    _append_iree_dispatch_profile_lines(lines, report)
    _append_rocprof_kernel_trace_lines(lines, report)
    _append_rocprof_counter_collection_lines(lines, report)
    _append_rocblas_log_lines(lines, report)
    _append_comparison_scorecard_lines(lines, report)
    _append_comparison_lines(lines, report)
    _append_disassembly_report_lines(lines, report)
    _append_amdhsa_metadata_lines(lines, report)
    return "\n".join(lines) + "\n"


def format_rocblas_replay_script(
    report: Mapping[str, Any],
    executable: str = "rocblas-bench",
    extra_arguments: Sequence[str] = (),
) -> str:
    """Formats rocBLAS profile rows as a directly runnable replay script."""

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
    ]
    rocblas_logs = report.get("rocblas_logs", {})
    if isinstance(rocblas_logs, Mapping):
        for name, rocblas_log_value in rocblas_logs.items():
            rocblas_log = _as_mapping(rocblas_log_value)
            trace_rows = rocblas_log.get("trace_rows", [])
            if not isinstance(trace_rows, list):
                continue
            for trace_value in trace_rows:
                trace_row = _as_mapping(trace_value)
                arguments = trace_row.get("rocblas_bench_arguments")
                if not isinstance(arguments, list) or not arguments:
                    continue
                argument_strings = [
                    argument for argument in arguments if isinstance(argument, str)
                ]
                if len(argument_strings) != len(arguments):
                    continue
                shape_key = trace_row.get("shape_key", "?")
                call_count = trace_row.get("call_count", "?")
                lines.append("")
                lines.append(f"# {name} {shape_key} calls={call_count}")
                lines.append(
                    shlex.join([executable, *argument_strings, *extra_arguments])
                )
    return "\n".join(lines) + "\n"


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--disassembly",
        action="append",
        default=[],
        type=_parse_named_path,
        help="Disassembly artifact as NAME=PATH, or PATH to use the stem as NAME.",
    )
    parser.add_argument(
        "--compile-report",
        action="append",
        default=[],
        type=_parse_named_path,
        help="Loom compile report JSON as NAME=PATH, or PATH to use the stem.",
    )
    parser.add_argument(
        "--benchmark-jsonl",
        action="append",
        default=[],
        type=_parse_named_path,
        help=("iree-benchmark-loom JSONL as NAME=PATH, or PATH to use the stem."),
    )
    parser.add_argument(
        "--iree-dispatch-profile",
        action="append",
        default=[],
        type=_parse_named_path,
        help=("IREE dispatch profile JSON as NAME=PATH, or PATH to use the stem."),
    )
    parser.add_argument(
        "--rocprof-kernel-trace",
        action="append",
        default=[],
        type=_parse_named_path,
        help=("rocprof kernel trace CSV as NAME=PATH, or PATH to use the stem."),
    )
    parser.add_argument(
        "--rocprof-counter-collection",
        action="append",
        default=[],
        type=_parse_named_path,
        help=("rocprof counter collection CSV as NAME=PATH, or PATH to use the stem."),
    )
    parser.add_argument(
        "--rocblas-log",
        action="append",
        default=[],
        type=_parse_named_path,
        help="rocBLAS log output as NAME=PATH, or PATH to use the stem.",
    )
    parser.add_argument(
        "--amdhsa-metadata",
        action="append",
        default=[],
        type=_parse_named_path,
        help=(
            "`llvm-readobj --notes` output as NAME=PATH, or PATH to use the "
            "stem as NAME."
        ),
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        type=_parse_comparison_spec,
        help=(
            "Named metric comparison as BASELINE=CANDIDATE using artifact names "
            "from this report."
        ),
    )
    parser.add_argument(
        "--amdhsa-metadata-regex",
        help="Regex filter applied to AMDHSA kernel name and symbol metadata.",
    )
    parser.add_argument(
        "--symbol-regex",
        help="Regex filter applied to per-symbol disassembly summaries.",
    )
    parser.add_argument(
        "--top-symbols",
        type=int,
        default=16,
        help="Number of top symbol summaries to include per disassembly.",
    )
    parser.add_argument(
        "--ordered-symbols",
        type=int,
        default=0,
        help=(
            "Number of address-ordered symbol and label summaries to include "
            "per disassembly."
        ),
    )
    parser.add_argument(
        "--symbol-weight",
        action="append",
        default=[],
        type=_parse_symbol_weight_spec,
        help=(
            "Weighted disassembly block rule as DISASSEMBLY_NAME=SYMBOL_REGEX=WEIGHT. "
            "Repeat to estimate dynamic instruction family counts from selected "
            "symbols or labels."
        ),
    )
    parser.add_argument(
        "--weighted-symbol-group",
        action="append",
        default=[],
        type=_parse_weighted_symbol_group_spec,
        help=(
            "Named weighted disassembly block rule as "
            "DISASSEMBLY_NAME/GROUP_NAME=SYMBOL_REGEX=WEIGHT. Repeat with "
            "the same group to estimate phase-specific dynamic instruction "
            "family counts from selected symbols or labels."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("json", "text", "rocblas-replay"),
        default="json",
        help="Output format.",
    )
    parser.add_argument(
        "--rocblas-bench-executable",
        default="rocblas-bench",
        help="Executable path used when --format=rocblas-replay.",
    )
    parser.add_argument(
        "--rocblas-replay-arg",
        action="append",
        default=[],
        help=(
            "Additional argument appended to each command emitted by "
            "--format=rocblas-replay. Repeat for flags and values; use "
            "--rocblas-replay-arg=--flag when the value begins with '-'."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(sys.argv[1:] if argv is None else argv)
    report = build_kernel_anatomy_report(
        disassembly_paths=args.disassembly,
        compile_report_paths=args.compile_report,
        benchmark_jsonl_paths=args.benchmark_jsonl,
        iree_dispatch_profile_paths=args.iree_dispatch_profile,
        rocprof_kernel_trace_paths=args.rocprof_kernel_trace,
        rocprof_counter_collection_paths=args.rocprof_counter_collection,
        rocblas_log_paths=args.rocblas_log,
        symbol_regex=args.symbol_regex,
        amdhsa_metadata_paths=args.amdhsa_metadata,
        amdhsa_metadata_regex=args.amdhsa_metadata_regex,
        symbol_weight_specs=args.symbol_weight,
        weighted_symbol_group_specs=args.weighted_symbol_group,
        top_symbol_count=args.top_symbols,
        ordered_symbol_count=args.ordered_symbols,
    )
    if args.compare:
        report["comparisons"] = build_kernel_anatomy_comparisons(report, args.compare)
        report["comparison_scorecard"] = build_kernel_anatomy_comparison_scorecard(
            report["comparisons"]
        )
    if args.format == "json":
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    elif args.format == "text":
        sys.stdout.write(format_text_report(report))
    else:
        sys.stdout.write(
            format_rocblas_replay_script(
                report,
                args.rocblas_bench_executable,
                args.rocblas_replay_arg,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
