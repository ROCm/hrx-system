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
import re
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


def _first_entry(report: Mapping[str, Any]) -> Mapping[str, Any]:
    entries = report.get("entries")
    if not isinstance(entries, Mapping):
        return report
    rows = entries.get("rows")
    if not isinstance(rows, list) or not rows:
        return report
    entry = rows[0]
    if not isinstance(entry, Mapping):
        return report
    return entry


def _summarize_loom_compile_report(
    report: Mapping[str, Any], path: Path | None = None
) -> dict[str, Any]:
    """Extracts kernel-economics fields from a Loom compile-report object."""

    entry = _first_entry(report)
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
    workload = entry.get("workload")
    if not isinstance(workload, Mapping):
        workload = {}
    wait_plan = entry.get("wait_plan")
    if not isinstance(wait_plan, Mapping):
        wait_plan = {}
    summary = {
        "function": report.get("function") or entry.get("function"),
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
        "target_resources": target_resources,
        "wait_plan": wait_plan,
    }
    if path is not None:
        summary["path"] = path.as_posix()
    return summary


def summarize_loom_compile_report(path: Path) -> dict[str, Any]:
    """Extracts kernel-economics fields from a Loom compile-report artifact."""

    report = _extract_compile_report(_read_json(path))
    return _summarize_loom_compile_report(report, path)


def summarize_loom_benchmark_jsonl(path: Path) -> dict[str, Any]:
    """Extracts timing and compile summaries from benchmark JSONL events."""

    benchmark_rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(_read_text(path).splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
        if not isinstance(event, Mapping) or event.get("row") != "benchmark":
            continue
        benchmark_result = event.get("benchmark_result")
        if not isinstance(benchmark_result, Mapping):
            continue
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
            compile_summary = _summarize_loom_compile_report(compile_report)
        benchmark_rows.append(
            {
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
        )
    return {
        "path": path.as_posix(),
        "benchmark_count": len(benchmark_rows),
        "benchmarks": benchmark_rows,
    }


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
    kernel_parameters: dict[str, str] = {}
    summary: dict[str, Any] = {
        "name": named_path.name,
        "paths": [named_path.path.as_posix()],
        "devices": [],
        "kernel_parameters": kernel_parameters,
        "timing_rows": timing_rows,
    }
    in_kernel_parameters = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("rocBLAS version:"):
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
            timing_rows.append(_parse_rocblas_timing_row(timing_header, stripped))
        elif in_kernel_parameters and ":" in stripped:
            key, value = stripped.split(":", 1)
            kernel_parameters[key.strip()] = value.strip()
    summary["selected_timing_row"] = _select_rocblas_timing_row(timing_rows)
    return summary


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


def _parse_rocblas_timing_row(header: Sequence[str], row: str) -> dict[str, Any]:
    values = next(csv.reader([row], skipinitialspace=True))
    return {
        key: _parse_numeric_scalar(value)
        for key, value in zip(header, values, strict=False)
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
    return {
        "name": named_path.name,
        "path": named_path.path.as_posix(),
        "whole_file": whole_file.metadata(),
        "symbol_count": len(blocks),
        "top_symbols": [block.metadata() for block in top_blocks],
        "ordered_symbols": [
            block.metadata() for block in _ordered_blocks(blocks, ordered_symbol_count)
        ],
    }


def build_kernel_anatomy_report(
    disassembly_paths: Sequence[NamedPath],
    compile_report_paths: Sequence[NamedPath],
    benchmark_jsonl_paths: Sequence[NamedPath] = (),
    iree_dispatch_profile_paths: Sequence[NamedPath] = (),
    rocblas_log_paths: Sequence[NamedPath] = (),
    symbol_regex: str | None = None,
    amdhsa_metadata_paths: Sequence[NamedPath] = (),
    amdhsa_metadata_regex: str | None = None,
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
    }


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
            "missing_baseline_metrics": sorted(
                set(candidate_metrics) - set(baseline_metrics)
            ),
            "missing_candidate_metrics": sorted(
                set(baseline_metrics) - set(candidate_metrics)
            ),
        }
    return comparisons


def _collect_comparison_metric_groups(
    report: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    _collect_compile_report_metric_groups(groups, report)
    _collect_benchmark_metric_groups(groups, report)
    _collect_iree_dispatch_profile_metric_groups(groups, report)
    _collect_rocblas_metric_groups(groups, report)
    _collect_disassembly_metric_groups(groups, report)
    _collect_metadata_metric_groups(groups, report)
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


def _collect_compile_report_metric_groups(
    groups: dict[str, dict[str, dict[str, Any]]], report: Mapping[str, Any]
) -> None:
    compile_reports = _as_mapping(report.get("loom_compile_reports"))
    for name, compile_report_value in compile_reports.items():
        compile_report = _as_mapping(compile_report_value)
        _record_metric(
            groups,
            name,
            "instruction_count",
            compile_report.get("instruction_count"),
            "compile_report",
        )
        _record_metric(
            groups,
            name,
            "code_byte_count",
            compile_report.get("code_byte_count"),
            "compile_report",
        )
        _record_metric(
            groups,
            name,
            "local_memory_bytes",
            compile_report.get("local_memory_bytes"),
            "compile_report",
        )
        _record_metric(
            groups,
            name,
            "private_memory_bytes",
            compile_report.get("private_memory_bytes"),
            "compile_report",
        )
        static_mix = _as_mapping(compile_report.get("static_instruction_mix"))
        _record_metric(
            groups,
            name,
            "wmma_count",
            static_mix.get("wmma_count"),
            "compile_report",
        )
        _record_metric(
            groups,
            name,
            "mfma_count",
            static_mix.get("mfma_count"),
            "compile_report",
        )
        _record_metric(
            groups,
            name,
            "vector_alu_count",
            static_mix.get("vector_alu_count"),
            "compile_report",
        )
        dynamic_mix = _as_mapping(compile_report.get("dynamic_instruction_mix"))
        _record_metric(
            groups,
            name,
            "dynamic_local_memory_count",
            dynamic_mix.get("local_memory_count"),
            "compile_report",
        )
        _record_metric(
            groups,
            name,
            "dynamic_local_memory_read_bytes",
            dynamic_mix.get("local_memory_read_bytes"),
            "compile_report",
        )
        _record_metric(
            groups,
            name,
            "dynamic_local_memory_write_bytes",
            dynamic_mix.get("local_memory_write_bytes"),
            "compile_report",
        )
        resources = _as_mapping(compile_report.get("target_resources"))
        vector = _as_mapping(resources.get("vector"))
        vector_final = _as_mapping(vector.get("final"))
        _record_metric(
            groups,
            name,
            "vgpr_count",
            vector_final.get("register_count"),
            "compile_report",
        )
        _record_metric(
            groups,
            name,
            "occupancy_percent",
            resources.get("occupancy_percent"),
            "compile_report",
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
        _record_metric(
            groups,
            name,
            "wmma_count",
            families.get("v_wmma"),
            "disassembly",
        )
        _record_metric(
            groups,
            name,
            "mfma_count",
            families.get("v_mfma"),
            "disassembly",
        )
        _record_metric(
            groups, name, "vector_alu_count", families.get("v_alu"), "disassembly"
        )
        for family in (
            "global_load",
            "global_store",
            "buffer_load",
            "buffer_store",
            "ds_read",
            "ds_write",
            "s_waitcnt",
            "s_barrier",
        ):
            _record_metric(
                groups, name, f"{family}_count", families.get(family), "disassembly"
            )
        memory = _as_mapping(whole_file.get("memory_byte_counts"))
        _record_metric(
            groups,
            name,
            "read_bytes",
            memory.get("read_bytes"),
            "disassembly",
        )
        _record_metric(
            groups, name, "write_bytes", memory.get("write_bytes"), "disassembly"
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
        lines.append(f"  {name}: benchmarks={benchmark_report.get('benchmark_count')}")
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
                f"correctness={correctness.get('failed_sample_count')}/"
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


def _format_rocblas_time_ms(timing: Mapping[str, Any]) -> str:
    time_us = timing.get("us")
    time_ms = time_us / 1000 if isinstance(time_us, (int, float)) else None
    return f"{time_ms:.6g}" if time_ms is not None else "?"


def _format_ratio(ratio: Any) -> str:
    return f"{ratio:.6g}x" if isinstance(ratio, (int, float)) else "?"


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
    _append_rocblas_log_lines(lines, report)
    _append_comparison_lines(lines, report)
    _append_disassembly_report_lines(lines, report)
    _append_amdhsa_metadata_lines(lines, report)
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
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output format.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(sys.argv[1:] if argv is None else argv)
    report = build_kernel_anatomy_report(
        disassembly_paths=args.disassembly,
        compile_report_paths=args.compile_report,
        benchmark_jsonl_paths=args.benchmark_jsonl,
        iree_dispatch_profile_paths=args.iree_dispatch_profile,
        rocblas_log_paths=args.rocblas_log,
        symbol_regex=args.symbol_regex,
        amdhsa_metadata_paths=args.amdhsa_metadata,
        amdhsa_metadata_regex=args.amdhsa_metadata_regex,
        top_symbol_count=args.top_symbols,
        ordered_symbol_count=args.ordered_symbols,
    )
    if args.compare:
        report["comparisons"] = build_kernel_anatomy_comparisons(report, args.compare)
    if args.format == "json":
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(format_text_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
