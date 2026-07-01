#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Summarizes ID4 generation benchmark rows with their plan metrics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import smoke_test


class GenerationBenchmarkSummaryError(ValueError):
    """Raised when benchmark summary inputs are missing or malformed."""


_LABEL_LOAD_STEPS_PATTERN = re.compile(
    r"\bparam_load_steps\[gather=(?P<gather>[0-9]+),"
    r"encode=(?P<encode>[0-9]+)\]"
)
_LABEL_LOAD_GROUPS_PATTERN = re.compile(
    r"\bparam_load_groups\[total=(?P<total>[0-9]+),"
    r"gather=(?P<gather>[0-9]+),encode=(?P<encode>[0-9]+)\]"
)
_LABEL_LOGICAL_LIVE_PATTERN = re.compile(
    r"\blogical_live\[boundary=(?P<boundary>[0-9]+)MiB,"
    r"taps=(?P<taps>[0-9]+)MiB,"
    r"resident=(?P<resident>[0-9]+)MiB,"
    r"phase_peak=(?P<phase_peak>[0-9]+)MiB,"
    r"stage_serial_peak=(?P<stage_serial_peak>[0-9]+)MiB,"
    r"selected_peak=(?P<selected_peak>[0-9]+)MiB\]"
)
_LABEL_STAGE_PATTERN = re.compile(
    r"\bstage\.(?P<name>[A-Za-z0-9_]+)\["
    r"param=(?P<parameter>[0-9]+)MiB,"
    r"src=(?P<source>[0-9]+)MiB,"
    r"src_direct=(?P<source_direct>[0-9]+)MiB,"
    r"src_encoded=(?P<source_encoded>[0-9]+)MiB,"
    r"load_steps=(?P<gather_load_steps>[0-9]+)/"
    r"(?P<encode_load_steps>[0-9]+),"
    r"load_groups=(?P<gather_load_groups>[0-9]+)/"
    r"(?P<encode_load_groups>[0-9]+),"
    r"local_hw=(?P<local_high_water>[0-9]+)MiB,"
    r"boundary=(?P<boundary>[0-9]+)MiB,"
    r"kernels=(?P<kernels>[0-9]+),"
    r"dispatches=(?P<dispatches>[0-9]+)\]"
)
_LABEL_TIMING_PATTERN = re.compile(
    r"\btiming_ms\[plan=(?P<plan>[0-9]+(?:\.[0-9]+)?),"
    r"prepare=(?P<prepare>[0-9]+(?:\.[0-9]+)?),"
    r"issue=(?P<issue>[0-9]+(?:\.[0-9]+)?),"
    r"begin=(?P<begin>[0-9]+(?:\.[0-9]+)?),"
    r"final_wait=(?P<final_wait>[0-9]+(?:\.[0-9]+)?),"
    r"total=(?P<total>[0-9]+(?:\.[0-9]+)?)\]"
)
_LABEL_PREFETCH_GROUPS_PATTERN = re.compile(
    r"\bprefetch_groups\[count=(?P<count>[0-9]+),"
    r"avg_regions=(?P<average>[0-9]+(?:\.[0-9]+)?),"
    r"max_regions=(?P<maximum>[0-9]+)\]"
)
_LABEL_DIRECT_GATHER_GROUPS_PATTERN = re.compile(
    r"\bdirect_gather_groups\[count=(?P<count>[0-9]+),"
    r"requests=(?P<requests>[0-9]+),"
    r"source=(?P<source>[0-9]+)MiB,target=(?P<target>[0-9]+)MiB,"
    r"max=(?P<maximum>[0-9]+)MiB\]"
)
_LABEL_ISSUE_ENCODE_WINDOW_PATTERN = re.compile(
    r"\bissue_encode_window\[count=(?P<count>[0-9]+),"
    r"staging=(?P<staging>[0-9]+)MiB,max=(?P<maximum>[0-9]+)MiB,"
    r"source=(?P<source>[0-9]+)MiB,target=(?P<target>[0-9]+)MiB,"
    r"chunks=(?P<chunks>[0-9]+),batches=(?P<batches>[0-9]+),"
    r"dispatches=(?P<dispatches>[0-9]+)\]"
)


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GenerationBenchmarkSummaryError(f"{context} must be a JSON object")
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise GenerationBenchmarkSummaryError(f"{context} must be a JSON array")
    return value


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise GenerationBenchmarkSummaryError(f"{context} must be a non-empty string")
    return value


def _require_number(value: Any, context: str) -> int | float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise GenerationBenchmarkSummaryError(f"{context} must be a number")
    return value


def _require_match(pattern: re.Pattern[str], text: str, context: str) -> re.Match[str]:
    match = pattern.search(text)
    if not match:
        raise GenerationBenchmarkSummaryError(f"{context} missing {pattern.pattern}")
    return match


def _label_token(label: str, key: str, context: str) -> str:
    match = re.search(rf"(?:^| ){re.escape(key)}=([^ ]+)", label)
    if not match:
        raise GenerationBenchmarkSummaryError(f"{context} missing {key}=...")
    return match.group(1)


def _label_unsigned(label: str, key: str, context: str) -> int:
    value = _label_token(label, key, context)
    if not value.isdecimal():
        raise GenerationBenchmarkSummaryError(
            f"{context} {key} must be an unsigned integer"
        )
    return int(value)


def _label_hex(label: str, key: str, context: str) -> int:
    value = _label_token(label, key, context)
    if not re.fullmatch(r"0x[0-9A-Fa-f]+", value):
        raise GenerationBenchmarkSummaryError(f"{context} {key} must be hex")
    return int(value, 16)


def _label_mib(label: str, key: str, context: str) -> int:
    value = _label_token(label, key, context)
    if not value.endswith("MiB") or not value[:-3].isdecimal():
        raise GenerationBenchmarkSummaryError(f"{context} {key} must be MiB")
    return int(value[:-3])


def _match_unsigned_group(match: re.Match[str], group: str, context: str) -> int:
    value = match.group(group)
    if not value.isdecimal():
        raise GenerationBenchmarkSummaryError(
            f"{context} {group} must be an unsigned integer"
        )
    return int(value)


def _match_float_group(match: re.Match[str], group: str, context: str) -> float:
    value = match.group(group)
    try:
        return float(value)
    except ValueError as exc:
        raise GenerationBenchmarkSummaryError(
            f"{context} {group} must be a number"
        ) from exc


def _ceil_mib(byte_length: int) -> int:
    return (byte_length + 1024 * 1024 - 1) // (1024 * 1024)


def _require_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise GenerationBenchmarkSummaryError(
            f"{context} mismatch: got {actual}, expected {expected}"
        )


def _parse_generation_benchmark_label(label: str, context: str) -> dict[str, Any]:
    load_steps_match = _require_match(
        _LABEL_LOAD_STEPS_PATTERN, label, f"{context}.label"
    )
    load_groups_match = _require_match(
        _LABEL_LOAD_GROUPS_PATTERN, label, f"{context}.label"
    )
    logical_live_match = _require_match(
        _LABEL_LOGICAL_LIVE_PATTERN, label, f"{context}.label"
    )
    prefetch_groups_match = _require_match(
        _LABEL_PREFETCH_GROUPS_PATTERN, label, f"{context}.label"
    )
    direct_gather_groups_match = _require_match(
        _LABEL_DIRECT_GATHER_GROUPS_PATTERN, label, f"{context}.label"
    )
    issue_encode_window_match = _require_match(
        _LABEL_ISSUE_ENCODE_WINDOW_PATTERN, label, f"{context}.label"
    )
    timing_match = _require_match(_LABEL_TIMING_PATTERN, label, f"{context}.label")

    runtime_stages: dict[str, Any] = {}
    for stage_match in _LABEL_STAGE_PATTERN.finditer(label):
        stage_name = stage_match.group("name")
        runtime_stages[stage_name] = {
            "parameter_mib": _match_unsigned_group(
                stage_match, "parameter", f"{context}.label.stage.{stage_name}"
            ),
            "source_mib": _match_unsigned_group(
                stage_match, "source", f"{context}.label.stage.{stage_name}"
            ),
            "source_direct_mib": _match_unsigned_group(
                stage_match, "source_direct", f"{context}.label.stage.{stage_name}"
            ),
            "source_encoded_mib": _match_unsigned_group(
                stage_match, "source_encoded", f"{context}.label.stage.{stage_name}"
            ),
            "parameter_gather_load_step_count": _match_unsigned_group(
                stage_match,
                "gather_load_steps",
                f"{context}.label.stage.{stage_name}",
            ),
            "parameter_encode_load_step_count": _match_unsigned_group(
                stage_match,
                "encode_load_steps",
                f"{context}.label.stage.{stage_name}",
            ),
            "parameter_gather_load_group_count": _match_unsigned_group(
                stage_match,
                "gather_load_groups",
                f"{context}.label.stage.{stage_name}",
            ),
            "parameter_encode_load_group_count": _match_unsigned_group(
                stage_match,
                "encode_load_groups",
                f"{context}.label.stage.{stage_name}",
            ),
            "local_high_water_mib": _match_unsigned_group(
                stage_match, "local_high_water", f"{context}.label.stage.{stage_name}"
            ),
            "boundary_mib": _match_unsigned_group(
                stage_match, "boundary", f"{context}.label.stage.{stage_name}"
            ),
            "kernel_count": _match_unsigned_group(
                stage_match, "kernels", f"{context}.label.stage.{stage_name}"
            ),
            "dispatch_count": _match_unsigned_group(
                stage_match, "dispatches", f"{context}.label.stage.{stage_name}"
            ),
        }
    if not runtime_stages:
        raise GenerationBenchmarkSummaryError(
            f"{context}.label must contain at least one stage.*[...] group"
        )

    return {
        "benchmark_scope": _label_token(label, "scope", f"{context}.label"),
        "prompt_label": _label_token(label, "prompt", f"{context}.label"),
        "generation_residency_request": _label_token(
            label, "residency_request", f"{context}.label"
        ),
        "generation_residency": _label_token(label, "residency", f"{context}.label"),
        "generation_issue_mode": _label_token(label, "issue", f"{context}.label"),
        "parameter_load_prefetch_region_distance": _label_unsigned(
            label, "prefetch_regions", f"{context}.label"
        ),
        "generation_resident_stage_mask": _label_hex(
            label, "resident_stage_mask", f"{context}.label"
        ),
        "generation_residency_budget_mib": _label_mib(
            label, "residency_budget", f"{context}.label"
        ),
        "runtime_qwen_weight_execution_strategy": _label_token(
            label, "qwen_weights", f"{context}.label"
        ),
        "runtime_parameter_total_mib": _label_mib(
            label, "param_total", f"{context}.label"
        ),
        "runtime_largest_parameter_mib": _label_mib(
            label, "param_largest", f"{context}.label"
        ),
        "runtime_parameter_source_mib": _label_mib(
            label, "param_source", f"{context}.label"
        ),
        "runtime_parameter_source_direct_mib": _label_mib(
            label, "param_source_direct", f"{context}.label"
        ),
        "runtime_parameter_source_encoded_mib": _label_mib(
            label, "param_source_encoded", f"{context}.label"
        ),
        "runtime_parameter_gather_load_step_count": _match_unsigned_group(
            load_steps_match, "gather", f"{context}.label"
        ),
        "runtime_parameter_encode_load_step_count": _match_unsigned_group(
            load_steps_match, "encode", f"{context}.label"
        ),
        "runtime_parameter_load_group_count": _match_unsigned_group(
            load_groups_match, "total", f"{context}.label"
        ),
        "runtime_parameter_gather_load_group_count": _match_unsigned_group(
            load_groups_match, "gather", f"{context}.label"
        ),
        "runtime_parameter_encode_load_group_count": _match_unsigned_group(
            load_groups_match, "encode", f"{context}.label"
        ),
        "runtime_local_high_water_total_mib": _label_mib(
            label, "local_hw_total", f"{context}.label"
        ),
        "runtime_local_high_water_largest_mib": _label_mib(
            label, "local_hw_largest", f"{context}.label"
        ),
        "runtime_boundary_mib": _label_mib(label, "boundary", f"{context}.label"),
        "runtime_kernel_count": _label_unsigned(label, "kernels", f"{context}.label"),
        "runtime_dispatch_count": _label_unsigned(
            label, "dispatches", f"{context}.label"
        ),
        "logical_live_boundary_mib": _match_unsigned_group(
            logical_live_match, "boundary", f"{context}.label"
        ),
        "logical_live_tap_mib": _match_unsigned_group(
            logical_live_match, "taps", f"{context}.label"
        ),
        "logical_live_resident_mib": _match_unsigned_group(
            logical_live_match, "resident", f"{context}.label"
        ),
        "logical_live_phase_peak_mib": _match_unsigned_group(
            logical_live_match, "phase_peak", f"{context}.label"
        ),
        "logical_live_stage_serial_peak_mib": _match_unsigned_group(
            logical_live_match, "stage_serial_peak", f"{context}.label"
        ),
        "logical_live_selected_peak_mib": _match_unsigned_group(
            logical_live_match, "selected_peak", f"{context}.label"
        ),
        "prefetch_group_submit_count": _match_unsigned_group(
            prefetch_groups_match, "count", f"{context}.label"
        ),
        "prefetch_group_average_region_distance": _match_float_group(
            prefetch_groups_match, "average", f"{context}.label"
        ),
        "prefetch_group_max_region_distance": _match_unsigned_group(
            prefetch_groups_match, "maximum", f"{context}.label"
        ),
        "direct_gather_group_count": _match_unsigned_group(
            direct_gather_groups_match, "count", f"{context}.label"
        ),
        "direct_gather_request_count": _match_unsigned_group(
            direct_gather_groups_match, "requests", f"{context}.label"
        ),
        "direct_gather_source_mib": _match_unsigned_group(
            direct_gather_groups_match, "source", f"{context}.label"
        ),
        "direct_gather_target_mib": _match_unsigned_group(
            direct_gather_groups_match, "target", f"{context}.label"
        ),
        "direct_gather_max_mib": _match_unsigned_group(
            direct_gather_groups_match, "maximum", f"{context}.label"
        ),
        "issue_encode_window_count": _match_unsigned_group(
            issue_encode_window_match, "count", f"{context}.label"
        ),
        "issue_encode_window_staging_mib": _match_unsigned_group(
            issue_encode_window_match, "staging", f"{context}.label"
        ),
        "issue_encode_window_staging_max_mib": _match_unsigned_group(
            issue_encode_window_match, "maximum", f"{context}.label"
        ),
        "issue_encode_window_source_mib": _match_unsigned_group(
            issue_encode_window_match, "source", f"{context}.label"
        ),
        "issue_encode_window_target_mib": _match_unsigned_group(
            issue_encode_window_match, "target", f"{context}.label"
        ),
        "issue_encode_window_chunk_count": _match_unsigned_group(
            issue_encode_window_match, "chunks", f"{context}.label"
        ),
        "issue_encode_window_batch_count": _match_unsigned_group(
            issue_encode_window_match, "batches", f"{context}.label"
        ),
        "issue_encode_window_dispatch_count": _match_unsigned_group(
            issue_encode_window_match, "dispatches", f"{context}.label"
        ),
        "timing_plan_ms": _match_float_group(timing_match, "plan", f"{context}.label"),
        "timing_prepare_ms": _match_float_group(
            timing_match, "prepare", f"{context}.label"
        ),
        "timing_issue_ms": _match_float_group(
            timing_match, "issue", f"{context}.label"
        ),
        "timing_begin_ms": _match_float_group(
            timing_match, "begin", f"{context}.label"
        ),
        "timing_final_wait_ms": _match_float_group(
            timing_match, "final_wait", f"{context}.label"
        ),
        "timing_total_ms": _match_float_group(
            timing_match, "total", f"{context}.label"
        ),
        "runtime_stages": runtime_stages,
    }


def _validate_label_plan_join(
    row: dict[str, Any], label_metrics: dict[str, Any], context: str
) -> None:
    _require_equal(
        label_metrics["prompt_label"],
        row["bucket"],
        f"{context}.label.prompt",
    )
    _require_equal(
        label_metrics["runtime_parameter_total_mib"],
        _ceil_mib(row["parameter_byte_length"]),
        f"{context}.label.param_total",
    )
    _require_equal(
        label_metrics["runtime_parameter_source_mib"],
        _ceil_mib(row["parameter_source_byte_length"]),
        f"{context}.label.param_source",
    )
    _require_equal(
        label_metrics["runtime_parameter_source_direct_mib"],
        _ceil_mib(row["parameter_direct_source_byte_length"]),
        f"{context}.label.param_source_direct",
    )
    _require_equal(
        label_metrics["runtime_parameter_source_encoded_mib"],
        _ceil_mib(row["parameter_encoded_source_byte_length"]),
        f"{context}.label.param_source_encoded",
    )
    _require_equal(
        label_metrics["runtime_parameter_gather_load_step_count"],
        row["parameter_gather_load_step_count"],
        f"{context}.label.param_load_steps.gather",
    )
    _require_equal(
        label_metrics["runtime_parameter_encode_load_step_count"],
        row["parameter_encode_load_step_count"],
        f"{context}.label.param_load_steps.encode",
    )
    _require_equal(
        label_metrics["runtime_parameter_load_group_count"],
        row["parameter_load_group_count"],
        f"{context}.label.param_load_groups.total",
    )
    _require_equal(
        label_metrics["runtime_parameter_gather_load_group_count"],
        row["parameter_gather_load_group_count"],
        f"{context}.label.param_load_groups.gather",
    )
    _require_equal(
        label_metrics["runtime_parameter_encode_load_group_count"],
        row["parameter_encode_load_group_count"],
        f"{context}.label.param_load_groups.encode",
    )
    _require_equal(
        label_metrics["runtime_boundary_mib"],
        _ceil_mib(row["stage_boundary_byte_length"]),
        f"{context}.label.boundary",
    )
    _require_equal(
        label_metrics["runtime_kernel_count"],
        sum(stage["kernel_count"] for stage in row["stages"].values()),
        f"{context}.label.kernels",
    )
    _require_equal(
        label_metrics["runtime_dispatch_count"],
        row["dispatch_count"],
        f"{context}.label.dispatches",
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise GenerationBenchmarkSummaryError(f"invalid JSON in {path}: {exc}") from exc
    return _require_object(payload, str(path))


def _benchmark_bucket(name: str) -> str:
    parts = name.split("/")
    if len(parts) < 2:
        raise GenerationBenchmarkSummaryError(
            f"benchmark name does not contain a prompt bucket: {name}"
        )
    if parts[-1] == "real_time" and len(parts) >= 3:
        return parts[-2]
    return parts[-1]


def summarize_generation_benchmark(
    benchmark_json_path: Path, plan_directory: Path
) -> dict[str, Any]:
    benchmark_json = _load_json(benchmark_json_path)
    rows: list[dict[str, Any]] = []
    for index, benchmark in enumerate(
        _require_list(benchmark_json.get("benchmarks"), "benchmarks")
    ):
        context = f"benchmarks[{index}]"
        benchmark_object = _require_object(benchmark, context)
        name = _require_string(benchmark_object.get("name"), f"{context}.name")
        bucket = _benchmark_bucket(name)
        plan_metrics = smoke_test.read_generation_plan_metrics(
            plan_directory / f"{bucket}.json"
        )
        plan_summary = plan_metrics["summary"]
        residency = plan_metrics["residency"]
        stages = plan_metrics["stages"]
        row = {
            "bucket": bucket,
            "benchmark": name,
            "real_time_ms": _require_number(
                benchmark_object.get("real_time"), f"{context}.real_time"
            ),
            "cpu_time_ms": _require_number(
                benchmark_object.get("cpu_time"), f"{context}.cpu_time"
            ),
            "iterations": _require_number(
                benchmark_object.get("iterations"), f"{context}.iterations"
            ),
            "qwen_token_count": plan_summary["qwen_token_count"],
            "qwen_token_capacity": plan_summary["qwen_token_capacity"],
            "image_token_count": plan_summary["image_token_count"],
            "conditioned_dit_token_count": plan_summary["conditioned_dit_token_count"],
            "conditioned_dit_token_capacity": plan_summary[
                "conditioned_dit_token_capacity"
            ],
            "unconditioned_dit_token_count": plan_summary[
                "unconditioned_dit_token_count"
            ],
            "unconditioned_dit_token_capacity": plan_summary[
                "unconditioned_dit_token_capacity"
            ],
            "denoise_step_count": plan_summary["denoise_step_count"],
            "dit_activation_format": plan_summary["dit_activation_format"],
            "dit_weight_execution_format": plan_summary["dit_weight_execution_format"],
            "qwen_weight_execution_strategy": plan_summary[
                "qwen_weight_execution_strategy"
            ],
            "dit_attention_implementation": plan_summary[
                "dit_attention_implementation"
            ],
            "dit_feed_forward_implementation": plan_summary[
                "dit_feed_forward_implementation"
            ],
            "parameter_byte_length": residency["total_stage_parameter_byte_length"],
            "parameter_source_byte_length": sum(
                stage["parameter_source_byte_length"] for stage in stages.values()
            ),
            "parameter_direct_source_byte_length": sum(
                stage["parameter_direct_source_byte_length"]
                for stage in stages.values()
            ),
            "parameter_encoded_source_byte_length": sum(
                stage["parameter_encoded_source_byte_length"]
                for stage in stages.values()
            ),
            "parameter_gather_load_step_count": sum(
                stage["parameter_gather_load_step_count"] for stage in stages.values()
            ),
            "parameter_encode_load_step_count": sum(
                stage["parameter_encode_load_step_count"] for stage in stages.values()
            ),
            "parameter_load_group_count": sum(
                stage["parameter_load_group_count"] for stage in stages.values()
            ),
            "parameter_gather_load_group_count": sum(
                stage["parameter_gather_load_group_count"] for stage in stages.values()
            ),
            "parameter_encode_load_group_count": sum(
                stage["parameter_encode_load_group_count"] for stage in stages.values()
            ),
            "phase_parameter_high_water_mark": residency[
                "phase_parameter_high_water_mark"
            ],
            "stage_boundary_byte_length": residency["total_stage_boundary_byte_length"],
            "shared_tensor_byte_length": sum(
                stage["shared_tensor_byte_length"] for stage in stages.values()
            ),
            "shared_tensor_count": sum(
                stage["shared_tensor_count"] for stage in stages.values()
            ),
            "stage_count": len(stages),
            "dispatch_count": sum(stage["dispatch_count"] for stage in stages.values()),
            "local_high_water_mark": sum(
                stage["memory_slab_high_water_mark"] for stage in stages.values()
            ),
            "stages": stages,
        }
        label = benchmark_object.get("label")
        if label is not None:
            label_metrics = _parse_generation_benchmark_label(
                _require_string(label, f"{context}.label"), context
            )
            _validate_label_plan_join(row, label_metrics, context)
            row.update(label_metrics)
        rows.append(row)
    return {
        "source": {
            "benchmark_json": str(benchmark_json_path),
            "plan_directory": str(plan_directory),
        },
        "rows": rows,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark_json",
        type=Path,
        required=True,
        help="Google Benchmark JSON emitted by session_live_benchmark.",
    )
    parser.add_argument(
        "--plan_dir",
        type=Path,
        required=True,
        help="Directory containing generation plan JSON files named by bucket.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path. Defaults to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        summary = summarize_generation_benchmark(args.benchmark_json, args.plan_dir)
    except (OSError, GenerationBenchmarkSummaryError, smoke_test.SmokeTestError) as exc:
        print(f"generation_benchmark_summary: {exc}", file=sys.stderr)
        return 1
    if args.output:
        with args.output.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
            file.write("\n")
    else:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
