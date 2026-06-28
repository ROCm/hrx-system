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
import sys
from pathlib import Path
from typing import Any

import smoke_test


class GenerationBenchmarkSummaryError(ValueError):
    """Raised when benchmark summary inputs are missing or malformed."""


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
        rows.append(
            {
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
                "conditioned_dit_token_count": plan_summary[
                    "conditioned_dit_token_count"
                ],
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
                "parameter_byte_length": residency["total_stage_parameter_byte_length"],
                "phase_parameter_high_water_mark": residency[
                    "phase_parameter_high_water_mark"
                ],
                "stage_boundary_byte_length": residency[
                    "total_stage_boundary_byte_length"
                ],
                "stage_count": len(stages),
                "dispatch_count": sum(
                    stage["dispatch_count"] for stage in stages.values()
                ),
                "local_high_water_mark": sum(
                    stage["memory_slab_high_water_mark"] for stage in stages.values()
                ),
                "stages": stages,
            }
        )
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
