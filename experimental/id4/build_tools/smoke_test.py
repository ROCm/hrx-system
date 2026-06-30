#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs an ID4 one-shot CLI smoke test and collects artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ID4_BINARY = Path("bazel-bin/experimental/id4/binding/cli/id4")

DEFAULT_PROMPT: dict[str, Any] = {
    "high_level_description": (
        "A realistic street photograph of three adults walking together through "
        "a modern city crosswalk on a clear afternoon. They look natural, "
        "healthy, and fully human, with ordinary clothing, normal hands, and "
        "relaxed expressions."
    ),
    "style_description": {
        "aesthetics": "naturalistic, candid, modern urban lifestyle",
        "lighting": "soft afternoon sunlight with realistic reflections",
        "photo": (
            "full-frame documentary street photography, eye-level camera, "
            "crisp focus, realistic skin texture"
        ),
        "medium": "photograph",
    },
    "compositional_deconstruction": {
        "background": (
            "A clean modern downtown street with glass storefronts, crosswalk "
            "stripes, and soft reflections in windows."
        ),
        "elements": [
            {
                "type": "obj",
                "bbox": [260, 170, 920, 390],
                "desc": (
                    "An adult man walking on the left side of the group, "
                    "wearing a navy jacket and dark trousers."
                ),
            },
            {
                "type": "obj",
                "bbox": [230, 390, 930, 610],
                "desc": (
                    "An adult woman walking in the center of the group, "
                    "wearing a tan coat and dark jeans."
                ),
            },
            {
                "type": "obj",
                "bbox": [260, 610, 920, 830],
                "desc": (
                    "An adult man walking on the right side of the group, "
                    "wearing a charcoal overshirt and sneakers."
                ),
            },
        ],
    },
}


class SmokeTestError(RuntimeError):
    """Raised when the smoke run or artifact validation fails."""


@dataclass(frozen=True)
class ImageMetrics:
    width: int
    height: int
    max_value: int
    payload_size: int
    minimum_channel_value: int
    maximum_channel_value: int
    mean_channel_value: float
    luma_stddev: float
    sampled_unique_rgb_count: int

    def to_json(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "max_value": self.max_value,
            "payload_size": self.payload_size,
            "minimum_channel_value": self.minimum_channel_value,
            "maximum_channel_value": self.maximum_channel_value,
            "mean_channel_value": self.mean_channel_value,
            "luma_stddev": self.luma_stddev,
            "sampled_unique_rgb_count": self.sampled_unique_rgb_count,
        }


PLAN_SUMMARY_INTEGER_FIELDS = (
    "qwen_token_count",
    "qwen_token_capacity",
    "image_token_count",
    "conditioned_dit_token_count",
    "conditioned_dit_token_capacity",
    "unconditioned_dit_token_count",
    "unconditioned_dit_token_capacity",
    "denoise_step_count",
    "dit_activation_format",
    "dit_weight_execution_format",
    "dit_attention_implementation",
    "dit_feed_forward_implementation",
)

PLAN_RESIDENCY_INTEGER_FIELDS = (
    "total_stage_parameter_byte_length",
    "phase_parameter_high_water_mark",
    "largest_stage_parameter_byte_length",
    "total_stage_boundary_byte_length",
)

PLAN_PHASE_INTEGER_FIELDS = (
    "parameter_byte_length",
    "largest_stage_parameter_byte_length",
    "constant_byte_length",
    "local_slab_byte_length",
    "local_high_water_mark",
    "stage_boundary_byte_length",
)

PLAN_STAGE_STATISTIC_FIELDS = (
    "parameter_slab_byte_length",
    "largest_parameter_slab_byte_length",
    "parameter_source_byte_length",
    "parameter_direct_source_byte_length",
    "parameter_encoded_source_byte_length",
    "parameter_gather_load_step_count",
    "parameter_encode_load_step_count",
    "constant_slab_byte_length",
    "memory_slab_byte_length",
    "memory_slab_high_water_mark",
    "boundary_tensor_byte_length",
    "diagnostic_tap_byte_length",
    "kernel_count",
    "region_count",
    "operation_count",
    "dispatch_count",
)

PROFILE_TOP_ROW_COUNT = 20

PROFILE_DISPATCH_RE = re.compile(
    r"^\s*dispatch\s+(?P<name>\S+)\s+count=(?P<count>\d+)\s+"
    r"total=(?P<total>\d+)\s+ticks\s+avg=(?P<average>\d+)\s+ticks\s*$"
)

PROFILE_QUEUE_RE = re.compile(
    r"^\s*(?P<family>host_queue|device_queue)\s+p=(?P<device>\d+)\s+"
    r"q=(?P<queue>\d+)\s+(?P<operation>\w+)\s+count=(?P<count>\d+)\s+"
    r"total=(?P<total>-|\d+)(?:\s+ticks)?\s+avg=(?P<average>-|\d+)"
    r"(?:\s+ticks)?(?:\s+invalid=(?P<invalid>\d+))?"
    r"(?:\s+operations=(?P<operations>\d+))?"
    r"(?:\s+payload=(?P<payload>\d+)B)?\s*$"
)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True, help="Artifact directory.")
    parser.add_argument("--id4_binary", default=str(DEFAULT_ID4_BINARY))
    parser.add_argument("--device", action="append", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--parameters", action="append", required=True)
    parser.add_argument("--request_json", help="Full request JSON to run.")
    parser.add_argument("--latent_width", type=int, default=8)
    parser.add_argument("--latent_height", type=int, default=8)
    parser.add_argument("--denoise_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument(
        "--dit_parameter_format",
        default="fp8_e4m3",
        choices=(
            "bf16",
            "fp8_e4m3",
        ),
    )
    parser.add_argument(
        "--dit_activation_format",
        default="bf16_linear_input",
        choices=("f32_canonical", "bf16_linear_input"),
    )
    parser.add_argument(
        "--dit_weight_execution_format",
        default="bf16_resident",
        choices=("bf16_resident", "fp8_direct"),
    )
    parser.add_argument(
        "--dit_attention_implementation",
        default="online_wmma",
        choices=("streaming", "materialized_wmma", "blocked_wmma", "online_wmma"),
    )
    parser.add_argument(
        "--dit_feed_forward_implementation",
        default="fused_product",
        choices=("fused_product", "pytorch_parity"),
    )
    parser.add_argument(
        "--generation_residency",
        default="issue_phases",
        choices=(
            "issue_phases",
            "selected_stage_bundles",
            "all_stage_bundles",
        ),
    )
    parser.add_argument(
        "--generation_issue_mode",
        default="full",
        choices=("full", "phases", "stage_serial"),
    )
    parser.add_argument(
        "--generation_resident_stage_bundles",
        default="",
        help=(
            "Comma-separated stage bundles retained by "
            "--generation_residency=selected_stage_bundles."
        ),
    )
    parser.add_argument(
        "--dit_fp8_source_residency",
        default="disabled",
        choices=("disabled", "dit_conditioned", "dit_unconditioned", "all"),
    )
    parser.add_argument("--dit_fp8_source_cache_budget_mib", type=int, default=0)
    parser.add_argument(
        "--dit_fp8_source_cache_miss_mode",
        default="retain",
        choices=("retain", "direct_on_pressure"),
    )
    parser.add_argument(
        "--vae_tiling_mode",
        default="memory_budget",
        choices=(
            "disabled",
            "explicit_tile_size",
            "relative_tile_size",
            "memory_budget",
        ),
    )
    parser.add_argument("--vae_tile_size_x", type=int, default=0)
    parser.add_argument("--vae_tile_size_y", type=int, default=0)
    parser.add_argument("--vae_relative_size_x", type=float, default=0.0)
    parser.add_argument("--vae_relative_size_y", type=float, default=0.0)
    parser.add_argument("--vae_memory_budget", type=int, default=536870912)
    parser.add_argument("--vae_overlap", type=float, default=0.5)
    parser.add_argument("--extra_id4_arg", action="append", default=[])
    return parser.parse_args(argv)


def validate_request_payload(request: dict[str, Any]) -> None:
    if "prompt" not in request:
        raise SmokeTestError("request_json must contain a prompt payload")
    generation = request.get("generation")
    if not isinstance(generation, dict):
        raise SmokeTestError("request_json must contain generation metadata")
    for field_name in (
        "latent_width",
        "latent_height",
        "denoise_steps",
        "seed",
        "guidance_scale",
    ):
        if field_name not in generation:
            raise SmokeTestError(
                f"request_json generation metadata is missing {field_name}"
            )


def load_request(args: argparse.Namespace) -> dict[str, Any]:
    if args.request_json:
        path = Path(args.request_json)
        with path.open(encoding="utf-8") as file:
            request = json.load(file)
        if not isinstance(request, dict):
            raise SmokeTestError("request_json must contain a JSON object")
        validate_request_payload(request)
        return request
    request = {
        "prompt": DEFAULT_PROMPT,
        "generation": {
            "latent_width": args.latent_width,
            "latent_height": args.latent_height,
            "denoise_steps": args.denoise_steps,
            "seed": args.seed,
            "guidance_scale": args.guidance_scale,
        },
    }
    validate_request_payload(request)
    return request


def write_compact_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, separators=(",", ":"))
        file.write("\n")


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SmokeTestError(f"{context} must be a JSON object")
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise SmokeTestError(f"{context} must be a JSON array")
    return value


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise SmokeTestError(f"{context} must be a string")
    return value


def _require_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise SmokeTestError(f"{context} must be a boolean")
    return value


def _require_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SmokeTestError(f"{context} must be an integer")
    return value


def _copy_integer_fields(
    value: dict[str, Any], field_names: tuple[str, ...], context: str
) -> dict[str, int]:
    return {
        field_name: _require_int(value.get(field_name), f"{context}.{field_name}")
        for field_name in field_names
    }


def _summarize_shape(value: Any, context: str) -> dict[str, Any]:
    shape = _require_object(value, context)
    rank = _require_int(shape.get("rank"), f"{context}.rank")
    dims = [
        _require_int(dim, f"{context}.dims[{index}]")
        for index, dim in enumerate(_require_list(shape.get("dims"), f"{context}.dims"))
    ]
    if rank != len(dims):
        raise SmokeTestError(
            f"{context}.rank is {rank}, but {context}.dims has {len(dims)} entries"
        )
    return {
        "rank": rank,
        "dims": dims,
    }


def _summarize_generation_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("kind") != "ideogram4_generation":
        raise SmokeTestError("plan.kind must be ideogram4_generation")

    summary = _require_object(plan.get("summary"), "plan.summary")
    plan_summary: dict[str, Any] = _copy_integer_fields(
        summary, PLAN_SUMMARY_INTEGER_FIELDS, "plan.summary"
    )
    plan_summary["diffusion_latent_shape"] = _summarize_shape(
        summary.get("diffusion_latent_shape"), "plan.summary.diffusion_latent_shape"
    )
    plan_summary["decoded_image_shape"] = _summarize_shape(
        summary.get("decoded_image_shape"), "plan.summary.decoded_image_shape"
    )
    plan_summary["vae_tiling"] = _require_object(
        summary.get("vae_tiling"), "plan.summary.vae_tiling"
    )

    residency = _require_object(plan.get("residency"), "plan.residency")
    plan_residency: dict[str, Any] = _copy_integer_fields(
        residency, PLAN_RESIDENCY_INTEGER_FIELDS, "plan.residency"
    )
    plan_residency["phases"] = []
    for phase_index, phase in enumerate(
        _require_list(residency.get("phases"), "plan.residency.phases")
    ):
        phase_context = f"plan.residency.phases[{phase_index}]"
        phase_object = _require_object(phase, phase_context)
        phase_summary: dict[str, Any] = {
            "name": _require_string(phase_object.get("name"), f"{phase_context}.name"),
            "stage_keys": [
                _require_string(stage_key, f"{phase_context}.stage_keys[{index}]")
                for index, stage_key in enumerate(
                    _require_list(
                        phase_object.get("stage_keys"), f"{phase_context}.stage_keys"
                    )
                )
            ],
            "repeated_per_denoise_step": _require_bool(
                phase_object.get("repeated_per_denoise_step"),
                f"{phase_context}.repeated_per_denoise_step",
            ),
        }
        phase_summary.update(
            _copy_integer_fields(phase_object, PLAN_PHASE_INTEGER_FIELDS, phase_context)
        )
        plan_residency["phases"].append(phase_summary)

    stages = _require_object(plan.get("stages"), "plan.stages")
    stage_summaries: dict[str, dict[str, int]] = {}
    for stage_key, stage in sorted(stages.items()):
        stage_context = f"plan.stages.{stage_key}"
        stage_object = _require_object(stage, stage_context)
        statistics = _require_object(
            stage_object.get("statistics"), f"{stage_context}.statistics"
        )
        stage_summaries[stage_key] = _copy_integer_fields(
            statistics, PLAN_STAGE_STATISTIC_FIELDS, f"{stage_context}.statistics"
        )

    return {
        "summary": plan_summary,
        "residency": plan_residency,
        "stages": stage_summaries,
    }


def read_generation_plan_metrics(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        plan = json.load(file)
    return _summarize_generation_plan(_require_object(plan, "plan"))


def _integer_match_group(match: re.Match[str], name: str) -> int:
    return int(match.group(name))


def _profile_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        -int(row["total_ticks"]),
        -int(row["count"]),
        str(row.get("name", row.get("operation", ""))),
    )


def _profile_top_rows(rows: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
    ranked_rows = sorted(rows.values(), key=_profile_sort_key)
    return ranked_rows[:PROFILE_TOP_ROW_COUNT]


def read_profile_metrics(path: Path) -> dict[str, Any]:
    dispatch_rows: dict[str, dict[str, Any]] = {}
    queue_rows: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    dispatch_row_count = 0
    queue_row_count = 0
    invalid_queue_row_count = 0
    unknown_timed_row_count = 0

    with path.open(encoding="utf-8") as file:
        for line in file:
            dispatch_match = PROFILE_DISPATCH_RE.match(line)
            if dispatch_match:
                dispatch_row_count += 1
                name = dispatch_match.group("name")
                count = _integer_match_group(dispatch_match, "count")
                total_ticks = _integer_match_group(dispatch_match, "total")
                row = dispatch_rows.setdefault(
                    name,
                    {
                        "name": name,
                        "count": 0,
                        "total_ticks": 0,
                        "average_ticks": 0,
                    },
                )
                row["count"] += count
                row["total_ticks"] += total_ticks
                row["average_ticks"] = row["total_ticks"] // row["count"]
                continue

            queue_match = PROFILE_QUEUE_RE.match(line)
            if queue_match:
                queue_row_count += 1
                if queue_match.group("total") == "-":
                    invalid_queue_row_count += 1
                    continue
                family = queue_match.group("family")
                device = _integer_match_group(queue_match, "device")
                queue = _integer_match_group(queue_match, "queue")
                operation = queue_match.group("operation")
                count = _integer_match_group(queue_match, "count")
                total_ticks = _integer_match_group(queue_match, "total")
                operations_match = queue_match.group("operations")
                operations = int(operations_match) if operations_match else count
                payload = queue_match.group("payload")
                row = queue_rows.setdefault(
                    (family, device, queue, operation),
                    {
                        "family": family,
                        "device": device,
                        "queue": queue,
                        "operation": operation,
                        "count": 0,
                        "operation_count": 0,
                        "payload_bytes": 0,
                        "total_ticks": 0,
                        "average_ticks": 0,
                    },
                )
                row["count"] += count
                row["operation_count"] += operations
                row["payload_bytes"] += int(payload) if payload else 0
                row["total_ticks"] += total_ticks
                row["average_ticks"] = row["total_ticks"] // row["count"]
                continue

            if " total=" in line and " ticks" in line:
                unknown_timed_row_count += 1

    return {
        "dispatches": {
            "row_count": dispatch_row_count,
            "function_count": len(dispatch_rows),
            "total_count": sum(row["count"] for row in dispatch_rows.values()),
            "top_by_total_ticks": _profile_top_rows(dispatch_rows),
        },
        "queue_operations": {
            "row_count": queue_row_count,
            "timed_row_count": queue_row_count - invalid_queue_row_count,
            "invalid_row_count": invalid_queue_row_count,
            "operation_count": sum(
                row["operation_count"] for row in queue_rows.values()
            ),
            "top_by_total_ticks": _profile_top_rows(queue_rows),
        },
        "unknown_timed_row_count": unknown_timed_row_count,
    }


def require_empty_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise SmokeTestError(f"output_dir must be empty: {path}")


def build_id4_command(args: argparse.Namespace, artifact_dir: Path) -> list[str]:
    command = [
        args.id4_binary,
        f"--tokenizer={args.tokenizer}",
        f"--prompt_json_file={artifact_dir / 'request.json'}",
        f"--output={artifact_dir / 'image.ppm'}",
        f"--dit_parameter_format={args.dit_parameter_format}",
        f"--dit_activation_format={args.dit_activation_format}",
        f"--dit_weight_execution_format={args.dit_weight_execution_format}",
        f"--dit_attention_implementation={args.dit_attention_implementation}",
        f"--dit_feed_forward_implementation={args.dit_feed_forward_implementation}",
        f"--generation_residency={args.generation_residency}",
        f"--generation_issue_mode={args.generation_issue_mode}",
        f"--dit_fp8_source_residency={args.dit_fp8_source_residency}",
        f"--dit_fp8_source_cache_budget_mib={args.dit_fp8_source_cache_budget_mib}",
        f"--dit_fp8_source_cache_miss_mode={args.dit_fp8_source_cache_miss_mode}",
        f"--vae_tiling_mode={args.vae_tiling_mode}",
        f"--dump_plan={artifact_dir / 'plan.json'}",
        f"--dump_diagnostics={artifact_dir / 'diagnostics'}",
        f"--profile_output={artifact_dir / 'profile.txt'}",
    ]
    if args.generation_resident_stage_bundles:
        command.append(
            "--generation_resident_stage_bundles="
            f"{args.generation_resident_stage_bundles}"
        )
    if args.vae_tiling_mode == "explicit_tile_size":
        command.extend(
            [
                f"--vae_tile_size_x={args.vae_tile_size_x}",
                f"--vae_tile_size_y={args.vae_tile_size_y}",
                f"--vae_overlap={args.vae_overlap}",
            ]
        )
    elif args.vae_tiling_mode == "relative_tile_size":
        command.extend(
            [
                f"--vae_relative_size_x={args.vae_relative_size_x}",
                f"--vae_relative_size_y={args.vae_relative_size_y}",
                f"--vae_overlap={args.vae_overlap}",
            ]
        )
    elif args.vae_tiling_mode == "memory_budget":
        command.extend(
            [
                f"--vae_memory_budget={args.vae_memory_budget}",
                f"--vae_overlap={args.vae_overlap}",
            ]
        )
    for device in args.device:
        command.append(f"--device={device}")
    for parameter in args.parameters:
        command.append(f"--parameters={parameter}")
    command.extend(args.extra_id4_arg)
    return command


def _read_ppm_token(data: bytes, position: int) -> tuple[bytes, int]:
    while position < len(data) and chr(data[position]).isspace():
        position += 1
    if position < len(data) and data[position] == ord("#"):
        line_end = data.find(b"\n", position)
        if line_end == -1:
            raise SmokeTestError("PPM comment is missing a newline")
        return _read_ppm_token(data, line_end + 1)
    token_start = position
    while position < len(data) and not chr(data[position]).isspace():
        position += 1
    if token_start == position:
        raise SmokeTestError("PPM header token is missing")
    return data[token_start:position], position


def read_ppm_metrics(path: Path) -> ImageMetrics:
    data = path.read_bytes()
    magic, position = _read_ppm_token(data, 0)
    if magic != b"P6":
        raise SmokeTestError("output image is not a binary PPM")
    width_token, position = _read_ppm_token(data, position)
    height_token, position = _read_ppm_token(data, position)
    max_value_token, position = _read_ppm_token(data, position)
    if position >= len(data) or not chr(data[position]).isspace():
        raise SmokeTestError("PPM header is missing payload separator")
    position += 1
    width = int(width_token)
    height = int(height_token)
    max_value = int(max_value_token)
    payload = data[position:]
    expected_payload_size = width * height * 3
    if len(payload) != expected_payload_size:
        raise SmokeTestError(
            f"PPM payload has {len(payload)} bytes, expected {expected_payload_size}"
        )
    channel_sum = sum(payload)
    mean = channel_sum / len(payload)
    luma_values = []
    sampled_values = set()
    for pixel in range(width * height):
        base = pixel * 3
        red = payload[base]
        green = payload[base + 1]
        blue = payload[base + 2]
        luma_values.append(0.2126 * red + 0.7152 * green + 0.0722 * blue)
        if pixel % 97 == 0:
            sampled_values.add((red, green, blue))
    luma_mean = sum(luma_values) / len(luma_values)
    luma_variance = sum((value - luma_mean) ** 2 for value in luma_values)
    luma_stddev = math.sqrt(luma_variance / len(luma_values))
    return ImageMetrics(
        width=width,
        height=height,
        max_value=max_value,
        payload_size=len(payload),
        minimum_channel_value=min(payload),
        maximum_channel_value=max(payload),
        mean_channel_value=mean,
        luma_stddev=luma_stddev,
        sampled_unique_rgb_count=len(sampled_values),
    )


def validate_image(metrics: ImageMetrics, request: dict[str, Any]) -> None:
    generation = request.get("generation")
    if not isinstance(generation, dict):
        raise SmokeTestError("request generation metadata is missing")
    expected_width = int(generation["latent_width"]) * 16
    expected_height = int(generation["latent_height"]) * 16
    if metrics.width != expected_width or metrics.height != expected_height:
        raise SmokeTestError(
            "output dimensions "
            f"{metrics.width}x{metrics.height} do not match expected "
            f"{expected_width}x{expected_height}"
        )
    if metrics.max_value != 255:
        raise SmokeTestError(f"output max value is {metrics.max_value}, expected 255")
    if metrics.maximum_channel_value - metrics.minimum_channel_value < 16:
        raise SmokeTestError("output image has too little channel dynamic range")
    if metrics.luma_stddev < 4.0:
        raise SmokeTestError("output image has too little luma variation")
    if metrics.sampled_unique_rgb_count < 8:
        raise SmokeTestError("output image has too few sampled RGB values")


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = Path(args.output_dir)
    require_empty_output_dir(artifact_dir)
    request = load_request(args)
    request_path = artifact_dir / "request.json"
    write_compact_json(request_path, request)
    command = build_id4_command(args, artifact_dir)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (artifact_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (artifact_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    summary: dict[str, Any] = {
        "state": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "command": command,
        "artifacts": {
            "request": str(request_path),
            "image": str(artifact_dir / "image.ppm"),
            "plan": str(artifact_dir / "plan.json"),
            "diagnostics": str(artifact_dir / "diagnostics"),
            "profile": str(artifact_dir / "profile.txt"),
            "stdout": str(artifact_dir / "stdout.txt"),
            "stderr": str(artifact_dir / "stderr.txt"),
        },
    }
    if completed.returncode == 0:
        try:
            summary["plan_metrics"] = read_generation_plan_metrics(
                artifact_dir / "plan.json"
            )
            summary["profile_metrics"] = read_profile_metrics(
                artifact_dir / "profile.txt"
            )
            metrics = read_ppm_metrics(artifact_dir / "image.ppm")
            summary["image_metrics"] = metrics.to_json()
            validate_image(metrics, request)
        except (OSError, ValueError, SmokeTestError) as exc:
            summary["state"] = "failed"
            summary["validation_error"] = str(exc)
    with (artifact_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    if completed.returncode != 0:
        raise SmokeTestError(f"id4 exited with status {completed.returncode}")
    if summary["state"] != "ok":
        raise SmokeTestError(summary["validation_error"])
    return summary


def main(argv: list[str]) -> int:
    args = parse_arguments(argv)
    try:
        summary = run_smoke(args)
    except (OSError, ValueError, SmokeTestError) as exc:
        print(f"id4 smoke test failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
