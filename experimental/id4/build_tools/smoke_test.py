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
        default="pytorch_parity",
        choices=("fused_product", "pytorch_parity"),
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
        f"--vae_tiling_mode={args.vae_tiling_mode}",
        f"--dump_plan={artifact_dir / 'plan.json'}",
        f"--dump_diagnostics={artifact_dir / 'diagnostics'}",
        f"--profile_output={artifact_dir / 'profile.txt'}",
    ]
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
