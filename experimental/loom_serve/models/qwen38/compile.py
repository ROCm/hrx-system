#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile Qwen stages into independent command and executable artifacts."""

import argparse
import json
import subprocess
from pathlib import Path


def compile_stage(arguments, stage):
    source = Path(__file__).resolve().parent
    output = arguments.output.resolve() / stage
    for directory in (
        output,
        output / "commands",
        output / "requests",
        output / "kernels",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    token_capacity = 1 if stage == "decode" else 512
    config = {
        "runner.qwen38.prefill_token_count": arguments.prefill_capacity,
        "runner.qwen38.span_capacity": arguments.span_capacity,
        "ggml.linear_q4k_q8_1_x4.token_capacity": token_capacity,
        "ggml.linear_q4k_q8_1_x4.output_capacity": 48,
        "ggml.linear_q5k_q8_1_x4.token_capacity": token_capacity,
        "ggml.linear_q5k_q8_1_x4.output_capacity": 17408,
        "ggml.linear_q6k_f32_decode.output_capacity": 5120,
        "ggml.linear_q6k_q8_1_x4.token_capacity": token_capacity,
        "ggml.linear_q6k_q8_1_x4.output_capacity": 248320,
        "ggml.linear_q8_0_q8_1_x4.token_capacity": 1,
        "ggml.linear_q8_0_q8_1_x4.output_capacity": 1024,
        "ggml.quantize_q8_1_x4.group_capacity": 136 * token_capacity,
        "qwen38.attention.cache_capacity": arguments.context_capacity,
        "qwen38.attention.decode_split_count": arguments.decode_splits,
        "qwen38.greedy_argmax.output_capacity": 248320,
    }
    configuration = output / "config.json"
    configuration.write_text(json.dumps(config, indent=2) + "\n")
    primary_name, root = {
        "prefill": ("prefill.loom", "qwen38_prefill"),
        "decode": ("programs/qwen38/model_decode.loom", "qwen38_text_decode_greedy"),
        "epoch": ("epoch.loom", "qwen38_epoch"),
    }[stage]
    primary = source / primary_name
    libraries = sorted(source.glob("kernels/**/*.loom")) + sorted(
        source.glob("programs/**/*.loom")
    )
    subprocess.run(
        [
            str(arguments.loom_link),
            str(primary),
            *[f"--library={library}" for library in libraries if library != primary],
            f"--root=@{root}",
            f"--target={arguments.target}",
            f"--config-file={configuration}",
            "--strip-check",
            "--to=bc",
            f"--output={output / 'linked.loombc'}",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(arguments.loom_compile),
            str(output / "linked.loombc"),
            "--product=command",
            "--format=loom-command",
            f"--root=@{root}",
            "--pipeline=canonicalize",
            f"--config-file={configuration}",
            f"--output={output / 'manifest.json'}",
            f"--emit-command-artifacts={output / 'commands'}",
            f"--emit-kernel-requests={output / 'requests'}",
        ],
        check=True,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    for request in sorted({entry["source_request"] for entry in manifest["entries"]}):
        print(f"Compiling {stage}/{request}", flush=True)
        subprocess.run(
            [
                str(arguments.loom_compile),
                str(output / "requests" / request),
                f"--target={arguments.target}",
                "--format=amdgpu-hsaco",
                f"--config-file={configuration}",
                f"--output={output / 'kernels' / (Path(request).stem + '.hsaco')}",
            ],
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-capacity", type=int, default=16384)
    parser.add_argument("--prefill-capacity", type=int, default=512)
    parser.add_argument("--span-capacity", type=int, default=4)
    parser.add_argument("--decode-splits", type=int, default=10)
    parser.add_argument("--target", default="amdgpu:gfx1151")
    parser.add_argument(
        "--stage", choices=("prefill", "decode", "epoch", "both", "all"), default="both"
    )
    parser.add_argument(
        "--loom-link",
        type=Path,
        default=Path("bazel-bin/loom/src/loom/tools/loom-link/loom-link"),
    )
    parser.add_argument(
        "--loom-compile",
        type=Path,
        default=Path("bazel-bin/loom/src/loom/tools/loom-compile/loom-compile"),
    )
    arguments = parser.parse_args()
    if not 1 <= arguments.prefill_capacity <= 512:
        parser.error("prefill capacity must be in [1, 512]")
    if not 1 <= arguments.span_capacity <= 8:
        parser.error("span capacity must be in [1, 8]")
    if not 1 <= arguments.decode_splits <= 64:
        parser.error("decode splits must be in [1, 64]")
    if not arguments.prefill_capacity <= arguments.context_capacity <= 262144:
        parser.error("context capacity must contain prefill and be at most 262144")
    stages = {
        "both": ("prefill", "decode"),
        "all": ("prefill", "decode", "epoch"),
    }.get(arguments.stage, (arguments.stage,))
    for stage in stages:
        compile_stage(arguments, stage)


if __name__ == "__main__":
    main()
