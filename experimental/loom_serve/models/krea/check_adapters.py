# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify block LoRA factors, fused epilogues, and checkpoint domains.

CPU/F64 checks each native matrix contraction. Epilogue and whole-projection
composition must then match the qualified operands bit-for-bit. Two executions
share resident weights; zero strength preserves the base bits. This is a
component accuracy check, not an image or throughput benchmark.
"""

import argparse
import json
import pathlib
import subprocess

import numpy as np
import torch
from safetensors import safe_open

parser = argparse.ArgumentParser()
parser.add_argument("--checker", nargs="+", required=True)
parser.add_argument("--model", type=pathlib.Path, required=True)
parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
parser.add_argument("--adapter", type=pathlib.Path, required=True)
parser.add_argument("--reference", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
torch.set_grad_enabled(False)


def load(path):
    return torch.from_numpy(np.fromfile(path, dtype="<u2")).view(torch.bfloat16)


def encode(value):
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite adapter tensor")
    return value.contiguous().view(torch.uint16).numpy().astype("<u2").tobytes()


with (
    safe_open(str(args.checkpoint), framework="pt") as weights,
    safe_open(str(args.adapter), framework="pt") as adapter,
):
    for phase in ("base", "style"):
        for component, source, weight_key, adapter_key, inputs, outputs in (
            ("query", "attention_input", "attn.wq", "attn.to_q", 6144, 6144),
            ("key", "attention_input", "attn.wk", "attn.to_k", 6144, 1536),
            ("value", "attention_input", "attn.wv", "attn.to_v", 6144, 1536),
            ("gate", "attention_input", "attn.gate", "attn.to_gate", 6144, 6144),
            (
                "attention_output",
                "attention_context",
                "attn.wo",
                "attn.to_out.0",
                6144,
                6144,
            ),
            (
                "feed_forward_gate",
                "feed_forward_input",
                "mlp.gate",
                "ff.gate",
                6144,
                16384,
            ),
            (
                "feed_forward_up",
                "feed_forward_input",
                "mlp.up",
                "ff.up",
                6144,
                16384,
            ),
            (
                "feed_forward_down",
                "feed_forward_product",
                "mlp.down",
                "ff.down",
                16384,
                6144,
            ),
        ):
            captured = load(args.reference / f"{phase}-block0-{source}.bf16")
            rows, remainder = divmod(captured.numel(), inputs)
            if remainder or rows < 16 or rows % 16:
                raise ValueError(f"invalid tiled input: {phase} {component}")
            matrix = weights.get_tensor(f"blocks.0.{weight_key}.weight")
            prefix = f"transformer.transformer_blocks.0.{adapter_key}.lora_"
            down = adapter.get_tensor(prefix + "A.weight")
            up = adapter.get_tensor(prefix + "B.weight")
            if (
                matrix.dtype != torch.bfloat16
                or matrix.shape != (outputs, inputs)
                or down.dtype != torch.float32
                or down.shape != (32, inputs)
                or up.dtype != torch.float32
                or up.shape != (outputs, 32)
            ):
                raise ValueError(f"unexpected checkpoint shape/dtype: {component}")
            down, up = down.bfloat16().double(), up.bfloat16().double()
            for count in (16, rows):
                directory = args.output / f"{phase}-{component}-{count}"
                directory.mkdir()
                value = captured.reshape(rows, inputs)[-count:]
                input_path = directory / "input.bf16"
                input_path.write_bytes(encode(value))
                strength_path = directory / "strength.f32"
                paths = {"input": input_path, "strength": strength_path}

                def execute(name, expected, arguments, checkpoints, *, exact=False):
                    # Reuse one oracle file; retained operands have distinct paths.
                    expected_path = args.output / "expected.bf16"
                    value_plane = component == "value" and name in (
                        "base",
                        "adapter_add",
                        "adapted",
                    )
                    stored_expected = expected.T if value_plane else expected
                    expected_path.write_bytes(encode(stored_expected))
                    actual_path = directory / f"{name}.actual.bf16"
                    print(
                        json.dumps(
                            dict(
                                phase=phase,
                                component=component,
                                rows=count,
                                check=name,
                            )
                        ),
                        flush=True,
                    )
                    command = [
                        *args.checker,
                        "--model=" + str(args.model / "qualification"),
                        "--weight_policy=" + str(args.model / "weights.loom"),
                        *["--weights=" + str(path) for path in checkpoints],
                        "--root=qualify.block_"
                        + component
                        + ("" if name == "base" else "_" + name),
                        f"--config=krea2.block_tokens={count}",
                        *["--input=" + str(paths[key]) for key in arguments],
                        "--expected=" + str(expected_path),
                        "--actual=" + str(actual_path),
                    ]
                    if exact:
                        command += ["--atol=0", "--rtol=0"]
                    subprocess.run(command, check=True)
                    actual = load(actual_path).reshape_as(stored_expected)
                    if value_plane:
                        actual = actual.T
                    if exact and encode(actual) != encode(expected):
                        raise AssertionError(f"{component} {name}: non-bitwise fusion")
                    # The file retains physical V layout for later native inputs;
                    # the returned tensor is logical for independent arithmetic.
                    paths[name] = actual_path
                    return actual

                base = execute(
                    "base",
                    (value.double() @ matrix.double().T).bfloat16(),
                    ["input"],
                    [args.checkpoint],
                )
                low = execute(
                    "adapter_down",
                    (value.double() @ down.T).bfloat16(),
                    ["input"],
                    [args.adapter],
                )
                delta = execute(
                    "adapter_up",
                    (low.double() @ up.T).bfloat16(),
                    ["adapter_down"],
                    [args.adapter],
                )
                for strength in (0.0, 0.5, 1.0):
                    strength_path.write_bytes(
                        np.array([strength], dtype="<f4").tobytes()
                    )
                    print(json.dumps(dict(strength=strength)), flush=True)
                    expected = base if strength == 0 else base + delta * strength
                    fused = execute(
                        "adapter_add",
                        expected,
                        ["adapter_down", "strength", "base"],
                        [args.adapter],
                        exact=True,
                    )
                    execute(
                        "adapted",
                        fused,
                        ["input", "strength"],
                        [args.checkpoint, args.adapter],
                        exact=True,
                    )

print(
    "PASS: all eight LoRA projections, factor accuracy and exact fusion/composition.",
    flush=True,
)
