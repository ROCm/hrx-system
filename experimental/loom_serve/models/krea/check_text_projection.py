# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check text projection with F64 primitives and exact native composition.

The text encoder and text fusion remain external captured inputs. CPU-only
reference arithmetic loads just the projection weights, not the full DiT.
One reusable working directory holds intermediate tensors and the current
reproducer; final projections remain in the output directory.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file


def encode(value):
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite text projection tensor")
    return value.contiguous().view(torch.uint16).numpy().astype("<u2").tobytes()


def gelu(value):
    value = value.double()
    argument = math.sqrt(2 / math.pi) * (value + 0.044715 * value.pow(3))
    return (0.5 * value * (1 + argument.tanh())).bfloat16()


def report(phase, rows, name, actual, expected):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("nonfinite text projection comparison")
    error = actual.double() - expected.double()
    print(
        json.dumps(
            dict(
                phase=phase,
                rows=rows,
                check=name,
                relative_l2=float(
                    error.norm() / expected.double().norm().clamp_min(1e-30)
                ),
                maximum_absolute_error=float(error.abs().max()),
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    working = arguments.output / "working"
    working.mkdir()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.set_grad_enabled(False)
    base_checkpoint = arguments.checkpoint / "turbo.safetensors"
    adapter_checkpoint = arguments.adapter / "softwatercolor.safetensors"
    stages = (("first", "txtmlp.1", "linear_1"), ("second", "txtmlp.3", "linear_2"))
    with (
        safe_open(str(base_checkpoint), framework="pt") as weights,
        safe_open(str(adapter_checkpoint), framework="pt") as adapter,
    ):
        scale = weights.get_tensor("txtmlp.0.scale").bfloat16().add(1).double()
        matrices = {
            name: (
                weights.get_tensor(key + ".weight").bfloat16().double(),
                weights.get_tensor(key + ".bias").bfloat16().double(),
                adapter.get_tensor("transformer.txt_in." + target + ".lora_A.weight")
                .bfloat16()
                .double(),
                adapter.get_tensor("transformer.txt_in." + target + ".lora_B.weight")
                .bfloat16()
                .double(),
            )
            for name, key, target in stages
        }

    for phase in ("base", "style"):
        adapted = phase == "style"
        captured = load_file(
            str(arguments.reference / f"{phase}-text_fusion.safetensors")
        )["text_fusion"].reshape(512, 2560)
        external = load_file(
            str(arguments.reference / f"{phase}-text_projection.safetensors")
        )["text_projection"].reshape(512, 6144)
        if captured.dtype != torch.bfloat16 or external.dtype != torch.bfloat16:
            raise ValueError("captured text tensors must use the qualified BF16 format")
        for rows in (16, 80, 512):
            paths = {
                "input": working / "input.bf16",
                "strength": working / "strength.f32",
            }
            value = captured[:rows].contiguous()
            paths["input"].write_bytes(encode(value))
            paths["strength"].write_bytes(
                np.array([float(adapted)], dtype="<f4").tobytes()
            )

            def execute(name, root, expected, inputs, checkpoints, *, exact=False):
                expected_path = working / "expected.bf16"
                actual_path = working / (name + ".bf16")
                expected_path.write_bytes(encode(expected))
                invocation = [
                    *arguments.checker,
                    f"--model={arguments.model / 'qualification'}",
                    f"--weight_policy={arguments.model / 'weights.loom'}",
                    *[f"--weights={path}" for path in checkpoints],
                    f"--root={root}",
                    f"--config=krea2.text_tokens={rows}",
                    *[f"--input={paths[key]}" for key in inputs],
                    f"--expected={expected_path}",
                    f"--actual={actual_path}",
                    *(["--atol=0", "--rtol=0"] if exact else []),
                ]
                (working / "check.json").write_text(
                    json.dumps(
                        dict(phase=phase, rows=rows, check=name, command=invocation),
                        indent=2,
                    )
                    + "\n"
                )
                print(json.dumps(dict(phase=phase, rows=rows, check=name)), flush=True)
                result = subprocess.run(invocation, capture_output=True, text=True)
                print(result.stdout, end="", flush=True)
                print(result.stderr, end="", flush=True)
                result.check_returncode()
                records = [
                    json.loads(line)
                    for line in result.stdout.splitlines()
                    if line.startswith("{")
                ]
                comparisons = [record for record in records if "iteration" in record]
                if [record["iteration"] for record in comparisons] != [0, 1]:
                    raise AssertionError("both device executions must complete")
                if exact and (
                    any(record["different"] for record in comparisons)
                    or actual_path.read_bytes() != encode(expected)
                ):
                    raise AssertionError(f"{name}: native composition is not bitwise")
                actual = torch.from_numpy(np.fromfile(actual_path, dtype="<u2"))
                paths[name] = actual_path
                footprint = next(
                    record for record in records if "workspace_bytes" in record
                )
                return actual.view(torch.bfloat16).reshape_as(expected), footprint

            squared_mean = value.double().square().mean(-1, keepdim=True)
            oracle = (value.double() * (squared_mean + 1e-5).rsqrt() * scale).bfloat16()
            native, _ = execute(
                "normalized", "krea2.text_norm", oracle, ["input"], [base_checkpoint]
            )
            native_name = "normalized"
            for stage_index, (name, _, _) in enumerate(stages):
                if stage_index:
                    oracle = gelu(oracle)
                    native, _ = execute(
                        "activated", "krea2.text_gelu", gelu(native), [native_name], []
                    )
                    native_name = "activated"
                matrix, bias, down, up = matrices[name]
                base, _ = execute(
                    name + "_base",
                    "qualify.text_" + name,
                    (native.double() @ matrix.T + bias).bfloat16(),
                    [native_name],
                    [base_checkpoint],
                )
                oracle_base = (oracle.double() @ matrix.T + bias).bfloat16()
                if adapted:
                    low, _ = execute(
                        name + "_low",
                        "qualify.text_" + name + "_adapter_down",
                        (native.double() @ down.T).bfloat16(),
                        [native_name],
                        [adapter_checkpoint],
                    )
                    delta, _ = execute(
                        name + "_delta",
                        "qualify.text_" + name + "_adapter_up",
                        (low.double() @ up.T).bfloat16(),
                        [name + "_low"],
                        [adapter_checkpoint],
                    )
                    combined, _ = execute(
                        name + "_combined",
                        "qualify.text_" + name + "_adapter_add",
                        base + delta,
                        [name + "_low", "strength", name + "_base"],
                        [adapter_checkpoint],
                        exact=True,
                    )
                    native, _ = execute(
                        name,
                        "qualify.text_" + name + "_adapted",
                        combined,
                        [native_name, "strength"],
                        [base_checkpoint, adapter_checkpoint],
                        exact=True,
                    )
                    oracle_low = (oracle.double() @ down.T).bfloat16()
                    oracle = oracle_base + (oracle_low.double() @ up.T).bfloat16()
                    native_name = name
                else:
                    native, oracle = base, oracle_base
                    native_name = name + "_base"
            whole, footprint = execute(
                "projection",
                "text_projection_adapted" if adapted else "text_projection",
                native,
                ["input", "strength"] if adapted else ["input"],
                [base_checkpoint, adapter_checkpoint] if adapted else [base_checkpoint],
                exact=True,
            )
            expected_counts = dict(
                kernels=7 if adapted else 4,
                parameters=9 if adapted else 5,
                parameter_roots=2 if adapted else 1,
                parameter_bytes=216655872 if adapted else 213968896,
            )
            if any(footprint[key] != value for key, value in expected_counts.items()):
                raise AssertionError(f"text projection residency changed: {footprint}")
            if footprint["workspace_bytes"] > rows * (2560 + 6144 + 32) * 2:
                raise AssertionError(
                    f"text projection workspace exceeds live bound: {footprint}"
                )
            if not adapted:
                execute(
                    "zero",
                    "text_projection_adapted",
                    whole,
                    ["input", "strength"],
                    [base_checkpoint, adapter_checkpoint],
                    exact=True,
                )
            (arguments.output / f"{phase}-{rows}.bf16").write_bytes(encode(whole))
            report(phase, rows, "native_projection_vs_f64", whole, oracle)
            report(phase, rows, "native_projection_vs_external", whole, external[:rows])
            print(
                json.dumps(dict(phase=phase, rows=rows, accepted=True, **footprint)),
                flush=True,
            )

    print(
        "PASS: text projection primitives and exact base/zero/LoRA composition.",
        flush=True,
    )


if __name__ == "__main__":
    main()
