# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify an entire base or LoRA transformer block and command composition.

Unchanged reductions retain their independent F64 primitive gates. Ordinary
attention reports F64 distance and its gate must compose exactly with the
native ungated output. The whole command must equal that native chain
bit-for-bit. A second, wholly CPU chain measures accumulated error independently
of the external implementation. This is a numerical harness, not a benchmark.
"""

import argparse
import json
import pathlib
import subprocess
from contextlib import nullcontext

import numpy as np
import torch
from block_reference import evaluate_block
from safetensors import safe_open

parser = argparse.ArgumentParser()
parser.add_argument("--checker", nargs="+", required=True)
parser.add_argument("--model", type=pathlib.Path, required=True)
parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
parser.add_argument("--adapter", type=pathlib.Path)
parser.add_argument("--layer", type=int, choices=range(28), default=0)
parser.add_argument("--phase", choices=("base", "style"), default="base")
parser.add_argument("--strength", type=float, default=0.0)
parser.add_argument("--reference", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
if (args.phase, args.strength) not in (("base", 0.0), ("style", 1.0)):
    parser.error("whole-block captures require base/strength=0 or style/strength=1")
if args.phase == "style" and args.adapter is None:
    parser.error("the style capture requires --adapter")
args.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
torch.set_grad_enabled(False)


def load_bf16(path):
    return torch.from_numpy(np.fromfile(path, dtype="<u2")).view(torch.bfloat16)


def encode(value):
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite block tensor")
    return value.contiguous().view(torch.uint16).numpy().astype("<u2").tobytes()


def report(rows, name, actual, expected):
    actual, expected = actual.double(), expected.double()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("nonfinite block comparison")
    error = actual - expected
    result = dict(
        phase=args.phase,
        layer=args.layer,
        strength=args.strength,
        rows=rows,
        check=name,
        diagnostic=True,
        elements=actual.numel(),
        different=int((error != 0).sum()),
        outside_single_bf16_envelope=int(
            (error.abs() > 2**-13 + 2**-7 * expected.abs()).sum()
        ),
        maximum_absolute_error=float(error.abs().max()),
        relative_l2=float(error.norm() / expected.norm().clamp_min(1e-30)),
    )
    print(json.dumps(result), flush=True)
    return result


capture = f"{args.phase}-block{args.layer}-"
mask = torch.from_numpy(
    np.fromfile(args.reference / (capture + "mask.u8"), dtype="u1")
).bool()
rows = mask.numel()
if rows < 16 or rows % 16 or not mask[-16:].all():
    raise ValueError("block capture must end with at least 16 valid image tokens")
initial = load_bf16(args.reference / (capture + "input.bf16")).reshape(rows, 6144)
modulation = load_bf16(args.reference / (capture + "modulation.bf16")).reshape(6, 6144)
cosine = torch.from_numpy(
    np.fromfile(args.reference / (capture + "cosine.f32"), dtype="<f4")
).reshape(rows, 128)
sine = torch.from_numpy(
    np.fromfile(args.reference / (capture + "sine.f32"), dtype="<f4")
).reshape(rows, 128)

with (
    safe_open(str(args.checkpoint), framework="pt") as weights,
    (
        safe_open(str(args.adapter), framework="pt")
        if args.adapter is not None
        else nullcontext()
    ) as adapter,
):
    for count in (16, rows):
        start = rows - count
        selected_cosine, selected_sine = cosine[start:], sine[start:]
        selected_mask = mask[start:]
        prefix = args.output / str(count)
        paths = {}
        if adapter is not None:
            paths["strength"] = args.output / "strength.f32"
            paths["strength"].write_bytes(
                np.array([args.strength], dtype="<f4").tobytes()
            )
        for name, tensor, dtype in (
            ("input", initial[start:], None),
            ("modulation", modulation, None),
            ("cosine", selected_cosine, "<f4"),
            ("sine", selected_sine, "<f4"),
            ("mask", selected_mask, "u1"),
        ):
            path = prefix.with_suffix(f".{name}.input")
            path.write_bytes(
                encode(tensor)
                if dtype is None
                else tensor.numpy().astype(dtype).tobytes()
            )
            paths[name] = path

        def execute(
            component,
            expected,
            arguments,
            *,
            exact=False,
            diagnostic=False,
            checkpoints=None,
        ):
            # Retain native operands, not a duplicate oracle file for every stage.
            expected_path = args.output / "expected.bf16"
            actual_path = prefix.with_suffix(f".{component}.actual.bf16")
            value_plane = component in ("value", "value_adapted")
            stored_expected = expected.T if value_plane else expected
            expected_path.write_bytes(encode(stored_expected))
            command = [
                *args.checker,
                "--model=" + str(args.model / "qualification"),
                "--weight_policy=" + str(args.model / "weights.loom"),
                *[
                    "--weights=" + str(path)
                    for path in (
                        [args.checkpoint] if checkpoints is None else checkpoints
                    )
                ],
                "--root=qualify.block_" + component,
                f"--config=krea2.block_tokens={count}",
                f"--config=krea2.block_index={args.layer}",
                *["--input=" + str(paths[name]) for name in arguments],
                "--expected=" + str(expected_path),
                "--actual=" + str(actual_path),
            ]
            if exact:
                command += ["--atol=0", "--rtol=0"]
            elif diagnostic:
                command.append("--report_only")
            result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
            print(result.stdout, end="", flush=True)
            result.check_returncode()
            records = [
                json.loads(line)
                for line in result.stdout.splitlines()
                if line.startswith("{")
            ]
            comparisons = [record for record in records if "iteration" in record]
            if [record["iteration"] for record in comparisons] != [0, 1]:
                raise AssertionError(f"{component}: missing checker executions")
            if any(record["nonfinite"] for record in comparisons):
                raise AssertionError(f"{component}: nonfinite output")
            actual = load_bf16(actual_path).reshape_as(stored_expected)
            if value_plane:
                actual = actual.T
            if exact and (
                any(record["different"] for record in comparisons)
                or encode(actual) != encode(expected)
            ):
                raise AssertionError(f"{component}: non-bitwise composition")
            # Native consumers receive the unchanged physical output file.
            paths[component] = actual_path
            return actual

        def observe(
            name,
            expected,
            arguments,
            *,
            exact=False,
            parameters=("base",),
            result_name=None,
        ):
            print(json.dumps(dict(rows=count, component=name, native=True)), flush=True)
            checkpoint_paths = {"base": args.checkpoint, "adapter": args.adapter}
            if name == "attention_context":
                # The CPU sigmoid remains a diagnostic; fusion must reproduce
                # the actual gate on the rounded native attention output.
                cpu_expected = expected
                expected = execute(
                    "attention_gate",
                    cpu_expected,
                    ["attention_ungated", "gate"],
                    diagnostic=True,
                    checkpoints=[],
                )
                report(count, "native_attention_gate_vs_cpu", expected, cpu_expected)
            actual = execute(
                name,
                expected,
                arguments,
                exact=exact,
                diagnostic=name == "attention_ungated",
                checkpoints=[checkpoint_paths[key] for key in parameters],
            )
            if name == "attention_ungated":
                report(count, "attention_vs_f64_diagnostic", actual, expected)
            if result_name is not None:
                paths[result_name] = paths[name]
            return actual

        operands = (
            initial[start:],
            modulation,
            selected_cosine,
            selected_sine,
            selected_mask,
            weights,
            adapter,
            args.layer,
            args.strength,
        )
        oracle = evaluate_block(*operands)
        prefix.with_suffix(".oracle.bf16").write_bytes(encode(oracle))
        serial = evaluate_block(*operands, observe=observe)
        inputs = ["input", "modulation", "cosine", "sine", "mask"]
        combined = execute(
            "forward" if adapter is None else "forward_adapted",
            serial,
            inputs if adapter is None else [*inputs, "strength"],
            checkpoints=(
                [args.checkpoint]
                if adapter is None
                else [args.checkpoint, args.adapter]
            ),
            exact=True,
        )
        if adapter is not None and args.strength == 0:
            execute("forward", combined, inputs, exact=True)
        report(count, "whole_command_vs_f64", combined, oracle)
        if count == rows:
            external = load_bf16(args.reference / (capture + "output.bf16")).reshape_as(
                oracle
            )
            report(count, "external_vs_f64", external, oracle)

print(
    "PASS: whole block equals the checked native chain; ordinary attention and "
    "whole-chain F64 distances are diagnostic.",
    flush=True,
)
