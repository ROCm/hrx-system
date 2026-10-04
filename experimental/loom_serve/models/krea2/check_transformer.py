# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify an entire base or LoRA transformer block and command composition.

Each separately executed native component is checked against independent CPU
arithmetic. Reductions use F64; fusion checks retain the qualified native
reduction output. The whole command must equal that native chain bit-for-bit.
A second, wholly CPU chain measures accumulated error independently of the
external implementation. This is a numerical harness, not a benchmark.
"""

import argparse
import json
import math
import pathlib
import subprocess
from contextlib import nullcontext

import numpy as np
import torch
from safetensors import safe_open

parser = argparse.ArgumentParser()
parser.add_argument("--checker", nargs="+", required=True)
parser.add_argument("--model", type=pathlib.Path, required=True)
parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
parser.add_argument("--adapter", type=pathlib.Path)
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
        strength=args.strength,
        rows=rows,
        check=name,
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


capture = args.phase + "-block0-"
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
    coefficients = (
        modulation + weights.get_tensor("blocks.0.mod.lin").reshape(6, 6144).bfloat16()
    )

    def normalize(value, key):
        scale = weights.get_tensor("blocks.0." + key).bfloat16().add(1).double()
        precise = value.double()
        inverse = (precise.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
        return (precise * inverse * scale).bfloat16()

    def linear(value, key):
        matrix = weights.get_tensor("blocks.0." + key + ".weight").bfloat16()
        return (value.double() @ matrix.double().T).bfloat16()

    def affine(value, row):
        return value * (coefficients[row] + 1) + coefficients[row + 1]

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

        def rotate(value):
            pairs = value.float().reshape(count, -1, 64, 2)
            rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
            return (
                value.float() * selected_cosine[:, None, :]
                + rotated * selected_sine[:, None, :]
            ).bfloat16()

        def attention(query, key, value):
            value = value.reshape(count, 12, 128)
            result = torch.empty(count, 48, 128, dtype=torch.bfloat16)
            for head in range(48):
                scores = (
                    query[:, head].double() @ key[:, head // 4].double().T
                ) / math.sqrt(128)
                scores[:, ~selected_mask] = -torch.inf
                result[:, head] = (
                    scores.softmax(-1) @ value[:, head // 4].double()
                ).bfloat16()
            return result.reshape(count, 6144)

        def execute(component, expected, arguments, *, exact=False, checkpoints=None):
            # Retain native operands, not a duplicate oracle file for every stage.
            expected_path = args.output / "expected.bf16"
            actual_path = prefix.with_suffix(f".{component}.actual.bf16")
            expected_path.write_bytes(encode(expected))
            command = [
                *args.checker,
                "--model=" + str(args.model),
                *[
                    "--weights=" + str(path)
                    for path in (
                        [args.checkpoint] if checkpoints is None else checkpoints
                    )
                ],
                "--root=block0_" + component,
                f"--config=krea2.block_tokens={count}",
                *["--input=" + str(paths[name]) for name in arguments],
                "--expected=" + str(expected_path),
                "--actual=" + str(actual_path),
            ]
            if exact:
                command += ["--atol=0", "--rtol=0"]
            subprocess.run(command, check=True)
            actual = load_bf16(actual_path).reshape_as(expected)
            if exact and encode(actual) != encode(expected):
                raise AssertionError(f"{component}: non-bitwise composition")
            paths[component] = actual_path
            return actual

        def evaluate(*, native):
            def compute(name, expected, arguments, *, exact=False, checkpoints=None):
                print(
                    json.dumps(dict(rows=count, component=name, native=native)),
                    flush=True,
                )
                return (
                    execute(
                        name,
                        expected,
                        arguments,
                        exact=exact,
                        checkpoints=checkpoints,
                    )
                    if native
                    else expected
                )

            def project(name, key, adapter_key, value, argument):
                base = compute(name, linear(value, key), [argument])
                if adapter is None:
                    return base
                factor = "transformer.transformer_blocks.0." + adapter_key + ".lora_"
                down = adapter.get_tensor(factor + "A.weight").bfloat16().double()
                up = adapter.get_tensor(factor + "B.weight").bfloat16().double()
                low = compute(
                    name + "_adapter_down",
                    (value.double() @ down.T).bfloat16(),
                    [argument],
                    checkpoints=[args.adapter],
                )
                delta = compute(
                    name + "_adapter_up",
                    (low.double() @ up.T).bfloat16(),
                    [name + "_adapter_down"],
                    checkpoints=[args.adapter],
                )
                combined = compute(
                    name + "_adapted",
                    base if args.strength == 0 else base + delta * args.strength,
                    [argument, "strength"],
                    exact=True,
                    checkpoints=[args.checkpoint, args.adapter],
                )
                if native:
                    paths[name] = paths[name + "_adapted"]
                return combined

            source = initial[start:]
            normalized = compute("norm1", normalize(source, "prenorm.scale"), ["input"])
            attention_input = compute(
                "attention_input",
                affine(normalized, 0),
                ["input", "modulation"],
                exact=True,
            )
            projections = {}
            for name, key, adapter_key in (
                ("query", "wq", "to_q"),
                ("key", "wk", "to_k"),
                ("value", "wv", "to_v"),
                ("gate", "gate", "to_gate"),
            ):
                projections[name] = project(
                    name,
                    "attn." + key,
                    "attn." + adapter_key,
                    attention_input,
                    "attention_input",
                )
            for name, heads, key in (("query", 48, "qnorm"), ("key", 12, "knorm")):
                normalized = compute(
                    name + "_norm",
                    normalize(
                        projections[name].reshape(count, heads, 128),
                        "attn.qknorm." + key + ".scale",
                    ),
                    [name],
                )
                projections[name] = compute(
                    name + "_rotary",
                    rotate(normalized),
                    [name, "cosine", "sine"],
                    exact=True,
                )
            context = compute(
                "attention_ungated",
                attention(
                    projections["query"], projections["key"], projections["value"]
                ),
                ["query_rotary", "key_rotary", "value", "mask"],
            )
            context = compute(
                "attention_context",
                context * torch.sigmoid(projections["gate"].double()).bfloat16(),
                ["query_rotary", "key_rotary", "value", "mask", "gate"],
                exact=True,
            )
            update = project(
                "attention_output",
                "attn.wo",
                "attn.to_out.0",
                context,
                "attention_context",
            )
            residual = compute(
                "attention_residual",
                source + coefficients[2] * update,
                ["input", "attention_output", "modulation"],
                exact=True,
            )
            normalized = compute(
                "norm2", normalize(residual, "postnorm.scale"), ["attention_residual"]
            )
            feed_forward_input = compute(
                "feed_forward_input",
                affine(normalized, 3),
                ["attention_residual", "modulation"],
                exact=True,
            )
            gate = project(
                "feed_forward_gate",
                "mlp.gate",
                "ff.gate",
                feed_forward_input,
                "feed_forward_input",
            )
            up = project(
                "feed_forward_up",
                "mlp.up",
                "ff.up",
                feed_forward_input,
                "feed_forward_input",
            )
            activated = compute(
                "feed_forward_silu",
                torch.nn.functional.silu(gate.double()).bfloat16(),
                ["feed_forward_gate"],
            )
            product = compute(
                "feed_forward_product",
                activated * up,
                ["feed_forward_gate", "feed_forward_up"],
                exact=True,
            )
            update = project(
                "feed_forward_down",
                "mlp.down",
                "ff.down",
                product,
                "feed_forward_product",
            )
            return compute(
                "feed_forward_residual",
                residual + coefficients[5] * update,
                ["attention_residual", "feed_forward_down", "modulation"],
                exact=True,
            )

        oracle = evaluate(native=False)
        prefix.with_suffix(".oracle.bf16").write_bytes(encode(oracle))
        serial = evaluate(native=True)
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
        native_error = report(count, "whole_command_vs_f64", combined, oracle)
        if count == rows:
            external = load_bf16(args.reference / (capture + "output.bf16")).reshape_as(
                oracle
            )
            external_error = report(count, "external_vs_f64", external, oracle)
            if native_error["relative_l2"] > external_error["relative_l2"]:
                raise AssertionError("whole-block error exceeds the external baseline")

print(
    "PASS: whole block equals the qualified native chain and meets the F64 accuracy gate.",
    flush=True,
)
