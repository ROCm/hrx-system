# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify block residuals, post-attention modulation, and SwiGLU fusion."""

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
        raise ValueError("nonfinite pointwise tensor")
    return value.contiguous().view(torch.uint16).numpy().astype("<u2").tobytes()


with safe_open(str(args.checkpoint), framework="pt") as checkpoint:
    table = checkpoint.get_tensor("blocks.0.mod.lin").reshape(6, 6144).bfloat16()
    postnorm_scale = (
        checkpoint.get_tensor("blocks.0.postnorm.scale").bfloat16().add(1).double()
    )

for phase in ("base", "style"):
    modulation_path = args.reference / f"{phase}-block0-modulation.bf16"
    coefficients = load(modulation_path).reshape(6, 6144) + table
    residual_path = args.reference / f"{phase}-block0-attention_residual.bf16"
    rows, remainder = divmod(residual_path.stat().st_size, 6144 * 2)
    if remainder or rows < 16 or rows % 16:
        raise ValueError("invalid tiled block capture")
    for count in (16, rows):
        prefix = args.output / f"{phase}-{count}"

        def source(name, width=6144):
            path = args.reference / f"{phase}-block0-{name}.bf16"
            values = load(path).reshape(rows, width)[:count]
            if count != rows:
                path = prefix.with_suffix(f".{name}.input.bf16")
                path.write_bytes(encode(values))
            return path, values

        def run(component, inputs, expected, *, exact):
            expected_path = prefix.with_suffix(f".{component}.expected.bf16")
            actual_path = prefix.with_suffix(f".{component}.actual.bf16")
            expected_path.write_bytes(encode(expected))
            print(
                json.dumps(
                    dict(phase=phase, rows=count, component=component, exact=exact)
                ),
                flush=True,
            )
            command = [
                *args.checker,
                "--model=" + str(args.model),
                "--weights=" + str(args.checkpoint),
                "--root=block0_" + component,
                f"--config=krea2.block_tokens={count}",
                *["--input=" + str(path) for path in inputs],
                "--expected=" + str(expected_path),
                "--actual=" + str(actual_path),
            ]
            if exact:
                command += ["--atol=0", "--rtol=0"]
            subprocess.run(command, check=True)
            actual = load(actual_path).reshape_as(expected)
            if exact:
                assert encode(actual) == encode(expected), component
            return actual

        for component, initial, update, coefficient, external in (
            ("attention_residual", "input", "attention", 2, "attention_residual"),
            (
                "feed_forward_residual",
                "attention_residual",
                "feed_forward",
                5,
                "output",
            ),
        ):
            initial_path, initial_values = source(initial)
            update_path, update_values = source(update)
            expected = initial_values + coefficients[coefficient] * update_values
            _, external_values = source(external)
            assert encode(expected) == encode(external_values), (
                "residual interpretation differs from the external model"
            )
            run(
                component,
                [initial_path, update_path, modulation_path],
                expected,
                exact=True,
            )

        initial_path, initial_values = source("attention_residual")
        precise_values = initial_values.double()
        normalized = (
            precise_values
            * (precise_values.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
            * postnorm_scale
        ).bfloat16()
        qualified_norm = run("norm2", [initial_path], normalized, exact=False)
        _, external_norm = source("norm2")
        _, external_affine = source("feed_forward_input")

        def affine(value):
            return value * (coefficients[3] + 1) + coefficients[4]

        assert encode(affine(external_norm)) == encode(external_affine), (
            "post-attention affine interpretation differs from the external model"
        )
        run(
            "feed_forward_input",
            [initial_path, modulation_path],
            affine(qualified_norm),
            exact=True,
        )

        gate_path, gate = source("feed_forward_gate", 16384)
        up_path, up = source("feed_forward_up", 16384)
        oracle_silu = torch.nn.functional.silu(gate.double()).bfloat16()
        qualified_silu = run("feed_forward_silu", [gate_path], oracle_silu, exact=False)
        _, external_product = source("feed_forward_product", 16384)
        assert encode(torch.nn.functional.silu(gate) * up) == encode(
            external_product
        ), "SwiGLU interpretation differs from the external model"
        run(
            "feed_forward_product",
            [gate_path, up_path],
            qualified_silu * up,
            exact=True,
        )
print(
    "PASS: gated residuals, post-attention affine fusion, and SwiGLU fusion.",
    flush=True,
)
