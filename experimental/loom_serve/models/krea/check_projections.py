# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify block-zero BF16 contractions against an independent F64 oracle.

The base captures exclude LoRA updates. All contraction shapes are qualified
at the complete captured sequence and an independently JITed 16-row prefix.
Library differences are recorded separately: its reduction order is not the
accuracy contract. F64 weights exist only in this CPU qualification script.
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
parser.add_argument("--reference", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
torch.set_grad_enabled(False)


def load(path):
    return torch.from_numpy(np.fromfile(path, dtype="<u2")).view(torch.bfloat16)


def encode(value):
    return value.contiguous().view(torch.uint16).numpy().astype("<u2").tobytes()


for component, source, expected, weight_name, inputs, outputs in (
    ("query", "attention_input", "query", "attn.wq", 6144, 6144),
    ("key", "attention_input", "key", "attn.wk", 6144, 1536),
    ("value", "attention_input", "value", "attn.wv", 6144, 1536),
    ("gate", "attention_input", "gate", "attn.gate", 6144, 6144),
    ("attention_output", "attention_context", "attention", "attn.wo", 6144, 6144),
    (
        "feed_forward_gate",
        "feed_forward_input",
        "feed_forward_gate",
        "mlp.gate",
        6144,
        16384,
    ),
    ("feed_forward_up", "feed_forward_input", "feed_forward_up", "mlp.up", 6144, 16384),
    (
        "feed_forward_down",
        "feed_forward_product",
        "feed_forward",
        "mlp.down",
        16384,
        6144,
    ),
):
    source_path = args.reference / f"base-block0-{source}.bf16"
    expected_path = args.reference / f"base-block0-{expected}.bf16"
    rows, remainder = divmod(source_path.stat().st_size, inputs * 2)
    if remainder or rows < 16 or rows % 16:
        raise ValueError(f"invalid tiled input length: {source_path}")
    if expected_path.stat().st_size != rows * outputs * 2:
        raise ValueError(f"projection output length disagrees: {expected_path}")
    with safe_open(str(args.checkpoint), framework="pt") as checkpoint:
        weight = checkpoint.get_tensor(f"blocks.0.{weight_name}.weight")
        if weight.dtype != torch.bfloat16 or weight.shape != (outputs, inputs):
            raise ValueError(f"unexpected BF16 contraction weight: {weight_name}")
        oracle = (
            load(source_path).reshape(rows, inputs).double() @ weight.double().T
        ).bfloat16()
    if not torch.isfinite(oracle).all():
        raise ValueError(f"nonfinite F64 contraction oracle: {component}")
    oracle_path = args.output / f"{component}-oracle.bf16"
    oracle_path.write_bytes(encode(oracle.T if component == "value" else oracle))
    library = load(expected_path).reshape_as(oracle).float()
    error = library - oracle.float()
    print(
        json.dumps(
            dict(
                component=component,
                check="external_library_vs_f64",
                elements=oracle.numel(),
                different=int((error != 0).sum()),
                outside_tolerance=int(
                    (
                        error.abs() > 0.0001220703125 + 0.0078125 * oracle.float().abs()
                    ).sum()
                ),
                maximum_absolute_error=float(error.abs().max()),
                relative_l2=float(error.norm() / oracle.float().norm()),
            )
        ),
        flush=True,
    )
    for count in (rows, 16):
        if count == rows:
            input_path = source_path
            reference_path = oracle_path
        else:
            input_path = args.output / f"{component}-input-{count}.bf16"
            reference_path = args.output / f"{component}-reference-{count}.bf16"
            with source_path.open("rb") as stream:
                input_path.write_bytes(stream.read(count * inputs * 2))
            # Select logical rows before encoding this shape's physical stride.
            selected = oracle[:count]
            reference_path.write_bytes(
                encode(selected.T if component == "value" else selected)
            )
        print(json.dumps(dict(component=component, rows=count)), flush=True)
        subprocess.run(
            [
                *args.checker,
                "--model=" + str(args.model / "qualification"),
                "--weight_policy=" + str(args.model / "weights.loom"),
                "--weights=" + str(args.checkpoint),
                "--root=qualify.block_" + component,
                f"--config=krea2.block_tokens={count}",
                "--input=" + str(input_path),
                "--expected=" + str(reference_path),
                "--actual=" + str(args.output / f"{component}-{count}.bf16"),
            ],
            check=True,
        )
print(
    "PASS: all eight block projections against F64 at full and 16-row JIT shapes.",
    flush=True,
)
