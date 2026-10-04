# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify noncausal GQA and its BF16 sigmoid gate on real Krea inputs.

The independent CPU oracle uses F64 scores, softmax, and value contraction.
It never expands KV heads or retains all heads' score matrices simultaneously.
Small shapes use image coordinates; the 80-row case begins with a completely
masked 64-key tile and ends with a partial 16-key tile.
"""

import argparse
import json
import math
import pathlib
import subprocess

import numpy as np
import torch
from diffusers.models.embeddings import apply_rotary_emb

parser = argparse.ArgumentParser()
parser.add_argument("--checker", nargs="+", required=True)
parser.add_argument("--model", type=pathlib.Path, required=True)
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
        raise ValueError("nonfinite attention tensor")
    return value.contiguous().view(torch.uint16).numpy().astype("<u2").tobytes()


def report(phase, rows, check, actual, expected):
    actual, expected = actual.float(), expected.float()
    error = actual - expected
    print(
        json.dumps(
            dict(
                phase=phase,
                rows=rows,
                check=check,
                elements=actual.numel(),
                different=int((error != 0).sum()),
                outside_tolerance=int(
                    (error.abs() > 0.0001220703125 + 0.0078125 * expected.abs()).sum()
                ),
                maximum_absolute_error=float(error.abs().max()),
                relative_l2=float(error.norm() / expected.norm()),
            )
        ),
        flush=True,
    )


for phase in ("base", "style"):
    mask = torch.from_numpy(
        np.fromfile(args.reference / f"{phase}-block0-mask.u8", dtype="u1")
    ).bool()
    rows = mask.numel()
    if rows < 80 or rows % 16 or not mask[-16:].all():
        raise ValueError("reference must contain tiled text and image rows")
    masked = torch.nonzero(~mask).flatten()
    if masked.numel() < 64:
        raise ValueError("reference must contain 64 masked text rows")
    cosine = torch.from_numpy(
        np.fromfile(args.reference / f"{phase}-block0-cosine.f32", dtype="<f4")
    ).reshape(rows, 128)
    sine = torch.from_numpy(
        np.fromfile(args.reference / f"{phase}-block0-sine.f32", dtype="<f4")
    ).reshape(rows, 128)
    query = apply_rotary_emb(
        load(args.reference / f"{phase}-block0-query_norm.bf16").reshape(
            1, rows, 48, 128
        ),
        (cosine, sine),
        sequence_dim=1,
    ).squeeze(0)
    key = apply_rotary_emb(
        load(args.reference / f"{phase}-block0-key_norm.bf16").reshape(
            1, rows, 12, 128
        ),
        (cosine, sine),
        sequence_dim=1,
    ).squeeze(0)
    value = load(args.reference / f"{phase}-block0-value.bf16").reshape(rows, 12, 128)
    gate = load(args.reference / f"{phase}-block0-gate.bf16").reshape(rows, 48, 128)
    for indices in (
        torch.arange(rows - 16, rows),
        torch.cat((masked[:64], torch.arange(rows - 16, rows))),
        torch.arange(rows),
    ):
        count = indices.numel()
        selected_query = query[indices]
        selected_key = key[indices]
        selected_value = value[indices]
        selected_mask = mask[indices]
        selected_gate = gate[indices]
        expected = torch.empty(count, 48, 128, dtype=torch.bfloat16)
        for head in range(48):
            scores = (
                selected_query[:, head].double() @ selected_key[:, head // 4].double().T
            ) / math.sqrt(128)
            scores[:, ~selected_mask] = -torch.inf
            expected[:, head] = (
                scores.softmax(dim=-1) @ selected_value[:, head // 4].double()
            ).bfloat16()
        prefix = args.output / f"{phase}-{count}"
        input_paths = []
        for name, tensor in (
            ("query", selected_query),
            ("key", selected_key),
            ("value", selected_value),
        ):
            path = prefix.with_suffix(f".{name}.bf16")
            path.write_bytes(encode(tensor))
            input_paths.append(path)
        mask_path = prefix.with_suffix(".mask.u8")
        mask_path.write_bytes(selected_mask.numpy().astype("u1").tobytes())
        input_paths.append(mask_path)
        expected_path = prefix.with_suffix(".expected.bf16")
        expected_path.write_bytes(encode(expected))
        actual_path = prefix.with_suffix(".actual.bf16")
        command = [
            *args.checker,
            "--model=" + str(args.model / "qualification"),
            f"--config=krea2.block_tokens={count}",
            *["--input=" + str(path) for path in input_paths],
        ]
        print(
            json.dumps(dict(phase=phase, rows=count, check="attention_vs_f64")),
            flush=True,
        )
        subprocess.run(
            command
            + [
                "--root=qualify.block_attention_ungated",
                "--expected=" + str(expected_path),
                "--actual=" + str(actual_path),
            ],
            check=True,
        )
        qualified = load(actual_path).reshape_as(expected)
        expected_gated = qualified * torch.sigmoid(selected_gate)
        expected_gated_path = prefix.with_suffix(".gated.expected.bf16")
        expected_gated_path.write_bytes(encode(expected_gated))
        gate_path = prefix.with_suffix(".gate.bf16")
        gate_path.write_bytes(encode(selected_gate))
        actual_gated_path = prefix.with_suffix(".gated.actual.bf16")
        print(
            json.dumps(dict(phase=phase, rows=count, check="attention_gate_fusion")),
            flush=True,
        )
        subprocess.run(
            command
            + [
                "--root=qualify.block_attention_context",
                "--input=" + str(gate_path),
                "--expected=" + str(expected_gated_path),
                "--actual=" + str(actual_gated_path),
                "--atol=0",
                "--rtol=0",
            ],
            check=True,
        )
        actual_gated = load(actual_gated_path).reshape_as(expected)
        assert encode(actual_gated) == encode(expected_gated), "non-bitwise gate fusion"
        f64_gated = expected * torch.sigmoid(selected_gate)
        report(phase, count, "gated_vs_f64", actual_gated, f64_gated)
        if count == rows:
            external = load(
                args.reference / f"{phase}-block0-attention_context.bf16"
            ).reshape_as(expected)
            report(phase, count, "external_vs_f64", external, f64_gated)
print("PASS: masked GQA and bitwise sigmoid-gate fusion.", flush=True)
