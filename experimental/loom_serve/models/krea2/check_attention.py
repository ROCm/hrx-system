# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check ordinary BF16 DiT attention math, masking, replay and gate fusion.

Representable analytic cases have exact answers. Real captured inputs report
independent F64 and ordinary K64 CPU diagnostics, not an expanded-precision
trajectory requirement. Exact replay, masked-key independence and composition
with the native gate remain hard checks. All CPU references process one head
at a time without expanding KV heads or retaining every score matrix.
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
    actual, expected = actual.double(), expected.double()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError(f"nonfinite attention comparison: {check}")
    error = actual - expected
    print(
        json.dumps(
            dict(
                phase=phase,
                rows=rows,
                check=check,
                diagnostic=True,
                elements=actual.numel(),
                different=int((error != 0).sum()),
                outside_tolerance=int(
                    (error.abs() > 0.0001220703125 + 0.0078125 * expected.abs()).sum()
                ),
                maximum_absolute_error=float(error.abs().max()),
                relative_l2=float(error.norm() / expected.norm().clamp_min(1e-30)),
            )
        ),
        flush=True,
    )


def write_inputs(label, query, key, value, mask):
    paths = []
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        path = args.output / f"{label}.{name}.bf16"
        stored = tensor.reshape(mask.numel(), -1).T if name == "value" else tensor
        path.write_bytes(encode(stored))
        paths.append(path)
    mask_path = args.output / f"{label}.mask.u8"
    mask_path.write_bytes(mask.numpy().astype("u1").tobytes())
    return [*paths, mask_path]


def execute(label, component, inputs, expected, *, exact):
    # One oracle scratch file is sufficient; native outputs retain their names.
    expected_path = args.output / "expected.bf16"
    actual_path = args.output / f"{label}.actual.bf16"
    expected_path.write_bytes(encode(expected))
    print(
        json.dumps(dict(check=label, rows=expected.shape[0], exact=exact)),
        flush=True,
    )
    result = subprocess.run(
        [
            *args.checker,
            "--model=" + str(args.model / "qualification"),
            "--root=qualify.block_" + component,
            f"--config=krea2.block_tokens={expected.shape[0]}",
            *["--input=" + str(path) for path in inputs],
            "--expected=" + str(expected_path),
            "--actual=" + str(actual_path),
            *(["--atol=0", "--rtol=0"] if exact else ["--report_only"]),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    print(result.stdout, end="", flush=True)
    result.check_returncode()
    records = [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    ]
    comparisons = [record for record in records if "iteration" in record]
    if [record["iteration"] for record in comparisons] != [0, 1]:
        raise AssertionError(f"{label}: checker did not complete both executions")
    if any(
        record["nonfinite"] or record["elements"] != expected.numel()
        for record in comparisons
    ):
        raise AssertionError(f"{label}: invalid output extent or nonfinite values")
    actual = load(actual_path).reshape_as(expected)
    if not torch.isfinite(actual).all():
        raise AssertionError(f"{label}: nonfinite saved output")
    if exact and (
        any(record["different"] for record in comparisons)
        or actual_path.read_bytes() != encode(expected)
    ):
        raise AssertionError(f"{label}: output is not bitwise equal")
    return actual


def ordinary_reference(query, key, value, mask):
    """Model K64/BF16 probability boundaries, not native WMMA/exp rounding."""
    count = mask.numel()
    scale = 1 / math.sqrt(128)
    result = torch.empty(count, 48, 128, dtype=torch.bfloat16)
    for head in range(48):
        query_head = query[:, head].float()
        key_head = key[:, head // 4].float()
        value_head = value[:, head // 4].float()
        maximum = torch.full((count,), -1e30, dtype=torch.float32)
        denominator = torch.zeros(count, dtype=torch.float32)
        numerator = torch.zeros(count, 128, dtype=torch.float32)
        for start in range(0, count, 64):
            end = min(start + 64, count)
            scores = torch.zeros(count, end - start, dtype=torch.float32)
            for channel in range(0, 128, 16):
                scores = scores + (
                    query_head[:, channel : channel + 16]
                    @ key_head[start:end, channel : channel + 16].T
                )
            scores[:, ~mask[start:end]] = -1e30
            next_maximum = torch.maximum(maximum, scores.max(-1).values)
            probability = ((scores - next_maximum[:, None]) * scale).exp()
            probability[:, ~mask[start:end]] = 0
            old_scale = ((maximum - next_maximum) * scale).exp()
            denominator = denominator * old_scale + probability.sum(-1)
            rounded = probability.bfloat16().float()
            product = torch.zeros_like(numerator)
            for offset in range(0, end - start, 16):
                product = product + (
                    rounded[:, offset : offset + 16]
                    @ value_head[start + offset : start + offset + 16]
                )
            numerator = numerator * old_scale[:, None] + product
            maximum = next_maximum
        result[:, head] = (numerator * denominator.reciprocal()[:, None]).bfloat16()
    return result


# Each neighboring key pair balances around an integer mean. Both selected V
# rows and the 64-key average are exactly BF16-representable, away from ties.
keys = torch.arange(80)[:, None, None]
heads = torch.arange(12)[None, :, None]
channels = torch.arange(128)[None, None, :]
mean_value = 8 + 3 * heads + channels.remainder(19)
deviation = (1 - 2 * keys.remainder(2)) * (1 + (keys // 2).remainder(3))
analytic_value = (mean_value + deviation).bfloat16()
analytic_query = (
    (torch.arange(80 * 48 * 128).reshape(80, 48, 128).remainder(17) - 8) / 8
).bfloat16()
analytic_key = (
    (torch.arange(80 * 12 * 128).reshape(80, 12, 128).remainder(13) - 6) / 8
).bfloat16()
for selected_key in (0, 79, None):
    label = (
        "analytic-average64" if selected_key is None else f"analytic-key{selected_key}"
    )
    mask = torch.zeros(80, dtype=torch.bool)
    if selected_key is None:
        mask[16:] = True
        query, key = torch.zeros_like(analytic_query), torch.zeros_like(analytic_key)
        expected_row = mean_value[0].bfloat16()
    else:
        mask[selected_key] = True
        query, key = analytic_query, analytic_key
        expected_row = analytic_value[selected_key]
    expected = expected_row.repeat_interleave(4, dim=0)[None].expand(80, -1, -1)
    inputs = write_inputs(label, query, key, analytic_value, mask)
    execute(label, "attention_ungated", inputs, expected, exact=True)
    gate_path = args.output / f"{label}.gate.bf16"
    gate_path.write_bytes(encode(torch.zeros_like(expected)))
    execute(
        label + "-gated",
        "attention_context",
        [*inputs, gate_path],
        expected * 0.5,
        exact=True,
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
        print(
            json.dumps(
                dict(phase=phase, rows=count, check="cpu_attention_diagnostics")
            ),
            flush=True,
        )
        precise = torch.empty(count, 48, 128, dtype=torch.float64)
        for head in range(48):
            scores = (
                selected_query[:, head].double() @ selected_key[:, head // 4].double().T
            ) / math.sqrt(128)
            scores[:, ~selected_mask] = -torch.inf
            precise[:, head] = (
                scores.softmax(dim=-1) @ selected_value[:, head // 4].double()
            )
        expected = precise.bfloat16()
        label = f"{phase}-{count}"
        inputs = write_inputs(
            label, selected_query, selected_key, selected_value, selected_mask
        )
        actual = execute(label, "attention_ungated", inputs, expected, exact=False)
        execute(label + "-replay", "attention_ungated", inputs, actual, exact=True)
        ordinary = ordinary_reference(
            selected_query, selected_key, selected_value, selected_mask
        )
        report(phase, count, "attention_vs_bf16_f64", actual, expected)
        report(phase, count, "attention_vs_continuous_f64", actual, precise)
        report(phase, count, "attention_vs_ordinary_k64_cpu", actual, ordinary)
        report(phase, count, "ordinary_k64_cpu_vs_continuous_f64", ordinary, precise)
        gate_path = args.output / f"{label}.gate.bf16"
        gate_path.write_bytes(encode(selected_gate))
        cpu_gated = actual * torch.sigmoid(selected_gate)
        native_gated = execute(
            label + "-gate-only",
            "attention_gate",
            [args.output / f"{label}.actual.bf16", gate_path],
            cpu_gated,
            exact=False,
        )
        report(phase, count, "native_gate_vs_cpu_sigmoid", native_gated, cpu_gated)
        actual_gated = execute(
            label + "-gated",
            "attention_context",
            [*inputs, gate_path],
            native_gated,
            exact=True,
        )
        execute(
            label + "-gated-replay",
            "attention_context",
            [*inputs, gate_path],
            actual_gated,
            exact=True,
        )
        if count == 80:
            perturbed_key, perturbed_value = (
                selected_key.clone(),
                selected_value.clone(),
            )
            perturbed_key[~selected_mask] = 3
            perturbed_value[~selected_mask] = -7
            perturbed = write_inputs(
                label + "-masked",
                selected_query,
                perturbed_key,
                perturbed_value,
                selected_mask,
            )
            execute(
                label + "-masked",
                "attention_ungated",
                perturbed,
                actual,
                exact=True,
            )
            execute(
                label + "-masked-gated",
                "attention_context",
                [*perturbed, gate_path],
                actual_gated,
                exact=True,
            )
        f64_gated = expected * torch.sigmoid(selected_gate)
        report(phase, count, "gated_vs_f64_attention_cpu_gate", actual_gated, f64_gated)
        if count == rows:
            external = load(
                args.reference / f"{phase}-block0-attention_context.bf16"
            ).reshape_as(expected)
            report(phase, count, "external_vs_f64", external, f64_gated)
print(
    "PASS: exact attention math, mask independence, fresh replay and native gate fusion; "
    "real-input F64 and ordinary-K64 distances are diagnostics.",
    flush=True,
)
