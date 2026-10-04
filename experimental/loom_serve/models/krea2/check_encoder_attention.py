# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify causal GQA on real early/late encoder Q/K/V and padding.

The CPU F64 oracle owns one head's scores at a time. Native outputs must also
remain bitwise unchanged when masked K/V change, and earlier query rows must
remain exact when future K/V change inside a key tile. One overwritten fixture
and four retained attention outputs use less than 96 MiB, without weights.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
from check_text_fusion_blocks import encode, load_bf16, report


def attention(query, key, value, mask):
    rows, heads, width = query.shape
    positions = torch.arange(rows)
    visible = (positions[:, None] >= positions[None, :]) & mask[None, :]
    result = torch.empty_like(query)
    for head in range(heads):
        scores = query[:, head].double() @ key[:, head // 4].double().T
        scores /= math.sqrt(width)
        scores.masked_fill_(~visible, -torch.inf)
        result[:, head] = (scores.softmax(-1) @ value[:, head // 4].double()).bfloat16()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--normalization_results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    working = arguments.output / "working"
    working.mkdir()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.set_grad_enabled(False)
    original_mask = torch.from_numpy(
        np.fromfile(arguments.reference / "mask.u8", dtype="u1")
    ).bool()
    if (
        original_mask.shape != (546,)
        or not original_mask[0]
        or not original_mask[-5:].all()
    ):
        raise ValueError("qualification requires the canonical prefix and live suffix")
    for texts in (16, 512):
        rows = texts + 48
        count = min(rows, 546)
        mask = torch.zeros(rows, dtype=torch.bool)
        mask[:count] = original_mask[:count]
        for layer in (0, 34):
            context = dict(text_tokens=texts, encoder_rows=rows, layer=layer)
            query = load_bf16(
                arguments.normalization_results / f"layer{layer}-{rows}-query.bf16",
                (rows, 32, 128),
            )
            key = load_bf16(
                arguments.normalization_results / f"layer{layer}-{rows}-key.bf16",
                (rows, 8, 128),
            )
            value = torch.zeros(rows, 8, 128, dtype=torch.bfloat16)
            value[:count] = load_bf16(
                arguments.reference / f"layer{layer}.value.bf16", (546, 8, 128)
            )[:count]

            def execute(name, expected, selected_key, selected_value, *, exact=False):
                paths = []
                for index, operand in enumerate(
                    (query, selected_key, selected_value, mask)
                ):
                    path = working / f"input-{index}"
                    path.write_bytes(encode(operand))
                    paths.append(path)
                expected_path = working / "expected.bf16"
                actual_path = working / "actual.bf16"
                expected_bytes = encode(expected)
                expected_path.write_bytes(expected_bytes)
                invocation = [
                    *arguments.checker,
                    f"--model={arguments.model}",
                    "--root=krea2.encoder_attention",
                    f"--config=krea2.text_tokens={texts}",
                    *[f"--input={path}" for path in paths],
                    f"--expected={expected_path}",
                    f"--actual={actual_path}",
                    *(["--atol=0", "--rtol=0"] if exact else []),
                ]
                (working / "check.json").write_text(
                    json.dumps(
                        dict(**context, check=name, command=invocation), indent=2
                    )
                    + "\n"
                )
                print(json.dumps(dict(**context, check=name, exact=exact)), flush=True)
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
                    raise AssertionError("both attention executions must complete")
                if exact and (
                    any(record["different"] for record in comparisons)
                    or actual_path.read_bytes() != expected_bytes
                ):
                    raise AssertionError("masked K/V affect attention output")
                footprint = next(
                    record for record in records if "workspace_bytes" in record
                )
                if footprint != dict(
                    workspace_bytes=0,
                    parameter_bytes=0,
                    parameter_roots=0,
                    parameters=0,
                    kernels=1,
                ):
                    raise AssertionError(f"unexpected attention residency: {footprint}")
                return load_bf16(actual_path, expected.shape)

            expected = attention(query, key, value, mask)
            native = execute("causal_vs_f64", expected, key, value)
            if encode(native[0]) != encode(value[0].repeat_interleave(4, dim=0)):
                raise AssertionError("first query must return its sole visible value")
            (arguments.output / f"layer{layer}-{rows}.bf16").write_bytes(encode(native))
            external = load_bf16(
                arguments.reference / f"layer{layer}.attention.bf16", (546, 32, 128)
            )[:count]
            report(context, "native_vs_external", native[:count], external)

            masked_key, masked_value = key.clone(), value.clone()
            masked_key[~mask] = 31
            masked_value[~mask] = -47
            execute("masked_values_exact", native, masked_key, masked_value, exact=True)

            # Row 21 is inside both the query and key tile. Altering every
            # following K/V must not affect rows 0..20, including row 20 itself.
            cut = 21
            future_key, future_value = key.clone(), value.clone()
            future_key[cut:] = (-future_key[cut:].float() + 3).bfloat16()
            future_value[cut:] = (future_value[cut:].float() * -2 + 5).bfloat16()
            future_expected = attention(query, future_key, future_value, mask)
            future = execute(
                "future_values_vs_f64", future_expected, future_key, future_value
            )
            if encode(future[:cut]) != encode(native[:cut]):
                raise AssertionError("future K/V affect earlier query rows")
            if encode(future[cut:]) == encode(native[cut:]):
                raise AssertionError("future perturbation did not exercise a live key")
            print(
                json.dumps(
                    dict(
                        **context,
                        check="causal_prefix_exact",
                        rows=cut,
                        elements=future[:cut].numel(),
                        different=0,
                    )
                ),
                flush=True,
            )

    print(
        "PASS: causal GQA, masked-value isolation and exact prefix causality.",
        flush=True,
    )


if __name__ == "__main__":
    main()
