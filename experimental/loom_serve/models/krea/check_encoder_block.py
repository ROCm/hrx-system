# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify native encoder blocks and adjacent-layer in-place composition.

Every stage uses an independent F64 primitive with explicit BF16 boundaries.
The complete command must then reproduce those qualified native stages exactly,
including fused rotary/product boundaries. Layers 0 and 34 use real canonical
inputs; layer 1 consumes native layer 0 to prove the advancing hidden lifetime.
One overwritten fixture and final blocks retain less than 128 MiB.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from check_text_fusion_blocks import encode, load_bf16, report
from encoder_reference import evaluate_encoder_block
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    working = arguments.output / "working"
    working.mkdir()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.set_grad_enabled(False)
    checkpoint = arguments.checkpoint / "text_encoder/model.safetensors"
    original_mask = torch.from_numpy(
        np.fromfile(arguments.reference / "mask.u8", dtype="u1")
    ).bool()
    if original_mask.shape != (546,) or not original_mask[0]:
        raise ValueError("qualification requires the canonical 546-row mask")
    with safe_open(str(checkpoint), framework="pt") as weights:
        for texts in (16, 512):
            rows = texts + 48
            count = min(rows, 546)

            def captured(name, width, fill=0):
                source = load_bf16(arguments.reference / (name + ".bf16"), (546, width))
                value = torch.full((rows, width), fill, dtype=torch.bfloat16)
                value[:count] = source[:count]
                return value

            cosine = captured("cosine", 128, fill=1)
            sine = captured("sine", 128)
            mask = torch.zeros(rows, dtype=torch.bool)
            mask[:count] = original_mask[:count]
            first = captured("layer0.input", 2560)
            initial = first
            prior_native = None
            for layer in (0, 1, 34):
                context = dict(text_tokens=texts, encoder_rows=rows, layer=layer)
                if layer == 1:
                    initial = prior_native
                elif layer == 34:
                    initial = captured("layer34.input", 2560)

                def execute(name, expected, operands, *, parameters=True, exact=False):
                    paths = []
                    for index, value in enumerate(operands):
                        path = working / f"input-{index}"
                        path.write_bytes(encode(value))
                        paths.append(path)
                    expected_path = working / "expected.bf16"
                    actual_path = working / "actual.bf16"
                    expected_bytes = encode(expected)
                    expected_path.write_bytes(expected_bytes)
                    prefix = "qualify" if parameters or name == "silu" else "krea2"
                    invocation = [
                        *arguments.checker,
                        f"--model={arguments.model / 'qualification'}",
                        f"--weight_policy={arguments.model / 'weights.loom'}",
                        *([f"--weights={checkpoint}"] if parameters else []),
                        f"--root={prefix}.encoder_{name}",
                        f"--config=krea2.text_tokens={texts}",
                        f"--config=qualify.encoder_layer={layer}",
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
                    print(
                        json.dumps(dict(**context, check=name, exact=exact)), flush=True
                    )
                    result = subprocess.run(invocation, capture_output=True, text=True)
                    print(result.stdout, end="", flush=True)
                    print(result.stderr, end="", flush=True)
                    result.check_returncode()
                    records = [
                        json.loads(line)
                        for line in result.stdout.splitlines()
                        if line.startswith("{")
                    ]
                    comparisons = [
                        record for record in records if "iteration" in record
                    ]
                    if [record["iteration"] for record in comparisons] != [0, 1]:
                        raise AssertionError("both native executions must complete")
                    if exact and (
                        any(record["different"] for record in comparisons)
                        or actual_path.read_bytes() != expected_bytes
                    ):
                        raise AssertionError(
                            f"{name}: native composition differs bitwise"
                        )
                    footprint = next(
                        record for record in records if "workspace_bytes" in record
                    )
                    if name in ("block", "pair"):
                        layers = 2 if name == "pair" else 1
                        expected_footprint = dict(
                            parameter_bytes=201861632 * layers,
                            parameter_roots=1,
                            parameters=11 * layers,
                            kernels=11,
                        )
                        # The live FFN extent is 44,032 bytes per row. The
                        # current best-fit layout leaves a 1,024-byte tail
                        # between its factors; better packing remains valid.
                        if footprint["workspace_bytes"] > rows * 45056 or any(
                            footprint[key] != value
                            for key, value in expected_footprint.items()
                        ):
                            raise AssertionError(
                                f"{name}: unexpected block residency: {footprint}"
                            )
                    elif footprint["workspace_bytes"] != 0 or footprint["kernels"] != 1:
                        raise AssertionError(
                            f"{name}: primitive requires unexpected scratch"
                        )
                    return load_bf16(actual_path, expected.shape)

                operands = (initial, cosine, sine, mask, weights, layer)
                oracle = evaluate_encoder_block(*operands)
                native = evaluate_encoder_block(*operands, observe=execute)
                whole = execute(
                    "block", native, [initial, cosine, sine, mask], exact=True
                )
                report(context, "block_vs_f64", whole, oracle)
                if layer in (0, 34):
                    external = captured(f"layer{layer}.output", 2560)
                    report(
                        context, "block_vs_external", whole[:count], external[:count]
                    )
                if layer == 1:
                    execute("pair", whole, [first, cosine, sine, mask], exact=True)
                (arguments.output / f"layer{layer}-{rows}.bf16").write_bytes(
                    encode(whole)
                )
                prior_native = whole

    print(
        "PASS: encoder blocks, exact native stages and in-place adjacent layers.",
        flush=True,
    )


if __name__ == "__main__":
    main()
