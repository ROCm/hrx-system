# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify embedding, 35 in-place decoder layers and strided tap collection.

Exact embedding lookup and independently executed native decoder blocks define
the composition oracle. Local and accumulated F64/canonical differences remain
diagnostics, separate from the primitive arithmetic gates in check_encoder_block.
The 16-token case is a causal prefix of the canonical request, not a newly
tokenized short prompt. The complete 512-token case retains the live suffix.
One overwritten fixture plus final taps and request data use less than 160 MiB.
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
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--encoder_reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    working = arguments.output / "working"
    working.mkdir()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.set_grad_enabled(False)
    checkpoint = arguments.checkpoint / "text_encoder/model.safetensors"
    original_ids = torch.from_numpy(
        np.fromfile(arguments.encoder_reference / "tokens.i32", dtype="<i4")
    )
    original_mask = torch.from_numpy(
        np.fromfile(arguments.encoder_reference / "mask.u8", dtype="u1")
    )
    if (
        original_ids.shape != (546,)
        or original_mask.shape != (546,)
        or (original_ids < 0).any()
        or (original_ids >= 151936).any()
        or (original_mask > 1).any()
        or not original_mask[0]
        or not original_mask[-5:].all()
    ):
        raise ValueError("expected canonical 546-row tokens and binary causal mask")
    capture = load_file(str(arguments.reference / "inputs.safetensors"))
    canonical = capture["encoder_hidden"]
    if canonical.shape != (1, 512, 12, 2560) or canonical.dtype != torch.bfloat16:
        raise ValueError("expected canonical BF16[1,512,12,2560] taps")
    if not torch.equal(capture["encoder_mask"].reshape(-1), original_mask[34:].bool()):
        raise ValueError("encoder key mask differs from the fusion input mask")

    with safe_open(str(checkpoint), framework="pt") as weights:
        embedding = weights.get_tensor("language_model.embed_tokens.weight")
        for texts in (16, 512):
            rows, logical_rows = texts + 48, texts + 34
            context = dict(text_tokens=texts, encoder_rows=rows)
            token_ids = original_ids[:logical_rows].contiguous()
            mask = torch.zeros(rows, dtype=torch.bool)
            mask[:logical_rows] = original_mask[:logical_rows].bool()
            tables = []
            for name, fill in (("cosine", 1), ("sine", 0)):
                table = torch.full((rows, 128), fill, dtype=torch.bfloat16)
                table[:logical_rows] = load_bf16(
                    arguments.encoder_reference / (name + ".bf16"), (546, 128)
                )[:logical_rows]
                tables.append(table)
            cosine, sine = tables
            request = [token_ids.numpy().astype("<i4").tobytes(), *tables, mask]
            request_paths = []
            for name, value in zip(
                ("tokens.i32", "cosine.bf16", "sine.bf16", "mask.u8"), request
            ):
                path = arguments.output / f"{texts}-{name}"
                path.write_bytes(value if isinstance(value, bytes) else encode(value))
                request_paths.append(path)

            def execute(root, expected, operands, *, layer=None, exact=False):
                inputs = []
                for index, value in enumerate(operands):
                    if isinstance(value, Path):
                        inputs.append(value)
                    else:
                        path = working / f"input-{index}"
                        path.write_bytes(encode(value))
                        inputs.append(path)
                expected_path = working / "expected.bf16"
                expected_bytes = encode(expected)
                expected_path.write_bytes(expected_bytes)
                actual_path = working / "actual.bf16"
                invocation = [
                    *arguments.checker,
                    f"--model={arguments.model / 'qualification'}",
                    f"--weight_policy={arguments.model / 'weights.loom'}",
                    f"--weights={checkpoint}",
                    f"--root={root}",
                    f"--config=krea2.text_tokens={texts}",
                    *(
                        [f"--config=qualify.encoder_layer={layer}"]
                        if layer is not None
                        else []
                    ),
                    *[f"--input={path}" for path in inputs],
                    f"--expected={expected_path}",
                    f"--actual={actual_path}",
                    *(["--atol=0", "--rtol=0"] if exact else ["--report_only"]),
                ]
                record = dict(**context, root=root, layer=layer, exact=exact)
                (working / "check.json").write_text(
                    json.dumps(dict(**record, command=invocation), indent=2) + "\n"
                )
                print(json.dumps(record), flush=True)
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
                    raise AssertionError("both native executions must complete")
                if exact and (
                    any(record["different"] for record in comparisons)
                    or actual_path.read_bytes() != expected_bytes
                ):
                    raise AssertionError(f"{root}: native composition differs bitwise")
                footprint = next(
                    record for record in records if "workspace_bytes" in record
                )
                if root == "text_encoder":
                    expected_footprint = dict(
                        parameter_bytes=7843069440,
                        parameters=386,
                        parameter_roots=1,
                        kernels=13,
                    )
                    maximum_workspace = rows * (5120 + 45056)
                elif root == "krea2.encoder_embedding":
                    expected_footprint = dict(
                        parameter_bytes=777912320,
                        parameters=1,
                        parameter_roots=1,
                        kernels=1,
                    )
                    maximum_workspace = 0
                else:
                    expected_footprint = dict(
                        parameter_bytes=201861632,
                        parameters=11,
                        parameter_roots=1,
                        kernels=11,
                    )
                    maximum_workspace = rows * 45056
                if footprint["workspace_bytes"] > maximum_workspace or any(
                    footprint[key] != value for key, value in expected_footprint.items()
                ):
                    raise AssertionError(f"{root}: unexpected residency: {footprint}")
                return load_bf16(actual_path, expected.shape)

            embedded = torch.zeros(rows, 2560, dtype=torch.bfloat16)
            embedded[:logical_rows] = embedding[token_ids.long()]
            native = execute(
                "krea2.encoder_embedding", embedded, request_paths[:1], exact=True
            )
            oracle = embedded
            native_taps = torch.empty(texts, 12, 2560, dtype=torch.bfloat16)
            oracle_taps = torch.empty_like(native_taps)
            for layer in range(35):
                local = evaluate_encoder_block(
                    native, cosine, sine, mask, weights, layer
                )
                oracle = (
                    local
                    if layer == 0
                    else evaluate_encoder_block(
                        oracle, cosine, sine, mask, weights, layer
                    )
                )
                native = execute(
                    "qualify.encoder_block",
                    local,
                    [native, *request_paths[1:]],
                    layer=layer,
                )
                report(
                    dict(**context, layer=layer), "block_vs_local_f64", native, local
                )
                if layer % 3 == 1:
                    native_taps[:, layer // 3] = native[34:logical_rows]
                    oracle_taps[:, layer // 3] = oracle[34:logical_rows]
            whole = execute("text_encoder", native_taps, request_paths, exact=True)
            report(context, "taps_vs_f64_chain", whole, oracle_taps)
            report(context, "taps_vs_canonical", whole, canonical[0, :texts])
            live = mask[34:logical_rows]
            report(context, "live_taps_vs_f64_chain", whole[live], oracle_taps[live])
            report(
                context,
                "live_taps_vs_canonical",
                whole[live],
                canonical[0, :texts][live],
            )
            (arguments.output / f"taps-{texts}.bf16").write_bytes(encode(whole))

    print(
        "PASS: exact native embedding and 35-layer encoder tap composition.", flush=True
    )


if __name__ == "__main__":
    main()
