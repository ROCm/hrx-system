# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check queued stack-to-head composition against qualified native outputs.

Consumes reference_stack.py and check_head.py artifacts. No external model or
new arithmetic oracle is needed: every output bit must equal the separately
qualified native chain. Only final velocity tensors are written.
"""

import argparse
import json
import pathlib
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--checker", nargs="+", required=True)
parser.add_argument("--model", type=pathlib.Path, required=True)
parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
parser.add_argument("--adapter", type=pathlib.Path, required=True)
parser.add_argument("--reference", type=pathlib.Path, required=True)
parser.add_argument("--head_results", type=pathlib.Path, required=True)
parser.add_argument("--image_rows", type=int, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
if args.image_rows < 16 or args.image_rows % 16:
    parser.error("--image_rows must describe complete 16-row image tiles")
args.output.mkdir(parents=True, exist_ok=False)

for phase, adapted in (("base", False), ("base", True), ("style", True)):
    name = "zero" if phase == "base" and adapted else phase
    head = args.head_results / f"{phase}-native-{args.image_rows}"
    expected = head / "velocity.bf16"
    actual = args.output / f"{name}-velocity.bf16"
    initial = args.reference / f"{phase}-stack-input.bf16"
    tokens, remainder = divmod(initial.stat().st_size, 6144 * 2)
    if remainder or tokens % 16 or tokens < args.image_rows:
        raise ValueError("combined capture does not contain the complete image suffix")
    if expected.stat().st_size != args.image_rows * 64 * 2:
        raise ValueError("head capture has an unexpected velocity shape")
    prefix = f"{phase}-block27-"
    inputs = [
        initial,
        args.reference / (prefix + "modulation.bf16"),
        args.reference / (prefix + "cosine.f32"),
        args.reference / (prefix + "sine.f32"),
        args.reference / (prefix + "mask.u8"),
        head / "embedding.bf16",
    ]
    checkpoints = [args.checkpoint / "turbo.safetensors"]
    if adapted:
        checkpoints.append(args.adapter / "softwatercolor.safetensors")
        inputs.append(head / "strength.f32")
    command = [
        *args.checker,
        "--model=" + str(args.model),
        "--weight_policy=" + str(args.model / "weights.loom"),
        *["--weights=" + str(path) for path in checkpoints],
        "--root=transformer_velocity" + ("_adapted" if adapted else ""),
        f"--config=krea2.block_tokens={tokens}",
        f"--config=krea2.image_tokens={args.image_rows}",
        *["--input=" + str(path) for path in inputs],
        "--expected=" + str(expected),
        "--actual=" + str(actual),
        "--atol=0",
        "--rtol=0",
    ]
    print(json.dumps(dict(phase=name, check="queued_velocity")), flush=True)
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    print(result.stdout, end="", flush=True)
    result.check_returncode()
    records = [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    ]
    comparisons = [record for record in records if "iteration" in record]
    if [record["iteration"] for record in comparisons] != [0, 1] or any(
        record["different"] for record in comparisons
    ):
        raise AssertionError("both queued executions must equal the native chain")
    if actual.read_bytes() != expected.read_bytes():
        raise AssertionError("non-bitwise stack-to-head composition")
    footprint = next(record for record in records if "workspace_bytes" in record)
    expected_footprint = dict(
        kernels=21 if adapted else 13,
        parameters=818 if adapted else 368,
        parameter_roots=2 if adapted else 1,
        parameter_bytes=24747553024 if adapted else 24317366528,
    )
    if any(footprint[key] != value for key, value in expected_footprint.items()):
        raise AssertionError(
            f"stack-to-head parameter/kernel sharing changed: {footprint}"
        )
    # Qualified block peak storage scales linearly with the tiled token count.
    # Composition adds one combined state, not another block or a suffix copy.
    block_row_bytes = 144640 if adapted else 144384
    maximum_workspace = tokens * (block_row_bytes + 6144 * 2)
    if footprint["workspace_bytes"] > maximum_workspace:
        raise AssertionError(f"stack-to-head workspace exceeds live bound: {footprint}")
    print(
        json.dumps(dict(phase=name, check="velocity_accepted", **footprint)), flush=True
    )

print(
    "PASS: queued stack-to-head base, zero and LoRA outputs equal the native chain.",
    flush=True,
)
