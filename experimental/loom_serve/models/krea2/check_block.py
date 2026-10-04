# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify normalization independently, then the exact modulation fusion.

The scalar BF16 modulation rounds after each multiply/add. A single BF16
normalization step can become several output steps after cancellation, so a
relative tolerance on only the final tensor is not its numerical contract.
This check does not replace a complete block/model accuracy comparison.
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
    words = np.fromfile(path, dtype="<u2")
    return torch.from_numpy(words).view(torch.bfloat16)


def encoded(value):
    assert torch.isfinite(value).all(), "nonfinite component value"
    return value.contiguous().view(torch.uint16).numpy().astype("<u2").tobytes()


with safe_open(str(args.checkpoint), framework="pt") as checkpoint:
    # The executed reference loads the table as BF16, despite F32 file storage.
    table = checkpoint.get_tensor("blocks.0.mod.lin").reshape(6, 6144).bfloat16()

for phase in ("base", "style"):
    source = args.reference / f"{phase}-block0-input.bf16"
    modulation = args.reference / f"{phase}-block0-modulation.bf16"
    expected_norm = args.reference / f"{phase}-block0-norm1.bf16"
    actual_norm = args.output / f"{phase}-norm1.bf16"
    normalized = load(expected_norm).reshape(-1, 6144)
    rows = normalized.shape[0]
    common = [
        *args.checker,
        "--model=" + str(args.model),
        "--weights=" + str(args.checkpoint),
        f"--config=krea2.block_tokens={rows}",
        "--input=" + str(source),
    ]
    print(json.dumps(dict(phase=phase, check="independent_normalization")), flush=True)
    subprocess.run(
        common
        + [
            "--root=block0_norm1",
            "--expected=" + str(expected_norm),
            "--actual=" + str(actual_norm),
        ],
        check=True,
    )
    coefficients = load(modulation).reshape(6, 6144) + table
    scale = coefficients[0] + 1
    shift = coefficients[1]

    def affine(value):
        return value * scale + shift

    # Establish the independent pointwise interpretation against the actual
    # external model before using it to check the fused command.
    reference_affine = args.reference / f"{phase}-block0-attention_input.bf16"
    assert encoded(affine(normalized)) == reference_affine.read_bytes(), (
        "pointwise oracle differs from the external model"
    )

    # Normalization was independently qualified above. Exact equivalence here
    # checks fusion, rounding boundaries, indexing and parameter interpretation
    # without pretending different F32 reduction orders are bit-identical.
    composed = affine(load(actual_norm).reshape_as(normalized))
    composed_path = args.output / f"{phase}-unfused.bf16"
    composed_path.write_bytes(encoded(composed))
    actual_path = args.output / f"{phase}-fused.bf16"
    print(json.dumps(dict(phase=phase, check="exact_fusion_equivalence")), flush=True)
    subprocess.run(
        common
        + [
            "--root=block0_attention_input",
            "--input=" + str(modulation),
            "--expected=" + str(composed_path),
            "--actual=" + str(actual_path),
            "--atol=0",
            "--rtol=0",
        ],
        check=True,
    )
    assert actual_path.read_bytes() == composed_path.read_bytes(), "non-bitwise fusion"
    reference = load(reference_affine).float()
    actual = load(actual_path).float()
    error = actual - reference
    print(
        json.dumps(
            dict(
                phase=phase,
                check="external_fused_difference",
                elements=actual.numel(),
                different=int((actual != reference).sum()),
                maximum_absolute_error=float(error.abs().max()),
                relative_l2=float(error.norm() / reference.norm()),
            )
        ),
        flush=True,
    )
print("PASS: independent normalization and bitwise modulation fusion.", flush=True)
