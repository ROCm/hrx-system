# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify normalization, modulation fusion, and normalized rotary embeddings.

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
from diffusers.models.embeddings import apply_rotary_emb
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
    head_scales = {
        name: checkpoint.get_tensor(f"blocks.0.attn.qknorm.{letter}norm.scale")
        .bfloat16()
        .add(1)
        .double()
        for name, letter in (("query", "q"), ("key", "k"))
    }

for phase in ("base", "style"):
    source = args.reference / f"{phase}-block0-input.bf16"
    modulation = args.reference / f"{phase}-block0-modulation.bf16"
    expected_norm = args.reference / f"{phase}-block0-norm1.bf16"
    actual_norm = args.output / f"{phase}-norm1.bf16"
    normalized = load(expected_norm).reshape(-1, 6144)
    rows = normalized.shape[0]
    common = [
        *args.checker,
        "--model=" + str(args.model / "qualification"),
        "--weight_policy=" + str(args.model / "weights.loom"),
        "--weights=" + str(args.checkpoint),
        f"--config=krea2.block_tokens={rows}",
        "--input=" + str(source),
    ]
    print(json.dumps(dict(phase=phase, check="independent_normalization")), flush=True)
    subprocess.run(
        common
        + [
            "--root=qualify.block_norm1",
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
            "--root=qualify.block_attention_input",
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
    cosine_path = args.reference / f"{phase}-block0-cosine.f32"
    sine_path = args.reference / f"{phase}-block0-sine.f32"
    cosine = torch.from_numpy(np.fromfile(cosine_path, dtype="<f4")).reshape(rows, 128)
    sine = torch.from_numpy(np.fromfile(sine_path, dtype="<f4")).reshape(rows, 128)
    for name, heads in (("query", 48), ("key", 12)):
        source_path = args.reference / f"{phase}-block0-{name}.bf16"
        values = load(source_path).reshape(rows, heads, 128).double()
        normalized = (
            values
            * (values.square().mean(dim=-1, keepdim=True) + 1e-5).rsqrt()
            * head_scales[name]
        ).bfloat16()
        external_norm = load(
            args.reference / f"{phase}-block0-{name}_norm.bf16"
        ).reshape_as(normalized)
        error = external_norm.float() - normalized.float()
        print(
            json.dumps(
                dict(
                    phase=phase,
                    component=name,
                    check="external_head_norm_vs_f64",
                    different=int((error != 0).sum()),
                    relative_l2=float(error.norm() / normalized.float().norm()),
                )
            ),
            flush=True,
        )

        expected_norm_path = args.output / f"{phase}-{name}-norm-f64.bf16"
        actual_norm_path = args.output / f"{phase}-{name}-norm.bf16"
        expected_norm_path.write_bytes(encoded(normalized))
        subprocess.run(
            [
                *args.checker,
                "--model=" + str(args.model / "qualification"),
                "--weight_policy=" + str(args.model / "weights.loom"),
                "--weights=" + str(args.checkpoint),
                f"--config=krea2.block_tokens={rows}",
                f"--root=qualify.block_{name}_norm",
                "--input=" + str(source_path),
                "--expected=" + str(expected_norm_path),
                "--actual=" + str(actual_norm_path),
            ],
            check=True,
        )
        qualified_norm = load(actual_norm_path).reshape_as(normalized)

        # Evaluate the published interleaved rotation independently, and check
        # that interpretation against the pinned library's actual operation.
        # As with modulation, exact fusion uses independently qualified norm
        # results; cancellation can amplify a single BF16 normalization step.
        pairs = qualified_norm.float().reshape(rows, heads, 64, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        expected = (
            qualified_norm.float() * cosine[:, None, :] + rotated * sine[:, None, :]
        ).bfloat16()
        library_rotated = apply_rotary_emb(
            qualified_norm.unsqueeze(0), (cosine, sine), sequence_dim=1
        ).squeeze(0)
        assert encoded(expected) == encoded(library_rotated), (
            "rotary oracle disagreement"
        )
        f64_rotated = (
            apply_rotary_emb(normalized.unsqueeze(0), (cosine, sine), sequence_dim=1)
            .squeeze(0)
            .float()
        )
        for count in (rows, 16):
            # The small specialization takes the last image rows, not the
            # unrotated text prefix. Tables retain their original coordinates.
            start = rows - count
            prefix = args.output / f"{phase}-{name}-rotary-{count}"
            input_path = source_path
            cosine_input = cosine_path
            sine_input = sine_path
            if count != rows:
                input_path = prefix.with_suffix(".input.bf16")
                input_path.write_bytes(encoded(values[start:].bfloat16()))
                cosine_input = prefix.with_suffix(".cosine.f32")
                cosine_input.write_bytes(cosine[start:].numpy().astype("<f4").tobytes())
                sine_input = prefix.with_suffix(".sine.f32")
                sine_input.write_bytes(sine[start:].numpy().astype("<f4").tobytes())
            expected_path = prefix.with_suffix(".expected.bf16")
            expected_path.write_bytes(encoded(expected[start:]))
            print(
                json.dumps(
                    dict(phase=phase, component=name, check="fused_rotary", rows=count)
                ),
                flush=True,
            )
            subprocess.run(
                [
                    *args.checker,
                    "--model=" + str(args.model / "qualification"),
                    "--weight_policy=" + str(args.model / "weights.loom"),
                    "--weights=" + str(args.checkpoint),
                    f"--config=krea2.block_tokens={count}",
                    f"--root=qualify.block_{name}_rotary",
                    "--input=" + str(input_path),
                    "--input=" + str(cosine_input),
                    "--input=" + str(sine_input),
                    "--expected=" + str(expected_path),
                    "--actual=" + str(prefix.with_suffix(".actual.bf16")),
                    "--atol=0",
                    "--rtol=0",
                ],
                check=True,
            )
            actual_path = prefix.with_suffix(".actual.bf16")
            assert actual_path.read_bytes() == expected_path.read_bytes(), (
                "non-bitwise rotary fusion"
            )
            actual = load(actual_path).reshape(count, heads, 128).float()
            reference = f64_rotated[start:]
            error = actual - reference
            print(
                json.dumps(
                    dict(
                        phase=phase,
                        component=name,
                        check="rotary_vs_f64_normalization",
                        rows=count,
                        different=int((error != 0).sum()),
                        outside_single_bf16_envelope=int(
                            (
                                error.abs()
                                > 0.0001220703125 + 0.0078125 * reference.abs()
                            ).sum()
                        ),
                        relative_l2=float(error.norm() / reference.norm()),
                    )
                ),
                flush=True,
            )
print(
    "PASS: normalization, exact modulation fusion, and normalized rotary embeddings.",
    flush=True,
)
