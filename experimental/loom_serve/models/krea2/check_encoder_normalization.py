# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify encoder RMS and half-split rotary on real early/late layer tensors.

Independent F64 primitives retain every BF16 tensor-rounding boundary. The
fused head transform must match separately executed native normalization and
rotation bit-for-bit, twice. One overwritten fixture and final head outputs
retain less than 96 MiB; original checkpoint and reference files are reused.
"""

import argparse
import json
import subprocess
from pathlib import Path

import torch
from check_text_fusion_blocks import encode, load_bf16, report
from safetensors import safe_open


def normalize(value, weight):
    value = value.double()
    inverse = (value.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    rounded = (value * inverse).bfloat16()
    return (rounded.double() * weight.double()).bfloat16()


def rotate(value, cosine, sine):
    cosine, sine = cosine.double().unsqueeze(1), sine.double().unsqueeze(1)
    rotated = torch.cat((-value[..., 64:], value[..., :64]), dim=-1).double()
    direct = (value.double() * cosine).bfloat16()
    cross = (rotated * sine).bfloat16()
    return (direct.double() + cross.double()).bfloat16()


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
    with safe_open(str(checkpoint), framework="pt") as parameters:
        for texts in (16, 512):
            rows = texts + 48

            def captured(name, shape, fill=0):
                original = load_bf16(
                    arguments.reference / (name + ".bf16"), (546, *shape)
                )
                value = torch.full((rows, *shape), fill, dtype=torch.bfloat16)
                count = min(rows, 546)
                value[:count] = original[:count]
                return value

            cosine = captured("cosine", (128,), fill=1)
            sine = captured("sine", (128,))
            for layer in (0, 34):
                context = dict(text_tokens=texts, encoder_rows=rows, layer=layer)
                prefix = f"language_model.layers.{layer}."

                def execute(name, expected, operands, *, weight_bytes=0, exact=False):
                    inputs = []
                    for index, value in enumerate(operands):
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
                        f"--root=qualify.encoder_{name}",
                        f"--weight_policy={arguments.model / 'weights.loom'}",
                        *([f"--weights={checkpoint}"] if weight_bytes else []),
                        f"--config=krea2.text_tokens={texts}",
                        f"--config=qualify.encoder_layer={layer}",
                        *[f"--input={path}" for path in inputs],
                        f"--expected={expected_path}",
                        f"--actual={actual_path}",
                        *(["--atol=0", "--rtol=0"] if exact else []),
                    ]
                    (working / "check.json").write_text(
                        json.dumps(
                            dict(**context, root=name, command=invocation), indent=2
                        )
                        + "\n"
                    )
                    print(
                        json.dumps(dict(**context, root=name, exact=exact)), flush=True
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
                            f"{name}: fused head transform differs bitwise"
                        )
                    footprint = next(
                        record for record in records if "workspace_bytes" in record
                    )
                    expected_footprint = dict(
                        workspace_bytes=0,
                        kernels=1,
                        parameters=int(bool(weight_bytes)),
                        parameter_roots=int(bool(weight_bytes)),
                        parameter_bytes=weight_bytes,
                    )
                    if any(
                        footprint[key] != value
                        for key, value in expected_footprint.items()
                    ):
                        raise AssertionError(
                            f"{name}: unexpected primitive residency: {footprint}"
                        )
                    return load_bf16(actual_path, expected.shape)

                for name, source, weight_key in (
                    ("norm1", "input", "input_layernorm.weight"),
                    ("norm2", "residual", "post_attention_layernorm.weight"),
                ):
                    value = captured(f"layer{layer}.{source}", (2560,))
                    weight = parameters.get_tensor(prefix + weight_key)
                    native = execute(
                        name, normalize(value, weight), [value], weight_bytes=5120
                    )
                    report(
                        context,
                        name + "_vs_external",
                        native,
                        captured(f"layer{layer}.{name}", (2560,)),
                    )

                for name, heads, weight_key in (
                    ("query", 32, "self_attn.q_norm.weight"),
                    ("key", 8, "self_attn.k_norm.weight"),
                ):
                    value = captured(f"layer{layer}.{name}", (heads, 128))
                    weight = parameters.get_tensor(prefix + weight_key)
                    normalized = execute(
                        name + "_norm",
                        normalize(value, weight),
                        [value],
                        weight_bytes=256,
                    )
                    rotated = execute(
                        name + "_rotate",
                        rotate(normalized, cosine, sine),
                        [normalized, cosine, sine],
                    )
                    fused = execute(
                        name + "_rotary",
                        rotated,
                        [value, cosine, sine],
                        weight_bytes=256,
                        exact=True,
                    )
                    report(
                        context,
                        name + "_fused_vs_f64",
                        fused,
                        rotate(normalize(value, weight), cosine, sine),
                    )
                    report(
                        context,
                        name + "_norm_vs_external",
                        normalized,
                        captured(f"layer{layer}.{name}_norm", (heads, 128)),
                    )
                    (arguments.output / f"layer{layer}-{rows}-{name}.bf16").write_bytes(
                        encode(fused)
                    )

    print(
        "PASS: encoder RMS and half-split rotary primitives and exact fusion.",
        flush=True,
    )


if __name__ == "__main__":
    main()
