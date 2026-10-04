# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify spatial attention and its in-place native VAE prefix."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    QwenImageAttentionBlock,
)
from safetensors import safe_open


def encode(value):
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite VAE tensor")
    return value.float().contiguous().numpy().astype("<f4").tobytes()


def normalize(value, weight):
    value = value.double()
    normalized = value / value.square().sum(1, keepdim=True).sqrt().clamp_min(1e-12)
    return (
        normalized * math.sqrt(value.shape[1]) * weight.double().view(1, -1, 1, 1)
    ).float()


def convolve(value, layer):
    return F.conv2d(value.double(), layer.weight.double(), layer.bias.double()).float()


def pack_qkv(qkv):
    """Encode channel-first Q/K and token-first V in three equal-size planes."""
    channels = qkv.shape[1] // 3
    query, key, value = qkv.reshape(3, channels, -1)
    return torch.cat(
        (query.flatten(), key.flatten(), value.T.contiguous().flatten())
    ).reshape_as(qkv)


def unpack_qkv(packed):
    """Recover logical channel-first values from the production physical layout."""
    channels = packed.shape[1] // 3
    tokens = packed.numel() // (3 * channels)
    planes = packed.reshape(3, channels, tokens)
    logical = planes.clone()
    logical[2] = planes[2].reshape(tokens, channels).T
    return logical.reshape_as(packed)


def attend(qkv):
    channels = qkv.shape[1] // 3
    query, key, value = qkv.reshape(3, channels, -1).double()
    output = torch.empty_like(query)
    for start in range(0, query.shape[1], 128):
        scores = (query[:, start : start + 128].T @ key) / math.sqrt(channels)
        output[:, start : start + 128] = (scores.softmax(-1) @ value.T).T
    return output.float().reshape(1, channels, *qkv.shape[2:])


def report(name, actual, expected):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("nonfinite VAE attention comparison")
    error = actual.double() - expected.double()
    relative_l2 = float(error.norm() / expected.double().norm())
    print(
        json.dumps(
            dict(
                check=name,
                relative_l2=relative_l2,
                maximum_absolute_error=float(error.abs().max()),
            )
        ),
        flush=True,
    )
    return relative_l2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input_results", type=Path, required=True)
    parser.add_argument("--residual_results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.set_grad_enabled(False)
    checkpoint = arguments.checkpoint / "vae" / "diffusion_pytorch_model.safetensors"
    block = QwenImageAttentionBlock(384).eval()
    with safe_open(str(checkpoint), framework="pt") as parameters:
        block.load_state_dict(
            {
                key: parameters.get_tensor("decoder.mid_block.attentions.0." + key)
                for key in block.state_dict()
            }
        )

    def run(name, root, inputs, expected, configuration, *, weights=False, exact=False):
        directory = arguments.output / name
        directory.mkdir()
        input_paths = []
        for ordinal, value in enumerate(inputs):
            if isinstance(value, Path):
                input_paths.append(value)
            else:
                path = directory / f"input-{ordinal}.f32"
                path.write_bytes(encode(value))
                input_paths.append(path)
        (directory / "expected.f32").write_bytes(encode(expected))
        print(json.dumps(dict(check=name, root=root)), flush=True)
        result = subprocess.run(
            [
                *arguments.checker,
                f"--model={arguments.model / 'qualification'}",
                f"--root={root}",
                "--output_type=f32",
                *[f"--config={key}={value}" for key, value in configuration.items()],
                *(
                    [
                        f"--weights={checkpoint}",
                        f"--weight_policy={arguments.model / 'weights.loom'}",
                    ]
                    if weights
                    else []
                ),
                *[f"--input={path}" for path in input_paths],
                f"--expected={directory}/expected.f32",
                f"--actual={directory}/actual.f32",
                "--atol=0" if exact else "--atol=0.00002",
                "--rtol=0" if exact else "--rtol=0.00002",
            ],
            capture_output=True,
            text=True,
        )
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", flush=True)
        if result.returncode:
            raise RuntimeError(f"{name}: component checker exited {result.returncode}")
        records = [
            json.loads(line)
            for line in result.stdout.splitlines()
            if line.startswith("{")
        ]
        comparisons = [record for record in records if "iteration" in record]
        if [record["iteration"] for record in comparisons] != [0, 1]:
            raise AssertionError("both device executions must complete")
        if exact and any(record["different"] for record in comparisons):
            raise AssertionError(f"{name}: exact native composition differs")
        actual = torch.from_numpy(
            np.fromfile(directory / "actual.f32", dtype="<f4")
        ).reshape(expected.shape)
        return actual, next(record for record in records if "workspace_bytes" in record)

    generator = torch.Generator().manual_seed(71)
    for channels in (1, 3, 128, 129, 256, 257, 384, 385, 512):
        for tokens in (1, 127, 128, 129, 257):
            # Increasing key magnitudes force the online maximum to change
            # between tiles; odd channel counts exercise every masked tail.
            qkv = torch.randn(1, 3 * channels, 1, tokens, generator=generator)
            qkv[:, channels : 2 * channels] *= torch.linspace(0.5, 3, tokens)
            _, reflection = run(
                f"family-{channels}x{tokens}",
                "qualify.image_attention",
                (pack_qkv(qkv),),
                attend(qkv),
                {
                    "qualify.attention_channels": channels,
                    "qualify.attention_tokens": tokens,
                },
            )
            if reflection["workspace_bytes"] or reflection["kernels"] != 1:
                raise AssertionError(f"attention materialized scratch: {reflection}")

    for phase in ("base", "style"):
        for height, width in ((2, 6), (48, 48)):
            prefix = f"{phase}-{height}x{width}"
            fixture = arguments.input_results / (prefix + "-composed")
            state = torch.from_numpy(
                np.fromfile(
                    arguments.residual_results / (prefix + "-residual") / "actual.f32",
                    dtype="<f4",
                )
            ).reshape(1, 384, height, width)
            configuration = {
                "krea2.latent_height": height,
                "krea2.latent_width": width,
            }
            normalized, _ = run(
                prefix + "-norm",
                "krea2.vae_attention_norm",
                (state,),
                normalize(state, block.norm.gamma),
                configuration,
                weights=True,
            )
            packed_qkv, _ = run(
                prefix + "-qkv",
                "krea2.vae_attention_qkv",
                (normalized,),
                pack_qkv(convolve(normalized, block.to_qkv)),
                configuration,
                weights=True,
            )
            attended, _ = run(
                prefix + "-core",
                "krea2.vae_attention_core",
                (packed_qkv,),
                attend(unpack_qkv(packed_qkv)),
                configuration,
            )
            projected, _ = run(
                prefix + "-projection",
                "krea2.vae_attention_projection",
                (attended,),
                convolve(attended, block.proj),
                configuration,
                weights=True,
            )
            expected = projected + state
            run(
                prefix + "-add",
                "krea2.vae_attention_projection_add",
                (attended, state),
                expected,
                configuration,
                weights=True,
                exact=True,
            )
            composed, reflection = run(
                prefix + "-attention",
                "vae_spatial_attention",
                (state,),
                expected,
                configuration,
                weights=True,
                exact=True,
            )
            feature_bytes = height * width * 384 * 4
            if (
                reflection["kernels"] != 4
                or reflection["parameters"] != 5
                or reflection["workspace_bytes"] != 4 * feature_bytes
            ):
                raise AssertionError(f"unexpected attention reflection: {reflection}")
            _, complete = run(
                prefix + "-prefix",
                "vae_input_attention",
                (fixture / "input-0", fixture / "input-1"),
                expected,
                configuration,
                weights=True,
                exact=True,
            )
            if (
                complete["kernels"] != 10
                or complete["parameters"] != 15
                or complete["workspace_bytes"]
                > 6 * feature_bytes + 2 * height * width * 16 * 4
            ):
                raise AssertionError(f"unexpected prefix reflection: {complete}")
            oracle_qkv = convolve(normalize(state, block.norm.gamma), block.to_qkv)
            oracle = convolve(attend(oracle_qkv), block.proj) + state
            relative_l2 = report(prefix + "-native-vs-f64", composed, oracle)
            if relative_l2 > 0.00002:
                raise AssertionError(
                    f"{prefix}: accumulated attention error {relative_l2}"
                )
            external = block(state.unsqueeze(2)).squeeze(2)
            report(prefix + "-external-vs-f64", external, oracle)
            report(prefix + "-native-vs-external", composed, external)
    print(
        "PASS: native VAE attention primitives, online tile tails, and in-place prefix.",
        flush=True,
    )


if __name__ == "__main__":
    main()
