# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify a VAE residual and in-place prefix on real native feature tensors."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    QwenImageResidualBlock,
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


def activate(value):
    return F.silu(value.double()).float()


def convolve(value, layer):
    return F.conv2d(
        value.to(torch.bfloat16).double(),
        layer.weight[:, :, -1].to(torch.bfloat16).double(),
        layer.bias.double(),
        padding=1,
    ).float()


def report(name, actual, expected):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("nonfinite VAE residual comparison")
    error = actual.double() - expected.double()
    print(
        json.dumps(
            dict(
                check=name,
                relative_l2=float(error.norm() / expected.double().norm()),
                maximum_absolute_error=float(error.abs().max()),
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input_results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.set_grad_enabled(False)
    checkpoint = arguments.checkpoint / "vae" / "diffusion_pytorch_model.safetensors"
    block = QwenImageResidualBlock(384, 384).eval()
    with safe_open(str(checkpoint), framework="pt") as parameters:
        block.load_state_dict(
            {
                key: parameters.get_tensor("decoder.mid_block.resnets.0." + key)
                for key in block.state_dict()
            }
        )

    def run(name, root, inputs, expected, height, width, *, ordinal=1, exact=False):
        directory = arguments.output / name
        directory.mkdir()
        input_paths = []
        for i, value in enumerate(inputs):
            if isinstance(value, Path):
                input_paths.append(value)
            else:
                path = directory / f"input-{i}.f32"
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
                f"--weights={checkpoint}",
                f"--weight_policy={arguments.model / 'weights.loom'}",
                f"--config=krea2.latent_height={height}",
                f"--config=krea2.latent_width={width}",
                "--config=qualify.vae_block=0",
                f"--config=qualify.vae_ordinal={ordinal}",
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

    for phase in ("base", "style"):
        for height, width in ((2, 6), (48, 48)):
            prefix = f"{phase}-{height}x{width}"
            fixture = arguments.input_results / (prefix + "-composed")
            state = torch.from_numpy(
                np.fromfile(fixture / "actual.f32", dtype="<f4")
            ).reshape(1, 384, height, width)
            normalized, _ = run(
                prefix + "-norm1",
                "qualify.vae_mid_norm",
                (state,),
                activate(normalize(state, block.norm1.gamma)),
                height,
                width,
            )
            first, _ = run(
                prefix + "-conv1",
                "qualify.vae_mid_convolution",
                (normalized,),
                convolve(normalized, block.conv1),
                height,
                width,
            )
            activated, _ = run(
                prefix + "-norm2",
                "qualify.vae_mid_norm",
                (first,),
                activate(normalize(first, block.norm2.gamma)),
                height,
                width,
                ordinal=2,
            )
            second, _ = run(
                prefix + "-conv2",
                "qualify.vae_mid_convolution",
                (activated,),
                convolve(activated, block.conv2),
                height,
                width,
                ordinal=2,
            )
            expected = second + state
            run(
                prefix + "-add",
                "qualify.vae_mid_convolution_add",
                (activated, state),
                expected,
                height,
                width,
                exact=True,
            )
            composed, reflection = run(
                prefix + "-residual",
                "qualify.vae_mid_residual",
                (state,),
                expected,
                height,
                width,
                exact=True,
            )
            feature_bytes = height * width * 384 * 4
            if (
                reflection["kernels"] != 3
                or reflection["parameters"] != 6
                or reflection["workspace_bytes"] != 2 * feature_bytes
            ):
                raise AssertionError(f"unexpected residual reflection: {reflection}")
            _, complete = run(
                prefix + "-prefix",
                "vae_input_residual",
                (fixture / "input-0", fixture / "input-1"),
                expected,
                height,
                width,
                exact=True,
            )
            if (
                complete["kernels"] != 6
                or complete["parameters"] != 10
                or complete["workspace_bytes"]
                > 2 * feature_bytes + 2 * height * width * 16 * 4
            ):
                raise AssertionError(f"unexpected prefix reflection: {complete}")
            oracle_first = convolve(
                activate(normalize(state, block.norm1.gamma)), block.conv1
            )
            oracle = (
                convolve(
                    activate(normalize(oracle_first, block.norm2.gamma)), block.conv2
                )
                + state
            )
            report(prefix + "-native-vs-f64", composed, oracle)
            relative_l2 = (
                composed.double() - oracle.double()
            ).norm() / oracle.double().norm()
            if relative_l2 > 0.00002:
                raise AssertionError(
                    f"{prefix}: accumulated residual error {relative_l2}"
                )
            external = block(state.unsqueeze(2)).squeeze(2)
            report(prefix + "-external-vs-f64", external, oracle)
            report(prefix + "-native-vs-external", composed, external)
    print(
        "PASS: native VAE residual primitives, fusion, and in-place prefix composition.",
        flush=True,
    )


if __name__ == "__main__":
    main()
