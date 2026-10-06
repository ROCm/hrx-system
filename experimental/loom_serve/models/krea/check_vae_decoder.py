# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify native still-image VAE decoding through final RGB.

Only one active primitive fixture is retained: its metadata, inputs, expected
output and actual output are overwritten after each completed check. Final
native/reference pixels and PNGs remain. This bounds temporary disk usage while
preserving the complete failing primitive if a check exits unsuccessfully.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLQwenImage
from PIL import Image


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


def convolve(value, layer, *, matrix=False):
    weight = layer.weight
    if weight.ndim == 5:
        weight = weight[:, :, -1]
    if matrix:
        value = value.to(torch.bfloat16).float()
        weight = weight.to(torch.bfloat16).float()
    return F.conv2d(
        value.double(),
        weight.double(),
        layer.bias.double(),
        padding=weight.shape[-1] // 2,
    ).float()


def residual(value, block):
    skip = (
        convolve(value, block.conv_shortcut) if block.in_dim != block.out_dim else value
    )
    first = convolve(
        activate(normalize(value, block.norm1.gamma)),
        block.conv1,
        matrix=True,
    )
    return (
        convolve(
            activate(normalize(first, block.norm2.gamma)), block.conv2, matrix=True
        )
        + skip
    )


def image_pixels(value):
    return (
        (value[0].permute(1, 2, 0) / 2 + 0.5)
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .numpy()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input_results", type=Path, required=True)
    parser.add_argument("--attention_results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    working = arguments.output / "working"
    working.mkdir()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.set_grad_enabled(False)
    checkpoint = arguments.checkpoint / "vae" / "diffusion_pytorch_model.safetensors"
    vae = AutoencoderKLQwenImage.from_pretrained(
        arguments.checkpoint / "vae", torch_dtype=torch.float32
    ).eval()

    def run(
        name, root, inputs, expected, configuration, *, weights=True, mode="element"
    ):
        input_paths = []
        for ordinal, value in enumerate(inputs):
            if isinstance(value, Path):
                input_paths.append(value)
            else:
                path = working / f"input-{ordinal}.f32"
                path.write_bytes(encode(value))
                input_paths.append(path)
        (working / "expected.f32").write_bytes(encode(expected))
        command = [
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
            f"--expected={working}/expected.f32",
            f"--actual={working}/actual.f32",
            "--atol=0" if mode == "exact" else "--atol=0.00002",
            "--rtol=0" if mode == "exact" else "--rtol=0.00002",
            *(["--relative_l2_tolerance=0.00002"] if mode == "chain" else []),
        ]
        (working / "check.json").write_text(
            json.dumps(dict(check=name, mode=mode, command=command), indent=2) + "\n"
        )
        print(json.dumps(dict(check=name, root=root, mode=mode)), flush=True)
        result = subprocess.run(command, capture_output=True, text=True)
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
        if mode == "exact" and any(record["different"] for record in comparisons):
            raise AssertionError(f"{name}: exact native composition differs")
        actual = torch.from_numpy(
            np.fromfile(working / "actual.f32", dtype="<f4")
        ).reshape(expected.shape)
        return actual, next(record for record in records if "workspace_bytes" in record)

    for phase in ("base", "style"):
        for height, width in ((2, 6), (48, 48)):
            prefix = f"{phase}-{height}x{width}"
            fixture = arguments.input_results / (prefix + "-composed")
            attention = torch.from_numpy(
                np.fromfile(
                    arguments.attention_results
                    / (prefix + "-attention")
                    / "actual.f32",
                    dtype="<f4",
                )
            ).reshape(1, 384, height, width)
            configuration = {
                "krea2.latent_height": height,
                "krea2.latent_width": width,
            }
            state, _ = run(
                prefix + "-mid1",
                "qualify.vae_mid_residual",
                (attention,),
                residual(attention, vae.decoder.mid_block.resnets[1]),
                {**configuration, "qualify.vae_block": 1},
                mode="chain",
            )
            for stage, up_block in enumerate(vae.decoder.up_blocks):
                for block_index, block in enumerate(up_block.resnets):
                    state, _ = run(
                        prefix + f"-up{stage}-residual{block_index}",
                        "qualify.vae_up_residual",
                        (state,),
                        residual(state, block),
                        {
                            **configuration,
                            "qualify.vae_stage": stage,
                            "qualify.vae_up_block": block_index,
                        },
                        mode="chain",
                    )
                if up_block.upsamplers is not None:
                    enlarged = F.interpolate(
                        state, scale_factor=2, mode="nearest-exact"
                    )
                    expected = convolve(
                        enlarged, up_block.upsamplers[0].resample[1], matrix=True
                    )
                    state, _ = run(
                        prefix + f"-up{stage}-resize",
                        "qualify.vae_up_resize",
                        (state,),
                        expected,
                        {**configuration, "qualify.vae_resize_stage": stage},
                    )
            normalized, _ = run(
                prefix + "-output-norm",
                "krea2.vae_output_norm",
                (state,),
                activate(normalize(state, vae.decoder.norm_out.gamma)),
                configuration,
            )
            decoded, _ = run(
                prefix + "-output-convolution",
                "krea2.vae_output_convolution",
                (normalized,),
                convolve(normalized, vae.decoder.conv_out),
                configuration,
            )
            expected = decoded.clamp(-1, 1)
            run(
                prefix + "-clamp",
                "krea2.vae_clamp",
                (decoded,),
                expected,
                configuration,
                weights=False,
                mode="exact",
            )
            run(
                prefix + "-body",
                "vae_decoder_body",
                (attention,),
                expected,
                configuration,
                mode="exact",
            )
            complete, reflection = run(
                prefix + "-complete",
                "vae_decode",
                (fixture / "input-0", fixture / "input-1"),
                expected,
                configuration,
                mode="exact",
            )
            largest_feature_bytes = 96 * height * width * 64 * 4
            if (
                reflection["parameters"] != 104
                or reflection["kernels"] != 30
                or reflection["workspace_bytes"]
                > 4 * largest_feature_bytes + 4 * height * width * 384 * 4
            ):
                raise AssertionError(f"unexpected complete decoder: {reflection}")
            (arguments.output / (prefix + "-native.f32")).write_bytes(encode(complete))

            packed = torch.from_numpy(
                np.fromfile(fixture / "input-0", dtype="<u2")
            ).view(torch.bfloat16)
            latent = (
                packed.reshape(1, height // 2, width // 2, 16, 2, 2)
                .permute(0, 3, 1, 4, 2, 5)
                .reshape(1, 16, height, width)
                .float()
            )
            affine = torch.from_numpy(
                np.fromfile(fixture / "input-1", dtype="<f4")
            ).reshape(2, 16)
            latent = latent / affine[0].view(1, 16, 1, 1) + affine[1].view(1, 16, 1, 1)
            external = vae.decode(latent.unsqueeze(2)).sample.squeeze(2)
            (arguments.output / (prefix + "-external.f32")).write_bytes(
                encode(external)
            )
            relative_l2 = float(
                (complete.double() - external.double()).norm()
                / external.double().norm()
            )
            if not math.isfinite(relative_l2):
                raise AssertionError(f"{prefix}: nonfinite decoder error {relative_l2}")
            native_pixels = image_pixels(complete)
            external_pixels = image_pixels(external)
            Image.fromarray(native_pixels).save(
                arguments.output / (prefix + "-native.png")
            )
            Image.fromarray(external_pixels).save(
                arguments.output / (prefix + "-external.png")
            )
            pixel_difference = native_pixels.astype(np.float64) - external_pixels
            print(
                json.dumps(
                    dict(
                        check=prefix + "-final-image",
                        relative_l2=relative_l2,
                        pixel_rmse=float(np.sqrt(np.mean(pixel_difference**2))),
                        different_channels=int(np.count_nonzero(pixel_difference)),
                        maximum_pixel_difference=float(
                            np.max(np.abs(pixel_difference))
                        ),
                        reflection=reflection,
                    )
                ),
                flush=True,
            )
    print(
        "PASS: full native still-image VAE, exact composition and final RGB.",
        flush=True,
    )


if __name__ == "__main__":
    main()
