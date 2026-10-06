# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify the native F32 VAE input path on actual denoised latent values."""

import argparse
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageCausalConv3d
from safetensors import safe_open


def encode(value):
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite VAE tensor")
    if value.dtype == torch.bfloat16:
        return value.contiguous().view(torch.uint16).numpy().astype("<u2").tobytes()
    return value.float().contiguous().numpy().astype("<f4").tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--denoise_results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.set_grad_enabled(False)
    checkpoint = arguments.checkpoint / "vae" / "diffusion_pytorch_model.safetensors"
    config = json.loads((arguments.checkpoint / "vae" / "config.json").read_text())
    mean = torch.tensor(config["latents_mean"], dtype=torch.float32)
    inverse_std = 1 / torch.tensor(config["latents_std"], dtype=torch.float32)
    affine = torch.stack((inverse_std, mean))

    def run(name, root, inputs, expected, configuration, *, weights=False, exact=False):
        directory = arguments.output / name
        directory.mkdir()
        input_paths = []
        for ordinal, value in enumerate(inputs):
            path = directory / f"input-{ordinal}"
            path.write_bytes(encode(value))
            input_paths.append(path)
        (directory / "expected.f32").write_bytes(encode(expected))
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
        print(json.dumps(dict(check=name, root=root)), flush=True)
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
            raise AssertionError(f"{name}: exact composition differs in representation")
        actual = torch.from_numpy(
            np.fromfile(directory / "actual.f32", dtype="<f4")
        ).reshape(expected.shape)
        reflection = next(record for record in records if "workspace_bytes" in record)
        return actual, reflection

    # All fresh-frame storage variants use the same config-free convolution.
    # Odd spatial/channel tails and a one-row image stress padding and masking.
    generator = torch.Generator().manual_seed(193)
    for height, width in ((1, 2), (17, 19)):
        for extent, planes, upsample in itertools.product((1, 3), (1, 3), (1, 2)):
            value = torch.randn(1, 3, height, width, generator=generator)
            weight = torch.randn(5, 3, planes, extent, extent, generator=generator)
            bias = torch.randn(5, generator=generator)
            resized = F.interpolate(
                value.double(), scale_factor=upsample, mode="nearest-exact"
            )
            expected = F.conv2d(
                resized, weight[:, :, -1].double(), bias.double(), padding=extent // 2
            ).float()
            run(
                f"family-{height}x{width}-k{extent}-t{planes}-s{upsample}",
                "qualify.image_convolution",
                (value, weight, bias),
                expected,
                {
                    "qualify.height": height,
                    "qualify.width": width,
                    "qualify.inputs": 3,
                    "qualify.outputs": 5,
                    "qualify.extent": extent,
                    "qualify.planes": planes,
                    "qualify.upsample": upsample,
                },
            )

    post_quant = QwenImageCausalConv3d(16, 16, 1).eval()
    conv_in = QwenImageCausalConv3d(16, 384, 3, padding=1).eval()
    with safe_open(str(checkpoint), framework="pt") as parameters:
        for prefix, module in (
            ("post_quant_conv", post_quant),
            ("decoder.conv_in", conv_in),
        ):
            module.load_state_dict(
                {
                    name: parameters.get_tensor(f"{prefix}.{name}")
                    for name in ("weight", "bias")
                }
            )

    for phase in ("base", "style"):
        stored = torch.from_numpy(
            np.fromfile(
                arguments.denoise_results / phase / "trajectory.bf16", dtype="<u2"
            )
        ).view(torch.bfloat16)
        if stored.numel() != 576 * 64:
            raise ValueError("qualification expects a completed 384x384 trajectory")
        latent = (
            stored.reshape(1, 24, 24, 16, 2, 2)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(1, 16, 48, 48)
        )
        for height, width in ((2, 6), (48, 48)):
            cropped = latent[:, :, :height, :width].contiguous()
            packed = (
                cropped.reshape(1, 16, height // 2, 2, width // 2, 2)
                .permute(0, 2, 4, 1, 3, 5)
                .contiguous()
            )
            configuration = {"krea2.latent_height": height, "krea2.latent_width": width}
            name = f"{phase}-{height}x{width}"
            expected = cropped.float() / inverse_std.view(1, 16, 1, 1) + mean.view(
                1, 16, 1, 1
            )
            unpacked, _ = run(
                name + "-unpack",
                "krea2.vae_unpack",
                (packed, affine),
                expected,
                configuration,
                exact=True,
            )
            projected_oracle = F.conv2d(
                unpacked.double(),
                post_quant.weight[:, :, -1].double(),
                post_quant.bias.double(),
            ).float()
            projected, _ = run(
                name + "-post-quant",
                "krea2.vae_post_quant",
                (unpacked,),
                projected_oracle,
                configuration,
                weights=True,
            )
            features_oracle = F.conv2d(
                projected.double(),
                conv_in.weight[:, :, -1].double(),
                conv_in.bias.double(),
                padding=1,
            ).float()
            features, _ = run(
                name + "-conv-in",
                "krea2.vae_conv_in",
                (projected,),
                features_oracle,
                configuration,
                weights=True,
            )
            composed, reflection = run(
                name + "-composed",
                "vae_input",
                (packed, affine),
                features,
                configuration,
                weights=True,
                exact=True,
            )
            if (
                reflection["kernels"] != 3
                or reflection["parameters"] != 4
                or reflection["parameter_roots"] != 1
            ):
                raise AssertionError(f"unexpected VAE input reflection: {reflection}")
            if reflection["workspace_bytes"] != 2 * height * width * 16 * 4:
                raise AssertionError(f"unexpected VAE input workspace: {reflection}")
            external = conv_in(post_quant(expected.unsqueeze(2))).squeeze(2)
            error = composed.double() - external.double()
            print(
                json.dumps(
                    dict(
                        phase=phase,
                        shape=[height, width],
                        check="native_prefix_vs_cpu_vae",
                        relative_l2=float(error.norm() / external.double().norm()),
                        maximum_absolute_error=float(error.abs().max()),
                        **reflection,
                    )
                ),
                flush=True,
            )
    print(
        "PASS: actual latent-to-F32-VAE input and complete convolution family.",
        flush=True,
    )


if __name__ == "__main__":
    main()
