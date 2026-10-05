# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify native Krea trajectories from captured text and fixed packed noise.

Python's per-step native calls establish the composition oracle only. The
production denoise command owns all steps and time conditioning on-device.
Final preview images use the independent CPU VAE, not a native Loom decoder.
"""

import argparse
import json
import math
import pathlib
import subprocess

import numpy as np
import torch
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    Krea2Transformer2DModel,
)
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

parser = argparse.ArgumentParser()
parser.add_argument("--checker", nargs="+", required=True)
parser.add_argument("--model", type=pathlib.Path, required=True)
parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
parser.add_argument("--adapter", type=pathlib.Path, required=True)
parser.add_argument("--reference", type=pathlib.Path, required=True)
parser.add_argument("--stack_reference", type=pathlib.Path, required=True)
parser.add_argument("--time_results", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--phase", choices=("base", "style", "both"), default="both")
args = parser.parse_args()
if not 1 <= args.steps <= 8:
    parser.error("--steps selects a prefix of the eight-step Turbo trajectory")
args.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.set_grad_enabled(False)


def load(path, dtype="<u2"):
    value = torch.from_numpy(np.fromfile(path, dtype=dtype))
    return value.view(torch.bfloat16) if dtype == "<u2" else value


def encode(value):
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite denoising tensor")
    return value.contiguous().view(torch.uint16).numpy().astype("<u2").tobytes()


def report(phase, step, name, actual, expected):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("nonfinite trajectory comparison")
    error = actual.double() - expected.double()
    print(
        json.dumps(
            dict(
                phase=phase,
                step=step,
                check=name,
                relative_l2=float(
                    error.norm() / expected.double().norm().clamp_min(1e-30)
                ),
                maximum_absolute_error=float(error.abs().max()),
            )
        ),
        flush=True,
    )


scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    args.checkpoint / "scheduler", local_files_only=True
)
scheduler.set_timesteps(8, device="cpu", sigmas=np.linspace(1.0, 1 / 8, 8), mu=1.15)
times = (scheduler.timesteps / scheduler.config.num_train_timesteps).bfloat16()
deltas = scheduler.sigmas[1:] - scheduler.sigmas[:-1]
initial = load(args.reference / "initial_latents.bf16").reshape(-1, 64)
image_rows = initial.shape[0]
grid = math.isqrt(image_rows)
if grid * grid != image_rows or image_rows % 16:
    raise ValueError("qualification requires complete tiles from a square image")
transformer = (
    Krea2Transformer2DModel.from_single_file(
        str(args.checkpoint / "turbo.safetensors"),
        config=str(args.checkpoint / "transformer"),
        torch_dtype=torch.bfloat16,
        device="cuda",
        local_files_only=True,
    )
    .eval()
    .requires_grad_(False)
)
base_checkpoint = args.checkpoint / "turbo.safetensors"
adapter_checkpoint = args.adapter / "softwatercolor.safetensors"
with safe_open(str(base_checkpoint), framework="pt") as weights:
    image_weight = weights.get_tensor("first.weight").bfloat16().double()
    image_bias = weights.get_tensor("first.bias").bfloat16().double()
with safe_open(str(adapter_checkpoint), framework="pt") as weights:
    image_down = (
        weights.get_tensor("transformer.img_in.lora_A.weight").bfloat16().double()
    )
    image_up = (
        weights.get_tensor("transformer.img_in.lora_B.weight").bfloat16().double()
    )

trajectories = {}
for phase in ("base", "style"):
    if args.phase not in (phase, "both"):
        continue
    adapted = phase == "style"
    if adapted:
        transformer.load_lora_adapter(
            str(args.adapter),
            weight_name="softwatercolor.safetensors",
            local_files_only=True,
            adapter_name="watercolor",
        )
        transformer.set_adapters("watercolor", weights=1.0)
    directory = args.output / phase
    directory.mkdir()
    combined_initial = load(args.stack_reference / f"{phase}-stack-input.bf16").reshape(
        -1, 6144
    )
    tokens = combined_initial.shape[0]
    text_rows = tokens - image_rows
    if text_rows < 0 or tokens % 16:
        raise ValueError("captured sequence does not contain the image suffix")
    text = combined_initial[:text_rows].contiguous()
    prefix = f"{phase}-block27-"
    paths = {
        "sample": directory / "sample.bf16",
        "text": directory / "text.bf16",
        "times": directory / "times.bf16",
        "deltas": directory / "deltas.f32",
        "strength": directory / "strength.f32",
        "cosine": args.stack_reference / (prefix + "cosine.f32"),
        "sine": args.stack_reference / (prefix + "sine.f32"),
        "mask": args.stack_reference / (prefix + "mask.u8"),
        "conditioning": args.time_results / f"{phase}-8" / "conditioning.bf16",
    }
    paths["text"].write_bytes(encode(text))
    paths["times"].write_bytes(encode(times[: args.steps]))
    paths["deltas"].write_bytes(deltas.numpy().astype("<f4").tobytes())
    paths["strength"].write_bytes(np.array([float(adapted)], dtype="<f4").tobytes())
    if load(args.time_results / f"{phase}-8" / "times.bf16").tolist() != times.tolist():
        raise ValueError("qualified conditioning does not use this Turbo schedule")
    conditioning = load(paths["conditioning"])
    if conditioning.numel() != 16 * (6144 + 36864):
        raise ValueError("qualified conditioning has the wrong physical layout")
    embeddings = conditioning[: 16 * 6144].reshape(16, 6144)
    modulations = conditioning[16 * 6144 :].reshape(16, 36864)
    rotary = tuple(
        load(paths[name], "<f4").reshape(tokens, 128).cuda()
        for name in ("cosine", "sine")
    )
    mask = load(paths["mask"], "u1").bool().reshape(1, 1, 1, tokens).cuda()
    checkpoints = (
        [base_checkpoint, adapter_checkpoint] if adapted else [base_checkpoint]
    )

    def execute(
        name, root, expected, inputs, *, comparison="exact", step=0, weights=None
    ):
        expected_path = directory / "expected.bf16"
        actual_path = directory / (name + ".bf16")
        expected_path.write_bytes(encode(expected))
        command = [
            *args.checker,
            "--model="
            + str(
                args.model / "qualification"
                if root.startswith("qualify.")
                else args.model
            ),
            "--weight_policy=" + str(args.model / "weights.loom"),
            *[
                "--weights=" + str(path)
                for path in (checkpoints if weights is None else weights)
            ],
            "--root=" + root,
            f"--config=krea2.block_tokens={tokens}",
            f"--config=krea2.image_tokens={image_rows}",
            f"--config=krea2.time_count={args.steps}",
            *["--input=" + str(paths[key]) for key in inputs],
            "--expected=" + str(expected_path),
            "--actual=" + str(actual_path),
        ]
        if root.startswith("qualify."):
            command.append(f"--config=krea2.step_index={step}")
        if comparison == "exact":
            command += ["--atol=0", "--rtol=0"]
        elif comparison == "report":
            command.append("--report_only")
        print(json.dumps(dict(phase=phase, step=step, component=name)), flush=True)
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
        print(result.stdout, end="", flush=True)
        result.check_returncode()
        records = [
            json.loads(line)
            for line in result.stdout.splitlines()
            if line.startswith("{")
        ]
        comparisons = [record for record in records if "iteration" in record]
        if [record["iteration"] for record in comparisons] != [0, 1]:
            raise AssertionError("checker did not complete both executions")
        if comparison == "exact" and (
            any(record["different"] for record in comparisons)
            or actual_path.read_bytes() != encode(expected)
        ):
            raise AssertionError("native trajectory composition is not bitwise")
        paths[name] = actual_path
        footprint = next(record for record in records if "workspace_bytes" in record)
        return load(actual_path).reshape_as(expected), footprint

    sample = initial
    for step in range(args.steps):
        paths["sample"].write_bytes(encode(sample))
        image_expected = (sample.double() @ image_weight.T + image_bias).bfloat16()
        if adapted:
            base, _ = execute(
                "image_base",
                "image_projection",
                image_expected,
                ["sample"],
                comparison="elementwise",
                step=step,
                weights=[base_checkpoint],
            )
            low, _ = execute(
                "image_low",
                "krea2.image_adapter_down",
                (sample.double() @ image_down.T).bfloat16(),
                ["sample"],
                comparison="elementwise",
                step=step,
                weights=[adapter_checkpoint],
            )
            delta, _ = execute(
                "image_delta",
                "qualify.image_adapter_up",
                (low.double() @ image_up.T).bfloat16(),
                ["image_low"],
                comparison="elementwise",
                step=step,
                weights=[adapter_checkpoint],
            )
            projected, _ = execute(
                "image",
                "image_projection_adapted",
                (base.float() + delta.float()).bfloat16(),
                ["sample", "strength"],
                step=step,
            )
        else:
            projected, _ = execute(
                "image",
                "image_projection",
                image_expected,
                ["sample"],
                comparison="elementwise",
                step=step,
            )
        if (
            step == 0
            and encode(projected)
            != (args.reference / f"{phase}-image_projection.bf16").read_bytes()
        ):
            raise AssertionError(
                "first image projection changed from the qualified capture"
            )
        combined = torch.cat((text, projected))
        for name, value in (
            ("combined", combined),
            ("modulation", modulations[step]),
            ("embedding", embeddings[step]),
        ):
            paths[name] = directory / (name + ".bf16")
            paths[name].write_bytes(encode(value))
        external = combined[None].cuda()
        external_modulation = modulations[step].reshape(1, 1, 36864).cuda()
        for block in transformer.transformer_blocks:
            external = block(external, external_modulation, rotary, mask)
        external = (
            transformer.final_layer(
                external[:, text_rows:], embeddings[step].reshape(1, 1, 6144).cuda()
            )
            .cpu()
            .reshape(image_rows, 64)
        )
        velocity_inputs = [
            "combined",
            "modulation",
            "cosine",
            "sine",
            "mask",
            "embedding",
        ]
        if adapted:
            velocity_inputs.append("strength")
        velocity, _ = execute(
            "velocity",
            "transformer_velocity_adapted" if adapted else "transformer_velocity",
            external,
            velocity_inputs,
            comparison="report",
            step=step,
        )
        report(
            phase, step, "native_velocity_vs_external_same_inputs", velocity, external
        )
        # The scalar F32 delta is not cast to BF16; only its product is rounded.
        update = (velocity.float() * deltas[step]).bfloat16().float()
        expected = (sample.float() + update).bfloat16()
        sample, _ = execute(
            "update",
            "qualify.denoise_update",
            expected,
            ["sample", "velocity", "deltas"],
            step=step,
            weights=[],
        )
        if step in (0, args.steps - 1):
            inputs = [
                "sample",
                "text",
                "conditioning",
                "cosine",
                "sine",
                "mask",
                "deltas",
            ]
            if adapted:
                inputs.append("strength")
            execute(
                "step",
                "qualify.denoise_step_adapted" if adapted else "qualify.denoise_step",
                sample,
                inputs,
                step=step,
            )
        captured = load_file(str(args.reference / f"{phase}-step-{step}.safetensors"))[
            "latents"
        ].reshape_as(sample)
        report(phase, step, "native_trajectory_vs_captured_external", sample, captured)
        (directory / f"step-{step}.bf16").write_bytes(encode(sample))

    paths["sample"].write_bytes(encode(initial))
    inputs = ["sample", "text", "times", "cosine", "sine", "mask", "deltas"]
    if adapted:
        inputs.append("strength")
    whole, footprint = execute(
        "trajectory", "denoise_adapted" if adapted else "denoise", sample, inputs
    )
    expected_counts = dict(
        kernels=35 if adapted else 21,
        parameters=834 if adapted else 376,
        parameter_roots=2 if adapted else 1,
        parameter_bytes=25821294848 if adapted else 25382416640,
    )
    if any(footprint[key] != value for key, value in expected_counts.items()):
        raise AssertionError(
            f"denoising does not share kernels and parameters: {footprint}"
        )
    # The best-fit transient planner does not extend a retired tail hole.
    # Include one conditioning-scratch-sized fragment, independent of steps.
    time_scratch = 16 * (512 + 2 * 12288 + (64 if adapted else 0))
    maximum_workspace = (
        tokens * ((144640 if adapted else 144384) + 12288)
        + image_rows * 128
        + 16 * 86016
        + time_scratch
    )
    if footprint["workspace_bytes"] > maximum_workspace:
        raise AssertionError(
            f"trajectory workspace grows beyond live storage: {footprint}"
        )
    if not adapted:
        zero, _ = execute(
            "zero",
            "denoise_adapted",
            whole,
            [*inputs, "strength"],
            weights=[base_checkpoint, adapter_checkpoint],
        )
        if encode(zero) != encode(whole):
            raise AssertionError("zero adapter changes the complete trajectory")
    trajectories[phase] = whole
    print(
        json.dumps(
            dict(
                phase=phase, check="trajectory_accepted", steps=args.steps, **footprint
            )
        ),
        flush=True,
    )

del transformer
if args.steps == 8:
    vae = (
        AutoencoderKLQwenImage.from_pretrained(
            args.checkpoint / "vae", torch_dtype=torch.float32, local_files_only=True
        )
        .eval()
        .requires_grad_(False)
    )
    processor = VaeImageProcessor(vae_scale_factor=8)
    mean = torch.tensor(vae.config.latents_mean).reshape(1, 16, 1, 1, 1)
    inverse_std = 1.0 / torch.tensor(vae.config.latents_std).reshape(1, 16, 1, 1, 1)
    for phase, packed in trajectories.items():
        latent = (
            packed.reshape(1, grid, grid, 16, 2, 2)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(1, 16, 1, grid * 2, grid * 2)
            .float()
        )
        latent = latent / inverse_std + mean
        decoded = vae.decode(latent, return_dict=False)[0][:, :, 0]
        image = processor.postprocess(decoded, output_type="np")[0]
        if not np.isfinite(image).all():
            raise ValueError("nonfinite decoded trajectory")
        pixels = np.rint(image * 255).astype(np.uint8)
        path = args.output / f"{phase}-native-denoise-external-vae.png"
        Image.fromarray(pixels).save(path)
        reference_pixels = np.array(Image.open(args.reference / (phase + ".png")))
        error = pixels.astype(np.float64) - reference_pixels.astype(np.float64)
        mse = float(np.square(error).mean())
        print(
            json.dumps(
                dict(
                    phase=phase,
                    check="preview_external_vae",
                    path=str(path),
                    pixel_rmse=math.sqrt(mse),
                    psnr_db=10 * math.log10(255**2 / mse) if mse else None,
                )
            ),
            flush=True,
        )
    if len(trajectories) == 2 and encode(trajectories["base"]) == encode(
        trajectories["style"]
    ):
        raise AssertionError("active LoRA does not change the denoised latent")

print(
    "PASS: native denoising matches staged composition; text conditioning and VAE remain external.",
    flush=True,
)
