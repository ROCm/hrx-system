# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Independent fresh-prompt image reference, consuming only native initial noise.

The encoder, denoiser and request math are canonical Transformers/Diffusers;
VAE decoding uses the previously qualified CPU/F32 reference path. Retains
only final RGB and PNGs, below 16 MiB. This is not a deployment dependency.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    Krea2Pipeline,
    Krea2Transformer2DModel,
)
from PIL import Image
from transformers import AutoTokenizer, Qwen3VLModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.set_grad_enabled(False)
    root = args.checkpoint
    print("Loading canonical reference components.", flush=True)
    transformer = (
        Krea2Transformer2DModel.from_single_file(
            str(root / "turbo.safetensors"),
            config=str(root / "transformer"),
            torch_dtype=torch.bfloat16,
            device="cuda",
            local_files_only=True,
        )
        .eval()
        .requires_grad_(False)
    )
    encoder = (
        Qwen3VLModel.from_pretrained(
            root / "text_encoder",
            dtype=torch.bfloat16,
            device_map={"": "cuda"},
            local_files_only=True,
        )
        .eval()
        .requires_grad_(False)
    )
    vae = (
        AutoencoderKLQwenImage.from_pretrained(
            root / "vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        .to("cuda")
        .eval()
        .requires_grad_(False)
    )
    pipeline = Krea2Pipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
            root / "scheduler", local_files_only=True
        ),
        vae=vae,
        text_encoder=encoder,
        tokenizer=AutoTokenizer.from_pretrained(
            root / "tokenizer", local_files_only=True
        ),
        transformer=transformer,
        is_distilled=True,
    )
    pipeline.set_progress_bar_config(disable=True)
    original = [
        (parameter, parameter.data_ptr(), parameter._version)
        for parameter in transformer.parameters()
    ]
    requests = {}
    for name in ("canonical-512", "novel-32"):
        directory = args.requests / name
        request = json.loads((directory / "request.json").read_text())
        print(
            json.dumps(dict(event="encode", case=name, prompt=request["prompt"])),
            flush=True,
        )
        features, mask = pipeline.get_text_hidden_states(
            [request["prompt"]], max_sequence_length=request["text_tokens"]
        )
        noise = (
            torch.frombuffer(
                bytearray((directory / "input-0").read_bytes()), dtype=torch.bfloat16
            )
            .reshape(1, -1, 64)
            .to("cuda")
        )
        assert torch.isfinite(features).all() and torch.isfinite(noise).all()
        requests[name] = request, features, mask, noise
    denoised = {}
    for phase in ("base", "style"):
        if phase == "style":
            transformer.load_lora_adapter(
                str(args.adapter),
                weight_name="softwatercolor.safetensors",
                local_files_only=True,
                adapter_name="watercolor",
            )
            transformer.set_adapters("watercolor", weights=1.0)
        for name, (request, features, mask, noise) in requests.items():
            print(json.dumps(dict(event="denoise", case=name, phase=phase)), flush=True)
            packed = pipeline(
                prompt_embeds=features,
                prompt_embeds_mask=mask,
                latents=noise.clone(),
                height=request["height"],
                width=request["width"],
                num_inference_steps=8,
                guidance_scale=0.0,
                output_type="latent",
            ).images
            assert torch.isfinite(packed).all()
            denoised[(name, phase)] = packed.cpu().contiguous()
            assert all(
                parameter.data_ptr() == pointer and parameter._version == version
                for parameter, pointer, version in original
            )
    vae.to("cpu")
    mean = torch.tensor(vae.config.latents_mean).reshape(1, 16, 1, 1, 1)
    inverse = (1 / torch.tensor(vae.config.latents_std)).reshape(1, 16, 1, 1, 1)
    for (name, phase), packed in denoised.items():
        request = requests[name][0]
        print(json.dumps(dict(event="decode", case=name, phase=phase)), flush=True)
        latents = pipeline._unpack_latents(
            packed, request["height"], request["width"]
        ).float()
        decoded = vae.decode(latents / inverse + mean, return_dict=False)[0][0, :, 0]
        assert torch.isfinite(decoded).all()
        rgb = decoded.clamp(-1, 1).numpy()
        (args.output / f"{name}-{phase}.f32").write_bytes(rgb.astype("<f4").tobytes())
        pixels = np.rint((rgb.transpose(1, 2, 0) / 2 + 0.5) * 255).astype(np.uint8)
        Image.fromarray(pixels).save(args.output / f"{name}-{phase}.png")
    print(
        "PASS: finite canonical fresh-prompt reference images, unchanged base weights.",
        flush=True,
    )


if __name__ == "__main__":
    main()
