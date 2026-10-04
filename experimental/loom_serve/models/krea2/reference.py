# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Krea 2 Turbo numerical reference, not the Loom implementation.

Run with exclusive use of the target device. Original checkpoint files are
reused in place; component tensors deliberately synchronize for inspection.
See README.md for pinned inputs, dependencies, precision, and qualification.
"""

import argparse
import collections
import importlib.metadata
import json
import pathlib
import time

import numpy as np
import torch
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    Krea2Pipeline,
    Krea2Transformer2DModel,
)
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3VLModel

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
parser.add_argument("--adapter", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--size", type=int, default=384)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.set_grad_enabled(False)
root = args.checkpoint
adapter = args.adapter
dtype = torch.bfloat16
started = time.monotonic()
events = (args.output / "events.jsonl").open("w")


def event(kind, **fields):
    record = {"event": kind, "elapsed_seconds": time.monotonic() - started, **fields}
    line = json.dumps(record)
    print(line, flush=True)
    events.write(line + "\n")
    events.flush()


event(
    "environment",
    packages={
        name: importlib.metadata.version(name)
        for name in ("torch", "diffusers", "transformers", "peft", "safetensors")
    },
    device=torch.cuda.get_device_name(),
    hip=torch.version.hip,
)
event("load_transformer")
transformer = (
    Krea2Transformer2DModel.from_single_file(
        str(root / "turbo.safetensors"),
        config=str(root / "transformer"),
        torch_dtype=dtype,
        device="cuda",
        local_files_only=True,
    )
    .eval()
    .requires_grad_(False)
)
base_parameters = [
    (name, parameter, parameter.data_ptr(), parameter._version)
    for name, parameter in transformer.named_parameters()
]
event(
    "transformer_loaded",
    parameter_dtypes=dict(
        collections.Counter(
            str(parameter.dtype) for _, parameter, _, _ in base_parameters
        )
    ),
    norm_dtype=str(transformer.transformer_blocks[0].norm1.weight.dtype),
)
event("load_encoder")
encoder = (
    Qwen3VLModel.from_pretrained(
        root / "text_encoder",
        dtype=dtype,
        device_map={"": "cuda"},
        local_files_only=True,
    )
    .eval()
    .requires_grad_(False)
)
tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
event("load_vae")
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
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    root / "scheduler",
    local_files_only=True,
)
pipeline = Krea2Pipeline(
    scheduler=scheduler,
    vae=vae,
    text_encoder=encoder,
    tokenizer=tokenizer,
    transformer=transformer,
    is_distilled=True,
)
pipeline.set_progress_bar_config(disable=True)
prompt = "A deer grazing in the forest, Art Deco watercolor style"
event("encode_prompt", prompt=prompt)
features, mask = pipeline.get_text_hidden_states([prompt], max_sequence_length=512)
assert torch.isfinite(features).all(), "nonfinite encoder output"
noise = pipeline.prepare_latents(
    1,
    16,
    args.size,
    args.size,
    dtype,
    torch.device("cuda"),
    torch.Generator(device="cpu").manual_seed(42),
)
save_file(
    {
        "encoder_hidden": features.cpu().contiguous(),
        "encoder_mask": mask.cpu().contiguous(),
        "initial_latents": noise.cpu().contiguous(),
    },
    str(args.output / "inputs.safetensors"),
)
(args.output / "initial_latents.bf16").write_bytes(
    noise.cpu().contiguous().view(torch.uint16).numpy().astype("<u2").tobytes()
)
event(
    "inputs_ready",
    hidden_shape=list(features.shape),
    mask_valid=int(mask.sum()),
    latent_shape=list(noise.shape),
    peak_device_bytes=torch.cuda.max_memory_allocated(),
)

capture = {"phase": "base", "enabled": True}


def tap(name):
    def hook(module, arguments, output):
        if capture["enabled"]:
            assert torch.isfinite(output).all(), name
            save_file(
                {name: output.detach().cpu().contiguous()},
                str(args.output / (capture["phase"] + "-" + name + ".safetensors")),
            )
            if name == "image_projection":
                (
                    args.output / (capture["phase"] + "-image_projection.bf16")
                ).write_bytes(
                    output.detach()
                    .cpu()
                    .contiguous()
                    .view(torch.uint16)
                    .numpy()
                    .astype("<u2")
                    .tobytes()
                )

    return hook


def step(pipeline, index, timestep, values):
    latents = values["latents"]
    assert torch.isfinite(latents).all(), "nonfinite step output"
    save_file(
        {"latents": latents.cpu().contiguous()},
        str(args.output / (capture["phase"] + "-step-" + str(index) + ".safetensors")),
    )
    capture["enabled"] = False
    event("step", phase=capture["phase"], index=index, timestep=float(timestep))
    return values


denoised = {}
for phase in ("base", "zero", "style"):
    capture.update(phase=phase, enabled=phase != "zero")
    if phase == "zero":
        event("load_adapter")
        transformer.load_lora_adapter(
            str(adapter),
            weight_name="softwatercolor.safetensors",
            local_files_only=True,
            adapter_name="watercolor",
        )
        event(
            "adapter_loaded",
            parameter_dtypes=dict(
                collections.Counter(
                    str(parameter.dtype)
                    for name, parameter in transformer.named_parameters()
                    if ".lora_A." in name or ".lora_B." in name
                )
            ),
        )
        transformer.set_adapters("watercolor", weights=0.0)
    elif phase == "style":
        transformer.set_adapters("watercolor", weights=1.0)
    # Adapter injection replaces Linear modules. Hooks observe the outer
    # module including its update, not the wrapped base-layer output.
    handles = [
        module.register_forward_hook(tap(name))
        for name, module in (
            ("image_projection", transformer.img_in),
            ("text_fusion", transformer.text_fusion),
            ("text_projection", transformer.txt_in),
            ("block0", transformer.transformer_blocks[0]),
        )
    ]
    event("generate", phase=phase, size=args.size, steps=8)
    latents = pipeline(
        prompt_embeds=features,
        prompt_embeds_mask=mask,
        latents=noise.clone(),
        height=args.size,
        width=args.size,
        num_inference_steps=8,
        guidance_scale=0.0,
        callback_on_step_end=step,
        output_type="latent",
    ).images
    for name, parameter, pointer, version in base_parameters:
        assert parameter.data_ptr() == pointer and parameter._version == version, name
    denoised[phase] = latents.cpu().contiguous()
    for handle in handles:
        handle.remove()
    if phase == "zero":
        assert torch.equal(denoised["base"], denoised["zero"]), (
            "zero adapter changes denoised latents"
        )
        event("zero_latents_match_base")

# F32 CPU VAE decoding is the explicit reference precision, independent of
# the Loom deployment or a GPU library's convolution algorithm selection.
# Decode only after GPU generation finishes so moving this component cannot
# change Diffusers' inferred execution device during the denoising loop.
vae.to("cpu")
images = {}
event("vae_oracle", device="cpu", dtype=str(vae.dtype))
mean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1)
inverse_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1)
for phase, packed in denoised.items():
    latents = pipeline._unpack_latents(packed, args.size, args.size).float()
    latents = latents / inverse_std + mean
    decoded = vae.decode(latents, return_dict=False)[0][:, :, 0]
    image = pipeline.image_processor.postprocess(decoded, output_type="np")[0]
    assert np.isfinite(image).all(), "nonfinite decoded image"
    images[phase] = image.copy()
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(
        args.output / (phase + ".png")
    )
    event(
        "image",
        phase=phase,
        shape=list(image.shape),
        minimum=float(image.min()),
        maximum=float(image.max()),
        peak_device_bytes=torch.cuda.max_memory_allocated(),
    )
    if phase == "zero":
        assert np.array_equal(images["base"], image), "zero adapter changes base output"
        event("zero_matches_base")
assert not np.array_equal(images["base"], images["style"]), "adapter has no effect"
event(
    "complete",
    adapter_mean_absolute_difference=float(
        np.abs(images["base"] - images["style"]).mean()
    ),
)
events.close()
