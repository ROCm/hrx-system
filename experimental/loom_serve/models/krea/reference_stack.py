# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Capture the complete external DiT stack and its final block's boundaries.

Consumes the first-step block-zero capture from reference.py --block-details.
Only the independent transformer is loaded; text encoding and VAE decoding are
not repeated. This is numerical evidence, not the Loom execution path.
"""

import argparse
import json
import pathlib

import numpy as np
import torch
from diffusers import Krea2Transformer2DModel

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
parser.add_argument("--adapter", type=pathlib.Path, required=True)
parser.add_argument("--reference", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.set_grad_enabled(False)


def load(name, dtype):
    value = torch.from_numpy(np.fromfile(args.reference / name, dtype=dtype))
    if dtype == "<u2":
        value = value.view(torch.bfloat16)
    elif dtype == "u1":
        value = value.bool()
    return value.to("cuda")


def save(directory, name, value):
    if not torch.isfinite(value).all():
        raise ValueError(f"nonfinite reference tensor: {name}")
    value = value.cpu().contiguous()
    if value.dtype == torch.bfloat16:
        data = value.view(torch.uint16).numpy().astype("<u2").tobytes()
        suffix = "bf16"
    elif value.dtype == torch.float32:
        data = value.numpy().astype("<f4").tobytes()
        suffix = "f32"
    elif value.dtype == torch.bool:
        data = value.numpy().astype("u1").tobytes()
        suffix = "u8"
    else:
        raise ValueError(f"unexpected tensor dtype: {value.dtype}")
    path = directory / f"{name}.{suffix}"
    path.write_bytes(data)
    print(json.dumps(dict(path=str(path), shape=list(value.shape))), flush=True)


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
if len(transformer.transformer_blocks) != 28:
    raise ValueError("qualification requires the pinned 28-layer Krea transformer")
base_outputs = {}
for phase in ("base", "zero", "style"):
    if phase == "zero":
        transformer.load_lora_adapter(
            str(args.adapter),
            weight_name="softwatercolor.safetensors",
            local_files_only=True,
            adapter_name="watercolor",
        )
        transformer.set_adapters("watercolor", weights=0.0)
    elif phase == "style":
        transformer.set_adapters("watercolor", weights=1.0)
    capture_phase = "base" if phase == "zero" else phase
    prefix = f"{capture_phase}-block0-"
    mask = load(prefix + "mask.u8", "u1")
    tokens = mask.numel()
    if tokens <= 16 or tokens % 16 or not mask[-16:].all():
        raise ValueError("capture requires a tiled sequence ending in image tokens")
    initial = load(prefix + "input.bf16", "<u2").reshape(1, tokens, 6144)
    modulation = load(prefix + "modulation.bf16", "<u2").reshape(1, 1, 6 * 6144)
    cosine = load(prefix + "cosine.f32", "<f4").reshape(tokens, 128)
    sine = load(prefix + "sine.f32", "<f4").reshape(tokens, 128)
    for count in (16, tokens):
        selected = slice(tokens - count, tokens)
        value = initial[:, selected].contiguous()
        rotary = (cosine[selected].contiguous(), sine[selected].contiguous())
        selected_mask = mask[selected].reshape(1, 1, 1, count)
        directory = args.output / str(count)
        directory.mkdir(exist_ok=True)
        if phase != "zero":
            save(directory, f"{phase}-stack-input", value)
        for layer, block in enumerate(transformer.transformer_blocks):
            if layer == 27 and phase != "zero":
                for name, tensor in (
                    ("input", value),
                    ("modulation", modulation),
                    ("cosine", rotary[0]),
                    ("sine", rotary[1]),
                    ("mask", selected_mask),
                ):
                    save(directory, f"{phase}-block27-{name}", tensor)
            value = block(value, modulation, rotary, selected_mask)
            if not torch.isfinite(value).all():
                raise ValueError(f"nonfinite {phase} layer {layer} output")
        if phase == "base":
            base_outputs[count] = value.clone()
        elif phase == "zero":
            if not torch.equal(value, base_outputs[count]):
                raise AssertionError("zero-strength stack differs from base")
            continue
        save(directory, f"{phase}-block27-output", value)
        if phase == "style" and torch.equal(value, base_outputs[count]):
            raise AssertionError("active adapter did not change the stack output")
print(
    "PASS: external 28-layer base/zero/style captures are finite and distinct.",
    flush=True,
)
