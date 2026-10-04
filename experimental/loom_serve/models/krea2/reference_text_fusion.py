# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Capture canonical text-fusion stages without loading the full DiT.

The final base/LoRA outputs must exactly reproduce reference.py's captures
before these intermediates are accepted. Only the 49 fusion base tensors
and 66 adapter tensors are loaded. Retained BF16 stage outputs total 135 MiB
at 512 tokens; the existing encoder taps are not duplicated.
"""

import argparse
import json
from pathlib import Path

import torch
from diffusers.loaders.single_file_utils import (
    convert_krea2_transformer_checkpoint_to_diffusers,
)
from diffusers.models.transformers.transformer_krea2 import Krea2TextFusion
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora.layer import LoraLayer
from peft.utils import set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import load_file


def encode(value):
    value = value.detach().cpu().contiguous()
    if value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
        raise ValueError("fusion capture requires finite BF16 tensors")
    return value.view(torch.uint16).numpy().astype("<u2").tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.set_grad_enabled(False)
    configuration = json.loads(
        (arguments.checkpoint / "transformer/config.json").read_text()
    )
    with torch.device("meta"):
        fusion = Krea2TextFusion(
            num_text_layers=configuration["num_text_layers"],
            dim=configuration["text_hidden_dim"],
            num_heads=configuration["text_num_attention_heads"],
            num_kv_heads=configuration["text_num_key_value_heads"],
            intermediate_size=configuration["text_intermediate_size"],
            num_layerwise_blocks=configuration["num_layerwise_text_blocks"],
            num_refiner_blocks=configuration["num_refiner_text_blocks"],
            eps=configuration["norm_eps"],
        )
    with safe_open(
        str(arguments.checkpoint / "turbo.safetensors"), framework="pt"
    ) as source:
        original = {
            name: source.get_tensor(name).bfloat16()
            for name in source.keys()
            if name.startswith("txtfusion.")
        }
    converted = convert_krea2_transformer_checkpoint_to_diffusers(original)
    fusion.load_state_dict(
        {name.removeprefix("text_fusion."): value for name, value in converted.items()},
        strict=True,
        assign=True,
    )
    del converted, original
    fusion = fusion.eval().requires_grad_(False).to("cuda")
    base_parameters = [
        (name, parameter, parameter.data_ptr(), parameter._version)
        for name, parameter in fusion.named_parameters()
    ]
    if len(base_parameters) != 49:
        raise ValueError("the pinned fusion module has 49 base parameters")
    captured = load_file(str(arguments.reference / "inputs.safetensors"))
    features = captured["encoder_hidden"].to("cuda")
    mask = captured["encoder_mask"].bool().to("cuda")
    if features.shape != (1, 512, 12, 2560) or mask.numel() != 512:
        raise ValueError("qualification requires the pinned 512-token capture")
    mask = mask.reshape(1, 1, 1, 512)
    base_bytes = None
    for phase in ("base", "zero", "style"):
        if phase == "zero":
            targets = [
                name
                for name, module in fusion.named_modules()
                if isinstance(module, torch.nn.Linear)
            ]
            if len(targets) != 33:
                raise ValueError("the pinned adapter covers all 33 fusion linears")
            inject_adapter_in_model(
                LoraConfig(r=32, lora_alpha=32, target_modules=targets),
                fusion,
                adapter_name="watercolor",
            )
            with safe_open(
                str(arguments.adapter / "softwatercolor.safetensors"), framework="pt"
            ) as source:
                adapter = {
                    name.removeprefix("transformer.text_fusion."): source.get_tensor(
                        name
                    ).bfloat16()
                    for name in source.keys()
                    if name.startswith("transformer.text_fusion.")
                }
            if len(adapter) != 66:
                raise ValueError("the pinned adapter has 66 fusion tensors")
            loaded = set_peft_model_state_dict(
                fusion, adapter, adapter_name="watercolor"
            )
            if loaded.unexpected_keys or any(
                "lora_" in name for name in loaded.missing_keys
            ):
                raise ValueError(f"adapter binding mismatch: {loaded}")
            del adapter
            fusion.eval().requires_grad_(False)
            if any(
                parameter.dtype != torch.bfloat16 for parameter in fusion.parameters()
            ):
                raise ValueError("reference base and adapter must remain BF16")
        if phase != "base":
            for module in fusion.modules():
                if isinstance(module, LoraLayer):
                    module.set_scale("watercolor", float(phase == "style"))

        def tap(name):
            def capture(module, inputs, output):
                path = arguments.output / f"{phase}-{name}.bf16"
                path.write_bytes(encode(output))
                print(
                    json.dumps(
                        dict(
                            phase=phase,
                            stage=name,
                            shape=list(output.shape),
                            bytes=path.stat().st_size,
                        )
                    ),
                    flush=True,
                )

            return capture

        handles = []
        if phase != "zero":
            for name, module in (
                ("layerwise0", fusion.layerwise_blocks[0]),
                ("layerwise1", fusion.layerwise_blocks[1]),
                ("projected", fusion.projector),
                ("refiner0", fusion.refiner_blocks[0]),
                ("refiner1", fusion.refiner_blocks[1]),
            ):
                handles.append(module.register_forward_hook(tap(name)))
        value = fusion(features, attention_mask=mask)
        for handle in handles:
            handle.remove()
        actual_bytes = encode(value)
        if phase == "zero":
            if actual_bytes != base_bytes:
                raise AssertionError("zero-strength fusion differs from base")
        else:
            expected = load_file(
                str(arguments.reference / f"{phase}-text_fusion.safetensors")
            )["text_fusion"]
            if actual_bytes != encode(expected):
                error = value.cpu().double() - expected.double()
                print(
                    json.dumps(
                        dict(
                            phase=phase,
                            relative_l2=float(error.norm() / expected.double().norm()),
                            maximum_absolute_error=float(error.abs().max()),
                        )
                    ),
                    flush=True,
                )
                raise AssertionError(
                    "standalone fusion does not reproduce the complete reference"
                )
            if phase == "base":
                base_bytes = actual_bytes
        for name, parameter, pointer, version in base_parameters:
            if parameter.data_ptr() != pointer or parameter._version != version:
                raise AssertionError(f"base parameter changed: {name}")
        print(
            json.dumps(dict(phase=phase, exact=True, base_immutable=True)), flush=True
        )
    print(
        "PASS: standalone canonical fusion reproduces base/zero/LoRA captures.",
        flush=True,
    )


if __name__ == "__main__":
    main()
