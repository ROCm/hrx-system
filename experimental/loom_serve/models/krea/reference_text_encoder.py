# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Extract canonical Qwen3-VL text primitives without loading the DiT or VAE.

The complete twelve-tap result must reproduce the original Krea capture
bit-for-bit. Decoder hooks establish hidden-state tuple indexing, and a second
execution omits the unused final decoder/norm to qualify that graph boundary.
Only layer-zero/layer-34 primitives and request inputs are retained, below
192 MiB. These are external reference tensors, not native Loom outputs.
"""

import argparse
import json
from pathlib import Path

import torch
from diffusers import Krea2Pipeline
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen3VLModel


def encode(value):
    value = value.detach().cpu().contiguous()
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError("nonfinite encoder tensor")
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().astype("<u2").tobytes(), "bf16"
    if value.dtype == torch.int32:
        return value.numpy().astype("<i4").tobytes(), "i32"
    if value.dtype == torch.bool:
        return value.numpy().astype("u1").tobytes(), "u8"
    raise ValueError(f"unsupported encoder capture type: {value.dtype}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.set_grad_enabled(False)
    events = [
        json.loads(line)
        for line in (arguments.reference / "events.jsonl").read_text().splitlines()
    ]
    prompts = [event["prompt"] for event in events if event["event"] == "encode_prompt"]
    if len(prompts) != 1:
        raise ValueError("qualification requires one captured prompt")
    expected = load_file(str(arguments.reference / "inputs.safetensors"))
    expected_taps = expected["encoder_hidden"]
    expected_mask = expected["encoder_mask"]
    if expected_taps.shape != (1, 512, 12, 2560):
        raise ValueError("qualification requires the original 512-token encoder taps")

    encoder = (
        Qwen3VLModel.from_pretrained(
            arguments.checkpoint / "text_encoder",
            dtype=torch.bfloat16,
            device_map={"": "cuda"},
            local_files_only=True,
        )
        .eval()
        .requires_grad_(False)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        arguments.checkpoint / "tokenizer", local_files_only=True
    )
    pipeline = Krea2Pipeline(
        scheduler=None,
        vae=None,
        text_encoder=encoder,
        tokenizer=tokenizer,
        transformer=None,
        is_distilled=True,
    )
    language = encoder.language_model
    selected = tuple(pipeline.text_encoder_select_layers)
    if selected != tuple(range(2, 36, 3)) or len(language.layers) != 36:
        raise ValueError("the pinned encoder uses twelve taps from 36 decoder layers")
    original_parameters = [
        (name, parameter, parameter.data_ptr(), parameter._version)
        for name, parameter in encoder.named_parameters()
    ]
    inventory = {}
    retained_bytes = 0

    def record(name, value):
        nonlocal retained_bytes
        data, suffix = encode(value)
        if name in inventory:
            raise AssertionError(f"duplicate reference capture: {name}")
        retained_bytes += len(data)
        if retained_bytes > 192 * 1024 * 1024:
            raise AssertionError("encoder fixtures exceed the declared retention bound")
        filename = name + "." + suffix
        (arguments.output / filename).write_bytes(data)
        inventory[name] = dict(file=filename, shape=list(value.shape), bytes=len(data))
        print(json.dumps(dict(stage=name, **inventory[name])), flush=True)

    def output_hook(name):
        def capture(module, inputs, output):
            record(name, output)

        return capture

    def input_hook(name):
        def capture(module, inputs):
            record(name, inputs[0])

        return capture

    def capture_request(module, inputs, keywords):
        ids = keywords["input_ids"]
        mask = keywords["attention_mask"]
        positions = keywords["position_ids"]
        if ids.shape != (1, 546) or mask.shape != ids.shape:
            raise ValueError("the canonical padded request has 546 encoder positions")
        if positions.shape != (3, 1, 546) or any(
            not torch.equal(positions[0], positions[axis]) for axis in (1, 2)
        ):
            raise ValueError("text-only rotary positions must agree across all axes")
        if not torch.equal(positions[0], (mask.long().cumsum(-1) - 1).clamp_min(0)):
            raise AssertionError("middle padding must not advance rotary position")
        record("tokens", ids.int())
        record("mask", mask.bool())
        record("positions", positions[0].int())

    def capture_rotary(module, inputs, output):
        record("cosine", output[0])
        record("sine", output[1])

    decoder_taps = {}

    def tap_hook(index):
        def capture(module, inputs, output):
            decoder_taps[index] = output.detach().cpu().clone()

        return capture

    def verify_tuple(module, inputs, output):
        for index in selected:
            if encode(output.hidden_states[index])[0] != encode(decoder_taps[index])[0]:
                raise AssertionError(f"hidden tuple {index} is not decoder {index - 1}")
        print(
            json.dumps(dict(decoder_tap_indices=list(selected), exact=True)), flush=True
        )

    handles = [
        encoder.register_forward_pre_hook(capture_request, with_kwargs=True),
        language.embed_tokens.register_forward_hook(output_hook("embedding")),
        language.rotary_emb.register_forward_hook(capture_rotary),
        encoder.register_forward_hook(verify_tuple),
    ]
    for index in selected:
        handles.append(
            language.layers[index - 1].register_forward_hook(tap_hook(index))
        )
    for index in (0, 34):
        layer = language.layers[index]
        prefix = f"layer{index}."
        handles.append(layer.register_forward_pre_hook(input_hook(prefix + "input")))
        handles.append(layer.register_forward_hook(output_hook(prefix + "output")))
        for name, module in (
            ("norm1", layer.input_layernorm),
            ("query", layer.self_attn.q_proj),
            ("key", layer.self_attn.k_proj),
            ("value", layer.self_attn.v_proj),
            ("query_norm", layer.self_attn.q_norm),
            ("key_norm", layer.self_attn.k_norm),
            ("attention_out", layer.self_attn.o_proj),
            ("norm2", layer.post_attention_layernorm),
            ("gate", layer.mlp.gate_proj),
            ("up", layer.mlp.up_proj),
            ("down", layer.mlp.down_proj),
        ):
            handles.append(module.register_forward_hook(output_hook(prefix + name)))
        for name, module in (
            ("attention", layer.self_attn.o_proj),
            ("residual", layer.post_attention_layernorm),
            ("product", layer.mlp.down_proj),
        ):
            handles.append(module.register_forward_pre_hook(input_hook(prefix + name)))

    taps, mask = pipeline.get_text_hidden_states(
        [prompts[0]], 512, torch.device("cuda")
    )
    for handle in handles:
        handle.remove()
    if encode(taps)[0] != encode(expected_taps)[0] or not torch.equal(
        mask.cpu(), expected_mask
    ):
        raise AssertionError(
            "standalone encoder does not reproduce the complete reference"
        )
    print(
        json.dumps(dict(check="canonical_taps", exact=True, elements=taps.numel())),
        flush=True,
    )
    del taps
    decoder_taps.clear()

    # The consumer ends at decoder 34's output, before the final decoder/norm.
    # Re-executing that exact prefix verifies the selected dependency boundary.
    language.layers = torch.nn.ModuleList(list(language.layers[:35]))
    language.norm = torch.nn.Identity()
    taps, mask = pipeline.get_text_hidden_states(
        [prompts[0]], 512, torch.device("cuda")
    )
    if encode(taps)[0] != encode(expected_taps)[0] or not torch.equal(
        mask.cpu(), expected_mask
    ):
        raise AssertionError("encoder prefix changes the consumer's selected taps")
    for name, parameter, pointer, version in original_parameters:
        if parameter.data_ptr() != pointer or parameter._version != version:
            raise AssertionError(f"encoder parameter changed: {name}")
    parameters = list(language.parameters())
    parameter_bytes = sum(value.numel() * value.element_size() for value in parameters)
    if len(parameters) != 386 or parameter_bytes != 7843069440:
        raise AssertionError("the selected text-only encoder weight domain changed")
    inventory["request"] = dict(prompt=prompts[0], retained_bytes=retained_bytes)
    (arguments.output / "tensors.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(
                check="encoder_prefix",
                exact=True,
                elements=taps.numel(),
                parameters=len(parameters),
                parameter_bytes=parameter_bytes,
                retained_bytes=retained_bytes,
                base_immutable=True,
            )
        ),
        flush=True,
    )
    print(
        "PASS: canonical encoder taps and the text-only prefix are exact.", flush=True
    )


if __name__ == "__main__":
    main()
