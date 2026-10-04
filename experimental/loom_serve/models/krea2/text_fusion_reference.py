# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Independent fusion-block arithmetic with F64 reductions and BF16 boundaries.

An observer may replace each primitive with its separately qualified native
output. Its operands are real tensors, so a caller can recycle disk fixtures
without retaining every intermediate on disk.
"""

import math

import torch


def evaluate_fusion_block(
    source, mask, weights, adapter, axis, layer, strength, *, observe=None
):
    rows = source.shape[0]
    prefix = f"txtfusion.{axis}_blocks.{layer}."
    adapter_prefix = f"transformer.text_fusion.{axis}_blocks.{layer}."

    def compute(name, expected, operands, **metadata):
        return (
            expected
            if observe is None
            else observe(name, expected, operands, **metadata)
        )

    def normalize(name, value, key, width):
        scale = weights.get_tensor(prefix + key).bfloat16().add(1).double()
        precise = value.double().reshape(-1, width)
        inverse = (precise.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
        expected = (precise * inverse * scale).bfloat16().reshape_as(value)
        return compute(name, expected, [value])

    def project(name, key, adapter_key, value):
        matrix = weights.get_tensor(prefix + key + ".weight").bfloat16().double()
        base = compute(name, (value.double() @ matrix.T).bfloat16(), [value])
        if adapter is None:
            return base
        factor = adapter_prefix + adapter_key + ".lora_"
        down = adapter.get_tensor(factor + "A.weight").bfloat16().double()
        up = adapter.get_tensor(factor + "B.weight").bfloat16().double()
        low = compute(
            name + "_adapter_down",
            (value.double() @ down.T).bfloat16(),
            [value],
            parameters=("adapter",),
        )
        delta = compute(
            name + "_adapter_up",
            (low.double() @ up.T).bfloat16(),
            [low],
            parameters=("adapter",),
        )
        return compute(
            name + "_adapted",
            base if strength == 0 else base + delta * strength,
            [value, torch.tensor([strength], dtype=torch.float32)],
            parameters=("base", "adapter"),
            exact=True,
        )

    normalized = normalize("norm1", source, "prenorm.scale", 2560)
    projections = {
        name: project(name, "attn." + key, "attn." + target, normalized)
        for name, key, target in (
            ("query", "wq", "to_q"),
            ("key", "wk", "to_k"),
            ("value", "wv", "to_v"),
            ("gate", "gate", "to_gate"),
        )
    }
    for name, key in (("query", "qnorm"), ("key", "knorm")):
        projections[name] = normalize(
            name + "_norm", projections[name], "attn.qknorm." + key + ".scale", 128
        )
    length = 12 if axis == "layerwise" else rows
    sequences = rows // length
    query, key, value = (
        projections[name].reshape(sequences, length, 20, 128)
        for name in ("query", "key", "value")
    )
    context = torch.empty_like(query)
    for head in range(20):
        scores = (
            query[:, :, head].double() @ key[:, :, head].double().transpose(-1, -2)
        ) / math.sqrt(128)
        if axis == "refiner":
            scores[:, :, ~mask] = -torch.inf
        context[:, :, head] = (
            scores.softmax(-1) @ value[:, :, head].double()
        ).bfloat16()
    attention_operands = [projections[name] for name in ("query", "key", "value")]
    if axis == "refiner":
        attention_operands.append(mask)
    context = compute(
        "attention_ungated",
        context.reshape(rows, 2560),
        attention_operands,
        parameters=(),
    )
    context = compute(
        "attention_context",
        context * projections["gate"].double().sigmoid().bfloat16(),
        [*attention_operands, projections["gate"]],
        parameters=(),
        exact=True,
    )
    update = project("attention_output", "attn.wo", "attn.to_out.0", context)
    residual = compute(
        "residual", source + update, [source, update], parameters=(), exact=True
    )
    normalized = normalize("norm2", residual, "postnorm.scale", 2560)
    gate = project("feed_forward_gate", "mlp.gate", "ff.gate", normalized)
    up = project("feed_forward_up", "mlp.up", "ff.up", normalized)
    activated = compute(
        "silu",
        torch.nn.functional.silu(gate.double()).bfloat16(),
        [gate],
        parameters=(),
    )
    product = compute("product", activated * up, [gate, up], parameters=(), exact=True)
    update = project("feed_forward_down", "mlp.down", "ff.down", product)
    return compute(
        "residual", residual + update, [residual, update], parameters=(), exact=True
    )
