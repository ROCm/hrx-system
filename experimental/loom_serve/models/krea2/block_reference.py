# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Independent F64 block arithmetic with explicit BF16 tensor boundaries.

An optional observer substitutes independently checked native component
outputs, allowing the same arithmetic to test both whole blocks and fusion.
Checkpoint handles and every model operand are supplied by the caller.
"""

import math

import torch


def evaluate_block(
    source,
    modulation,
    cosine,
    sine,
    mask,
    weights,
    adapter,
    layer,
    strength,
    *,
    observe=None,
):
    count = source.shape[0]

    def compute(name, expected, arguments, **metadata):
        return (
            expected
            if observe is None
            else observe(name, expected, arguments, **metadata)
        )

    coefficients = (
        modulation
        + weights.get_tensor(f"blocks.{layer}.mod.lin").reshape(6, 6144).bfloat16()
    )

    def normalize(value, key):
        scale = weights.get_tensor(f"blocks.{layer}." + key).bfloat16().add(1).double()
        precise = value.double()
        inverse = (precise.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
        return (precise * inverse * scale).bfloat16()

    def linear(value, key):
        matrix = weights.get_tensor(f"blocks.{layer}." + key + ".weight").bfloat16()
        return (value.double() @ matrix.double().T).bfloat16()

    def affine(value, row):
        return value * (coefficients[row] + 1) + coefficients[row + 1]

    def rotate(value):
        pairs = value.float().reshape(count, -1, 64, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        return (
            value.float() * cosine[:, None, :] + rotated * sine[:, None, :]
        ).bfloat16()

    def attention(query, key, value):
        value = value.reshape(count, 12, 128)
        result = torch.empty(count, 48, 128, dtype=torch.bfloat16)
        for head in range(48):
            scores = (
                query[:, head].double() @ key[:, head // 4].double().T
            ) / math.sqrt(128)
            scores[:, ~mask] = -torch.inf
            result[:, head] = (
                scores.softmax(-1) @ value[:, head // 4].double()
            ).bfloat16()
        return result.reshape(count, 6144)

    def project(name, key, adapter_key, value, argument):
        base = compute(name, linear(value, key), [argument])
        if adapter is None:
            return base
        factor = f"transformer.transformer_blocks.{layer}." + adapter_key + ".lora_"
        down = adapter.get_tensor(factor + "A.weight").bfloat16().double()
        up = adapter.get_tensor(factor + "B.weight").bfloat16().double()
        low = compute(
            name + "_adapter_down",
            (value.double() @ down.T).bfloat16(),
            [argument],
            parameters=("adapter",),
        )
        delta = compute(
            name + "_adapter_up",
            (low.double() @ up.T).bfloat16(),
            [name + "_adapter_down"],
            parameters=("adapter",),
        )
        combined = compute(
            name + "_adapted",
            base if strength == 0 else base + delta * strength,
            [argument, "strength"],
            exact=True,
            parameters=("base", "adapter"),
            result_name=name,
        )
        return combined

    normalized = compute("norm1", normalize(source, "prenorm.scale"), ["input"])
    attention_input = compute(
        "attention_input",
        affine(normalized, 0),
        ["input", "modulation"],
        exact=True,
    )
    projections = {}
    for name, key, adapter_key in (
        ("query", "wq", "to_q"),
        ("key", "wk", "to_k"),
        ("value", "wv", "to_v"),
        ("gate", "gate", "to_gate"),
    ):
        projections[name] = project(
            name,
            "attn." + key,
            "attn." + adapter_key,
            attention_input,
            "attention_input",
        )
    for name, heads, key in (("query", 48, "qnorm"), ("key", 12, "knorm")):
        normalized = compute(
            name + "_norm",
            normalize(
                projections[name].reshape(count, heads, 128),
                "attn.qknorm." + key + ".scale",
            ),
            [name],
        )
        projections[name] = compute(
            name + "_rotary",
            rotate(normalized),
            [name, "cosine", "sine"],
            exact=True,
        )
    context = compute(
        "attention_ungated",
        attention(projections["query"], projections["key"], projections["value"]),
        ["query_rotary", "key_rotary", "value", "mask"],
        parameters=(),
    )
    context = compute(
        "attention_context",
        context * torch.sigmoid(projections["gate"].double()).bfloat16(),
        ["query_rotary", "key_rotary", "value", "mask", "gate"],
        exact=True,
        parameters=(),
    )
    update = project(
        "attention_output",
        "attn.wo",
        "attn.to_out.0",
        context,
        "attention_context",
    )
    residual = compute(
        "attention_residual",
        source + coefficients[2] * update,
        ["input", "attention_output", "modulation"],
        exact=True,
    )
    normalized = compute(
        "norm2", normalize(residual, "postnorm.scale"), ["attention_residual"]
    )
    feed_forward_input = compute(
        "feed_forward_input",
        affine(normalized, 3),
        ["attention_residual", "modulation"],
        exact=True,
    )
    gate = project(
        "feed_forward_gate",
        "mlp.gate",
        "ff.gate",
        feed_forward_input,
        "feed_forward_input",
    )
    up = project(
        "feed_forward_up",
        "mlp.up",
        "ff.up",
        feed_forward_input,
        "feed_forward_input",
    )
    activated = compute(
        "feed_forward_silu",
        torch.nn.functional.silu(gate.double()).bfloat16(),
        ["feed_forward_gate"],
        parameters=(),
    )
    product = compute(
        "feed_forward_product",
        activated * up,
        ["feed_forward_gate", "feed_forward_up"],
        exact=True,
        parameters=(),
    )
    update = project(
        "feed_forward_down",
        "mlp.down",
        "ff.down",
        product,
        "feed_forward_product",
    )
    return compute(
        "feed_forward_residual",
        residual + coefficients[5] * update,
        ["attention_residual", "feed_forward_down", "modulation"],
        exact=True,
    )
