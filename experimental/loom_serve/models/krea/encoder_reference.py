# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Independent F64 encoder block primitives with explicit BF16 boundaries."""

from check_encoder_attention import attention
from check_encoder_normalization import normalize, rotate


def evaluate_encoder_block(value, cosine, sine, mask, weights, layer, observe=None):
    """Evaluate a real decoder block, optionally replacing each qualified stage."""
    prefix = f"language_model.layers.{layer}."
    rows = value.shape[0]

    def emit(name, expected, operands, *, parameters=True, exact=False):
        if observe is None:
            return expected
        return observe(name, expected, operands, parameters=parameters, exact=exact)

    def linear(name, key, operand):
        weight = weights.get_tensor(prefix + key + ".weight")
        expected = (operand.double().reshape(rows, -1) @ weight.double().T).bfloat16()
        return emit(name, expected, [operand])

    def head(name, operand, heads):
        projected = operand.reshape(rows, heads, 128)
        weight_key = (
            "self_attn.q_norm.weight" if name == "query" else "self_attn.k_norm.weight"
        )
        weight = weights.get_tensor(prefix + weight_key)
        normalized = emit(name + "_norm", normalize(projected, weight), [projected])
        return emit(
            name + "_rotary",
            rotate(normalized, cosine, sine),
            [projected, cosine, sine],
            exact=True,
        )

    normalized = emit(
        "norm1",
        normalize(value, weights.get_tensor(prefix + "input_layernorm.weight")),
        [value],
    )
    query = head("query", linear("query", "self_attn.q_proj", normalized), 32)
    key = head("key", linear("key", "self_attn.k_proj", normalized), 8)
    values = linear("value", "self_attn.v_proj", normalized).reshape(rows, 8, 128)
    context = emit(
        "attention",
        attention(query, key, values, mask),
        [query, key, values, mask],
        parameters=False,
    )
    update = linear("attention_output", "self_attn.o_proj", context)
    residual = emit(
        "residual",
        (value.double() + update.double()).bfloat16(),
        [value, update],
        parameters=False,
    )
    normalized = emit(
        "norm2",
        normalize(
            residual, weights.get_tensor(prefix + "post_attention_layernorm.weight")
        ),
        [residual],
    )
    gate = linear("feed_forward_gate", "mlp.gate_proj", normalized)
    up = linear("feed_forward_up", "mlp.up_proj", normalized)
    activated = emit(
        "silu",
        (gate.double() * gate.double().sigmoid()).bfloat16(),
        [gate],
        parameters=False,
    )
    product = emit(
        "product",
        (activated.double() * up.double()).bfloat16(),
        [gate, up],
        parameters=False,
        exact=True,
    )
    update = linear("feed_forward_down", "mlp.down_proj", product)
    return emit(
        "residual",
        (residual.double() + update.double()).bfloat16(),
        [residual, update],
        parameters=False,
    )
