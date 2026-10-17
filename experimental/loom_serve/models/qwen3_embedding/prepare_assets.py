#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface-hub==0.35.3", "numpy==2.5.4",
#   "safetensors==0.8.0", "torch==2.12.0+cpu",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///

"""Prepare pinned Qwen3 embedding assets; serving needs no Python model."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

CHECKPOINT = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
LAYER_COUNT = 28
COLUMN_COUNT = 8
RECORD_COUNT = 120
RECORD_BYTES = 18496
CONSTANT_BYTES = 8192
CONSTANT_BASE = COLUMN_COUNT * LAYER_COUNT * RECORD_COUNT * RECORD_BYTES
EMBEDDING_BASE = CONSTANT_BASE + (LAYER_COUNT + 2) * CONSTANT_BYTES
ROTARY_BASE = EMBEDDING_BASE + 151669 * 1024 * 2
BANK_BYTES = ROTARY_BASE + 128 * 128 * 2
EXPECTED_BANK_SHA256 = (
    "14122288fb4a329131d3dbde969fac3d139b5c89811b56ac1cc245bbf5e66579"
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bf16_words(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)
    )


def pack_panel(tensor: torch.Tensor) -> np.ndarray:
    """Encode N8/K8 blocks, with one shared exponent per eight K lanes.

    Native BFP16EBS8 MACs consume each exponent followed by eight signed
    mantissas. A positive rounding carry increases the shared exponent; -128
    remains representable. Each 128x128 panel has 64 bytes of FIFO lookahead.
    """
    values = tensor.float().numpy().reshape(16, 8, 16, 8).transpose(0, 2, 1, 3).copy()
    exponents = ((values.view(np.uint32) >> 23) & 255).max(-1, keepdims=True)
    scales = np.ldexp(1.0, exponents.astype(np.int32) - 133)
    carry = np.rint(values.astype(np.float64) / scales).max(-1, keepdims=True) > 127
    exponents += carry
    scales *= np.where(carry, 2.0, 1.0)
    mantissas = np.rint(values / scales).clip(-128, 127).astype(np.int8)
    records = np.concatenate(
        (exponents.astype(np.uint8), mantissas.view(np.uint8)), axis=-1
    )
    panel = np.zeros(RECORD_BYTES, np.uint8)
    panel[:18432] = records.reshape(-1)
    return panel


def pack_layer(checkpoint, layer: int, output: np.ndarray) -> None:
    prefix = f"layers.{layer}."
    weights = {
        name: checkpoint.get_tensor(prefix + f"self_attn.{name}_proj.weight")
        for name in ("q", "k", "v", "o")
    }
    weights.update(
        {
            name: checkpoint.get_tensor(prefix + f"mlp.{name}_proj.weight")
            for name in ("gate", "up", "down")
        }
    )
    for column in range(COLUMN_COUNT):
        record = 0
        # Q-low/K, Q-high/V, O, three gate/up panel pairs, and down.
        for group, chunks in enumerate((8, 8, 16, 8, 8, 8, 24)):
            for chunk in range(chunks):
                for role in range(1 if group in (2, 6) else 2):
                    if group < 2:
                        name = "q" if role == 0 else ("k" if group == 0 else "v")
                        panel = column + group * 8 if role == 0 else column
                    elif group in (2, 6):
                        name = "o" if group == 2 else "down"
                        panel = column
                    else:
                        name = "gate" if role == 0 else "up"
                        panel = column + (group - 3) * 8
                    tile = weights[name][
                        panel * 128 : (panel + 1) * 128, chunk * 128 : (chunk + 1) * 128
                    ]
                    output[column, record] = pack_panel(tile)
                    record += 1


def silu_table() -> np.ndarray:
    """BF16 SiLU values for magnitudes 0x3b80..0x4300, by sign."""
    magnitude = np.arange(0x3B80, 0x4301, dtype=np.uint16)
    positive = (magnitude.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    values = np.zeros(4096, np.uint16)
    for sign, offset in ((1, 0), (-1, 2048)):
        argument = sign * positive
        exponential = np.exp(-positive)
        result = np.where(argument >= 0, argument, argument * exponential) / (
            1 + exponential
        )
        values[offset : offset + magnitude.size] = bf16_words(torch.from_numpy(result))
    return values


def rotary_table() -> np.ndarray:
    """Preserve the publisher's F32 power, reciprocal, and angle round points."""
    powers = torch.arange(0, 128, 2, dtype=torch.int64).float() / 128
    inverse = 1.0 / (1000000.0**powers)
    positions = torch.arange(128).reshape(1, 1, 128).float()
    angles = (inverse.reshape(1, 64, 1) @ positions).transpose(1, 2)
    # Keep the duplicated width used by the publisher before sin/cos, which
    # also fixes the CPU vector-math path used to prepare the qualified table.
    repeated = torch.cat((angles, angles), dim=-1)
    cosine = repeated.cos().to(torch.bfloat16)[0, :, :64]
    sine = repeated.sin().to(torch.bfloat16)[0, :, :64]
    return bf16_words(torch.cat((cosine, sine), dim=1))


def write_bank(checkpoint_path: Path, path: Path) -> str:
    # Keep the tensor payload, including the safetensors header, aligned to the
    # pipeline's 64-byte binding contract. The command binds one resident tensor.
    header = json.dumps(
        {
            "resident": {
                "dtype": "I8",
                "shape": [BANK_BYTES],
                "data_offsets": [0, BANK_BYTES],
            }
        }
    ).encode()
    header += b" " * (-(8 + len(header)) % 64)
    with path.open("xb") as stream:
        stream.write(struct.pack("<Q", len(header)))
        stream.write(header)
        stream.truncate(8 + len(header) + BANK_BYTES)
    bank = np.memmap(
        path, mode="r+", dtype=np.uint8, offset=8 + len(header), shape=(BANK_BYTES,)
    )
    bank[:] = 0
    panels = bank[:CONSTANT_BASE].reshape(
        COLUMN_COUNT, LAYER_COUNT, RECORD_COUNT, RECORD_BYTES
    )
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        for layer in range(LAYER_COUNT):
            pack_layer(checkpoint, layer, panels[:, layer])
            prefix = f"layers.{layer}."
            base = CONSTANT_BASE + layer * CONSTANT_BYTES
            for name, offset in (
                ("self_attn.q_norm.weight", 0),
                ("self_attn.k_norm.weight", 256),
                ("input_layernorm.weight", 512),
                ("post_attention_layernorm.weight", 2560),
            ):
                values = (
                    bf16_words(checkpoint.get_tensor(prefix + name))
                    .view(np.uint8)
                    .reshape(-1)
                )
                bank[base + offset : base + offset + values.size] = values
            print(f"Packed layer {layer + 1}/{LAYER_COUNT}", flush=True)
        final = bf16_words(checkpoint.get_tensor("norm.weight")).view(np.uint8)
        final_base = CONSTANT_BASE + LAYER_COUNT * CONSTANT_BYTES + 512
        bank[final_base : final_base + final.size] = final
        table_base = CONSTANT_BASE + (LAYER_COUNT + 1) * CONSTANT_BYTES
        bank[table_base : table_base + CONSTANT_BYTES] = silu_table().view(np.uint8)
        embedding = (
            bf16_words(checkpoint.get_tensor("embed_tokens.weight"))
            .view(np.uint8)
            .reshape(-1)
        )
        bank[EMBEDDING_BASE:ROTARY_BASE] = embedding
        bank[ROTARY_BASE:] = rotary_table().view(np.uint8).reshape(-1)
    bank.flush()
    result = hashlib.sha256(memoryview(bank)).hexdigest()
    del panels, bank
    return result


def prepare(output: Path, name: str, offline: bool) -> Path:
    output = output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Choose a new deployment directory: {output}")
    if not name:
        raise ValueError("The deployment needs a nonempty model name")
    package = Path(__file__).resolve().parent
    deployment = json.loads((package / "model.json").read_text())
    assets = {
        filename: Path(
            hf_hub_download(
                CHECKPOINT, filename, revision=REVISION, local_files_only=offline
            )
        ).resolve()
        for filename in ("model.safetensors", "tokenizer.json", "config.json")
    }
    config = json.loads(assets["config.json"].read_text())
    expected = {
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "head_dim": 128,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "vocab_size": 151669,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
        "rope_scaling": None,
        "hidden_act": "silu",
        "attention_bias": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"Unsupported checkpoint {key}: {config.get(key)!r}; expected {value!r}"
            )
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True)
    with torch.inference_mode():
        bank_digest = write_bank(
            assets["model.safetensors"], output / "weights.safetensors"
        )
    if bank_digest != EXPECTED_BANK_SHA256:
        raise RuntimeError(
            f"Prepared bank {bank_digest} differs from the qualified layout "
            f"{EXPECTED_BANK_SHA256}; no model document was published"
        )
    (output / "tokenizer.json").symlink_to(assets["tokenizer.json"])
    provenance = {
        "checkpoint": CHECKPOINT,
        "revision": REVISION,
        "inputs": {filename: digest(path) for filename, path in assets.items()},
        "resident_bytes": BANK_BYTES,
        "resident_sha256": bank_digest,
        "weights_sha256": digest(output / "weights.safetensors"),
        "sources": {path.name: digest(path) for path in sorted(package.glob("*.loom"))},
        "converter_sha256": digest(Path(__file__)),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    deployment["name"] = name
    deployment["base_directory"] = str(package)
    deployment["definition"]["assets"] = str(output)
    # Publish only after all source references and parameter assets are ready.
    document = output / "model.json"
    document.write_text(json.dumps(deployment, indent=2) + "\n")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="qwen3-embedding-0.6b")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only pinned files already in the Hugging Face cache",
    )
    arguments = parser.parse_args()
    print(prepare(arguments.output, arguments.name, arguments.offline))


if __name__ == "__main__":
    main()
