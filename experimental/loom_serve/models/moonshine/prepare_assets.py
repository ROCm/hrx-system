#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface-hub==2.2.0", "safetensors==0.8.0"]
# ///

"""Prepare a pinned Moonshine deployment without a Python inference model."""

import argparse
import ctypes
import hashlib
import json
import struct
from pathlib import Path

CHECKPOINT = "moonshine-ai/moonshine-streaming-tiny"
REVISION = "f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a"


def rotary_frequencies() -> bytes:
    """Match the reference's F32 power and reciprocal round points.

    Thirty-two of each head's forty dimensions rotate. This table is constant
    across decoder positions; kernels evaluate each request's own angles.
    """

    def f32(value: float) -> float:
        return struct.unpack("<f", struct.pack("<f", value))[0]

    return struct.pack(
        "<16f", *(f32(1.0 / f32(10000.0 ** (index / 16))) for index in range(16))
    )


def prepare(output: Path, name: str, offline: bool) -> Path:
    from huggingface_hub import hf_hub_download
    from safetensors import TensorSpec, serialize

    output = output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Choose a new deployment directory: {output}")
    if not name:
        raise ValueError("The deployment needs a nonempty model name")
    package = Path(__file__).resolve().parent
    deployment = json.loads((package / "model.json").read_text())
    assets = {}
    records = {}
    for filename in ("model.safetensors", "tokenizer.json", "config.json"):
        path = Path(
            hf_hub_download(
                CHECKPOINT, filename, revision=REVISION, local_files_only=offline
            )
        ).resolve()
        assets[filename] = path
        with path.open("rb") as source:
            records[filename] = hashlib.file_digest(source, "sha256").hexdigest()
    frequency_data = ctypes.create_string_buffer(rotary_frequencies())
    frequencies = bytes(
        serialize(
            {
                "inverse": TensorSpec(
                    dtype="float32",
                    shape=[16],
                    data_ptr=ctypes.addressof(frequency_data),
                    data_len=64,
                )
            }
        )
    )
    deployment["name"] = name
    deployment["base_directory"] = str(package)
    deployment["definition"]["assets"] = str(output)
    # A new directory preserves any running deployment. Publish its model
    # document only after every referenced asset is complete.
    output.mkdir(parents=True)
    for filename, path in assets.items():
        (output / filename).symlink_to(path)
    (output / "rotary.safetensors").write_bytes(frequencies)
    (output / "provenance.json").write_text(
        json.dumps(
            {"checkpoint": CHECKPOINT, "revision": REVISION, "sha256": records},
            indent=2,
        )
        + "\n"
    )
    model = output / "model.json"
    model.write_text(json.dumps(deployment, indent=2) + "\n")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="moonshine-streaming-tiny")
    parser.add_argument("--offline", action="store_true")
    arguments = parser.parse_args()
    print(prepare(arguments.output, arguments.name, arguments.offline))


if __name__ == "__main__":
    main()
