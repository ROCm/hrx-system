#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.2.4", "onnx==1.19.1"]
# ///

"""Prepare both pinned Silero rate families for native Loom execution."""

import argparse
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

REVISION = "5cd7945676eb32225748052e2e6a0580e4686a08"
CHECKPOINT_URL = (
    f"https://raw.githubusercontent.com/snakers4/silero-vad/{REVISION}/"
    "src/silero_vad/data/silero_vad.onnx"
)
CHECKPOINT_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
CHECKPOINT_BYTES = 2327524
RATE_BRANCHES = ((8000, "else_branch"), (16000, "then_branch"))
MATRIX_PARAMETERS = (
    ("stft_weights", "stft.forward_basis_buffer"),
    ("encoder0_weights", "encoder.0.reparam_conv.weight"),
    ("encoder1_weights", "encoder.1.reparam_conv.weight"),
    ("encoder2_weights", "encoder.2.reparam_conv.weight"),
    ("encoder3_weights", "encoder.3.reparam_conv.weight"),
    ("input_gates_weights", "decoder.rnn.weight_ih"),
    ("recurrent_gates_weights", "decoder.rnn.weight_hh"),
)
BIAS_PARAMETERS = tuple(
    (f"encoder{layer}_bias", f"encoder.{layer}.reparam_conv.bias") for layer in range(4)
)


def checkpoint_bytes(path: Path | None) -> bytes:
    if path is None:
        with urllib.request.urlopen(CHECKPOINT_URL) as response:
            data = response.read(CHECKPOINT_BYTES + 1)
    else:
        data = path.expanduser().read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != CHECKPOINT_BYTES or digest != CHECKPOINT_SHA256:
        raise ValueError(f"Silero checkpoint integrity mismatch: {digest}")
    return data


def parameters(graph):
    import numpy as np
    import onnx
    from onnx import numpy_helper

    # The checksum fixes the exported graph and original IFGO gate ordering.
    names = {source for _, source in MATRIX_PARAMETERS + BIAS_PARAMETERS}
    names.update(
        (
            "decoder.rnn.bias_ih",
            "decoder.rnn.bias_hh",
            "decoder.decoder.2.weight",
            "decoder.decoder.2.bias",
        )
    )
    original = {
        attribute.t.name: numpy_helper.to_array(attribute.t)
        for node in graph.node
        for attribute in node.attribute
        if attribute.type == onnx.AttributeProto.TENSOR and attribute.t.name in names
    }
    result = {}
    for name, source in MATRIX_PARAMETERS:
        value = original[source]
        matrix = value.reshape(value.shape[0], -1)
        # Twelve output accumulators share each sixteen-column activation load.
        padding = (-matrix.shape[0]) % 12
        result[name] = np.pad(matrix, ((0, padding), (0, 0)))
    result.update((name, original[source]) for name, source in BIAS_PARAMETERS)
    result["gate_bias"] = np.add(
        original["decoder.rnn.bias_ih"], original["decoder.rnn.bias_hh"]
    )
    result["head_weight"] = original["decoder.decoder.2.weight"].reshape(128)
    result["head_bias"] = original["decoder.decoder.2.bias"].reshape(1)
    return result


def safetensors_bytes(tensors) -> bytes:
    """Serialize the package's contiguous F32 parameters in Safetensors format."""
    header = {}
    payloads = []
    offset = 0
    for name, value in tensors.items():
        payload = value.astype("<f4", copy=False).tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(value.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads.append(payload)
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + b"".join(payloads)


def prepare(output: Path, name: str, checkpoint: Path | None, device: int | None):
    output = output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Choose a new deployment directory: {output}")
    if not name:
        raise ValueError("The deployment needs a nonempty name")
    if device is not None and device < 0:
        raise ValueError("The device index must be nonnegative")
    data = checkpoint_bytes(checkpoint)
    import onnx

    model = onnx.load_model_from_string(data)
    selector = next(node for node in model.graph.node if node.op_type == "If")
    branches = {attribute.name: attribute.g for attribute in selector.attribute}
    packed = {
        rate: safetensors_bytes(parameters(branches[branch]))
        for rate, branch in RATE_BRANCHES
    }
    package = Path(__file__).resolve().parent
    documents = {}
    for rate, _ in RATE_BRANCHES:
        document = json.loads((package / f"model_{rate}.json").read_text())
        document["name"] = f"{name}-{rate}"
        document["base_directory"] = str(package)
        document["definition"]["assets"] = str(output)
        if device is not None:
            document["device"] = device
        documents[rate] = document

    # Publish deployment documents only after all referenced assets are complete.
    output.mkdir(parents=True)
    (output / "silero_vad.onnx").write_bytes(data)
    (output / "LICENSE.silero").write_bytes((package / "LICENSE.silero").read_bytes())
    for rate, payload in packed.items():
        (output / f"silero-{rate}.safetensors").write_bytes(payload)
    provenance = {
        "url": CHECKPOINT_URL,
        "revision": REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "sha256": {
            f"silero-{rate}.safetensors": hashlib.sha256(payload).hexdigest()
            for rate, payload in packed.items()
        },
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    paths = []
    for rate, document in documents.items():
        path = output / f"model_{rate}.json"
        path.write_text(json.dumps(document, indent=2) + "\n")
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--name", default="silero-vad", help="Prefix for the two model names"
    )
    parser.add_argument(
        "--onnx", type=Path, help="Use the pinned local checkpoint offline"
    )
    parser.add_argument(
        "--device", type=int, help="Device index in a multi-device server"
    )
    args = parser.parse_args()
    for path in prepare(args.output, args.name, args.onnx, args.device):
        print(path)


if __name__ == "__main__":
    main()
