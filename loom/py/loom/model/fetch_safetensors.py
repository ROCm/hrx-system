# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Byte-preserving safetensors repack with bounded streaming storage."""

from __future__ import annotations

import hashlib
import io
import json
import os
import struct
from contextlib import ExitStack, nullcontext
from pathlib import Path

from .fetch_range import RangeReader
from .fetch_source import FetchError
from .manifest import ManifestError, _object, _unique_object

_HEADER_LIMIT = 16 * 1024 * 1024
# Safetensors dtype widths describe encoded bytes, never arithmetic conversion.
DTYPE_BITS = {
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    **dict.fromkeys(
        (
            "BOOL",
            "U8",
            "I8",
            "F8_E5M2",
            "F8_E4M3",
            "F8_E8M0",
            "F8_E4M3FNUZ",
            "F8_E5M2FNUZ",
        ),
        8,
    ),
    **dict.fromkeys(("I16", "U16", "F16", "BF16"), 16),
    **dict.fromkeys(("I32", "U32", "F32"), 32),
    **dict.fromkeys(("I64", "U64", "F64", "C64"), 64),
}


def _header(content, data_length):
    try:
        metadata = json.loads(content.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ManifestError(
            "safetensors header requires finite-depth UTF-8 JSON"
        ) from error
    if not isinstance(metadata, dict):
        raise ManifestError("safetensors header requires an object")
    tensors = []
    for name, tensor in metadata.items():
        try:
            name.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ManifestError("safetensors tensor names require UTF-8") from error
        if name == "__metadata__":
            if not isinstance(tensor, dict) or any(
                not isinstance(value, str) for value in tensor.values()
            ):
                raise ManifestError("safetensors metadata values must be strings")
            try:
                for key, value in tensor.items():
                    key.encode("utf-8")
                    value.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ManifestError("safetensors metadata requires UTF-8") from error
            continue
        _object(tensor, "safetensors." + name, ("dtype", "shape", "data_offsets"))
        dtype, shape, offsets = tensor["dtype"], tensor["shape"], tensor["data_offsets"]
        if not isinstance(dtype, str) or dtype not in DTYPE_BITS:
            raise ManifestError(f"unsupported safetensors dtype {dtype!r}")
        if not isinstance(shape, list) or any(
            type(size) is not int or not 0 <= size < 2**64 for size in shape
        ):
            raise ManifestError("safetensors shape requires unsigned 64-bit dimensions")
        elements = 1
        for size in shape:
            elements *= size
            if elements >= 2**64:
                raise ManifestError("safetensors shape product overflows 64 bits")
        bits = elements * DTYPE_BITS[dtype]
        if bits >= 2**64 or bits % 8:
            raise ManifestError(
                "safetensors tensor size overflows or is not byte-aligned"
            )
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                type(offset) is not int or not 0 <= offset < 2**64 for offset in offsets
            )
        ):
            raise ManifestError(
                "safetensors data_offsets requires two unsigned offsets"
            )
        begin, end = offsets
        if end - begin != bits // 8:
            raise ManifestError(
                "safetensors tensor shape/dtype disagrees with its byte range"
            )
        tensors.append((begin, end, name))
    tensors.sort()
    position = 0
    for begin, end, _ in tensors:
        if begin != position:
            raise ManifestError("safetensors data ranges overlap or contain holes")
        position = end
    if position != data_length:
        raise ManifestError("safetensors tensor ranges do not cover the source payload")
    return metadata, [name for _, _, name in tensors]


def repack(
    source,
    transform,
    destination: Path | None,
    preserve: Path | None,
    *,
    endpoint=None,
    token=False,
):
    """Hashes final bytes, writing them only when the cache needs a new result."""
    if preserve is not None:
        preserve.mkdir(exist_ok=True)
    preserved = []

    def copy_range(reader, offset, length, output):
        with ExitStack() as stack:
            digest = hashlib.sha256()
            retained = (
                stack.enter_context((preserve / str(len(preserved))).open("wb"))
                if preserve is not None
                else None
            )
            for chunk in reader.chunks(offset, length):
                output.write(chunk)
                if retained is not None:
                    retained.write(chunk)
                    digest.update(chunk)
            if retained is not None:
                retained.flush()
                os.fsync(retained.fileno())
                preserved.append(
                    {
                        "offset": offset,
                        "length": length,
                        "sha256": digest.hexdigest(),
                        "path": str(len(preserved)),
                    }
                )

    def read_range(reader, offset, length):
        content = io.BytesIO()
        copy_range(reader, offset, length, content)
        return content.getvalue()

    with RangeReader(source, endpoint=endpoint, token=token) as reader:
        header_length = struct.unpack("<Q", read_range(reader, 0, 8))[0]
        if header_length > _HEADER_LIMIT or 8 + header_length > reader.size:
            raise FetchError(
                "safetensors header exceeds its source or the 16 MiB limit"
            )
        base = 8 + header_length
        metadata, order = _header(
            read_range(reader, 8, header_length), reader.size - base
        )
        if "tensors" in transform:
            missing = set(transform["tensors"]) - (metadata.keys() - {"__metadata__"})
            if missing:
                raise ManifestError(
                    "safetensors source is missing tensors: "
                    + ", ".join(sorted(missing))
                )
            order = sorted(
                transform["tensors"],
                key=lambda name: (-DTYPE_BITS[metadata[name]["dtype"]], name),
            )
        result = {}
        if "__metadata__" in metadata:
            result["__metadata__"] = metadata["__metadata__"]
        transfers = []
        position = 0
        for name in order:
            tensor = metadata[name]
            begin, end = tensor["data_offsets"]
            result[name] = {
                **tensor,
                "data_offsets": [position, position + end - begin],
            }
            transfers.append((base + begin, end - begin))
            position += end - begin
        if "tensors" not in transform:
            # Preserve original payload order so a full checkpoint is one streaming
            # request, with no tensor-sized buffers or second local hashing pass.
            transfers = [(base, reader.size - base)]
        encoded = json.dumps(
            result, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        encoded += b" " * (-(8 + len(encoded)) % transform["alignment"])
        if len(encoded) > _HEADER_LIMIT:
            raise FetchError("aligned safetensors header exceeds the 16 MiB limit")
        digest = hashlib.sha256()
        with (
            destination.open("wb")
            if destination is not None
            else nullcontext() as output
        ):

            class Sink:
                def write(self, chunk):
                    digest.update(chunk)
                    if output is not None:
                        output.write(chunk)

            sink = Sink()
            sink.write(struct.pack("<Q", len(encoded)))
            sink.write(encoded)
            for offset, length in transfers:
                copy_range(reader, offset, length, sink)
            if output is not None:
                output.flush()
                os.fsync(output.fileno())
        return {
            "size": 8 + len(encoded) + position,
            "sha256": digest.hexdigest(),
            "source_size": reader.size,
            "ranges": preserved,
        }
