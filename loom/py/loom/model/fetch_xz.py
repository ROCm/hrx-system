# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stream an XZ source into the installed file's byte identity."""

from __future__ import annotations

import hashlib
import os
from contextlib import ExitStack
from pathlib import Path

from .fetch_progress import ByteProgress
from .fetch_source import FetchError, Source, open_source

_CHUNK_SIZE = 1024 * 1024
# Includes the 64 MiB dictionary of preset 9 and liblzma's decoder state.
_DECODER_MEMORY_LIMIT = 128 * 1024 * 1024


def _decode(chunks, expected_size):
    try:
        import lzma
    except ImportError:
        raise FetchError(
            "this Python installation lacks the standard lzma module needed for "
            "XZ files; use a Python installation built with liblzma support"
        ) from None

    def start_stream():
        return lzma.LZMADecompressor(lzma.FORMAT_XZ, memlimit=_DECODER_MEMORY_LIMIT)

    decoder = start_stream()
    padding = 0
    size = 0
    try:
        for chunk in chunks:
            while chunk or not decoder.needs_input:
                if decoder.eof:
                    # XZ permits concatenated streams with four-byte-aligned
                    # zero padding, including after the final stream.
                    data = chunk.lstrip(b"\0")
                    padding += len(chunk) - len(data)
                    if not data:
                        break
                    if padding % 4:
                        raise FetchError("invalid XZ stream padding")
                    padding = 0
                    chunk = data
                    decoder = start_stream()
                limit = _CHUNK_SIZE
                if expected_size is not None:
                    limit = min(limit, expected_size - size + 1)
                block = decoder.decompress(chunk, max_length=limit)
                chunk = decoder.unused_data
                size += len(block)
                if expected_size is not None and size > expected_size:
                    raise FetchError("XZ output exceeds its locked byte count")
                if block:
                    yield block
    except lzma.LZMAError as error:
        raise FetchError(f"cannot decode XZ source: {error}") from None
    if not decoder.eof:
        raise FetchError("truncated XZ source")
    if padding % 4:
        raise FetchError("invalid XZ stream padding")


def unpack_xz(
    source: Source,
    destination: Path | None,
    preserve: Path | None,
    expected_size: int | None,
    *,
    progress=None,
):
    """Hashes decoded bytes during the write and optionally retains the source."""
    received = 0
    retained = None
    source_digest = None
    size = 0
    digest = hashlib.sha256()
    counter = ByteProgress(progress, "transform")
    with ExitStack() as stack:
        stream, total = stack.enter_context(open_source(source, offline=False))
        output = stack.enter_context(destination.open("wb")) if destination else None
        if preserve:
            preserve.mkdir()
            retained = stack.enter_context((preserve / "0").open("wb"))
            source_digest = hashlib.sha256()

        def chunks():
            nonlocal received
            counter(0, total)
            for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
                received += len(chunk)
                if total is not None and received > total:
                    raise FetchError("XZ source exceeded its declared byte count")
                if retained is not None:
                    retained.write(chunk)
                    source_digest.update(chunk)
                counter(received, total)
                yield chunk
            if total is not None and received != total:
                raise FetchError("XZ transfer ended before its declared byte count")
            counter(received, received)

        for block in _decode(chunks(), expected_size):
            size += len(block)
            digest.update(block)
            if output is not None:
                output.write(block)
        for file in (output, retained):
            if file is not None:
                file.flush()
                os.fsync(file.fileno())
    ranges = []
    if source_digest is not None:
        ranges.append(
            {
                "offset": 0,
                "length": received,
                "sha256": source_digest.hexdigest(),
                "path": "0",
            }
        )
    return {
        "size": size,
        "sha256": digest.hexdigest(),
        "source_size": received,
        "ranges": ranges,
    }
