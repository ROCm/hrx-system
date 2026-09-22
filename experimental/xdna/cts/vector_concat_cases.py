# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Independent byte-slicing oracle for packed vector concatenation."""

import random
import sys
from pathlib import Path


def main():
    directory = Path(sys.argv[1])
    name = sys.argv[2]
    directory.mkdir(parents=True, exist_ok=True)
    if name.startswith("bytes_"):
        _, first, last = name.split("_")
        shapes = [(left, 64 - left) for left in range(int(first), int(last) + 1)]
        slot_bytes = 64
    else:
        byte_count = {
            "i8": 1,
            "f8E4M3": 1,
            "f8E5M2": 1,
            "i16": 2,
            "f16": 2,
            "bf16": 2,
            "i32": 4,
            "f32": 4,
            "i64": 8,
            "f64": 8,
        }[name]
        shapes = [
            (byte_count, 64 - byte_count),
            (byte_count, byte_count),
            (32, byte_count),
            (64, byte_count),
            (64, 64 - byte_count),
        ]
        if name != "f32":
            shapes.append((64, 64))
        slot_bytes = 128
    generator = random.Random(0xC04CA7)
    inputs = bytearray()
    expected = bytearray()
    edges = (
        0,
        2**64 - 1,
        1,
        0x8000000000000000,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000000001,
        0xFFF0000000000001,
        0x7FC0000180000000,
        0x7C017F817E017FC1,
    )
    for record in range(16):
        if record < 2:
            data = bytes((position + record * 128) & 255 for position in range(128))
        elif record < 12:
            data = b"".join(
                edges[(record + word) % len(edges)].to_bytes(8, "little")
                for word in range(16)
            )
        else:
            data = generator.randbytes(128)
        result = bytearray([0xA5] * (slot_bytes * len(shapes)))
        for slot, (left_bytes, right_bytes) in enumerate(shapes):
            payload = data[:left_bytes] + data[64 : 64 + right_bytes]
            offset = slot * slot_bytes
            result[offset : offset + len(payload)] = payload
        inputs += data
        expected += result
    guard = bytes([0xA5]) * 64
    (directory / "input.bin").write_bytes(inputs + guard)
    (directory / "output.bin").write_bytes(bytes([0xA5]) * len(expected) + guard)
    (directory / "expected.bin").write_bytes(expected + guard)


if __name__ == "__main__":
    main()
