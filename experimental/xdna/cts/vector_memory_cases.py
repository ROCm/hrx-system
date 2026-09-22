# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Independent raw-byte oracle for native 64-bit vector memory packets."""

import random
import sys
from pathlib import Path


def main():
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    generator = random.Random(0x640064)
    edges = [
        0,
        2**64 - 1,
        0x8000000000000000,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000000001,
        0xFFF0000000000001,
    ]
    edges += [1 << bit for bit in range(64)]
    inputs = bytearray()
    expected = bytearray()
    # The final two packets start at a 16-byte offset and use split loads.
    sizes = (16, 16, 32, 32, 64, 64, 32, 32)
    for record in range(80):
        words = [
            edges[(record + lane) % len(edges)]
            if record < len(edges)
            else generator.getrandbits(64)
            for lane in range(64)
        ]
        data = b"".join(word.to_bytes(8, "little") for word in words)
        result = bytearray([0xA5] * 512)
        for slot, size in enumerate(sizes):
            source = slot * 64 + (16 if slot >= 6 else 0)
            destination = slot * 64
            result[destination : destination + size] = data[source : source + size]
        inputs += data
        expected += result
    guard = bytes([0xA5]) * 64
    (directory / "input.bin").write_bytes(inputs + guard)
    (directory / "output.bin").write_bytes(bytes([0xA5]) * len(expected) + guard)
    (directory / "expected.bin").write_bytes(expected + guard)


if __name__ == "__main__":
    main()
