# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Independent bit and arithmetic oracles for streamed BF16 vector operations."""

import random
import struct
import sys
from pathlib import Path


def main():
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    generator = random.Random(0xBF16)
    inputs = bytearray()
    expected = bytearray()

    def bf16(value):
        return struct.unpack("<I", struct.pack("<f", value))[0] >> 16

    def pack16(values):
        return struct.pack("<8H", *values)

    edge_bits = [0, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0xFFA5, 0x0001, 0x8001]
    # Finite arithmetic is exact; sign operations also cover arbitrary BF16 bits.
    for record in range(64):
        table = [generator.randrange(-64, 65) / 4 for _ in range(8)]
        rhs = [generator.randrange(-64, 65) / 4 for _ in range(8)]
        codes = list(range(8))
        generator.shuffle(codes)
        bits = (
            edge_bits[record % 8 :] + edge_bits[: record % 8]
            if record < 8
            else [generator.randrange(65536) for _ in range(8)]
        )
        initial = [generator.randrange(-256, 256) / 16 for _ in range(4)]
        table_bits = [bf16(v) for v in table]
        rhs_bits = [bf16(v) for v in rhs]
        inputs += (
            pack16(table_bits)
            + pack16(rhs_bits)
            + pack16(codes)
            + pack16(bits)
            + struct.pack("<4f", *initial)
            + bytes([0xCD]) * 48
        )
        decoded = [table[c] for c in codes]
        decoded_bits = [table_bits[c] for c in codes]
        for values in (
            decoded_bits,
            [v ^ 0x8000 for v in bits],
            [v & 0x7FFF for v in bits],
            [(a & 0x7FFF) | (b & 0x8000) for a, b in zip(bits, rhs_bits)],
            decoded_bits[:5] + [table_bits[7]] + decoded_bits[6:],
            [table_bits[7]] * 8,
        ):
            expected += pack16(values)
        dots = [
            sum(decoded[i + j] * rhs[i + j] for j in range(2)) for i in range(0, 8, 2)
        ]
        expected += struct.pack("<4f", *dots) + struct.pack(
            "<4f", *[a + b for a, b in zip(dots, initial)]
        )
        expected += struct.pack("<64f", *[a * b + 0.0 for a in decoded for b in rhs])
    guard = bytes([0xA5]) * 64
    (directory / "input.bin").write_bytes(inputs + guard)
    (directory / "output.bin").write_bytes(bytes([0xA5]) * len(expected) + guard)
    (directory / "expected.bin").write_bytes(expected + guard)


if __name__ == "__main__":
    main()
