# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Independent integer oracles for word and wide shifts in streamed native packets."""

import random
import struct
import sys
from pathlib import Path


def main():
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    mask = (1 << 64) - 1
    edges = (0, 1, (1 << 31) - 1, 1 << 31, (1 << 32) - 1, 1 << 32, 1 << 63, mask)
    generator = random.Random(0xA1E2_0064)
    inputs = []
    expected = []
    for packet in range(16):
        for count in range(64):
            value = edges[packet] if packet < len(edges) else generator.getrandbits(64)
            low_count = count % 31 + 1
            high_count = count % 32 + 32
            low_zero_count = count & 31
            crossing_count = 31 + (count & 1)
            inputs.extend(
                (value, count, low_count, high_count, low_zero_count, crossing_count)
            )
            expected.extend(
                (value << amount) & mask
                for amount in (count, low_count, high_count, 0, 6, 31, 32, 63)
            )
            expected.append((value << 6) & 0xFFFFFFFF)
            word = value & 0xFFFFFFFF
            signed_word = word if word < (1 << 31) else word - (1 << 32)
            for amount in (count & 31, 1, 16, 31):
                expected.extend((word >> amount, (signed_word >> amount) & 0xFFFFFFFF))
            signed_value = value if value < (1 << 63) else value - (1 << 64)
            for amount in (
                count,
                low_count,
                high_count,
                0,
                6,
                31,
                32,
                63,
                low_zero_count,
                crossing_count,
            ):
                expected.extend((value >> amount, (signed_value >> amount) & mask))
    # Binding tails expose DMA writes beyond the declared pipeline views.
    guard = bytes([0xA5]) * 64
    (directory / "input.bin").write_bytes(
        struct.pack(f"<{len(inputs)}Q", *inputs) + guard
    )
    (directory / "output.bin").write_bytes(bytes([0xA5]) * (len(expected) * 8) + guard)
    (directory / "expected.bin").write_bytes(
        struct.pack(f"<{len(expected)}Q", *expected) + guard
    )


if __name__ == "__main__":
    main()
