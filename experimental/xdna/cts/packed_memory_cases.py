# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Independent byte oracles for imported packed fields on native XDNA."""

import struct
import sys
from pathlib import Path


def main():
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    inputs = bytearray()
    expected = bytearray()
    for packet in range(6):
        source = bytes((position + packet * 41) & 255 for position in range(256))
        result = bytearray([37]) * 256
        # Eight 15-byte records copy a contiguous body between distinct origins.
        result[5:125] = source[1:121]
        # Nested packed updates leave the child's internal and tail padding alone.
        struct.pack_into("<f", result, 29, 2.5)
        result[33] = 0
        struct.pack_into("<I", result, 34, 0x40000001)
        inputs.extend(source)
        expected.extend(result)
    guard = bytes([0xA5]) * 64
    (directory / "input.bin").write_bytes(inputs)
    (directory / "output.bin").write_bytes(bytes([0xA5]) * len(expected) + guard)
    (directory / "expected.bin").write_bytes(expected + guard)


if __name__ == "__main__":
    main()
