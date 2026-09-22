# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Raw-byte oracle for captured vector slices and exact partial stores."""

import random
import sys
from pathlib import Path


def main():
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    if sys.argv[2] == "wide":
        first = int(sys.argv[3])
        input_size = 128
        output_size = 1024
        copies = [
            (offset, slot * 64, min(64, 128 - offset))
            for slot, offset in enumerate(range(first, first + 16))
        ]
    else:
        element_size = int(sys.argv[2])
        lanes = 64 // element_size
        shapes = (
            (0, lanes // 2 + 1),
            (1, lanes - 1),
            (lanes // 4, lanes // 2),
            (lanes - 1, 1),
        )
        input_size = output_size = 256
        copies = [
            (slot * 64 + offset * element_size, slot * 64, count * element_size)
            for slot, (offset, count) in enumerate(shapes)
        ]
    generator = random.Random(0x511CE)
    inputs = bytearray()
    expected = bytearray()
    for record in range(16):
        data = (
            bytes((record * input_size + i) % 256 for i in range(input_size))
            if record < 2
            else generator.randbytes(input_size)
        )
        result = bytearray([0xA5] * output_size)
        for source, destination, length in copies:
            result[destination : destination + length] = data[source : source + length]
        inputs += data
        expected += result
    guard = bytes([0xA5]) * 64
    (directory / "input.bin").write_bytes(inputs + guard)
    (directory / "output.bin").write_bytes(bytes([0xA5]) * len(expected) + guard)
    (directory / "expected.bin").write_bytes(expected + guard)


if __name__ == "__main__":
    main()
