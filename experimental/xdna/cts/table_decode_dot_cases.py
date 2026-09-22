# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Independent signed-integer oracle for streamed nonlinear decode and dot."""

import random
import struct
import sys
from pathlib import Path


def main():
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    generator = random.Random(73091)
    inputs = bytearray()
    expected = bytearray()
    for record in range(16):
        table = generator.sample(range(-128, 128), 16)
        codes = list(range(16)) * 4
        generator.shuffle(codes)
        activation = [generator.randrange(-128, 128) for _ in range(64)]
        inputs.extend(struct.pack("<16b", *table) + bytes(48))
        inputs.extend(bytes(codes))
        inputs.extend(struct.pack("<64b", *activation))
        decoded = [table[index] for index in codes]
        expected.extend(struct.pack("<64b", *decoded))
        partial = [
            sum(decoded[begin + j] * activation[begin + j] for j in range(4))
            for begin in range(0, 64, 4)
        ]
        expected.extend(struct.pack("<16i", *partial))
    guard = bytes([0xA5]) * 64
    (directory / "input.bin").write_bytes(inputs + guard)
    (directory / "output.bin").write_bytes(bytes([0xCD]) * len(expected) + guard)
    (directory / "expected.bin").write_bytes(expected + guard)


if __name__ == "__main__":
    main()
