# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Independent integer oracles for packetized memory observation points."""

import argparse
import random
import struct
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "case",
        choices=(
            "store",
            "reduce",
            "two_reads",
            "shared_volatile",
            "shared_volatile_reverse",
        ),
    )
    arguments = parser.parse_args()
    directory = arguments.directory
    directory.mkdir(parents=True, exist_ok=True)
    name = arguments.case
    generator = random.Random(0xC4A7)
    inputs = bytearray()
    expected = bytearray()
    for record in range(16):
        values = [
            ((lane + record * 128) << 24) & 0xFFFFFFFF
            if record < 2
            else generator.getrandbits(32)
            for lane in range(128)
        ]
        inputs += struct.pack("<128I", *values)
        if name == "reduce":
            result = [(sum(values) + 53) & 0xFFFFFFFF] * 128
        elif name == "two_reads":
            result = [(value + 1234567) & 0xFFFFFFFF for value in values[:16]]
            result += [0] * 112
        else:
            result = [(value + 7) & 0xFFFFFFFF for value in values]
            if name.startswith("shared_volatile"):
                result = values + result
        expected += struct.pack(f"<{len(result)}I", *result)
    guard = bytes([0xA5]) * 64
    (directory / "input.bin").write_bytes(inputs + guard)
    (directory / "output.bin").write_bytes(bytes([0xCD]) * len(expected) + guard)
    (directory / "expected.bin").write_bytes(expected + guard)


if __name__ == "__main__":
    main()
