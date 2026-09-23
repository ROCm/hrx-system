# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Independent Boolean oracle for packed predicate-valued selection."""

import sys
from pathlib import Path


def main():
    directory = Path(sys.argv[1])
    for name, count in (("full", 64), ("partial", 9)):
        case_directory = directory / name
        case_directory.mkdir(parents=True, exist_ok=True)
        inputs = bytearray()
        expected = bytearray()
        for record in range(16):
            source = bytearray([0x7E] * 256)
            result = bytearray([0xA5] * 128)
            for lane in range(64):
                # Every lane sees all eight independent Boolean triples. The
                # high half has different patterns to expose swapped halves.
                bits = (lane + record + 3 * (lane // 32)) % 8
                source[lane] = (bits >> 2) & 1
                source[64 + lane] = (bits >> 1) & 1
                source[128 + lane] = bits & 1
            for lane in range(count):
                result[lane] = source[64 + lane] if source[lane] else source[128 + lane]
            inputs += source
            expected += result
        input_guard = bytes([0x79]) * 64
        output_guard = bytes([0x37]) * 64
        (case_directory / "input.bin").write_bytes(inputs + input_guard)
        (case_directory / "output.bin").write_bytes(
            bytes([0xCD]) * len(expected) + output_guard
        )
        (case_directory / "expected.bin").write_bytes(expected + output_guard)


if __name__ == "__main__":
    main()
