# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bit-exact independent oracles for packed register-table lookup."""

import random
import sys
from pathlib import Path

# Payload width, table entries, and index width for each native entry.
_CASES = {
    "i8_odd": (8, 3, 8),
    "i8_full": (8, 64, 8),
    "f16_odd": (16, 3, 16),
    "bf16_odd": (16, 5, 8),
    "i16_full": (16, 32, 16),
    "f32_odd": (32, 7, 32),
    "f32_full": (32, 16, 8),
    "fp8_e4m3": (8, 16, 8),
    "fp8_e5m2": (8, 16, 8),
    "f64_uniform": (64, 8, 8),
}


def main():
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    name = sys.argv[2]
    width, table_count, index_width = _CASES[name]
    lane_count = 512 // width
    element_bytes = width // 8
    generator = random.Random(0x7AB1E + width + table_count)
    inputs = bytearray()
    expected = bytearray()
    mask = (1 << width) - 1
    # Include both signs, all-ones, and single-bit payloads, then random bits.
    edges = [0, mask, 1 << (width - 1), (1 << (width - 1)) - 1]
    edges += [1 << bit for bit in range(width)]
    for record in range(64):
        table = [
            edges[(record + entry) % len(edges)]
            if record < len(edges)
            else generator.getrandbits(width)
            for entry in range(table_count)
        ]
        indices = [(record + lane) % table_count for lane in range(lane_count)]
        generator.shuffle(indices)
        table_data = b"".join(
            value.to_bytes(element_bytes, "little") for value in table
        )
        code_data = b"".join(
            value.to_bytes(index_width // 8, "little") for value in indices
        )
        inputs += table_data + bytes([0xCD]) * (64 - len(table_data))
        inputs += code_data + bytes([0xDC]) * (64 - len(code_data))
        selected = (
            [table[0]] * lane_count
            if name == "f64_uniform"
            else [table[index] for index in indices]
        )
        expected += b"".join(
            value.to_bytes(element_bytes, "little") for value in selected
        )
        expected += table[-1].to_bytes(element_bytes, "little") * lane_count
    guard = bytes([0xA5]) * 64
    (directory / "input.bin").write_bytes(inputs + guard)
    (directory / "output.bin").write_bytes(bytes([0xA5]) * len(expected) + guard)
    (directory / "expected.bin").write_bytes(expected + guard)


if __name__ == "__main__":
    main()
