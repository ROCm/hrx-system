# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Independent IEEE comparison oracles over raw floating-point encodings."""

import argparse
import itertools
import math
import random
import struct
from pathlib import Path


def decode(bits, element):
    if element == "f16":
        return struct.unpack("<e", struct.pack("<H", bits))[0]
    if element == "bf16":
        bits <<= 16
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def comparison_flags(lhs, rhs):
    unordered = math.isnan(lhs) or math.isnan(rhs)
    relations = (lhs == rhs, lhs > rhs, lhs >= rhs, lhs < rhs, lhs <= rhs, lhs != rhs)
    predicates = tuple(not unordered and value for value in relations)
    predicates += (not unordered,)
    predicates += tuple(unordered or value for value in relations)
    predicates += (unordered,)
    return sum(int(value) << index for index, value in enumerate(predicates))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("element", choices=("f16", "bf16", "f32"))
    arguments = parser.parse_args()
    element = arguments.element
    if element == "f16":
        sign, normal, one, infinity = 0x8000, 0x400, 0x3C00, 0x7C00
    elif element == "bf16":
        sign, normal, one, infinity = 0x8000, 0x80, 0x3F80, 0x7F80
    else:
        sign, normal, one, infinity = 0x80000000, 0x800000, 0x3F800000, 0x7F800000
    magnitudes = (
        0,
        1,
        2,
        3,
        normal - 2,
        normal - 1,
        normal,
        normal + 1,
        one - normal,
        one,
        one + normal,
        infinity - 1,
        infinity,
        infinity + 1,
        infinity + normal // 2,
        sign - 1,
    )
    edges = magnitudes + tuple(sign | value for value in magnitudes)
    if element == "f32":
        generator = random.Random(0xC04F)
        pairs = [
            (generator.getrandbits(32), generator.getrandbits(32)) for _ in range(8192)
        ]
        pairs += [(value, value) for value in edges]
        lane_count, encoding = 16, "I"
    else:
        # Both operands visit every encoding, including all signaling NaNs.
        # Equal pairs cover classification; a permutation covers numeric order.
        pairs = [(value, value) for value in range(65536)]
        pairs += [(value, (value * 25173 + 13849) & 0xFFFF) for value in range(65536)]
        lane_count, encoding = 32, "H"
    pairs += list(itertools.product(edges, repeat=2))

    inputs = bytearray()
    expected = bytearray()
    for offset in range(0, len(pairs), lane_count):
        packet = pairs[offset : offset + lane_count]
        inputs += struct.pack(f"<{lane_count}{encoding}", *(lhs for lhs, _ in packet))
        inputs += struct.pack(f"<{lane_count}{encoding}", *(rhs for _, rhs in packet))
        flags = [
            comparison_flags(decode(lhs, element), decode(rhs, element))
            for lhs, rhs in packet
        ]
        expected += struct.pack(f"<{lane_count}{encoding}", *flags)
    directory = arguments.directory
    directory.mkdir(parents=True, exist_ok=True)
    guard = bytes([0xA5]) * 64
    (directory / "input.bin").write_bytes(inputs + guard)
    (directory / "output.bin").write_bytes(bytes([0xCD]) * len(expected) + guard)
    (directory / "expected.bin").write_bytes(expected + guard)


if __name__ == "__main__":
    main()
