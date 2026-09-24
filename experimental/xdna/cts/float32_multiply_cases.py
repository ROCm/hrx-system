# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Binary32 multiplication oracle for streamed native vector workers."""

import math
import random
import struct
import sys
from pathlib import Path

_GUARD = bytes([0xA5]) * 64


def product_bits(left, right):
    # Two binary32 significands multiply exactly in binary64, including the
    # binary32 subnormal range. Packing performs the one required rounding.
    left_value = struct.unpack("<f", struct.pack("<I", left))[0]
    right_value = struct.unpack("<f", struct.pack("<I", right))[0]
    product = left_value * right_value
    try:
        encoded = struct.pack("<f", product)
    except OverflowError:
        encoded = struct.pack("<f", math.copysign(math.inf, product))
    return struct.unpack("<I", encoded)[0]


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    magnitudes = (
        0x00000000,
        0x00000001,
        0x007FFFFF,
        0x00800000,
        0x00800001,
        0x3F000000,
        0x3F000001,
        0x3F800000,
        0x3F800001,
        0x3FBFFFFF,
        0x3FC00000,
        0x40000000,
        0x7F7FFFFF,
        0x7F800000,
        0x7F800001,
        0x7FC00000,
    )
    edges = tuple(value | sign for sign in (0, 0x80000000) for value in magnitudes)
    pairs = [(left, right) for left in edges for right in edges]
    generator = random.Random(0xA1E2F032)
    pairs.extend(
        (generator.getrandbits(32), generator.getrandbits(32))
        for _ in range(8192 - len(pairs))
    )
    inputs = []
    for begin in range(0, len(pairs), 16):
        packet = pairs[begin : begin + 16]
        inputs.extend(left for left, _ in packet)
        inputs.extend(right for _, right in packet)
    expected = [product_bits(left, right) for left, right in pairs]
    (directory / "input.bin").write_bytes(
        struct.pack(f"<{len(inputs)}I", *inputs) + _GUARD
    )
    (directory / "output.bin").write_bytes(bytes([0xA5]) * (len(expected) * 4) + _GUARD)
    (directory / "expected.bin").write_bytes(
        struct.pack(f"<{len(expected)}I", *expected) + _GUARD
    )


def verify(directory, output_path):
    expected = (directory / "expected.bin").read_bytes()
    actual = output_path.read_bytes()
    if len(actual) != len(expected) or actual[-len(_GUARD) :] != _GUARD:
        raise ValueError("output length or binding guard changed")
    expected_values = struct.unpack(
        f"<{(len(expected) - len(_GUARD)) // 4}I", expected[: -len(_GUARD)]
    )
    actual_values = struct.unpack(f"<{len(expected_values)}I", actual[: -len(_GUARD)])
    for lane, (reference, result) in enumerate(zip(expected_values, actual_values)):
        # IEEE multiplication does not prescribe a NaN payload or sign.
        reference_nan = (reference & 0x7FFFFFFF) > 0x7F800000
        result_nan = (result & 0x7FFFFFFF) > 0x7F800000
        if reference != result and not (reference_nan and result_nan):
            raise ValueError(f"lane {lane}: expected {reference:08x}, got {result:08x}")


def main():
    action, directory = sys.argv[1:3]
    if action == "prepare":
        prepare(Path(directory))
    elif action == "verify":
        verify(Path(directory), Path(sys.argv[3]))
    else:
        raise ValueError(f"unknown action: {action}")


if __name__ == "__main__":
    main()
