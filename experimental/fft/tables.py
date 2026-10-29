# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Constructs immutable FP32 tables for the experimental real FFT helpers."""

import argparse
import math
import struct
from pathlib import Path


def validate_length(length: int) -> None:
    """Validates the radix family before allocating or constructing tables."""
    remaining = length
    if remaining <= 0 or remaining % 2:
        raise ValueError("real FFT length must be positive and even")
    for factor in (2, 5):
        while remaining % factor == 0:
            remaining //= factor
    if remaining != 1:
        raise ValueError("real FFT length must have only factors two and five")


def real_table(length: int) -> list[float]:
    """Returns stage-major twiddles for the frame-parallel real butterflies."""
    validate_length(length)
    result = []
    position = 1
    while position < length:
        remaining = length // position
        radix = 5 if remaining % 5 == 0 else 4 if remaining % 4 == 0 else 2
        groups = length // (position * radix)
        for component in range(1, radix):
            values = [0.0] * (position - 1)
            for frequency in range(1, (position - 1) // 2 + 1):
                phase = math.tau * component * groups * frequency / length
                values[2 * frequency - 2] = math.cos(phase)
                values[2 * frequency - 1] = math.sin(phase)
            result.extend(values)
        position *= radix
    return result


def frequency_table(length: int) -> list[float]:
    """Returns complex Stockham and real reconstruction twiddles for one frame."""
    validate_length(length)
    half = length // 2
    result = []
    previous = 1
    while previous < half:
        remaining = half // previous
        radix = 4 if remaining % 4 == 0 else 5 if remaining % 5 == 0 else 2
        for component in range(1, radix):
            phases = [
                -math.tau * component * position / (previous * radix)
                for position in range(previous)
            ]
            result.extend(map(math.cos, phases))
            result.extend(map(math.sin, phases))
        previous *= radix
    phases = [-math.tau * frequency / length for frequency in range(half + 1)]
    result.extend(map(math.cos, phases))
    result.extend(map(math.sin, phases))
    return result


def transform_table(length: int) -> list[float]:
    """Returns the combined frame-packet and single-frame table in buffer order."""
    return real_table(length) + frequency_table(length)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("length", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frequency-only", action="store_true")
    args = parser.parse_args()
    try:
        values = (
            frequency_table(args.length)
            if args.frequency_only
            else transform_table(args.length)
        )
    except ValueError as error:
        parser.error(str(error))
    args.output.write_bytes(struct.pack(f"<{len(values)}f", *values))


if __name__ == "__main__":
    main()
