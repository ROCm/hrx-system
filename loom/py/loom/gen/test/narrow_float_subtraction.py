# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Narrow subtraction checked against exact integer arithmetic and IEEE rounding."""

import argparse
import math
from pathlib import Path

from loom.gen.test.kernel_fixture import Arrays, Case, signed_bits


def subtract(lhs, rhs, fraction_bits, bias):
    """Subtract in units of the least subnormal, then round once to nearest-even."""
    infinity = ((1 << (15 - fraction_bits)) - 1) << fraction_bits
    left_magnitude, right_magnitude = lhs & 0x7FFF, rhs & 0x7FFF
    if left_magnitude > infinity or right_magnitude > infinity:
        return math.nan
    if left_magnitude == infinity and right_magnitude == infinity:
        return math.nan if (lhs ^ rhs) & 0x8000 == 0 else math.copysign(math.inf, -1 if lhs & 0x8000 else 1)
    if left_magnitude == infinity:
        return -math.inf if lhs & 0x8000 else math.inf
    if right_magnitude == infinity:
        return math.inf if rhs & 0x8000 else -math.inf

    def units(bits):
        exponent = (bits & 0x7FFF) >> fraction_bits
        significand = bits & ((1 << fraction_bits) - 1)
        if exponent:
            significand |= 1 << fraction_bits
        magnitude = significand << max(exponent - 1, 0)
        return -magnitude if bits & 0x8000 else magnitude

    exact = units(lhs) - units(rhs)
    if exact == 0:
        return -0.0 if lhs == 0x8000 and rhs == 0 else 0.0
    magnitude = abs(exact)
    shift = max(0, magnitude.bit_length() - fraction_bits - 1)
    rounded, remainder = divmod(magnitude, 1 << shift)
    divisor = 1 << shift
    if remainder * 2 > divisor or (remainder * 2 == divisor and rounded & 1):
        rounded += 1
    result = math.ldexp(float(rounded), shift + 1 - bias - fraction_bits)
    # Rounding the maximum finite significand upward produces infinity.
    if result >= math.ldexp(1.0, bias + 1):
        result = math.inf
    return -result if exact < 0 else result


def inputs(fraction_bits, bias):
    infinity = ((1 << (15 - fraction_bits)) - 1) << fraction_bits
    one = bias << fraction_bits
    positive = [
        0,
        1,
        2,
        (1 << fraction_bits) - 1,
        1 << fraction_bits,
        (1 << fraction_bits) + 1,
        one - 1,
        one,
        one + 1,
        one + 2,
        infinity - 1,
        infinity,
        infinity + 1,
        infinity + (1 << (fraction_bits - 1)),
    ]
    edges = positive + [bits | 0x8000 for bits in positive]
    pairs = [(lhs, rhs) for lhs in edges for rhs in edges]
    # Each operand visits every 16-bit payload; the odd multiplier permutes RHS.
    pairs.extend((bits, (bits * 40503 + 17) & 0xFFFF) for bits in range(65536))
    return [(lhs, rhs, subtract(lhs, rhs, fraction_bits, bias)) for lhs, rhs in pairs]


def checks(arrays, element, width, rows, nan):
    rows = [row for row in rows if math.isnan(row[2]) == nan]
    rows += [rows[0]] * (-len(rows) % width)
    count = len(rows)
    name = f"subtract_{element}_{width}"
    case = Case(arrays, name + ("_nan" if nan else "_bits"), "f32", count)
    left = [signed_bits(row[0], 16) for row in rows]
    right = [signed_bits(row[1], 16) for row in rows]
    case.array("lhs", left, "i16")
    case.array("rhs", right, "i16")
    case.scalar("count", count, "i32")
    case.launch(name, "%lhs, %rhs, %output, %count", f"tensor<{count}xi16>, tensor<{count}xi16>, tensor<{count}xf32>, i32")
    for argument, expected in (("lhs", left), ("rhs", right)):
        case.array(argument + "_expected", expected, "i16")
        case.lines.append(f"  check.expect.bitwise actual(%{argument}) expected(%{argument}_expected) : tensor<{count}xi16>")
    return case.finish([row[2] for row in rows], tolerance=0.0 if nan else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    options = parser.parse_args()
    arrays = Arrays(options.arrays, options.output.parent)
    cases = []
    for element, fraction_bits, bias in (("f16", 10, 15), ("bf16", 7, 127)):
        rows = inputs(fraction_bits, bias)
        cases.extend(f"kernel.decl @subtract_{element}_{width}() launch(%lhs: buffer, %rhs: buffer, %output: buffer, %count: i32)\n" for width in (1, 8))
        cases.extend(checks(arrays, element, width, rows, nan) for width in (1, 8) for nan in (False, True))
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text("\n".join(cases))


if __name__ == "__main__":
    main()
