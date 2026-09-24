# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Full-width vector bitwise truth pairs, constants, and scalar consumers."""

import argparse
from pathlib import Path

from loom.gen.test.kernel_fixture import Arrays, Case, signed_bits


def bitwise(arrays):
    lhs_values = []
    rhs_values = []
    expected = []
    for bit in range(64):
        mask = 1 << bit
        for truth in range(4):
            lhs_bits = (0xA57E19C368D20FB4 & ~mask) | ((truth & 1) << bit)
            rhs_bits = (0x39C6E0B58A47D12F & ~mask) | ((truth >> 1) << bit)
            lhs = [signed_bits(lhs_bits, 64), signed_bits(~lhs_bits, 64)]
            rhs = [signed_bits(rhs_bits, 64), signed_bits(~rhs_bits, 64)]
            lhs_values.extend(lhs)
            rhs_values.extend(rhs)
            conjunction = [a & b for a, b in zip(lhs, rhs, strict=True)]
            disjunction = [a | b for a, b in zip(lhs, rhs, strict=True)]
            difference = [a ^ b for a, b in zip(lhs, rhs, strict=True)]
            expected.extend(conjunction + disjunction + difference)
            expected.extend(value & 0xFFFFFFFF for value in lhs)
            expected.extend(value | -(1 << 63) | 1 for value in rhs)
            expected.extend(value ^ -0x100000000 for value in lhs)
            expected.extend([difference[0], disjunction[1], -123, -123])

    case = Case(arrays, "vector_i64_bitwise_values", "i64", len(expected))
    case.array("lhs", lhs_values)
    case.array("rhs", rhs_values)
    case.launch(
        "vector_i64_bitwise",
        "%lhs, %rhs, %output",
        "tensor<512xi64>, tensor<512xi64>, tensor<4096xi64>",
    )
    declaration = "kernel.decl @vector_i64_bitwise() launch(%lhs_input: buffer, %rhs_input: buffer, %output: buffer)\n\n"
    return declaration + case.finish(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    options = parser.parse_args()
    arrays = Arrays(options.arrays, options.output.parent)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(bitwise(arrays))


if __name__ == "__main__":
    main()
