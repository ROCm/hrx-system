# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks floating comparison legalization against numeric relations."""

import argparse
import itertools
import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Format, exponent bits, fraction bits, and shapes on both sides of a packet.
_FORMATS = (
    ("f8E4M3", 4, 3, ((4,), (2, 33))),
    ("f8E5M2", 5, 2, ((4,), (2, 33))),
    ("f16", 5, 10, ((4,), (3, 11))),
    ("bf16", 8, 7, ((4,), (3, 11))),
    ("f32", 8, 23, ((4,), (3, 7), (8, 8))),
    ("f64", 11, 52, ((4,), (3, 3))),
)
_PREDICATES = (
    "oeq",
    "ogt",
    "oge",
    "olt",
    "ole",
    "one",
    "ord",
    "ueq",
    "ugt",
    "uge",
    "ult",
    "ule",
    "une",
    "uno",
)


def _decode(bits, exponent_bits, fraction_bits, finite_only):
    sign = -1 if bits >> (exponent_bits + fraction_bits) else 1
    exponent_mask = (1 << exponent_bits) - 1
    exponent = (bits >> fraction_bits) & exponent_mask
    fraction = bits & ((1 << fraction_bits) - 1)
    if exponent == exponent_mask:
        if finite_only:
            if fraction == (1 << fraction_bits) - 1:
                return math.nan
        else:
            return math.nan if fraction else sign * math.inf
    bias = (1 << (exponent_bits - 1)) - 1
    if exponent == 0:
        return sign * math.ldexp(fraction, 1 - bias - fraction_bits)
    return sign * math.ldexp(
        (1 << fraction_bits) + fraction, exponent - bias - fraction_bits
    )


def _samples(exponent_bits, fraction_bits, finite_only):
    sign = 1 << (exponent_bits + fraction_bits)
    special = ((1 << exponent_bits) - 1) << fraction_bits
    one = ((1 << (exponent_bits - 1)) - 1) << fraction_bits
    maximum = special + (1 << fraction_bits) - 2 if finite_only else special - 1
    magnitudes = (
        0,
        1,
        (1 << fraction_bits) - 1,
        1 << fraction_bits,
        one - 1,
        one,
        one + 1,
        maximum,
        special,
        special + (1 << fraction_bits) - 1,
    )
    return sorted(
        {magnitude | negative for magnitude in magnitudes for negative in (0, sign)}
    )


def _relation(predicate, left, right):
    unordered = math.isnan(left) or math.isnan(right)
    if predicate == "ord":
        return not unordered
    if predicate == "uno":
        return unordered
    if unordered:
        return predicate.startswith("u")
    return {
        "eq": left == right,
        "ne": left != right,
        "lt": left < right,
        "le": left <= right,
        "gt": left > right,
        "ge": left >= right,
    }[predicate[1:]]


def _evaluate(body, left, right):
    """Evaluate the generic integer-vector IR produced by legalization."""
    values = {"%lhs": left, "%rhs": right}
    for line in body.strip().splitlines():
        line = line.strip()
        if line.startswith("func.return "):
            return values[line.split()[1]]
        destination, expression = line.split(" = ", 1)
        operation = expression.split()[0]
        inputs = re.findall(r"%[A-Za-z0-9_]+", expression.split(" : ", 1)[0])
        if operation == "vector.bitcast":
            values[destination] = values[inputs[0]]
            continue
        if operation == "vector.slice":
            shape = re.findall(r"vector<([0-9x]+)x[a-zA-Z0-9]+>", expression)[-1]
            count = math.prod(map(int, shape.split("x")))
            start = int(re.search(r"\[([0-9]+)\]", expression)[1])
            values[destination] = values[inputs[0]][start : start + count]
            continue
        types = re.findall(r"vector<([0-9x]+)xi([0-9]+)>", expression)
        shape, bit_count = types[-1]
        bit_count = int(bit_count)
        count = math.prod(map(int, shape.split("x")))
        mask = (1 << bit_count) - 1
        if operation == "vector.splati":
            result = [int(expression.split()[1]) & mask] * count
        elif operation.startswith("vector.concat<"):
            result = list(itertools.chain.from_iterable(values[v] for v in inputs))
        elif operation == "vector.select":
            result = [
                yes if condition else no
                for condition, yes, no in zip(*(values[v] for v in inputs), strict=True)
            ]
        elif operation == "vector.cmpi":
            predicate = expression.split()[1].rstrip(",")
            input_bits = int(types[0][1])
            sign = 1 << (input_bits - 1)
            result = []
            for lhs, rhs in zip(*(values[v] for v in inputs), strict=True):
                if predicate.startswith("s"):
                    lhs, rhs = (lhs ^ sign) - sign, (rhs ^ sign) - sign
                relation = predicate[1:] if predicate[0] in "su" else predicate
                result.append(
                    int(
                        {
                            "eq": lhs == rhs,
                            "ne": lhs != rhs,
                            "lt": lhs < rhs,
                            "le": lhs <= rhs,
                            "gt": lhs > rhs,
                            "ge": lhs >= rhs,
                        }[relation]
                    )
                )
        else:
            functions = {
                "vector.andi": lambda a, b: a & b,
                "vector.ori": lambda a, b: a | b,
                "vector.xori": lambda a, b: a ^ b,
                "vector.addi": lambda a, b: a + b,
                "vector.subi": lambda a, b: a - b,
            }
            if operation not in functions:
                raise AssertionError(f"Unexpected comparison operation: {line}")
            result = [
                functions[operation](a, b) & mask
                for a, b in zip(*(values[v] for v in inputs), strict=True)
            ]
        values[destination] = result
    raise AssertionError("Missing comparison return")


class FloatComparisonFamilyTest(unittest.TestCase):
    def test_numeric_relations(self):
        cases = []
        for name, exponent_bits, fraction_bits, shapes in _FORMATS:
            for dimensions, predicate, no_nan in itertools.product(
                shapes, _PREDICATES, (False, True)
            ):
                shape = "x".join(map(str, dimensions))
                symbol = f"compare_{name}_{shape}_{predicate}_{int(no_nan)}"
                value_type = f"vector<{shape}x{name}>"
                result_type = f"vector<{shape}xi1>"
                flags = "<nnan>" if no_nan else ""
                source = (
                    f"func.def @{symbol}(%lhs: {value_type}, %rhs: {value_type}) "
                    f"-> ({result_type}) {{\n"
                    f"  %result = vector.cmpf{flags} {predicate}, %lhs, %rhs : "
                    f"{value_type} -> {result_type}\n"
                    f"  func.return %result : {result_type}\n}}\n// ----\n"
                )
                cases.append(
                    (
                        symbol,
                        name,
                        exponent_bits,
                        fraction_bits,
                        math.prod(dimensions),
                        predicate,
                        no_nan,
                        source,
                    )
                )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "float_comparisons.loom-test"
            path.write_text(
                "// RUN: pass target=amd.xdna.aie2p:core "
                "target-legalize{mode=final}\n\n"
                + "\n// ====\n\n".join(case[-1] for case in cases),
                newline="\n",
            )
            result = subprocess.run(
                [_ARGS.checker, "--update", str(path)], capture_output=True, text=True
            )
            self.assertIn("updated", result.stdout + result.stderr)
            self.assertNotIn("error [", result.stdout + result.stderr)
            transformed = path.read_text()
        bodies = {}
        for section in transformed.split("// ===="):
            expected = section.split("// ----", 1)[1]
            match = re.search(
                r"func.def[^\n]*@(\w+)[^\n]*\{\n(.*?)^}", expected, re.M | re.S
            )
            self.assertIsNotNone(match, expected)
            bodies[match[1]] = match[2]
        self.assertEqual(len(bodies), len(cases))
        for (
            symbol,
            name,
            exponent_bits,
            fraction_bits,
            count,
            predicate,
            no_nan,
            _,
        ) in cases:
            with self.subTest(case=symbol):
                samples = _samples(exponent_bits, fraction_bits, name == "f8E4M3")
                decoded = {
                    bits: _decode(bits, exponent_bits, fraction_bits, name == "f8E4M3")
                    for bits in samples
                }
                if no_nan:
                    samples = [
                        bits for bits in samples if not math.isnan(decoded[bits])
                    ]
                pairs = list(itertools.product(samples, repeat=2))
                for begin in range(0, len(pairs), count):
                    packet = pairs[begin : begin + count]
                    packet += [packet[-1]] * (count - len(packet))
                    left, right = map(list, zip(*packet, strict=True))
                    expected = [
                        int(_relation(predicate, decoded[a], decoded[b]))
                        for a, b in packet
                    ]
                    self.assertEqual(
                        _evaluate(bodies[symbol], left, right),
                        expected,
                        f"{symbol}: operands {packet}",
                    )


_PARSER = argparse.ArgumentParser()
_PARSER.add_argument("checker")
_ARGS, _UNITTEST_ARGS = _PARSER.parse_known_args()

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *_UNITTEST_ARGS])
