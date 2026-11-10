# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks scalar lane order through the complete AIE2P constructor family."""

import argparse
import math
import re
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

# Maximum logical extent of the ordinary, predicate, and accumulator carriers.
# Logical rank does not change the maximum extent of a carrier family.
_ELEMENT_FAMILIES = (
    ("i1", 1, 128),
    ("i8", 8, 128),
    ("f8E4M3", 8, 128),
    ("f8E5M2", 8, 128),
    ("i16", 16, 64),
    ("f16", 16, 64),
    ("bf16", 16, 64),
    ("i32", 32, 64),
    ("index", 32, 64),
    ("offset", 32, 64),
    ("f32", 32, 64),
    ("i64", 64, 32),
    ("f64", 64, 16),
)


@dataclass(frozen=True)
class ConstructorCase:
    element_type: str
    element_bits: int
    dimensions: tuple[int, ...]

    @property
    def name(self) -> str:
        shape = "x".join(str(dimension) for dimension in self.dimensions)
        return f"construct_{self.element_type}_{shape}"

    @property
    def lane_order(self) -> list[int]:
        return [
            (lane * 5 + lane // 7) % 8 for lane in range(math.prod(self.dimensions))
        ]

    def source(self) -> str:
        shape = "x".join(str(dimension) for dimension in self.dimensions)
        result_type = f"vector<{shape}x{self.element_type}>"
        arguments = ", ".join(
            f"%input{index}: {self.element_type}" for index in range(8)
        )
        operands = ", ".join(f"%input{index}" for index in self.lane_order)
        return (
            f"func.def @{self.name}({arguments}) -> ({result_type}) {{\n"
            f"  %result = vector.from_elements {operands} : {result_type}\n"
            f"  func.return %result : {result_type}\n"
            "}"
        )


def _representative_lane_counts(element_bits: int, maximum_lanes: int) -> list[int]:
    packet_lanes = 64 if element_bits <= 8 else 512 // element_bits
    lane_counts = {1, 2, 3}
    for boundary in range(packet_lanes, maximum_lanes + 1, packet_lanes):
        lane_counts.update((boundary - 1, boundary, boundary + 1))
    return sorted(count for count in lane_counts if count <= maximum_lanes)


def _cases(exhaustive: bool) -> list[ConstructorCase]:
    cases = []
    for element_type, element_bits, maximum_lanes in _ELEMENT_FAMILIES:
        shaped_maximum = maximum_lanes
        lane_counts = (
            range(1, maximum_lanes + 1)
            if exhaustive
            else _representative_lane_counts(element_bits, maximum_lanes)
        )
        for lane_count in lane_counts:
            cases.append(ConstructorCase(element_type, element_bits, (lane_count,)))
            if exhaustive and lane_count <= shaped_maximum:
                leading = next(
                    (factor for factor in (3, 2) if lane_count % factor == 0), 1
                )
                cases.append(
                    ConstructorCase(
                        element_type, element_bits, (leading, lane_count // leading)
                    )
                )
        for dimensions in ((2, 2, 3), (1, 2, shaped_maximum // 2)):
            cases.append(ConstructorCase(element_type, element_bits, dimensions))
    return cases


def _input_payloads(element_bits: int) -> list[bytes]:
    if element_bits == 1:
        return [bytes([value]) for value in (1, 0, 1, 1, 0, 0, 0, 1)]
    return [
        bytes((index * 31 + byte * 19 + 13) & 255 for byte in range(element_bits // 8))
        for index in range(8)
    ]


def _evaluate(body: str, inputs: list[bytes]) -> bytes:
    # Interpret the emitted operand wiring using native byte-lane semantics.
    # Floating values remain payload bits; construction performs no arithmetic.
    values = {f"%input{index}": value for index, value in enumerate(inputs)}
    for line in body.strip().splitlines():
        line = line.strip()
        if line.startswith("return "):
            return values[line.split()[1]]
        destination, operation = line.split(" = ", 1)
        mnemonic = operation.split()[0]
        operands = re.findall(r"%[A-Za-z0-9_]+", operation.split(" : ", 1)[0])
        if mnemonic.startswith("vbcst."):
            width = int(mnemonic.split(".")[1]) // 8
            result = values[operands[0]][:width] * (64 // width)
        elif mnemonic.startswith("vpush."):
            _, direction, width_text = mnemonic.split(".")
            width = int(width_text) // 8
            if direction == "hi":
                result = values[operands[0]][width:] + values[operands[1]][:width]
            elif direction == "lo":
                result = values[operands[0]][:width] + values[operands[1]][:-width]
            else:
                raise AssertionError(f"Unexpected push direction: {line}")
        elif mnemonic.startswith("concat("):
            result = b"".join(values[operand] for operand in operands)
        elif mnemonic == "vmov.vector512.to.accumulator512":
            result = values[operands[0]]
        elif mnemonic == "mov.short":
            result = int(operation.split()[1]).to_bytes(4, "little", signed=True)
        elif mnemonic == "vsub.8":
            result = bytes(
                (left - right) & 255
                for left, right in zip(
                    values[operands[0]], values[operands[1]], strict=True
                )
            )
        elif mnemonic == "vlt.u8x64":
            # Predicate bits are represented as logical zero/one bytes here.
            result = bytes(
                left < right
                for left, right in zip(
                    values[operands[0]], values[operands[1]], strict=True
                )
            )
        else:
            raise AssertionError(f"Unexpected constructor instruction: {line}")
        values[destination] = result
    raise AssertionError("Missing constructor return")


class VectorConstructFamilyTest(unittest.TestCase):
    def test_transient_aggregate_extraction(self):
        # A captured load can exceed the register carrier domain. Extracting
        # one lane must consume that aggregate without reconstructing it as
        # a native shape alias. Predicate and address memory retain their
        # distinct source-memory diagnostics.
        cases = []
        for element_type, element_bits, maximum_lanes in _ELEMENT_FAMILIES:
            shaped_maximum = maximum_lanes
            shapes = (
                (maximum_lanes + 1,),
                (2, shaped_maximum // 2 + 1),
                (2, 2, shaped_maximum // 4 + 1),
            )
            for dimensions in shapes:
                shape = "x".join(str(dimension) for dimension in dimensions)
                indices = ", ".join(str(dimension - 1) for dimension in dimensions)
                zeros = ", ".join("0" for _ in dimensions)
                prefix = (
                    f"func.def @extract_{element_type}_{shape}(%input: buffer) "
                    f"-> ({element_type}) {{\n"
                    "  %input_offset = index.constant 0 : offset\n"
                    f"  %view = buffer.view %input[%input_offset] : buffer "
                    f"-> view<{shape}x{element_type}>\n"
                )
                constraint = {
                    "i1": "source_memory.element_width",
                    "index": "source_memory.element_byte_count",
                    "offset": "source_memory.element_byte_count",
                }.get(element_type)
                diagnostic = (
                    f'  // ERROR@+1: TARGET/003 {{constraint_key="{constraint}"}}\n'
                    if constraint
                    else ""
                )
                suffix = f"  func.return %result : {element_type}\n}}\n"
                source = (
                    prefix + diagnostic + f"  %loaded = vector.load %view[{zeros}] : "
                    f"view<{shape}x{element_type}> -> vector<{shape}x{element_type}>\n"
                    f"  %result = vector.extract %loaded[{indices}] : "
                    f"vector<{shape}x{element_type}> -> {element_type}\n" + suffix
                )
                if not constraint:
                    source += (
                        "\n// ----\n"
                        + prefix
                        + f"  %result = view.load %view[{indices}] : "
                        f"view<{shape}x{element_type}> -> {element_type}\n" + suffix
                    )
                cases.append(source)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transient_shapes.loom-test"
            path.write_text(
                "// RUN: pass target=amd.xdna.aie2p:core "
                "target-legalize{mode=final},canonicalize\n\n"
                + "\n// ====\n\n".join(cases),
                newline="\n",
            )
            result = subprocess.run(
                [_ARGS.checker, str(path)], capture_output=True, text=True
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_logical_lane_order(self):
        cases = _cases(_ARGS.exhaustive)
        source = "// RUN: emit source-low target=amd.xdna.aie2p:core output=low\n\n"
        source += "\n// ====\n".join(case.source() + "\n// ----\n" for case in cases)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vector_construct.loom-test"
            path.write_text(source, newline="\n")
            update = subprocess.run(
                [_ARGS.checker, "--update", str(path)], capture_output=True, text=True
            )
            self.assertIn("updated", update.stdout + update.stderr)
            output = path.read_text()

        functions = dict(
            re.findall(
                r"^low.func.def[^\n]*@(\w+)[^\n]*\{\n(.*?)^}", output, re.M | re.S
            )
        )
        self.assertEqual(len(functions), len(cases))
        for case in cases:
            with self.subTest(case=case.name):
                inputs = _input_payloads(case.element_bits)
                expected = b"".join(inputs[index] for index in case.lane_order)
                actual = _evaluate(functions[case.name], inputs)
                self.assertEqual(actual[: len(expected)], expected)


_PARSER = argparse.ArgumentParser()
_PARSER.add_argument("checker")
_PARSER.add_argument("--exhaustive", action="store_true")
_ARGS, _UNITTEST_ARGS = _PARSER.parse_known_args()

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *_UNITTEST_ARGS])
