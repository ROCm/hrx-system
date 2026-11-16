# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Audits shared packed-bitstream legalization across integer type families."""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from math import gcd
from pathlib import Path

_INTEGER_WIDTHS = (1, 8, 16, 32, 64)
_FIELD_WIDTHS = {
    1: (1,),
    8: (1, 2, 3, 4, 7, 8),
    16: (1, 3, 7, 8, 9, 15, 16),
    32: (1, 3, 7, 8, 9, 15, 16, 17, 31, 32),
    64: (1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64),
}
_BITSTREAM_OPERATIONS = ("bitpack", "bitunpacku", "bitunpacks")
_TARGET = "spirv:vulkan1.3+bda+extended-types"

_WIDE_MEMORY_SOURCE = """\
func.def public @pack_128(%input: buffer, %output: buffer) {
  %base = index.constant 0 : offset
  %inputs = buffer.view %input[%base] : buffer -> view<128xi8>
  %outputs = buffer.view %output[%base] : buffer -> view<64xi8>
  %bytes = vector.load %inputs[0] : view<128xi8> -> vector<128xi8>
  %packed = vector.bitpack<4> %bytes : vector<128xi8> -> vector<64xi8>
  vector.store %packed, %outputs[0] : vector<64xi8>, view<64xi8>
  func.return
}
"""


def _vector_type(lane_count: int, element_bit_count: int) -> str:
    return f"vector<{lane_count}xi{element_bit_count}>"


def _partition_lanes(lane_count: int) -> tuple[int, ...]:
    result = []
    for packet_lane_count in (4, 2, 1):
        while lane_count >= packet_lane_count:
            result.append(packet_lane_count)
            lane_count -= packet_lane_count
    assert lane_count == 0
    return tuple(result)


@dataclass(frozen=True)
class BitstreamCase:
    operation: str
    logical_element_bit_count: int
    field_bit_count: int
    storage_element_bit_count: int

    @property
    def name(self) -> str:
        return (
            f"audit_{self.operation}_i{self.logical_element_bit_count}_"
            f"q{self.field_bit_count}_i{self.storage_element_bit_count}"
        )

    def source(self) -> str:
        total_bit_count = (
            self.field_bit_count
            * self.storage_element_bit_count
            // gcd(self.field_bit_count, self.storage_element_bit_count)
        )
        logical_lane_count = total_bit_count // self.field_bit_count
        storage_lane_count = total_bit_count // self.storage_element_bit_count
        if self.operation == "bitpack":
            source_lane_count = logical_lane_count
            source_element_bit_count = self.logical_element_bit_count
            result_lane_count = storage_lane_count
            result_element_bit_count = self.storage_element_bit_count
        else:
            source_lane_count = storage_lane_count
            source_element_bit_count = self.storage_element_bit_count
            result_lane_count = logical_lane_count
            result_element_bit_count = self.logical_element_bit_count

        partition = _partition_lanes(source_lane_count)
        arguments = ", ".join(
            f"%element{index}: i{source_element_bit_count}"
            for index in range(source_lane_count)
        )
        result_type = _vector_type(result_lane_count, result_element_bit_count)
        lines = [
            f"func.def @{self.name}({arguments}) -> (i{result_element_bit_count}) {{"
        ]
        lane_offset = 0
        for index, lanes in enumerate(partition):
            elements = ", ".join(
                f"%element{lane}" for lane in range(lane_offset, lane_offset + lanes)
            )
            lines.append(
                f"  %part{index} = vector.from_elements {elements} : "
                f"{_vector_type(lanes, source_element_bit_count)}"
            )
            lane_offset += lanes
        if len(partition) == 1:
            source = "%part0"
        else:
            source = "%source"
            operands = ", ".join(f"%part{index}" for index in range(len(partition)))
            operand_types = ", ".join(
                _vector_type(lanes, source_element_bit_count) for lanes in partition
            )
            lines.append(
                f"  {source} = vector.concat<0> {operands} : {operand_types} "
                f"-> {_vector_type(source_lane_count, source_element_bit_count)}"
            )
        lines.extend(
            (
                f"  %result = vector.{self.operation}<{self.field_bit_count}> "
                f"{source} : "
                f"{_vector_type(source_lane_count, source_element_bit_count)} "
                f"-> {result_type}",
                f"  %sink = vector.extract %result[{result_lane_count - 1}] : "
                f"{result_type} -> i{result_element_bit_count}",
                f"  func.return %sink : i{result_element_bit_count}",
                "}",
            )
        )
        return "\n".join(lines)


def _case_shapes(exhaustive: bool) -> list[tuple[int, int, int]]:
    if exhaustive:
        return [
            (logical_width, field_width, storage_width)
            for logical_width, field_widths in _FIELD_WIDTHS.items()
            for field_width in field_widths
            for storage_width in _INTEGER_WIDTHS
        ]

    shapes = {
        (
            logical_width,
            1 if logical_width == 1 else min(3, logical_width),
            storage_width,
        )
        for logical_width in _INTEGER_WIDTHS
        for storage_width in _INTEGER_WIDTHS
    }
    for logical_width, field_widths in _FIELD_WIDTHS.items():
        for index, field_width in enumerate(field_widths):
            shapes.add(
                (
                    logical_width,
                    field_width,
                    _INTEGER_WIDTHS[index % len(_INTEGER_WIDTHS)],
                )
            )
    return sorted(shapes)


def _cases(exhaustive: bool) -> list[BitstreamCase]:
    return [
        BitstreamCase(operation, logical_width, field_width, storage_width)
        for logical_width, field_width, storage_width in _case_shapes(exhaustive)
        for operation in _BITSTREAM_OPERATIONS
    ]


def _test_source(cases: list[BitstreamCase]) -> str:
    return "\n// ====\n".join(
        f"// RUN: compile-report target={_TARGET} target-legalize{{mode=eager}}\n\n"
        + case.source()
        + "\n"
        for case in cases
    )


def _report_fields(line: str) -> dict[str, str]:
    return {
        match.group("key"): match.group("value")
        for match in re.finditer(r"(?P<key>[a-z_]+)=(?P<value>[^ ]+)", line)
    }


class BitstreamPacketLegalizationTest(unittest.TestCase):
    def _run_checker(self, source_path: Path, *, update: bool = False):
        arguments = [_ARGS.checker]
        if update:
            arguments.append("--update")
        arguments.append(str(source_path))
        return subprocess.run(arguments, capture_output=True, text=True)

    def test_family_uses_word_packets_without_scalarization(self):
        cases = _cases(_ARGS.exhaustive)
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "bitstream_family.loom-test"
            source_path.write_text(_test_source(cases), newline="\n")
            update_result = self._run_checker(source_path, update=True)
            update_output = update_result.stderr + update_result.stdout
            self.assertIn("updated", update_output, update_output)
            result = self._run_checker(source_path)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = source_path.read_text().partition("// ----\n")[2]

        expected = {(case.name, f"vector.{case.operation}") for case in cases}
        actual = set()
        for line in report.splitlines():
            if not line.startswith("COMPILE-REPORT: target_legalization["):
                continue
            fields = _report_fields(line)
            source_op = fields.get("source_op")
            if source_op not in {
                "vector.bitpack",
                "vector.bitunpacku",
                "vector.bitunpacks",
            }:
                continue
            key = (fields["function"], source_op)
            self.assertNotIn(key, actual)
            actual.add(key)
            self.assertEqual(fields["action"], "rewritten", line)
            self.assertEqual(fields["strategy"], "reference", line)
            self.assertEqual(fields["scalarized"], "false", line)
            self.assertGreater(int(fields["created_ops"]), 0, line)
        self.assertEqual(actual, expected)

    def test_bitcast_family_projects_both_packet_shapes(self):
        # Six lanes of the wider element exercise full packets and tails in
        # both directions. An 8:1 width ratio cannot fit both sides in SPIR-V's
        # four-lane carriers and retains the composed scalar lowering.
        types = ("i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64")
        cases = []
        expected = set()
        for source_type in types:
            for result_type in types:
                source_width = int(source_type.lstrip("bfi"))
                result_width = int(result_type.lstrip("bfi"))
                if source_type == result_type or max(source_width, result_width) > (
                    4 * min(source_width, result_width)
                ):
                    continue
                bit_count = 6 * max(source_width, result_width)
                source_vector = f"vector<{bit_count // source_width}x{source_type}>"
                result_vector = f"vector<{bit_count // result_width}x{result_type}>"
                name = f"cast_{source_type}_{result_type}"
                expected.add(name)
                cases.append(
                    f"// RUN: compile-report target={_TARGET} target-legalize{{mode=eager}}\n\n"
                    f"func.def @{name}(%input: {source_vector}) -> ({result_vector}) {{\n"
                    f"  %result = vector.bitcast %input : {source_vector} to {result_vector}\n"
                    f"  func.return %result : {result_vector}\n"
                    "}\n"
                )
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "bitcast_family.loom-test"
            source_path.write_text("\n// ====\n".join(cases), newline="\n")
            update = self._run_checker(source_path, update=True)
            self.assertIn("updated", update.stderr + update.stdout)
            result = self._run_checker(source_path)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = source_path.read_text()
        actual = set()
        for line in report.splitlines():
            if not line.startswith("COMPILE-REPORT: target_legalization["):
                continue
            fields = _report_fields(line)
            if fields.get("source_op") != "vector.bitcast":
                continue
            actual.add(fields["function"])
            self.assertEqual(fields["action"], "rewritten", line)
            self.assertEqual(fields["scalarized"], "false", line)
        self.assertEqual(actual, expected)

    def test_wide_memory_source_retains_word_packets(self):
        if _ARGS.exhaustive:
            self.skipTest("covered by the ordinary family target")
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "wide_memory.loom"
            artifact_path = directory_path / "wide_memory.spv"
            report_path = directory_path / "wide_memory.json"
            source_path.write_text(_WIDE_MEMORY_SOURCE, newline="\n")
            result = subprocess.run(
                [
                    _ARGS.compiler,
                    str(source_path),
                    "--format=spirv",
                    "--target=" + _TARGET,
                    "--output=" + str(artifact_path),
                    "--compile-report=details",
                    "--compile-report-output=" + str(report_path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(report_path.read_text())

        self.assertEqual(report["status"]["code"], 0)
        self.assertGreater(report["artifact_size"], 0)
        legalization = report["target_legalization"]
        self.assertEqual(legalization["unhandled_op_count"], 0)
        bitpack_rows = [
            row for row in legalization["rows"] if row["source_op"] == "vector.bitpack"
        ]
        self.assertEqual(len(bitpack_rows), 1)
        self.assertEqual(bitpack_rows[0]["action"], "rewritten")

        summaries = {
            (row["source_op"], row["descriptor_key"]): row
            for row in report["source_low"]["selection_summaries"]["rows"]
        }
        self.assertEqual(
            summaries[("vector.from_elements", "spirv.op_composite_construct.v4i8")][
                "selected_op_count"
            ],
            32,
        )
        self.assertEqual(
            summaries[("vector.bitcast", "spirv.op_bitcast.v4i8.i32")][
                "selected_op_count"
            ],
            32,
        )


_PARSER = argparse.ArgumentParser()
_PARSER.add_argument("checker")
_PARSER.add_argument("compiler")
_PARSER.add_argument("--exhaustive", action="store_true")
_ARGS, _UNITTEST_ARGS = _PARSER.parse_known_args()

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *_UNITTEST_ARGS])
