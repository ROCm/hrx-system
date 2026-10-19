# Copyright 2026 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Replay independent coefficient observations without a codec installation."""

import json
import unittest
from pathlib import Path

from sequential import InvalidJPEG, UnsupportedJPEG, decode, parse


class SequentialReferenceTest(unittest.TestCase):
    def test_public_oracle_coefficients_and_truncations(self):
        rows = [
            json.loads(line)
            for line in Path(__file__).with_name("probe.jsonl").read_text().splitlines()
            if line.startswith('{"kind":')
        ]
        expected = {
            (row["name"], row["mode"]): row for row in rows if row["kind"] == "result"
        }
        accepted = 0
        rejected = 0
        truncated = 0
        for row in rows:
            if row["kind"] != "input":
                continue
            name = row["name"]
            data = bytes.fromhex(row["hex"])
            with self.subTest(image=name):
                try:
                    frame = parse(data)
                except UnsupportedJPEG:
                    rejected += 1
                    continue
                planes, _ = decode(frame, True)
                oracle = expected[name, 0]
                self.assertTrue(oracle["complete"])
                self.assertFalse(oracle["warnings"])
                for component, plane, geometry, coefficients in zip(
                    frame.components,
                    planes,
                    oracle["components"],
                    oracle["coefficients"],
                    strict=True,
                ):
                    self.assertEqual(component.block_width, geometry["block_width"])
                    self.assertEqual(component.block_height, geometry["block_height"])
                    self.assertEqual(
                        list(component.quantization), coefficients["quantization"]
                    )
                    self.assertEqual(list(plane), coefficients["values"])
                accepted += 1
                if name in (
                    "ycbcr_420",
                    "ycbcr_420_restart_1",
                    "ycbcr_420_separate_scans",
                ):
                    for length in range(len(data)):
                        with self.assertRaises(InvalidJPEG):
                            decode(parse(data[:length]))
                        truncated += 1
        self.assertEqual(accepted, 14)
        self.assertEqual(rejected, 9)
        self.assertGreater(truncated, 1000)


if __name__ == "__main__":
    unittest.main()
