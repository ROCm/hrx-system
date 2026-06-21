# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import generate_fixture


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, sort_keys=True)
        file.write("\n")


def _write_manifest_record(trace_dir: Path, record: dict[str, object]) -> None:
    with (trace_dir / "manifest.jsonl").open("a", encoding="utf-8") as manifest:
        json.dump(record, manifest, sort_keys=True)
        manifest.write("\n")


class GenerateFixtureTest(unittest.TestCase):
    def test_generates_fixture_from_checked_in_plan_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan_root = root / "plans"
            trace_root = root / "traces"
            fixture_root = root / "fixtures"
            trace_dir = trace_root / "unit-trace"
            (trace_dir / "tensors").mkdir(parents=True)
            (trace_dir / "tensors" / "condition.bin").write_bytes(
                struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
            )
            _write_manifest_record(
                trace_dir,
                {
                    "dtype": "f32",
                    "file": "tensors/condition.bin",
                    "kind": "tensor",
                    "name": "condition",
                    "ordinal": 0,
                    "shape": [2, 3],
                    "stage": "qwen.encoder",
                },
            )
            _write_json(
                plan_root / "unit_fixture.json",
                {
                    "fixture_id": "unit_fixture",
                    "source_trace_id": "unit-trace",
                    "texts": [],
                    "tensors": [
                        {
                            "name": "condition",
                            "output": "qwen/condition.npy",
                            "role": "expected",
                            "slice": [
                                {"length": 1, "start": 1},
                                {"length": 2, "start": 0},
                            ],
                            "stage": "qwen.encoder",
                            "tolerance": {"atol": 0.0, "rtol": 0.0},
                        }
                    ],
                },
            )

            manifest = generate_fixture.generate_fixture(
                plan_root,
                "unit_fixture",
                trace_root,
                fixture_root,
            )

            self.assertEqual(manifest["fixture_id"], "unit_fixture")
            self.assertTrue((fixture_root / "unit_fixture" / "manifest.json").is_file())
            self.assertTrue(
                (fixture_root / "unit_fixture" / "qwen" / "condition.npy").is_file()
            )
            inventory = json.loads(
                (fixture_root / "unit_fixture" / "inventory.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(inventory["record_count"], 1)
            self.assertEqual(inventory["stages"][0]["stage"], "qwen.encoder")
            self.assertEqual(inventory["stages"][0]["records"][0]["shape"], [1, 2])

    def test_rejects_plan_with_mismatched_fixture_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plan_root = Path(temp_dir) / "plans"
            _write_json(
                plan_root / "expected_id.json",
                {
                    "fixture_id": "different_id",
                    "source_trace_id": "unit-trace",
                    "texts": [],
                    "tensors": [
                        {
                            "name": "condition",
                            "output": "condition.npy",
                            "role": "expected",
                            "slice": [{"length": 1, "start": 0}],
                            "stage": "qwen.encoder",
                            "tolerance": {"atol": 0.0, "rtol": 0.0},
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(
                generate_fixture.GenerateFixtureError,
                "declares fixture_id different_id",
            ):
                generate_fixture.load_fixture_plan(plan_root, "expected_id")


if __name__ == "__main__":
    unittest.main()
