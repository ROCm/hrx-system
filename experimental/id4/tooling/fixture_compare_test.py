#!/usr/bin/env python3
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

import fixture_compare
import trace_reduce


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, sort_keys=True)
        file.write("\n")


def _write_tensor_fixture(
    root: Path,
    file_name: str,
    dtype: str,
    shape: list[int],
    payload: bytes,
) -> None:
    trace_reduce.write_npy(root / file_name, dtype, shape, payload)


def _write_id4_tensor(
    root: Path,
    file_name: str,
    dtype: str,
    shape: list[int],
    payload: bytes,
) -> None:
    path = root / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps(
        {
            "byte_length": len(payload),
            "dtype": dtype,
            "kind": "tensor",
            "layout": "dense-row-major",
            "shape": shape,
            "storage_dtype": dtype,
            "storage_shape": shape,
            "version": 1,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    path.write_bytes(
        fixture_compare.ID4_TENSOR_MAGIC
        + struct.pack("<I", len(header))
        + header
        + payload
    )


class FixtureCompareTest(unittest.TestCase):
    def test_compares_expected_tensor_and_text_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture_dir = root / "fixture"
            actual_dir = root / "actual"

            _write_tensor_fixture(
                fixture_dir,
                "qwen/condition.npy",
                "f32",
                [2],
                b"\x00\x00\x80?\x00\x00\x00@",
            )
            _write_tensor_fixture(
                actual_dir,
                "qwen/condition.npy",
                "f32",
                [2],
                b"\x0a\xd7\x83?\x00\x00\x00@",
            )
            (fixture_dir / "qwen").mkdir(exist_ok=True)
            (actual_dir / "qwen").mkdir(exist_ok=True)
            (fixture_dir / "qwen" / "summary.txt").write_text(
                "condition\n", encoding="utf-8"
            )
            (actual_dir / "qwen" / "summary.txt").write_text(
                "condition\n", encoding="utf-8"
            )

            _write_json(
                fixture_dir / "manifest.json",
                {
                    "fixture_id": "unit_fixture",
                    "records": [
                        {
                            "dtype": "f32",
                            "file": "qwen/condition.npy",
                            "kind": "tensor",
                            "name": "condition",
                            "role": "expected",
                            "shape": [2],
                            "stage": "qwen.encoder",
                            "tolerance": {"atol": 0.05, "rtol": 0.0},
                        },
                        {
                            "file": "qwen/summary.txt",
                            "kind": "text",
                            "name": "summary",
                            "role": "expected",
                            "stage": "qwen.encoder",
                        },
                    ],
                    "schema_version": 1,
                    "source_trace": {"manifest_sha256": "abc123"},
                },
            )
            _write_json(
                actual_dir / "manifest.json",
                {
                    "records": [
                        {
                            "dtype": "f32",
                            "file": "qwen/condition.npy",
                            "kind": "tensor",
                            "name": "condition",
                            "role": "actual",
                            "shape": [2],
                            "stage": "qwen.encoder",
                        },
                        {
                            "file": "qwen/summary.txt",
                            "kind": "text",
                            "name": "summary",
                            "role": "actual",
                            "stage": "qwen.encoder",
                        },
                    ],
                    "run_id": "unit_run",
                    "schema_version": 1,
                },
            )

            report = fixture_compare.compare_fixtures(fixture_dir, actual_dir)

            self.assertEqual(report["fixture"]["fixture_id"], "unit_fixture")
            self.assertEqual(report["actual"]["run_id"], "unit_run")
            self.assertEqual(
                report["summary"],
                {
                    "compared_count": 2,
                    "expected_count": 2,
                    "failed_count": 0,
                    "passed_count": 2,
                },
            )
            condition = report["comparisons"][0]
            self.assertEqual(condition["kind"], "tensor")
            self.assertEqual(condition["status"], "pass")
            self.assertEqual(condition["mismatch_count"], 0)
            self.assertLess(condition["max_abs_error"], condition["atol"])

    def test_compares_expected_npy_against_actual_id4_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture_dir = root / "fixture"
            actual_dir = root / "actual"

            _write_tensor_fixture(
                fixture_dir,
                "condition.npy",
                "bf16",
                [2],
                b"\x80\x3f\x00\x40",
            )
            _write_id4_tensor(
                actual_dir,
                "condition.id4tensor",
                "bf16",
                [2],
                b"\x80\x3f\x00\x40",
            )
            _write_json(
                fixture_dir / "manifest.json",
                {
                    "fixture_id": "unit_fixture",
                    "records": [
                        {
                            "dtype": "bf16",
                            "file": "condition.npy",
                            "kind": "tensor",
                            "name": "condition",
                            "role": "expected",
                            "shape": [2],
                            "stage": "qwen.encoder",
                            "tolerance": {"atol": 0.0, "rtol": 0.0},
                        }
                    ],
                    "schema_version": 1,
                },
            )
            _write_json(
                actual_dir / "manifest.json",
                {
                    "records": [
                        {
                            "dtype": "bf16",
                            "file": "condition.id4tensor",
                            "kind": "tensor",
                            "name": "condition",
                            "role": "actual",
                            "shape": [2],
                            "stage": "qwen.encoder",
                        }
                    ],
                    "run_id": "unit_run",
                    "schema_version": 1,
                },
            )

            report = fixture_compare.compare_fixtures(fixture_dir, actual_dir)

            comparison = report["comparisons"][0]
            self.assertEqual(comparison["status"], "pass")
            self.assertEqual(comparison["dtype"], "bf16")
            self.assertEqual(comparison["mismatch_count"], 0)

    def test_reports_tensor_tolerance_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture_dir = root / "fixture"
            actual_dir = root / "actual"
            _write_tensor_fixture(
                fixture_dir,
                "condition.npy",
                "f32",
                [1],
                b"\x00\x00\x80?",
            )
            _write_tensor_fixture(
                actual_dir,
                "condition.npy",
                "f32",
                [1],
                b"\x00\x00\x00@",
            )
            _write_json(
                fixture_dir / "manifest.json",
                {
                    "fixture_id": "unit_fixture",
                    "records": [
                        {
                            "dtype": "f32",
                            "file": "condition.npy",
                            "kind": "tensor",
                            "name": "condition",
                            "role": "expected",
                            "shape": [1],
                            "stage": "qwen.encoder",
                            "tolerance": {"atol": 0.0, "rtol": 0.0},
                        }
                    ],
                    "schema_version": 1,
                },
            )
            _write_json(
                actual_dir / "manifest.json",
                {
                    "records": [
                        {
                            "dtype": "f32",
                            "file": "condition.npy",
                            "kind": "tensor",
                            "name": "condition",
                            "role": "actual",
                            "shape": [1],
                            "stage": "qwen.encoder",
                        }
                    ],
                    "schema_version": 1,
                },
            )

            report = fixture_compare.compare_fixtures(fixture_dir, actual_dir)

            self.assertEqual(report["summary"]["failed_count"], 1)
            comparison = report["comparisons"][0]
            self.assertEqual(comparison["status"], "fail")
            self.assertEqual(comparison["mismatch_count"], 1)
            self.assertEqual(comparison["first_mismatch"]["index"], 0)

    def test_compares_expected_slice_against_full_actual_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture_dir = root / "fixture"
            actual_dir = root / "actual"
            _write_tensor_fixture(
                fixture_dir,
                "condition_slice.npy",
                "i32",
                [2, 2],
                b"\x03\x00\x00\x00\x04\x00\x00\x00\x05\x00\x00\x00\x06\x00\x00\x00",
            )
            _write_tensor_fixture(
                actual_dir,
                "condition.npy",
                "i32",
                [3, 2],
                b"\x01\x00\x00\x00\x02\x00\x00\x00"
                b"\x03\x00\x00\x00\x04\x00\x00\x00"
                b"\x05\x00\x00\x00\x06\x00\x00\x00",
            )
            _write_json(
                fixture_dir / "manifest.json",
                {
                    "fixture_id": "unit_fixture",
                    "records": [
                        {
                            "dtype": "i32",
                            "file": "condition_slice.npy",
                            "kind": "tensor",
                            "name": "condition",
                            "role": "expected",
                            "shape": [2, 2],
                            "slice": [
                                {"start": 1, "length": 2},
                                {"start": 0, "length": 2},
                            ],
                            "stage": "qwen.encoder",
                        }
                    ],
                    "schema_version": 1,
                },
            )
            _write_json(
                actual_dir / "manifest.json",
                {
                    "records": [
                        {
                            "dtype": "i32",
                            "file": "condition.npy",
                            "kind": "tensor",
                            "name": "condition",
                            "role": "actual",
                            "shape": [3, 2],
                            "stage": "qwen.encoder",
                        }
                    ],
                    "schema_version": 1,
                },
            )

            report = fixture_compare.compare_fixtures(fixture_dir, actual_dir)

            comparison = report["comparisons"][0]
            self.assertEqual(comparison["status"], "pass")
            self.assertEqual(comparison["mismatch_count"], 0)
            self.assertEqual(comparison["actual_shape"], [3, 2])
            self.assertEqual(
                comparison["slice"],
                [
                    {"start": 1, "length": 2},
                    {"start": 0, "length": 2},
                ],
            )

    def test_reports_missing_actual_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture_dir = root / "fixture"
            actual_dir = root / "actual"
            _write_tensor_fixture(
                fixture_dir,
                "condition.npy",
                "i32",
                [1],
                b"\x01\x00\x00\x00",
            )
            _write_json(
                fixture_dir / "manifest.json",
                {
                    "fixture_id": "unit_fixture",
                    "records": [
                        {
                            "dtype": "i32",
                            "file": "condition.npy",
                            "kind": "tensor",
                            "name": "condition",
                            "role": "expected",
                            "shape": [1],
                            "stage": "qwen.encoder",
                        }
                    ],
                    "schema_version": 1,
                },
            )
            _write_json(
                actual_dir / "manifest.json",
                {
                    "records": [],
                    "schema_version": 1,
                },
            )

            report = fixture_compare.compare_fixtures(fixture_dir, actual_dir)

            self.assertEqual(report["summary"]["failed_count"], 1)
            self.assertEqual(
                report["comparisons"][0]["reason"], "missing_actual_record"
            )

    def test_rejects_expected_fixture_without_expected_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture_dir = root / "fixture"
            actual_dir = root / "actual"
            _write_json(
                fixture_dir / "manifest.json",
                {
                    "records": [
                        {
                            "file": "input.txt",
                            "kind": "text",
                            "name": "input",
                            "role": "input",
                            "stage": "qwen.prompt",
                        }
                    ],
                    "schema_version": 1,
                },
            )
            _write_json(
                actual_dir / "manifest.json",
                {
                    "records": [],
                    "schema_version": 1,
                },
            )

            with self.assertRaisesRegex(
                fixture_compare.FixtureCompareError,
                "contains no expected records",
            ):
                fixture_compare.compare_fixtures(fixture_dir, actual_dir)


if __name__ == "__main__":
    unittest.main()
