#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import math
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


def _write_result_manifest(
    root: Path,
    records: list[dict[str, object]],
) -> None:
    _write_json(
        root / "manifest.json",
        {
            "format": "id4tensor-v1",
            "records": records,
            "schema_version": 1,
        },
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

    def test_compares_result_tensor_manifests_by_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected_dir = root / "expected"
            actual_dir = root / "actual"

            _write_id4_tensor(
                expected_dir,
                "condition.id4tensor",
                "bf16",
                [2],
                b"\x80\x3f\x00\x40",
            )
            _write_id4_tensor(
                actual_dir,
                "condition.id4tensor",
                "bf16",
                [2],
                b"\x80\x3f\x40\x40",
            )
            _write_result_manifest(
                expected_dir,
                [
                    {
                        "byte_length": 4,
                        "dtype": "bf16",
                        "file": "condition.id4tensor",
                        "key": "dit_conditioned:layer0.qkv",
                        "ordinal": 0,
                        "shape": {"dims": [2], "rank": 1},
                    }
                ],
            )
            _write_result_manifest(
                actual_dir,
                [
                    {
                        "byte_length": 4,
                        "dtype": "bf16",
                        "file": "condition.id4tensor",
                        "key": "dit_conditioned:layer0.qkv",
                        "ordinal": 0,
                        "shape": {"dims": [2], "rank": 1},
                    }
                ],
            )

            report = fixture_compare.compare_result_tensors(
                expected_dir, actual_dir, atol=0.0, rtol=0.0
            )

            self.assertEqual(
                report["summary"],
                {
                    "actual_count": 1,
                    "compared_count": 1,
                    "expected_count": 1,
                    "failed_count": 1,
                    "passed_count": 0,
                },
            )
            comparison = report["comparisons"][0]
            self.assertEqual(comparison["stage"], "result")
            self.assertEqual(comparison["name"], "dit_conditioned:layer0.qkv")
            self.assertEqual(comparison["dtype"], "bf16")
            self.assertEqual(comparison["mismatch_count"], 1)
            self.assertEqual(comparison["mean_abs_error"], 0.5)
            self.assertAlmostEqual(comparison["rmse"], 0.5**0.5)
            self.assertEqual(comparison["max_abs_error_index"], 1)

    def test_result_tensor_compare_reports_extra_actual_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected_dir = root / "expected"
            actual_dir = root / "actual"

            _write_id4_tensor(
                expected_dir,
                "condition.id4tensor",
                "i32",
                [1],
                b"\x01\x00\x00\x00",
            )
            _write_id4_tensor(
                actual_dir,
                "condition.id4tensor",
                "i32",
                [1],
                b"\x01\x00\x00\x00",
            )
            _write_id4_tensor(
                actual_dir,
                "extra.id4tensor",
                "i32",
                [1],
                b"\x02\x00\x00\x00",
            )
            _write_result_manifest(
                expected_dir,
                [
                    {
                        "byte_length": 4,
                        "dtype": "i32",
                        "file": "condition.id4tensor",
                        "key": "conditioned_velocity",
                        "ordinal": 0,
                        "shape": {"dims": [1], "rank": 1},
                    }
                ],
            )
            _write_result_manifest(
                actual_dir,
                [
                    {
                        "byte_length": 4,
                        "dtype": "i32",
                        "file": "condition.id4tensor",
                        "key": "conditioned_velocity",
                        "ordinal": 0,
                        "shape": {"dims": [1], "rank": 1},
                    },
                    {
                        "byte_length": 4,
                        "dtype": "i32",
                        "file": "extra.id4tensor",
                        "key": "extra_tap",
                        "ordinal": 1,
                        "shape": {"dims": [1], "rank": 1},
                    },
                ],
            )

            report = fixture_compare.compare_result_tensors(
                expected_dir, actual_dir, atol=0.0, rtol=0.0
            )

            self.assertEqual(report["summary"]["failed_count"], 1)
            self.assertEqual(report["comparisons"][1]["name"], "extra_tap")
            self.assertEqual(report["comparisons"][1]["reason"], "extra_actual_record")

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

    def test_compares_float_tensor_with_aggregate_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture_dir = root / "fixture"
            actual_dir = root / "actual"
            _write_tensor_fixture(
                fixture_dir,
                "condition.npy",
                "f32",
                [4],
                struct.pack("<4f", 1.0, 2.0, 3.0, 4.0),
            )
            _write_tensor_fixture(
                actual_dir,
                "condition.npy",
                "f32",
                [4],
                struct.pack("<4f", 1.1, 2.2, 3.3, 4.4),
            )
            fixture_manifest = {
                "fixture_id": "unit_fixture",
                "records": [
                    {
                        "dtype": "f32",
                        "file": "condition.npy",
                        "kind": "tensor",
                        "name": "condition",
                        "role": "expected",
                        "shape": [4],
                        "stage": "qwen.encoder",
                        "tolerance": {
                            "mean_abs": 0.251,
                            "p99_abs": 0.401,
                            "max_abs": 0.401,
                        },
                    }
                ],
                "schema_version": 1,
            }
            _write_json(fixture_dir / "manifest.json", fixture_manifest)
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
                            "shape": [4],
                            "stage": "qwen.encoder",
                        }
                    ],
                    "schema_version": 1,
                },
            )

            report = fixture_compare.compare_fixtures(fixture_dir, actual_dir)

            comparison = report["comparisons"][0]
            self.assertEqual(comparison["status"], "pass")
            self.assertEqual(comparison["tolerance_mode"], "aggregate")
            self.assertAlmostEqual(comparison["mean_abs_error"], 0.25, places=6)
            self.assertAlmostEqual(comparison["p99_abs_error"], 0.4, places=6)
            self.assertAlmostEqual(comparison["max_abs_error"], 0.4, places=6)

            fixture_manifest["records"][0]["tolerance"]["max_abs"] = 0.39
            _write_json(fixture_dir / "manifest.json", fixture_manifest)
            report = fixture_compare.compare_fixtures(fixture_dir, actual_dir)
            self.assertEqual(report["comparisons"][0]["status"], "fail")

            fixture_manifest["records"][0]["tolerance"]["max_abs"] = 0.401
            _write_json(fixture_dir / "manifest.json", fixture_manifest)
            _write_tensor_fixture(
                actual_dir,
                "condition.npy",
                "f32",
                [4],
                struct.pack("<4f", 1.0, 2.0, 3.0, math.nan),
            )
            report = fixture_compare.compare_fixtures(fixture_dir, actual_dir)
            self.assertEqual(report["comparisons"][0]["status"], "fail")
            self.assertTrue(math.isinf(report["comparisons"][0]["max_abs_error"]))

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
