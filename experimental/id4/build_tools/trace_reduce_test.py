#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import ast
import json
import struct
import tempfile
import unittest
from pathlib import Path

import trace_reduce


def _write_manifest_record(trace_dir: Path, record: dict[str, object]) -> None:
    with (trace_dir / "manifest.jsonl").open("a", encoding="utf-8") as manifest:
        json.dump(record, manifest, sort_keys=True)
        manifest.write("\n")


def _read_npy(path: Path) -> tuple[str, tuple[int, ...], bytes]:
    payload = path.read_bytes()
    if payload[:6] != b"\x93NUMPY":
        raise AssertionError("missing NPY magic")
    if payload[6:8] != b"\x01\x00":
        raise AssertionError("unexpected NPY version")
    header_length = struct.unpack("<H", payload[8:10])[0]
    header = ast.literal_eval(payload[10 : 10 + header_length].decode("ascii"))
    data = payload[10 + header_length :]
    return header["descr"], tuple(header["shape"]), data


def _read_id4_tensor(path: Path) -> tuple[str, tuple[int, ...], bytes]:
    payload = path.read_bytes()
    if payload[:8] != trace_reduce.ID4_TENSOR_MAGIC:
        raise AssertionError("missing ID4 tensor magic")
    header_length = struct.unpack("<I", payload[8:12])[0]
    header = json.loads(payload[12 : 12 + header_length].decode("utf-8"))
    data = payload[12 + header_length :]
    return header["dtype"], tuple(header["shape"]), data


class TraceReduceTest(unittest.TestCase):
    def test_reduces_text_and_tensor_slices(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace_dir = root / "trace"
            output_dir = root / "fixtures"
            (trace_dir / "texts").mkdir(parents=True)
            (trace_dir / "tensors").mkdir()

            (trace_dir / "texts" / "prompt.txt").write_text(
                "structured prompt\n", encoding="utf-8"
            )
            token_payload = struct.pack("<6i", 10, 11, 12, 13, 14, 15)
            (trace_dir / "tensors" / "token_ids.bin").write_bytes(token_payload)
            matrix_values = [float(value) for value in range(20)]
            matrix_payload = struct.pack("<20f", *matrix_values)
            (trace_dir / "tensors" / "condition.bin").write_bytes(matrix_payload)

            _write_manifest_record(
                trace_dir,
                {
                    "file": "texts/prompt.txt",
                    "kind": "text",
                    "name": "wrapped_prompt",
                    "ordinal": 0,
                    "stage": "qwen.prompt",
                },
            )
            _write_manifest_record(
                trace_dir,
                {
                    "dtype": "i32",
                    "file": "tensors/token_ids.bin",
                    "kind": "tensor",
                    "name": "token_ids",
                    "ordinal": 1,
                    "shape": [6],
                    "stage": "qwen.prompt",
                },
            )
            _write_manifest_record(
                trace_dir,
                {
                    "dtype": "f32",
                    "file": "tensors/condition.bin",
                    "kind": "tensor",
                    "name": "condition",
                    "ordinal": 2,
                    "shape": [4, 5],
                    "stage": "qwen.encoder",
                },
            )

            plan = {
                "fixture_id": "unit_qwen_slice",
                "source_label": "unit trace",
                "texts": [
                    {
                        "name": "wrapped_prompt",
                        "output": "qwen/wrapped_prompt.txt",
                        "role": "metadata",
                        "stage": "qwen.prompt",
                    }
                ],
                "tensors": [
                    {
                        "name": "token_ids",
                        "output": "qwen/token_ids.npy",
                        "role": "input",
                        "slice": [{"length": 3, "start": 1}],
                        "stage": "qwen.prompt",
                    },
                    {
                        "name": "condition",
                        "output": "qwen/condition.npy",
                        "role": "expected",
                        "slice": [
                            {"length": 2, "start": 1},
                            {"length": 2, "start": 2},
                        ],
                        "stage": "qwen.encoder",
                        "tolerance": {"atol": 0.0, "rtol": 0.0},
                    },
                ],
            }

            manifest = trace_reduce.reduce_trace(trace_dir, output_dir, plan)

            self.assertEqual(manifest["fixture_id"], "unit_qwen_slice")
            self.assertEqual(manifest["source_trace"]["source_label"], "unit trace")
            self.assertEqual(len(manifest["records"]), 3)
            self.assertEqual(
                (output_dir / "qwen" / "wrapped_prompt.txt").read_text(
                    encoding="utf-8"
                ),
                "structured prompt\n",
            )

            dtype, shape, data = _read_id4_tensor(
                output_dir / "qwen" / "token_ids.id4tensor"
            )
            self.assertEqual(dtype, "i32")
            self.assertEqual(shape, (3,))
            self.assertEqual(struct.unpack("<3i", data), (11, 12, 13))

            descr, shape, data = _read_npy(output_dir / "qwen" / "condition.npy")
            self.assertEqual(descr, "<f4")
            self.assertEqual(shape, (2, 2))
            self.assertEqual(struct.unpack("<4f", data), (9.0, 13.0, 10.0, 14.0))

            condition_record = manifest["records"][2]
            self.assertEqual(manifest["records"][1]["file"], "qwen/token_ids.id4tensor")
            self.assertEqual(manifest["records"][1]["format"], "id4tensor-v1")
            self.assertEqual(condition_record["file"], "qwen/condition.npy")
            self.assertEqual(condition_record["format"], "npy-v1")
            self.assertEqual(condition_record["summary"]["finite_count"], 4)
            self.assertEqual(condition_record["summary"]["min"], 9.0)
            self.assertEqual(condition_record["summary"]["max"], 14.0)
            self.assertEqual(
                json.loads((output_dir / "manifest.json").read_text()),
                manifest,
            )
            inventory = json.loads((output_dir / "inventory.json").read_text())
            self.assertEqual(
                inventory["fixture_id"],
                "unit_qwen_slice",
            )
            self.assertEqual(inventory["kind_counts"], {"tensor": 2, "text": 1})
            self.assertEqual(
                inventory["role_counts"],
                {
                    "expected": 1,
                    "input": 1,
                    "metadata": 1,
                },
            )
            self.assertEqual(
                [stage["stage"] for stage in inventory["stages"]],
                ["qwen.encoder", "qwen.prompt"],
            )
            encoder_record = inventory["stages"][0]["records"][0]
            self.assertEqual(encoder_record["name"], "condition")
            self.assertEqual(encoder_record["dtype"], "f32")
            self.assertEqual(encoder_record["format"], "npy-v1")
            self.assertEqual(encoder_record["shape"], [2, 2])
            self.assertEqual(encoder_record["source_shape"], [4, 5])
            self.assertEqual(
                encoder_record["tolerance"],
                {"atol": 0.0, "rtol": 0.0},
            )
            prompt_records = inventory["stages"][1]["records"]
            self.assertEqual(
                [(record["kind"], record["name"]) for record in prompt_records],
                [("text", "wrapped_prompt"), ("tensor", "token_ids")],
            )
            self.assertEqual(prompt_records[1]["format"], "id4tensor-v1")

    def test_rejects_floating_expected_tensor_without_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace_dir = root / "trace"
            (trace_dir / "tensors").mkdir(parents=True)
            (trace_dir / "tensors" / "expected.bin").write_bytes(
                struct.pack("<2f", 1.0, 2.0)
            )
            _write_manifest_record(
                trace_dir,
                {
                    "dtype": "f32",
                    "file": "tensors/expected.bin",
                    "kind": "tensor",
                    "name": "expected",
                    "ordinal": 0,
                    "shape": [2],
                    "stage": "stage",
                },
            )
            plan = {
                "fixture_id": "missing_tolerance",
                "texts": [],
                "tensors": [
                    {
                        "name": "expected",
                        "output": "expected.npy",
                        "role": "expected",
                        "slice": [{"length": 2, "start": 0}],
                        "stage": "stage",
                    }
                ],
            }

            with self.assertRaisesRegex(
                trace_reduce.TraceReduceError,
                "tolerance is required",
            ):
                trace_reduce.reduce_trace(trace_dir, root / "out", plan)

    def test_reduces_bf16_tensor_to_exact_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace_dir = root / "trace"
            output_dir = root / "fixtures"
            (trace_dir / "tensors").mkdir(parents=True)
            (trace_dir / "tensors" / "hidden.bin").write_bytes(b"\x80\x3f\x00\x40")
            _write_manifest_record(
                trace_dir,
                {
                    "dtype": "bf16",
                    "file": "tensors/hidden.bin",
                    "kind": "tensor",
                    "name": "hidden",
                    "ordinal": 0,
                    "shape": [2],
                    "stage": "qwen.encoder",
                },
            )
            plan = {
                "fixture_id": "bf16_fixture",
                "texts": [],
                "tensors": [
                    {
                        "name": "hidden",
                        "output": "hidden.npy",
                        "role": "expected",
                        "slice": [{"length": 2, "start": 0}],
                        "stage": "qwen.encoder",
                        "tolerance": {"atol": 0.0, "rtol": 0.0},
                    }
                ],
            }

            manifest = trace_reduce.reduce_trace(trace_dir, output_dir, plan)

            dtype, shape, data = _read_id4_tensor(output_dir / "hidden.id4tensor")
            self.assertEqual(dtype, "bf16")
            self.assertEqual(shape, (2,))
            self.assertEqual(data, b"\x80\x3f\x00\x40")
            record = manifest["records"][0]
            self.assertEqual(record["file"], "hidden.id4tensor")
            self.assertEqual(record["format"], "id4tensor-v1")
            self.assertEqual(record["summary"]["finite_count"], 2)
            self.assertEqual(record["summary"]["min"], 1.0)
            self.assertEqual(record["summary"]["max"], 2.0)


if __name__ == "__main__":
    unittest.main()
