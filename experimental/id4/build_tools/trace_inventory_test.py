#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import trace_inventory


def _write_manifest_record(trace_dir: Path, record: dict[str, object]) -> None:
    with (trace_dir / "manifest.jsonl").open("a", encoding="utf-8") as manifest:
        json.dump(record, manifest, sort_keys=True)
        manifest.write("\n")


class TraceInventoryTest(unittest.TestCase):
    def test_builds_raw_trace_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace_dir = root / "trace"
            (trace_dir / "texts").mkdir(parents=True)
            (trace_dir / "tensors").mkdir()

            prompt_payload = b"structured prompt\n"
            (trace_dir / "texts" / "prompt.txt").write_bytes(prompt_payload)
            token_payload = struct.pack("<3i", 10, 11, 12)
            (trace_dir / "tensors" / "token_ids.bin").write_bytes(token_payload)
            condition_payload = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
            (trace_dir / "tensors" / "condition.bin").write_bytes(condition_payload)
            hidden_payload = struct.pack("<2f", 5.0, 6.0)
            (trace_dir / "tensors" / "hidden.bin").write_bytes(hidden_payload)

            _write_manifest_record(
                trace_dir,
                {
                    "bytes": len(prompt_payload),
                    "file": "texts/prompt.txt",
                    "kind": "text",
                    "name": "wrapped_prompt",
                    "ordinal": 0,
                    "producer": "conditioner",
                    "stage": "qwen.prompt",
                },
            )
            _write_manifest_record(
                trace_dir,
                {
                    "kind": "json",
                    "name": "selected_out_layers",
                    "ordinal": 1,
                    "producer": "conditioner",
                    "stage": "qwen.prompt",
                    "value": {"layers": [0, 3]},
                },
            )
            _write_manifest_record(
                trace_dir,
                {
                    "bytes": len(token_payload),
                    "dtype": "i32",
                    "elements": 3,
                    "file": "tensors/token_ids.bin",
                    "kind": "tensor",
                    "name": "token_ids",
                    "ordinal": 2,
                    "producer": "conditioner",
                    "shape": [3],
                    "stage": "qwen.prompt",
                    "summary": {
                        "finite_count": 3,
                        "inf_count": 0,
                        "max": 12,
                        "mean": 11,
                        "min": 10,
                        "nan_count": 0,
                    },
                },
            )
            _write_manifest_record(
                trace_dir,
                {
                    "bytes": len(condition_payload),
                    "dtype": "f32",
                    "elements": 4,
                    "file": "tensors/condition.bin",
                    "kind": "tensor",
                    "name": "condition",
                    "ordinal": 3,
                    "producer": "qwen3_vl",
                    "shape": [2, 2],
                    "stage": "qwen.encoder",
                },
            )
            _write_manifest_record(
                trace_dir,
                {
                    "bytes": len(hidden_payload),
                    "dtype": "f32",
                    "elements": 2,
                    "file": "tensors/hidden.bin",
                    "kind": "tensor",
                    "name": "ideogram4.cond.layers.0.hidden",
                    "ordinal": 4,
                    "producer": "dit",
                    "shape": [2],
                    "stage": "graph.debug",
                },
            )

            inventory = trace_inventory.build_trace_inventory(
                trace_dir, hash_payloads=True
            )

            self.assertEqual(inventory["record_count"], 5)
            self.assertEqual(
                inventory["kind_counts"], {"json": 1, "tensor": 3, "text": 1}
            )
            self.assertEqual(
                inventory["family_counts"],
                {
                    "ideogram4.dit.layer.boundary": 1,
                    "qwen.encoder.boundary": 1,
                    "qwen.prompt": 3,
                },
            )
            self.assertEqual(
                [stage["stage"] for stage in inventory["stages"]],
                ["qwen.prompt", "qwen.encoder", "graph.debug"],
            )
            self.assertEqual(
                inventory["coverage"]["qwen"],
                {
                    "has_condition": True,
                    "has_prompt_attention_mask": False,
                    "has_prompt_tokens": True,
                    "has_selected_hidden_states": False,
                    "internal_record_count": 0,
                    "selected_out_layers": [0, 3],
                },
            )
            self.assertEqual(
                inventory["coverage"]["ideogram4_dit"]["recorded_layer_boundaries"],
                [{"branch": "cond", "layer": 0, "record_count": 1}],
            )

            token_record = inventory["stages"][0]["records"][2]
            self.assertEqual(token_record["name"], "token_ids")
            self.assertEqual(token_record["summary"]["mean"], 11)
            self.assertEqual(
                token_record["file_sha256"],
                hashlib.sha256(token_payload).hexdigest(),
            )

    def test_writes_inventory_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace_dir = root / "trace"
            trace_dir.mkdir()
            _write_manifest_record(
                trace_dir,
                {
                    "kind": "json",
                    "name": "metadata",
                    "ordinal": 0,
                    "stage": "unit",
                    "value": {"value": 1},
                },
            )
            output_path = root / "out" / "inventory.json"

            inventory = trace_inventory.write_trace_inventory(
                trace_dir, output_path, hash_payloads=False
            )

            self.assertEqual(json.loads(output_path.read_text()), inventory)


if __name__ == "__main__":
    unittest.main()
