# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experimental.loom_serve.models.qwen import benchmark_service


class BenchmarkServiceTest(unittest.TestCase):
    def workload(self):
        return {
            "name": "review",
            "sessions": [
                {
                    "system": "Review the supplied code.",
                    "turns": [
                        {"content": "Read this file.", "max_tokens": 128},
                        {
                            "content": "Now consider this related file.",
                            "max_tokens": 192,
                        },
                    ],
                }
            ],
        }

    def test_loads_frozen_corpus_and_hashes_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"
            path.write_text(json.dumps(self.workload()))
            workload, digest = benchmark_service.load_workload(path)
            self.assertEqual(workload, self.workload())
            self.assertEqual(len(digest), 64)
            path.write_text(json.dumps(self.workload(), indent=2))
            self.assertNotEqual(benchmark_service.load_workload(path)[1], digest)

    def test_published_source_review_workload(self):
        workload, _ = benchmark_service.load_workload(
            Path(__file__).parent / "testdata/source_review.json"
        )
        self.assertEqual(len(workload["sessions"]), 8)
        self.assertTrue(
            all(len(session["turns"]) == 2 for session in workload["sessions"])
        )
        self.assertEqual(
            sum(
                turn["max_tokens"]
                for session in workload["sessions"]
                for turn in session["turns"]
            ),
            3072,
        )

    def test_rejects_invalid_turns_before_starting_workers(self):
        for turn in (
            {},
            {"content": "", "max_tokens": 2},
            {"content": "review", "max_tokens": True},
            {"content": "review", "max_tokens": 0},
        ):
            with self.subTest(turn=turn), tempfile.TemporaryDirectory() as directory:
                workload = self.workload()
                workload["sessions"][0]["turns"] = [turn]
                path = Path(directory) / "workload.json"
                path.write_text(json.dumps(workload))
                with self.assertRaises(ValueError):
                    benchmark_service.load_workload(path)

    def test_continuation_uses_actual_reply_and_checks_retention(self):
        calls = []

        def reply(address, session, messages, maximum):
            calls.append((copy.deepcopy(messages), maximum))
            return {
                "text": "First review" if len(calls) == 1 else "Revised review",
                "usage": {"prompt_tokens_details": {"cached_tokens": len(calls) - 1}},
            }

        with patch.object(benchmark_service, "request", side_effect=reply):
            result = benchmark_service.workload_client(
                SimpleNamespace(session_prefix="test"),
                None,
                0,
                threading.Barrier(1),
                self.workload(),
            )
        self.assertEqual([maximum for _, maximum in calls], [128, 192])
        self.assertEqual(
            calls[1][0][2], {"role": "assistant", "content": "First review"}
        )
        self.assertEqual(len(calls[0][0]), 2)
        self.assertNotEqual(
            result["turns"][0]["input_sha256"], result["turns"][1]["input_sha256"]
        )
        with (
            patch.object(
                benchmark_service,
                "request",
                return_value={
                    "text": "Review",
                    "usage": {"prompt_tokens_details": {"cached_tokens": 0}},
                },
            ),
            self.assertRaisesRegex(RuntimeError, "did not reuse retained state"),
        ):
            benchmark_service.workload_client(
                SimpleNamespace(session_prefix="test"),
                None,
                0,
                threading.Barrier(1),
                self.workload(),
            )


if __name__ == "__main__":
    unittest.main()
