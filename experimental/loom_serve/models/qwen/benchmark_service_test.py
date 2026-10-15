# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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

        def reply(address, session, messages, maximum, *, model):
            self.assertEqual(model, "reviewer")
            calls.append((copy.deepcopy(messages), maximum))
            return {
                "text": "First review" if len(calls) == 1 else "Revised review",
                "usage": {"prompt_tokens_details": {"cached_tokens": len(calls) - 1}},
            }

        with patch.object(benchmark_service, "request", side_effect=reply):
            result = benchmark_service.workload_client(
                SimpleNamespace(
                    session_prefix="test", model="reviewer", retention="required"
                ),
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
                SimpleNamespace(
                    session_prefix="test", model="reviewer", retention="required"
                ),
                None,
                0,
                threading.Barrier(1),
                self.workload(),
            )

    def test_scheduled_arrivals_and_retention_misses_are_explicit(self):
        workload = self.workload()
        workload["sessions"][0]["start_after_ms"] = 25
        workload["sessions"][0]["turns"][1]["pause_ms"] = 75
        with (
            patch.object(benchmark_service.time, "sleep") as sleep,
            patch.object(
                benchmark_service,
                "request",
                return_value={
                    "text": "Review",
                    "usage": {"prompt_tokens_details": {"cached_tokens": 0}},
                },
            ),
        ):
            result = benchmark_service.workload_client(
                SimpleNamespace(
                    session_prefix="test", model="reviewer", retention="observe"
                ),
                None,
                0,
                threading.Barrier(1),
                workload,
            )
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list], [0.025, 0.075]
        )
        self.assertEqual(len(result["turns"]), 2)
        self.assertTrue(
            all(
                turn["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
                for turn in result["turns"]
            )
        )

    def test_rejects_invalid_arrival_and_think_times(self):
        for field in ("start_after_ms", "pause_ms"):
            for value in (-1, True, 1.5, "10"):
                with (
                    self.subTest(field=field, value=value),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    workload = self.workload()
                    session = workload["sessions"][0]
                    target = (
                        session if field == "start_after_ms" else session["turns"][0]
                    )
                    target[field] = value
                    path = Path(directory) / "workload.json"
                    path.write_text(json.dumps(workload))
                    with self.assertRaisesRegex(ValueError, field):
                        benchmark_service.load_workload(path)

    def test_latency_distribution_keeps_absent_samples_distinct(self):
        self.assertEqual(benchmark_service.latency_distribution([None]), {"samples": 0})
        self.assertEqual(
            benchmark_service.latency_distribution([None, *range(1, 21)]),
            {
                "samples": 20,
                "minimum": 1,
                "p50": 10,
                "p95": 19,
                "p99": 20,
                "maximum": 20,
            },
        )

    def test_single_turn_sessions_have_no_followup_samples(self):
        summary = benchmark_service.followup_cache_summary([{"turns": [{}]}])
        self.assertEqual(summary["requests"], 0)
        for group in ("hits", "misses"):
            self.assertEqual(summary[group]["requests"], 0)
            self.assertEqual(summary[group]["cached_prompt_tokens"], 0)
            self.assertEqual(summary[group]["appended_prompt_tokens"], 0)
            self.assertEqual(summary[group]["request_latency_ms"], {"samples": 0})
            self.assertEqual(summary[group]["first_text_latency_ms"], {"samples": 0})

    def test_codeword_clients_extend_the_catalog_without_repeating_identity(self):
        for index, word in ((0, "MAPLE"), (7, "ONYX"), (8, "MAPLE-1"), (16, "MAPLE-2")):
            with self.subTest(index=index):
                calls = []

                def reply(address, session, messages, maximum, *, model):
                    self.assertEqual(model, "scout")
                    self.assertEqual(session, f"test-{index}")
                    calls.append(copy.deepcopy(messages))
                    return {
                        "text": "1\n2" if len(calls) == 1 else word,
                        "usage": {"prompt_tokens_details": {"cached_tokens": 1}},
                    }

                with patch.object(benchmark_service, "request", side_effect=reply):
                    result = benchmark_service.client(
                        SimpleNamespace(
                            session_prefix="test",
                            model="scout",
                            retention="required",
                            long_lines=1,
                            max_tokens=16,
                        ),
                        None,
                        index,
                        threading.Barrier(1),
                    )
                self.assertEqual(result["verified_codeword"], word)
                self.assertIn(f"codeword {word} for later", calls[0][1]["content"])
                self.assertEqual(calls[1][2], {"role": "assistant", "content": "1\n2"})

    def test_more_than_eight_clients_route_real_http_and_report_complete_samples(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.headers["X-Loom-Session"], body))
                index = int(self.headers["X-Loom-Session"].rsplit("-", 1)[1])
                # Initial prefix sharing is not retained follow-up reuse. Half
                # of the multi-turn clients lose their state between turns.
                cached = (
                    5
                    if index == 0
                    else 10
                    if index % 2 == 0 and len(body["messages"]) > 2
                    else 0
                )
                event = {
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "Review"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 2,
                        "prompt_tokens_details": {"cached_tokens": cached},
                    },
                }
                payload = (
                    "data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *arguments):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        owner = threading.Thread(target=server.serve_forever)
        owner.start()
        try:
            with (
                tempfile.TemporaryDirectory() as directory,
                redirect_stdout(io.StringIO()) as output,
            ):
                workload = self.workload()
                workload["sessions"] = [
                    copy.deepcopy(workload["sessions"][0]) for _ in range(12)
                ]
                workload["sessions"][0]["turns"] = workload["sessions"][0]["turns"][:1]
                for index in range(1, 12, 2):
                    workload["sessions"][index]["turns"].append(
                        {"content": "Now summarize your review.", "max_tokens": 32}
                    )
                path = Path(directory) / "workload.json"
                path.write_text(json.dumps(workload))
                benchmark_service.main(
                    [
                        f"--url=http://127.0.0.1:{server.server_port}",
                        "--model=reviewer",
                        "--clients=12",
                        "--retention=observe",
                        f"--workload={path}",
                    ]
                )
                records = [json.loads(line) for line in output.getvalue().splitlines()]
        finally:
            server.shutdown()
            owner.join()
            server.server_close()
        self.assertEqual(len(requests), 29)
        self.assertEqual(len({session for session, _ in requests}), 12)
        self.assertTrue(all(body["model"] == "reviewer" for _, body in requests))
        summary = records[-1]
        self.assertEqual(summary["event"], "summary")
        self.assertEqual(summary["requests"], 29)
        self.assertEqual(summary["cached_prompt_tokens"], 55)
        self.assertEqual(summary["appended_prompt_tokens"], 525)
        self.assertEqual(summary["output_tokens_including_eos"], 58)
        self.assertEqual(summary["request_latency_ms"]["samples"], 29)
        self.assertEqual(summary["first_text_latency_ms"]["samples"], 29)
        followups = summary["followup_cache"]
        self.assertEqual(followups["requests"], 17)
        for group, count, cached, appended in (
            ("hits", 5, 50, 50),
            ("misses", 12, 0, 240),
        ):
            self.assertEqual(followups[group]["requests"], count)
            self.assertEqual(followups[group]["cached_prompt_tokens"], cached)
            self.assertEqual(followups[group]["appended_prompt_tokens"], appended)
            self.assertEqual(followups[group]["request_latency_ms"]["samples"], count)
            self.assertEqual(
                followups[group]["first_text_latency_ms"]["samples"], count
            )
        for client in records[:-1]:
            for turn in client["turns"]:
                self.assertEqual(len(turn["output_chunk_ms"]), 1)
                self.assertLessEqual(turn["started_ns"], turn["completed_ns"])


if __name__ == "__main__":
    unittest.main()
