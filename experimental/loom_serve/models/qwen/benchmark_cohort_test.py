# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
import copy
import io
import json
import re
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from experimental.loom_serve.models.qwen import benchmark_cohort as benchmark


@contextlib.contextmanager
def endpoint(fault=None):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            status = 200
            content_type = "application/json"
            if self.path == "/apply-template":
                formatted = benchmark.prompt(body["messages"])
                payload = json.dumps(
                    {"prompt": formatted + ("bad" if fault == "template" else "")}
                ).encode()
            elif self.path == "/tokenize":
                if "messages" in body:
                    formatted = benchmark.prompt(body["messages"])
                else:
                    formatted = body["content"]
                payload = json.dumps({"tokens": list(formatted.encode())}).encode()
            elif self.path == "/v1/chat/completions":
                messages = body["messages"]
                first = len(messages) == 2
                agent = int(re.search(r"agent (\d+)", messages[1]["content"])[1])
                word = re.search(r"codeword ([A-Z0-9-]+)", messages[1]["content"])[1]
                text = benchmark.COUNTING if first else word
                reason = "stop"
                usage = {
                    "prompt_tokens": len(benchmark.prompt(messages).encode()),
                    "completion_tokens": len(text.split()) + 1,
                }
                if agent == 0:
                    if fault == "length" and first:
                        text, reason = "1\n2", "length"
                    elif fault == "recall" and not first:
                        text = "wrong"
                    elif fault == "usage":
                        usage["prompt_tokens"] += 1
                    elif fault == "http":
                        status = 503
                content_type = "text/event-stream"
                event = {
                    "model": body["model"],
                    "choices": [{"delta": {"content": text}, "finish_reason": reason}],
                    "usage": usage,
                }
                payload = (
                    b"data: " + json.dumps(event).encode() + b"\n\ndata: [DONE]\n\n"
                )
            else:
                status, payload = 404, b"unknown route"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}
        )
        thread.start()
        try:
            yield urlsplit(f"http://127.0.0.1:{server.server_port}"), requests
        finally:
            server.shutdown()
            thread.join()


class BenchmarkCohortTest(unittest.TestCase):
    def test_complete_protocol_on_each_tokenizer_adapter(self):
        for engine in ("loom", "llama", "vllm"):
            with self.subTest(engine=engine), endpoint() as (address, requests):
                result = benchmark.cohort(
                    address, engine, "reviewer", "measure8", 2, 32
                )
            self.assertEqual(result["failures"], [])
            self.assertEqual(result["completed_requests"], 4)
            self.assertEqual(result["verified_requests"], 4)
            self.assertGreater(result["output_tokens_per_second"], 0)
            self.assertGreater(
                len(result["responses"][1]["token_ids"][0]),
                len(result["responses"][0]["token_ids"][0]),
            )
            chats = [body for path, body in requests if path == "/v1/chat/completions"]
            self.assertEqual(
                sorted(body["max_tokens"] for body in chats), [16, 16, 128, 128]
            )
            self.assertTrue(all(body["temperature"] == 0 for body in chats))
            tokenizations = [body for path, body in requests if path == "/tokenize"]
            if engine == "vllm":
                self.assertTrue(
                    all(
                        body["chat_template_kwargs"] == {"enable_thinking": False}
                        for body in tokenizations
                    )
                )
            elif engine == "llama":
                self.assertTrue(
                    all(
                        body["parse_special"] and not body["add_special"]
                        for body in tokenizations
                    )
                )
            for item in result["responses"]:
                self.assertEqual(item["turns"][0]["text"], benchmark.COUNTING)
                self.assertEqual(item["turns"][1]["text"], item["word"])
                self.assertEqual(item["followup"][2]["content"], benchmark.COUNTING)
                self.assertGreaterEqual(
                    item["turns"][0]["started_ns"], result["started_ns"]
                )
                self.assertLessEqual(
                    item["turns"][1]["completed_ns"], result["completed_ns"]
                )

    def test_failure_keeps_successful_peers_and_raw_failed_outputs(self):
        for fault, completed, verified in (
            ("length", 3, 2),
            ("recall", 4, 3),
            ("usage", 3, 2),
            ("http", 2, 2),
        ):
            with self.subTest(fault=fault), endpoint(fault) as (address, _):
                result = benchmark.cohort(
                    address, "loom", "reviewer", "measure8", 2, 32
                )
            self.assertEqual(result["completed_requests"], completed)
            self.assertEqual(result["verified_requests"], verified)
            self.assertTrue(result["failures"])
            self.assertIsNone(result["output_tokens_per_second"])
            self.assertEqual(
                result["attempted_requests"], completed + (fault == "http")
            )
            self.assertTrue(result["responses"][0]["failures"])
            self.assertEqual(result["responses"][1]["failures"], [])
            self.assertEqual(len(result["responses"][1]["turns"]), 2)
            if fault == "length":
                self.assertEqual(result["responses"][0]["turns"][0]["text"], "1\n2")

    def test_template_mismatch_prevents_timed_submission(self):
        with endpoint("template") as (address, requests):
            result = benchmark.cohort(address, "llama", "reviewer", "measure1", 1, 32)
        self.assertTrue(result["failures"])
        self.assertEqual(result["completed_requests"], 0)
        self.assertFalse(any(path == "/v1/chat/completions" for path, _ in requests))

    def test_reference_checks_exact_inputs_outputs_and_counts(self):
        with endpoint() as (address, _):
            result = benchmark.cohort(address, "loom", "reviewer", "measure1", 1, 32)
        self.assertEqual(benchmark.compare(result, result), [])
        changed = copy.deepcopy(result)
        changed["responses"][0]["token_ids"][0][0] += 1
        changed["responses"][0]["turns"][0]["usage"]["completion_tokens"] += 1
        failures = benchmark.compare(result, changed)
        self.assertEqual(len(failures), 2)
        self.assertIn("token_ids differs", failures[0])
        self.assertIn("completion_tokens differs", failures[1])
        with endpoint() as (address, _):
            compared = benchmark.cohort(
                address, "loom", "reviewer", "measure1", 1, 32, reference=changed
            )
        self.assertEqual(compared["verified_requests"], 2)
        self.assertEqual(len(compared["failures"]), 2)
        self.assertIsNone(compared["output_tokens_per_second"])

    def test_cli_retains_failure_and_returns_nonzero(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            endpoint("recall") as (address, _),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            output = Path(directory) / "result.json"
            code = benchmark.main(
                [
                    "--url",
                    address.geturl(),
                    "--model",
                    "reviewer",
                    "--engine",
                    "loom",
                    "--clients",
                    "2",
                    "--no-warmup",
                    "--output",
                    str(output),
                ]
            )
            results = json.loads(output.read_text())
        self.assertEqual(code, 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["planned_requests"], 4)
        self.assertEqual(results[0]["completed_requests"], 4)
        self.assertEqual(results[0]["verified_requests"], 3)


if __name__ == "__main__":
    unittest.main()
