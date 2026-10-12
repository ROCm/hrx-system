# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
import http.server
import json
import threading
import unittest
from urllib.parse import urlsplit

from experimental.loom_serve.http.chat import request


@contextlib.contextmanager
def endpoint(payload, status=200):
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(
                {
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": json.loads(
                        self.rfile.read(int(self.headers["Content-Length"]))
                    ),
                }
            )
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    with http.server.HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}
        )
        thread.start()
        try:
            yield urlsplit(f"http://127.0.0.1:{server.server_port}"), requests
        finally:
            server.shutdown()
            thread.join()


def event(delta=None, reason=None, *, model="reviewer", usage=None):
    result = {
        "model": model,
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": reason}],
    }
    if usage is not None:
        result["usage"] = usage
    return result


def stream(*events, done=True):
    payload = b": keepalive\n\n" + b"".join(
        b"data: " + json.dumps(value).encode() + b"\n\n" for value in events
    )
    return payload + (b"data: [DONE]\n\n" if done else b"")


class ChatTest(unittest.TestCase):
    def test_complete_text_and_route_are_copied_with_timestamps(self):
        usage = {"prompt_tokens": 17, "completion_tokens": 4, "total_tokens": 21}
        messages = [{"role": "user", "content": "Read this code."}]
        payload = stream(
            event({"role": "assistant"}),
            event({"content": "héllo\n"}),
            event({"content": "world"}),
            event(reason="stop"),
            {"model": "reviewer", "choices": [], "usage": usage},
        )
        with endpoint(payload) as (address, requests):
            result = request(
                address, "agent-7", messages, 32, model="reviewer", checkpoint="base"
            )
        self.assertEqual(result["text"], "héllo\nworld")
        self.assertEqual(result["tool_calls"], [])
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["usage"], usage)
        self.assertEqual(len(result["output_chunk_ms"]), 2)
        self.assertLessEqual(result["first_text_ms"], result["request_ms"])
        self.assertEqual(
            result["request_ms"],
            (result["completed_ns"] - result["started_ns"]) / 1e6,
        )
        self.assertEqual(requests[0]["path"], "/v1/chat/completions")
        self.assertEqual(requests[0]["headers"]["X-Loom-Session"], "agent-7")
        self.assertEqual(requests[0]["headers"]["X-Loom-Checkpoint"], "base")
        self.assertEqual(
            requests[0]["body"],
            {
                "model": "reviewer",
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": 0,
                "max_tokens": 32,
            },
        )

    def test_tool_fragments_are_joined_without_inventing_text_latency(self):
        tools = [{"type": "function", "function": {"name": "read"}}]
        usage = {"prompt_tokens": 12, "completion_tokens": 9}
        payload = stream(
            event(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-7",
                            "type": "function",
                            "function": {"name": "read", "arguments": '{"path":'},
                        }
                    ]
                }
            ),
            event(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '"a.c"}'},
                        }
                    ]
                }
            ),
            event(reason="tool_calls", usage=usage),
        )
        with endpoint(payload) as (address, requests):
            result = request(address, "agent", [], 32, model="reviewer", tools=tools)
        self.assertEqual(result["text"], "")
        self.assertIsNone(result["first_text_ms"])
        self.assertEqual(len(result["output_chunk_ms"]), 2)
        self.assertEqual(
            result["tool_calls"],
            [
                {
                    "id": "call-7",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"a.c"}'},
                }
            ],
        )
        self.assertEqual(requests[0]["body"]["tools"], tools)
        self.assertNotIn("X-Loom-Checkpoint", requests[0]["headers"])

    def test_length_is_reported_for_the_workload_oracle_to_judge(self):
        payload = stream(
            event(
                {"content": "partial"},
                "length",
                usage={"prompt_tokens": 3, "completion_tokens": 1},
            )
        )
        with endpoint(payload) as (address, _):
            result = request(address, "agent", [], 1, model="reviewer")
        self.assertEqual(result["finish_reason"], "length")

    def test_failures_do_not_become_successful_samples(self):
        usage = {"prompt_tokens": 3, "completion_tokens": 1}
        for payload, diagnostic in (
            (
                stream(event({"content": "partial"}, "stop", usage=usage), done=False),
                "Incomplete successful SSE",
            ),
            (
                stream(event({"content": "partial"}, "stop")),
                "Incomplete successful SSE",
            ),
            (
                stream(event({"content": "partial"}, usage=usage)),
                "Incomplete successful SSE",
            ),
            (
                stream(event({"content": "wrong"}, "stop", model="other", usage=usage)),
                "wrong model",
            ),
            (stream({"error": {"message": "device failed"}}), "device failed"),
            (stream(event(reason="tool_calls", usage=usage)), "Tool deltas"),
            (
                stream(
                    event({"tool_calls": [{"index": 2}]}, "tool_calls", usage=usage)
                ),
                "indexes are not contiguous",
            ),
        ):
            with self.subTest(diagnostic=diagnostic), endpoint(payload) as (address, _):
                with self.assertRaisesRegex(RuntimeError, diagnostic):
                    request(address, "agent", [], 1, model="reviewer")
        with endpoint(b'{"error":"overloaded"}', status=503) as (address, _):
            with self.assertRaisesRegex(RuntimeError, "HTTP 503.*overloaded"):
                request(address, "agent", [], 1, model="reviewer")


if __name__ == "__main__":
    unittest.main()
