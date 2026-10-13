# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Actual HTTP chat response forms over a source-JIT CPU model."""

import http.client
import json
import shutil
import struct
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from loom_serve_testing.http import control, running_server

SERVER = Path(sys.argv.pop(1)).resolve()
PACKAGE = Path(sys.argv.pop(1)).resolve().parent


class ClientTest(unittest.TestCase):
    def request(self, address, body, session=None, expected=200):
        connection = http.client.HTTPConnection(address.hostname, address.port)
        headers = {"Content-Type": "application/json"}
        if session is not None:
            headers["X-Loom-Session"] = session
        try:
            connection.request(
                "POST", "/v1/chat/completions", json.dumps(body), headers
            )
            response = connection.getresponse()
            payload = response.read()
            self.assertEqual(response.status, expected, payload)
            if body.get("stream") and expected == 200:
                self.assertEqual(
                    response.getheader("Content-Type"), "text/event-stream"
                )
                lines = [
                    line[6:]
                    for line in payload.splitlines()
                    if line.startswith(b"data: ")
                ]
                self.assertEqual(lines.pop(), b"[DONE]")
                return [json.loads(line) for line in lines]
            self.assertEqual(response.getheader("Content-Type"), "application/json")
            return json.loads(payload)
        finally:
            connection.close()

    @staticmethod
    def body(**options):
        return {
            "model": "text",
            "messages": [{"role": "user", "content": "abxa"}],
            "max_tokens": 2,
            **options,
        }

    def check_completion(self, result, text, finish, inputs, outputs):
        self.assertTrue(result["id"].startswith("loom-"))
        self.assertEqual(result["object"], "chat.completion")
        self.assertGreater(result["created"], 0)
        self.assertEqual(result["model"], "text")
        self.assertEqual(
            result["choices"],
            [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish,
                    "logprobs": None,
                }
            ],
        )
        usage = result["usage"]
        self.assertEqual(usage["prompt_tokens"], inputs)
        self.assertEqual(usage["completion_tokens"], outputs)
        self.assertEqual(usage["total_tokens"], inputs + outputs)
        self.assertLessEqual(usage["prompt_tokens_details"]["cached_tokens"], inputs)

    def check_stream(self, events, *, include_usage):
        chunks = events[:-1] if include_usage else events
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")
        for event in events:
            self.assertEqual(event["id"], events[0]["id"])
            self.assertEqual(event["created"], events[0]["created"])
            self.assertGreater(event["created"], 0)
            self.assertEqual(event["model"], "text")
            self.assertEqual(event["object"], "chat.completion.chunk")
        for chunk in chunks:
            if include_usage:
                self.assertIsNone(chunk["usage"])
            else:
                self.assertNotIn("usage", chunk)
        if include_usage:
            self.assertEqual(events[-1]["choices"], [])
            self.assertEqual(events[-1]["usage"]["completion_tokens"], 2)
        return "".join(
            chunk["choices"][0]["delta"].get("content", "") for chunk in chunks
        )

    def test_output_forms_share_execution_history_and_failure_semantics(self):
        for workers in (1, 4):
            with (
                self.subTest(workers=workers),
                tempfile.TemporaryDirectory() as temporary,
            ):
                directory = Path(temporary)
                header = b'{"bias":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}'
                header += b" " * (-len(header) % 8)
                (directory / "weights.safetensors").write_bytes(
                    struct.pack("<Q", len(header)) + header + struct.pack("<I", 2)
                )
                declaration = {
                    "name": "text",
                    "base_directory": str(PACKAGE),
                    "definition": {
                        **json.loads((PACKAGE / "model.json").read_text())[
                            "definition"
                        ],
                        "weights": str(directory / "weights.safetensors"),
                        "tokenizer": str(PACKAGE / "tokenizer.json"),
                        "rows": 4,
                        "context_capacity": 96,
                        "pool_capacity": 256,
                        "prefill_capacity": 32,
                        "checkpoint_capacity": 2,
                        "max_tokens": 16,
                        "draft_depth": 0,
                    },
                }
                (directory / "model.json").write_text(json.dumps(declaration))
                log = directory / "server.log"
                with running_server(
                    [
                        str(SERVER),
                        "--model=model.json",
                        "--device=task",
                        "--pool_backing=fixed",
                        f"--workers={workers}",
                        "--listen=127.0.0.1:0",
                    ],
                    log,
                    cwd=directory,
                ) as (process, bound):
                    self.assertIsNotNone(bound, log.read_text())
                    address = urlsplit(f"http://{bound}")
                    for options in (
                        {},
                        {"stream": False},
                        {"stream": None},
                        {"stream": False, "stream_options": None},
                    ):
                        result = self.request(address, self.body(**options))
                        self.check_completion(result, "bx", "length", 4, 2)
                    result = self.request(address, self.body(max_tokens=16))
                    self.check_completion(result, "bxaaaxb", "stop", 4, 8)
                    result = self.request(
                        address,
                        self.body(
                            messages=[{"role": "user", "content": "a" * 11}],
                            max_tokens=16,
                        ),
                    )
                    self.check_completion(result, "", "stop", 11, 1)

                    # Real concurrent requests can use different output modes
                    # without changing the model's scheduling or math.
                    with ThreadPoolExecutor(max_workers=4) as clients:
                        requests = [
                            clients.submit(self.request, address, self.body()),
                            clients.submit(
                                self.request, address, self.body(stream=True)
                            ),
                            clients.submit(
                                self.request,
                                address,
                                self.body(
                                    stream=True, stream_options={"include_usage": True}
                                ),
                            ),
                            clients.submit(
                                self.request, address, self.body(stream=False)
                            ),
                        ]
                        self.check_completion(
                            requests[0].result(), "bx", "length", 4, 2
                        )
                        self.assertEqual(
                            self.check_stream(
                                requests[1].result(), include_usage=False
                            ),
                            "bx",
                        )
                        self.assertEqual(
                            self.check_stream(requests[2].result(), include_usage=True),
                            "bx",
                        )
                        self.check_completion(
                            requests[3].result(), "bx", "length", 4, 2
                        )

                    # A JSON completion is a valid retained prefix for SSE and
                    # vice versa. Full replay is the independent comparison.
                    first = self.request(address, self.body(), "retained")
                    followup = self.body(
                        messages=[
                            {"role": "user", "content": "abxa"},
                            first["choices"][0]["message"],
                            {"role": "user", "content": "b"},
                        ]
                    )
                    streamed = self.request(
                        address,
                        {
                            **followup,
                            "stream": True,
                            "stream_options": {"include_usage": True},
                        },
                        "retained",
                    )
                    text = self.check_stream(streamed, include_usage=True)
                    self.assertEqual(
                        streamed[-1]["usage"]["prompt_tokens_details"]["cached_tokens"],
                        5,
                    )
                    replayed = self.request(address, followup)
                    self.check_completion(replayed, text, "length", 8, 2)
                    last = self.request(
                        address,
                        self.body(
                            messages=[
                                *followup["messages"],
                                {"role": "assistant", "content": text},
                                {"role": "user", "content": "a"},
                            ]
                        ),
                        "retained",
                    )
                    self.assertEqual(
                        last["usage"]["prompt_tokens_details"]["cached_tokens"], 9
                    )

                    for options in (
                        {"stream": "false"},
                        {"stream_options": {}},
                        {"stream": False, "stream_options": {"include_usage": True}},
                        {"temperature": 0.7},
                        {"tool_choice": "required"},
                    ):
                        error = self.request(
                            address, self.body(**options), expected=400
                        )
                        self.assertIn("message", error["error"])

                    # A source package can fail while interpreting completed
                    # output. The native service must encode that real failure
                    # according to whether successful headers were published.
                    broken = directory / "broken"
                    broken.mkdir()
                    for name in (
                        "advance.loom",
                        "control.loom",
                        "prepare.loom",
                        "model.json",
                        "tokenizer.json",
                        "weights.loom",
                    ):
                        shutil.copyfile(PACKAGE / name, broken / name)
                    path = broken / "control.loom"
                    source = path.read_text()
                    marker = "%empty = global.load @no_tools : buffer\n  func.return %empty : buffer"
                    self.assertEqual(source.count(marker), 1)
                    path.write_text(
                        source.replace(marker, marker.replace("@no_tools", "@empty"))
                    )
                    control(
                        address,
                        "POST",
                        "/v1/models/register",
                        202,
                        body=json.dumps(
                            {
                                **declaration,
                                "name": "broken",
                                "base_directory": str(broken),
                            }
                        ),
                    )
                    failed = self.request(
                        address, self.body(model="broken"), expected=500
                    )
                    self.assertIn("message", failed["error"])
                    failed = self.request(
                        address, self.body(model="broken", stream=True)
                    )
                    self.assertIn("message", failed[-1]["error"])
                    result = self.request(address, self.body())
                    self.check_completion(result, "bx", "length", 4, 2)
                self.assertEqual(process.returncode, 0, log.read_text())


if __name__ == "__main__":
    unittest.main()
