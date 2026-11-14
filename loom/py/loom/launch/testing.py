# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A local stand-in for a served Loom model, for the launcher's own tests.

It answers the endpoints the launcher reads and the ones it forwards, with
configurable latency and token counts, so proxy accounting, client timing, chat
turns and benchmark rows can be tested without a device or a checkpoint.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass
class FakeModelConfig:
    """How the stand-in model behaves."""

    model: str = "test/model"
    completion_tokens: int = 8
    #: Seconds to wait before the first content token.
    first_token_delay: float = 0.0
    #: Seconds between content tokens.
    token_delay: float = 0.0
    #: Status to fail every completion with, when nonzero.
    fail_status: int = 0
    #: Emit a reasoning delta before the answer.
    reasoning: bool = False
    #: Report this many prompt tokens instead of counting words.
    prompt_tokens: int | None = None


@dataclass
class FakeOpenAIServer:
    """An HTTP server impersonating one prepared text deployment."""

    config: FakeModelConfig = field(default_factory=FakeModelConfig)
    requests: list[dict] = field(default_factory=list)
    _server: ThreadingHTTPServer | None = None
    _thread: threading.Thread | None = None

    def start(self) -> str:
        handler = _make_handler(self)
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="fake-loom-serve", daemon=True
        )
        self._thread.start()
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> FakeOpenAIServer:
        self.url = self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- observed state -----------------------------------------------------
    def completions(self) -> list[dict]:
        return [
            entry
            for entry in self.requests
            if entry.get("path") == "/v1/chat/completions"
        ]


def _make_handler(server: FakeOpenAIServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "fake-loom-serve/1"

        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/healthz":
                self._json(200, {"status": "ready"})
                return
            if path == "/v1/models":
                self._json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": server.config.model,
                                "object": "model",
                                "owned_by": "local",
                                "loom": {
                                    "kind": "text",
                                    "endpoint": "/v1/chat/completions",
                                },
                            }
                        ],
                    },
                )
                return
            if path == "/v1/status":
                self._json(
                    200,
                    {
                        "models": {
                            server.config.model: {
                                "state": "ready",
                                "name": server.config.model,
                            }
                        }
                    },
                )
                return
            self._json(404, {"error": {"message": "not found", "code": "NOT_FOUND"}})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                payload = {}
            server.requests.append(
                {
                    "path": path,
                    "payload": payload,
                    "headers": {
                        name.lower(): value for name, value in self.headers.items()
                    },
                }
            )
            if path != "/v1/chat/completions":
                self._json(
                    404, {"error": {"message": "not found", "code": "NOT_FOUND"}}
                )
                return
            config = server.config
            if config.fail_status:
                self._json(
                    config.fail_status,
                    {
                        "error": {
                            "message": "the model refused the request",
                            "type": "loom_request_error",
                            "code": "FAILED_PRECONDITION",
                        }
                    },
                )
                return
            prompt_tokens = config.prompt_tokens
            if prompt_tokens is None:
                prompt_tokens = sum(
                    len(str((message or {}).get("content") or "").split())
                    for message in payload.get("messages") or []
                )
            completion_tokens = config.completion_tokens
            if payload.get("stream"):
                self._stream(prompt_tokens, completion_tokens)
            else:
                self._json(
                    200,
                    _completion_document(config, prompt_tokens, completion_tokens),
                )

        def _stream(self, prompt_tokens: int, completion_tokens: int) -> None:
            config = server.config
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()

            def event(document: dict) -> None:
                data = b"data: " + json.dumps(document).encode("utf-8") + b"\n\n"
                frame = f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n"
                self.wfile.write(frame)
                self.wfile.flush()

            event(_delta_chunk(config, role="assistant", content=""))
            if config.reasoning:
                if config.first_token_delay:
                    time.sleep(config.first_token_delay)
                event(_delta_chunk(config, reasoning="Considering the request."))
            for index in range(completion_tokens):
                if index == 0 and config.first_token_delay:
                    time.sleep(config.first_token_delay)
                if config.token_delay:
                    time.sleep(config.token_delay)
                event(_delta_chunk(config, content=f" token{index}"))
            event(
                {
                    "object": "chat.completion.chunk",
                    "model": config.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            event(
                {
                    "object": "chat.completion.chunk",
                    "model": config.model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    },
                }
            )
            done = b"data: [DONE]\n\n"
            frame = f"{len(done):x}\r\n".encode("ascii") + done + b"\r\n"
            self.wfile.write(frame + b"0\r\n\r\n")
            self.wfile.flush()

        def _json(self, status: int, document: dict) -> None:
            payload = json.dumps(document).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()

        def do_DELETE(self) -> None:
            self.do_POST()

    return Handler


def _delta_chunk(
    config: FakeModelConfig,
    *,
    role: str | None = None,
    content: str = "",
    reasoning: str = "",
) -> dict:
    delta: dict = {}
    if role:
        delta["role"] = role
    if reasoning:
        delta["reasoning_content"] = reasoning
    if content:
        delta["content"] = content
    return {
        "object": "chat.completion.chunk",
        "model": config.model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }


def _completion_document(
    config: FakeModelConfig, prompt_tokens: int, completion_tokens: int
) -> dict:
    return {
        "object": "chat.completion",
        "model": config.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "".join(
                        f" token{index}" for index in range(completion_tokens)
                    ),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }
