# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The launcher's fronting proxy: one instrumented OpenAI endpoint.

The native server owns inference. This proxy sits in front of it and records
what a client experienced: time to first token, decode rate, token usage,
status and bytes, per request, as JSONL and as Prometheus counters. It also
serves the launcher's own observation routes under `/loom/`, so a client sees
one address for both inference and instrumentation.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from loom.launch.metrics import (
    LAUNCHER_PREFIX,
    Metrics,
    RequestRecord,
    route_name,
)
from loom.launch.store import JsonlWriter

#: Headers that describe this hop only and are never forwarded.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
        "accept-encoding",
    }
)


@dataclass(frozen=True)
class Upstream:
    """Where the native server listens."""

    host: str = "127.0.0.1"
    port: int = 8080
    unix_path: str | None = None

    def describe(self) -> dict:
        if self.unix_path:
            return {"unix": self.unix_path}
        return {"host": self.host, "port": self.port}

    def connect(self, timeout: float) -> http.client.HTTPConnection:
        if self.unix_path:
            connection = _UnixHTTPConnection(self.unix_path, timeout=timeout)
        else:
            connection = http.client.HTTPConnection(
                self.host, self.port, timeout=timeout
            )
        return connection


class _UnixHTTPConnection(http.client.HTTPConnection):
    """An HTTP connection to a Unix socket listener."""

    def __init__(self, path: str, timeout: float):
        super().__init__("unix", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._path)
        self.sock = sock


class Proxy:
    """An instrumented front end for one native server."""

    def __init__(
        self,
        upstream: Upstream,
        *,
        metrics: Metrics | None = None,
        request_log: JsonlWriter | None = None,
        bind_host: str = "127.0.0.1",
        bind_port: int = 0,
        status_provider: Callable[[], dict] | None = None,
        auth_token: str | None = None,
        request_timeout: float = 900.0,
    ):
        self.upstream = upstream
        self.metrics = metrics or Metrics()
        self.request_log = request_log
        self._status_provider = status_provider
        self._auth_token = auth_token
        self._request_timeout = request_timeout
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.bind_host = bind_host
        self.bind_port = bind_port

    # -- lifetime -----------------------------------------------------------
    def start(self) -> tuple[str, int]:
        handler = _make_handler(self)
        self._server = ThreadingHTTPServer((self.bind_host, self.bind_port), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="loom-launch-proxy",
            daemon=True,
        )
        self._thread.start()
        host, port = self._server.server_address[:2]
        self.bind_host, self.bind_port = host, port
        return host, port

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def address(self) -> str:
        return f"http://{self.bind_host}:{self.bind_port}"

    def status(self) -> dict:
        status = {
            "proxy": {
                "address": self.address,
                "upstream": self.upstream.describe(),
                "metrics": self.metrics.snapshot(),
            }
        }
        if self._status_provider is not None:
            try:
                status.update(self._status_provider())
            except Exception as exc:  # A failing probe must not break status.
                status["launch_error"] = str(exc)
        return status

    # -- request handling ---------------------------------------------------
    def forward(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        *,
        client: str | None,
        response_writer: "_ResponseWriter",
    ) -> RequestRecord:
        """Carries one request upstream and records what happened."""
        started_at = time.time()
        started = time.monotonic()
        route = route_name(path)
        record_id = headers.get("x-request-id") or uuid.uuid4().hex
        model: str | None = None
        stream_requested = False
        if body:
            try:
                payload = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = None
            if isinstance(payload, dict):
                candidate = payload.get("model")
                model = candidate if isinstance(candidate, str) else None
                stream_requested = bool(payload.get("stream"))
        self.metrics.begin()
        status = 0
        error: str | None = None
        ttft: float | None = None
        chunks = 0
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        cached_tokens: int | None = None
        response_bytes = 0
        try:
            connection = self.upstream.connect(self._request_timeout)
            try:
                outgoing = {
                    name: value
                    for name, value in headers.items()
                    if name.lower() not in _HOP_BY_HOP
                }
                outgoing.setdefault("Accept-Encoding", "identity")
                x_loom_model = headers.get("x-loom-model")
                if x_loom_model:
                    outgoing["X-Loom-Model"] = x_loom_model
                connection.request(method, path, body=body or None, headers=outgoing)
                response = connection.getresponse()
                self.metrics.set_upstream_up(True)
                status = response.status
                content_type = response.getheader("Content-Type") or ""
                streaming = content_type.startswith("text/event-stream")
                if streaming:
                    response_writer.begin_stream(status, response)
                    usage: dict = {}
                    while True:
                        line = response.readline()
                        if not line:
                            break
                        response_bytes += len(line)
                        response_writer.write_line(line)
                        payload = _event_payload(line)
                        if payload is None:
                            continue
                        chunks += 1
                        if ttft is None and _has_content(payload):
                            ttft = time.monotonic() - started
                        usage = _update_usage(usage, payload)
                    response_writer.end_stream()
                    prompt_tokens, completion_tokens, cached_tokens = _usage_totals(
                        usage
                    )
                else:
                    data = response.read()
                    response_bytes = len(data)
                    response_writer.write_response(status, response, data)
                    document = _json_document(data)
                    if isinstance(document, dict):
                        usage = document.get("usage")
                        if isinstance(usage, dict):
                            tokens = _usage_totals(usage)
                            prompt_tokens, completion_tokens, cached_tokens = tokens
                        candidate = document.get("model")
                        if model is None and isinstance(candidate, str):
                            model = candidate
            finally:
                connection.close()
        except (OSError, http.client.HTTPException) as exc:
            self.metrics.set_upstream_up(False)
            error = f"{type(exc).__name__}: {exc}"
            status = status or 502
            response_writer.error(
                502,
                f"upstream {self.upstream.describe()} failed: {exc}",
                error_id=record_id,
            )
        finally:
            self.metrics.end()
        duration = time.monotonic() - started
        record = RequestRecord(
            request_id=record_id,
            method=method,
            path=path,
            route=route,
            started_at=started_at,
            status=status,
            stream=stream_requested or response_writer.streamed,
            model=model,
            ttft_seconds=ttft,
            duration_seconds=duration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            chunks=chunks,
            request_bytes=len(body),
            response_bytes=response_bytes,
            client=client,
            error=error,
        )
        self.metrics.observe(record)
        if self.request_log is not None:
            self.request_log.write(record.describe())
        return record

    def authorized(self, headers: dict[str, str]) -> bool:
        if not self._auth_token:
            return True
        value = headers.get("authorization") or ""
        scheme, _, token = value.partition(" ")
        return scheme.lower() == "bearer" and token.strip() == self._auth_token


class _ResponseWriter:
    """Frames one response to the client, whichever shape the upstream chose."""

    def __init__(self, handler: BaseHTTPRequestHandler):
        self._handler = handler
        self.streamed = False
        self._started = False

    def begin_stream(self, status: int, response: http.client.HTTPResponse) -> None:
        """Starts a streamed response, framed with chunked transfer."""
        self.streamed = True
        self._handler.send_response(status)
        for name in ("Content-Type", "Cache-Control"):
            value = response.getheader(name)
            if value:
                self._handler.send_header(name, value)
        self._handler.send_header("X-Loom-Launch", "1")
        self._handler.send_header("Transfer-Encoding", "chunked")
        self._handler.send_header("Connection", "close")
        self._handler.close_connection = True
        self._handler.end_headers()
        self._started = True

    def write_response(
        self, status: int, response: http.client.HTTPResponse, data: bytes
    ) -> None:
        """Writes a complete response whose body was already read."""
        self._handler.send_response(status)
        for name in ("Content-Type", "Cache-Control"):
            value = response.getheader(name)
            if value:
                self._handler.send_header(name, value)
        self._handler.send_header("X-Loom-Launch", "1")
        self._handler.send_header("Content-Length", str(len(data)))
        self._handler.send_header("Connection", "close")
        self._handler.close_connection = True
        self._handler.end_headers()
        self._started = True
        self.write_body(data)

    def write_body(self, data: bytes) -> None:
        if not data:
            return
        try:
            self._handler.wfile.write(data)
            self._handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def write_line(self, line: bytes) -> None:
        frame = f"{len(line):x}\r\n".encode("ascii") + line + b"\r\n"
        self.write_body(frame)

    def end_stream(self) -> None:
        self.write_body(b"0\r\n\r\n")

    def error(self, status: int, message: str, *, error_id: str) -> None:
        if self._started:
            return
        payload = json.dumps(
            {
                "error": {
                    "message": message,
                    "type": "loom_launch_proxy_error",
                    "code": "UNAVAILABLE",
                    "request_id": error_id,
                }
            }
        ).encode("utf-8")
        try:
            self._handler.send_response(status)
            self._handler.send_header("Content-Type", "application/json")
            self._handler.send_header("Content-Length", str(len(payload)))
            self._handler.send_header("Connection", "close")
            self._handler.close_connection = True
            self._handler.end_headers()
            self._handler.wfile.write(payload)
            self._handler.wfile.flush()
            self._started = True
        except (BrokenPipeError, ConnectionResetError):
            pass


def _make_handler(proxy: Proxy) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "loom-launch/1"
        sys_version = ""

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def do_DELETE(self) -> None:
            self._dispatch("DELETE")

        def do_HEAD(self) -> None:
            self._dispatch("HEAD")

        def log_message(self, format: str, *args: object) -> None:
            # Request logging is structured; the default stderr line is noise.
            del format, args

        def _dispatch(self, method: str) -> None:
            headers = {name.lower(): value for name, value in self.headers.items()}
            body = self._read_body()
            path = self.path
            client = f"{self.client_address[0]}:{self.client_address[1]}"
            if path.startswith(LAUNCHER_PREFIX):
                self._serve_launcher(path, headers)
                return
            if not proxy.authorized(headers):
                self._write_json(
                    401,
                    {
                        "error": {
                            "message": "the launcher requires a bearer token",
                            "type": "loom_launch_auth_error",
                            "code": "PERMISSION_DENIED",
                        }
                    },
                )
                return
            writer = _ResponseWriter(self)
            proxy.forward(
                method, path, headers, body, client=client, response_writer=writer
            )

        def _read_body(self) -> bytes:
            length = self.headers.get("Content-Length")
            if length and length.isdigit():
                remaining = int(length)
                chunks = []
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 1 << 16))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                return b"".join(chunks)
            encoding = (self.headers.get("Transfer-Encoding") or "").lower()
            if "chunked" in encoding:
                chunks = []
                while True:
                    size_line = self.rfile.readline().strip()
                    if not size_line:
                        break
                    try:
                        size = int(size_line.split(b";")[0], 16)
                    except ValueError:
                        break
                    if size == 0:
                        self.rfile.readline()
                        break
                    chunks.append(self.rfile.read(size))
                    self.rfile.readline()
                return b"".join(chunks)
            return b""

        def _serve_launcher(self, path: str, headers: dict[str, str]) -> None:
            if not proxy.authorized(headers):
                self._write_json(
                    401,
                    {
                        "error": {
                            "message": "the launcher requires a bearer token",
                            "type": "loom_launch_auth_error",
                            "code": "PERMISSION_DENIED",
                        }
                    },
                )
                return
            target = path.split("?", 1)[0]
            if target == LAUNCHER_PREFIX + "metrics":
                payload = proxy.metrics.prometheus().encode("utf-8")
                self._write_raw(200, "text/plain; version=0.0.4", payload)
                return
            if target == LAUNCHER_PREFIX + "status":
                self._write_json(200, proxy.status())
                return
            if target == LAUNCHER_PREFIX + "requests":
                self._write_json(
                    200, {"object": "list", "data": proxy.metrics.recent()}
                )
                return
            self._write_json(
                404,
                {
                    "error": {
                        "message": f"{path} is not a launcher route",
                        "type": "loom_launch_error",
                        "code": "NOT_FOUND",
                    }
                },
            )

        def _write_json(self, status: int, document: dict) -> None:
            payload = json.dumps(document, indent=1).encode("utf-8")
            self._write_raw(status, "application/json", payload)

        def _write_raw(self, status: int, content_type: str, payload: bytes) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.close_connection = True
                self.end_headers()
                self.wfile.write(payload)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def _json_document(data: bytes) -> dict | None:
    try:
        document = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return document if isinstance(document, dict) else None


def _event_payload(line: bytes) -> dict | None:
    """Reads one SSE line's JSON payload, ignoring framing and `[DONE]`."""
    if not line.startswith(b"data:"):
        return None
    payload = line[len(b"data:") :].strip()
    if not payload or payload == b"[DONE]":
        return None
    try:
        document = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return document if isinstance(document, dict) else None


def _has_content(payload: dict) -> bool:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    choice = choices[0]
    if not isinstance(choice, dict):
        return False
    delta = choice.get("delta") or choice.get("message") or {}
    if not isinstance(delta, dict):
        return False
    if delta.get("reasoning_content"):
        return True
    if delta.get("content"):
        return True
    return bool(delta.get("tool_calls"))


def _update_usage(usage: dict, payload: dict) -> dict:
    candidate = payload.get("usage")
    if isinstance(candidate, dict):
        return candidate
    return usage


def _usage_totals(usage: dict) -> tuple[int | None, int | None, int | None]:
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    return (
        prompt if isinstance(prompt, int) else None,
        completion if isinstance(completion, int) else None,
        cached if isinstance(cached, int) else None,
    )
