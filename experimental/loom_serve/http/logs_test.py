# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import collections
import contextlib
import http.server
import json
import socket
import socketserver
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from experimental.loom_serve.http.logs import LogClient, ModelNotFound


def batch(model, records, next_position, *, identity="100:1", oldest=0):
    header = {
        "event": "stream",
        "scope": "server" if model is None else "model",
        "identity": identity,
        "oldest": oldest,
        "head": next_position,
        "capacity": 128,
        "partial": oldest != 0,
        "counters_available": False,
    }
    if model is not None:
        header["model"] = model
    footer = {
        "event": "cursor",
        "next": next_position,
        "value": f"{identity}:{next_position}",
    }
    return b"".join(
        json.dumps(record).encode() + b"\n" for record in [header, *records, footer]
    )


@contextlib.contextmanager
def endpoint(replies, *, unix_socket=None):
    replies = collections.deque(replies)
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, dict(self.headers)))
            status, payload = (
                replies.popleft() if replies else (500, b"unexpected poll")
            )
            self.send_response(status)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    server = (
        socketserver.UnixStreamServer(unix_socket, Handler)
        if unix_socket is not None
        else http.server.HTTPServer(("127.0.0.1", 0), Handler)
    )
    origin = (
        "http://not-a-tcp-host.invalid:8099"
        if unix_socket is not None
        else f"http://127.0.0.1:{server.server_port}"
    )
    with server:
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}
        )
        thread.start()
        try:
            yield (
                LogClient(origin, timeout=None, unix_socket=unix_socket),
                requests,
            )
        finally:
            server.shutdown()
            thread.join()


class LogsTest(unittest.TestCase):
    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "requires Unix sockets")
    def test_unix_path_is_literal_and_preserves_http_routing(self):
        replies = [
            (200, b'{"data":[{"id":"chat"}]}'),
            (200, batch("chat", [{"event": "stream_begin", "sequence": 0}], 1)),
            (200, batch("chat", [], 1)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "serve local%25#.sock")
            with endpoint(replies, unix_socket=path) as (client, requests):
                self.assertEqual(client.models(), ["chat"])
                client.poll("chat")
                client.poll("chat")
                self.assertEqual(client.cursors, {"chat": "100:1:1"})
                self.assertEqual(
                    [path for path, _ in requests], ["/v1/models", "/logz", "/logz"]
                )
                self.assertTrue(
                    all(
                        headers["Host"] == "not-a-tcp-host.invalid:8099"
                        for _, headers in requests
                    )
                )
                self.assertEqual(requests[-1][1]["X-Loom-Cursor"], "100:1:1")

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "requires Unix sockets")
    def test_unix_connect_failure_closes_socket_and_preserves_cursor(self):
        client = LogClient("http://localhost", timeout=2.5, unix_socket="missing.sock")
        client.cursors["chat"] = "100:1:1"
        with mock.patch.object(socket, "socket") as create_socket:
            connection = create_socket.return_value
            connection.connect.side_effect = FileNotFoundError("missing socket")
            with self.assertRaisesRegex(FileNotFoundError, "missing socket"):
                client.poll("chat")
            create_socket.assert_called_once_with(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout.assert_called_once_with(2.5)
            connection.connect.assert_called_once_with("missing.sock")
            connection.close.assert_called_once_with()
        self.assertEqual(client.cursors, {"chat": "100:1:1"})

    def test_unix_selection_rejects_tls_and_nonfilesystem_paths(self):
        for origin, path in (
            ("https://localhost", "serve.sock"),
            ("http://localhost", ""),
            ("http://localhost", "\0abstract"),
            ("http://localhost", 123),
        ):
            with self.subTest(origin=origin, path=path), self.assertRaises(ValueError):
                LogClient(origin, unix_socket=path)

    def test_independent_cursors_gaps_and_fresh_counters(self):
        start = {"event": "stream_begin", "sequence": 0}
        gap = {"event": "gap", "from": 1, "to": 10}
        counts = {"event": "counters", "time_ns": 22, "values": {"requests": 40}}
        replies = [
            (200, b'{"data":[{"id":"chat"},{"id":"image"}]}'),
            (200, batch("chat", [start], 1)),
            (200, batch("image", [start], 1, identity="100:2")),
            (
                200,
                batch(
                    "chat",
                    [gap, {"event": "request_end", "sequence": 10}, counts],
                    11,
                    oldest=10,
                ),
            ),
            (
                200,
                batch(
                    "chat",
                    [{"event": "gap", "from": 0, "to": 11}, counts],
                    11,
                    oldest=11,
                ),
            ),
        ]
        with endpoint(replies) as (client, requests):
            self.assertEqual(client.models(), ["chat", "image"])
            self.assertEqual(
                client.poll("chat").stream,
                {"scope": "model", "model": "chat", "identity": "100:1"},
            )
            client.poll("image")
            records = client.poll("chat").records
            self.assertEqual(records[1], gap)
            self.assertEqual(records[-2], counts)
            self.assertEqual(client.cursors, {"chat": "100:1:11", "image": "100:2:1"})
            other = LogClient(client.address.geturl(), timeout=None)
            self.assertEqual(other.poll("chat").records[1]["from"], 0)
            self.assertNotIn("X-Loom-Cursor", requests[1][1])
            self.assertNotIn("X-Loom-Cursor", requests[2][1])
            self.assertEqual(requests[3][1]["X-Loom-Cursor"], "100:1:1")
            self.assertNotIn("X-Loom-Cursor", requests[4][1])

    def test_restart_is_explicit_and_does_not_retry_forever(self):
        replies = [
            (200, batch(None, [], 0)),
            (409, b"changed incarnation"),
            (200, batch(None, [], 0, identity="200:3")),
            (409, b"changed again"),
            (409, b"bad fresh query"),
        ]
        with endpoint(replies) as (client, requests):
            client.poll()
            resumed = client.poll()
            self.assertEqual(
                resumed.records[0],
                {"event": "stream_reset", "previous_cursor": "100:1:0"},
            )
            self.assertEqual(resumed.stream["identity"], "200:3")
            self.assertNotIn("X-Loom-Cursor", requests[2][1])
            with self.assertRaisesRegex(RuntimeError, "HTTP 409"):
                client.poll()

    def test_malformed_batch_cannot_advance_cursor(self):
        good = batch("chat", [{"event": "stream_begin", "sequence": 0}], 1)
        invalid = [
            good[:-1],
            batch("chat", [{"event": "request_begin", "sequence": 7}], 8),
            batch("wrong-model", [], 1),
            batch("chat", [], 1, identity="200:1"),
            batch("chat", [{"event": "gap", "from": 0, "to": 2}], 2),
            batch("chat", [], 2),
            b"null\n",
            b"x" * (1024 * 1024 + 1),
        ]
        with endpoint([(200, good), *((200, payload) for payload in invalid)]) as (
            client,
            _,
        ):
            client.poll("chat")
            for payload in invalid:
                with self.subTest(payload=payload[:80]), self.assertRaises(ValueError):
                    client.poll("chat")
                self.assertEqual(client.cursors["chat"], "100:1:1")

    def test_model_removal_preserves_cursor_until_explicit_forget(self):
        for refresh in ([], [(409, b"changed incarnation")]):
            with (
                self.subTest(refresh=bool(refresh)),
                endpoint(
                    [
                        (200, batch("chat", [], 0)),
                        *refresh,
                        (404, b"removed"),
                        (200, batch("chat", [], 0, identity="100:2")),
                    ]
                ) as (client, requests),
            ):
                client.poll("chat")
                with self.assertRaises(ModelNotFound) as missing:
                    client.poll("chat")
                self.assertEqual(missing.exception.model, "chat")
                self.assertEqual(client.forget("chat"), "100:1:0")
                self.assertNotIn("chat", client.cursors)
                self.assertEqual(client.poll("chat").stream["identity"], "100:2")
                self.assertNotIn("X-Loom-Cursor", requests[-1][1])

    def test_missing_server_stream_is_not_model_churn(self):
        with endpoint([(404, b"no log endpoint")]) as (client, _):
            with self.assertRaisesRegex(RuntimeError, "HTTP 404") as failure:
                client.poll()
            self.assertNotIsInstance(failure.exception, ModelNotFound)

    def test_unavailable_observer_is_an_error_not_an_empty_stream(self):
        with endpoint([(501, b"no observer")]) as (client, _):
            with self.assertRaisesRegex(RuntimeError, "HTTP 501"):
                client.poll("chat")

    def test_origin_and_catalog_validation(self):
        for url in (
            "localhost:8080",
            "file:///tmp/x",
            "http://x/api",
            "http://x/?secret=1",
            "http://u:p@x",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                LogClient(url)
        with endpoint([(200, b'{"data":[{"id":"x"},{"id":"x"}]}')]) as (client, _):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                client.models()


if __name__ == "__main__":
    unittest.main()
