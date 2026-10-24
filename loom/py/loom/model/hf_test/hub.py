# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A controlled Hub HTTP dependency for tests of the real optional SDK."""

import hashlib
import http.server
import json
import os
import re
import sys
import threading
from contextlib import contextmanager
from unittest.mock import patch
from urllib.parse import unquote, urlsplit

import huggingface_hub.constants as hf_constants


class Hub:
    def __init__(self, root, files):
        # Tiny caller-owned payloads served by the fixture, keyed by repository path.
        self.files = files
        # Repository identity accepted by the HTTP fixture.
        self.repository = "fixture/model"
        # Immutable snapshot advertised by metadata and download responses.
        self.revision = "a" * 40
        # Publisher approval is changed explicitly by each test's user action.
        self.approved = False
        # Denial response, retaining the provider's explicit gate error code.
        self.denial = (403, "GatedRepo")
        # Optional synchronization of concurrent denied shard requests.
        self.denial_barrier = None
        # Provider retry guidance supplied with denial responses.
        self.retry_after = None
        # Selected request sequence; credentials are deliberately not recorded.
        self.requests = []
        # Actual byte-range requests, with no credential or signed redirect data.
        self.range_requests = []
        # Optional GET-only CDN redirects; metadata remains owned by the Hub.
        self.redirects = {}
        # Optional observer of actual requests for deterministic transfer races.
        self.request_observer = None
        # A selected body can be held until the test exercises cancellation.
        self.held_file = None
        # Signals that headers have been sent and the selected body is held.
        self.body_started = threading.Event()
        # Explicit release, also set during fixture teardown.
        self.release_body = threading.Event()
        # Controlled cancellation may disconnect before headers or body complete.
        self.allow_disconnect = False
        # SDK token file isolated from the executing user's credentials.
        self.credentials = root / "token"
        # SDK cache isolated from real models and retained only for the test.
        self.cache_dir = root / "hf-cache"
        # Bound loopback endpoint, populated by the serving context.
        self.endpoint = ""

    def source(self, path="code.loom", revision=None):
        return {
            "kind": "hf",
            "repository": self.repository,
            "revision": revision or self.revision,
            "path": path,
        }

    def environment(self):
        """Isolates an actual CLI subprocess's provider, credentials, and cache."""
        return {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(sys.path),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_ENDPOINT": self.endpoint,
            "HF_HUB_CACHE": str(self.cache_dir),
            "HF_TOKEN_PATH": str(self.credentials),
            "HF_TOKEN": "",
            "HUGGING_FACE_HUB_TOKEN": "",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "0",
            "HF_HUB_OFFLINE": "0",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }


@contextmanager
def running_hub(root, files=None):
    state = Hub(
        root,
        files
        if files is not None
        else {"code.loom": b"code", "weights.data": b"weights"},
    )
    with (
        patch.dict(os.environ, {"HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": ""}),
        patch.object(hf_constants, "HF_TOKEN_PATH", str(state.credentials)),
        patch.object(hf_constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", False),
        patch.object(hf_constants, "HF_HUB_DISABLE_PROGRESS_BARS", True),
    ):

        class Handler(http.server.BaseHTTPRequestHandler):
            def handle(self):
                try:
                    super().handle()
                except (ConnectionResetError, ConnectionAbortedError):
                    # Cancellation can also close an SDK connection before it
                    # sends its next request, outside the held body's writer.
                    if not state.allow_disconnect:
                        raise

            def log_message(self, *_):
                pass

            def response(
                self,
                status,
                content=b"",
                error="",
                *,
                metadata=False,
                head=False,
                hold=False,
                interval=None,
            ):
                digest = hashlib.sha256(content).hexdigest()
                total = len(content)
                if interval is not None:
                    first, last = interval
                    content = content[first : last + 1]
                    status = 206
                self.send_response(status)
                self.send_header("Content-Length", str(len(content)))
                if interval is not None:
                    self.send_header("Content-Range", f"bytes {first}-{last}/{total}")
                if error:
                    self.send_header("X-Error-Code", error)
                if status >= 400 and state.retry_after:
                    self.send_header("Retry-After", state.retry_after)
                if metadata:
                    self.send_header("X-Repo-Commit", state.revision)
                    self.send_header("ETag", '"' + digest + '"')
                self.end_headers()
                if not head:
                    if hold:
                        state.allow_disconnect = True
                        state.body_started.set()
                        state.release_body.wait()
                    try:
                        self.wfile.write(content)
                    except (
                        BrokenPipeError,
                        ConnectionResetError,
                        ConnectionAbortedError,
                    ):
                        if not state.allow_disconnect:
                            raise

            def serve_request(self, *, head=False):
                state.requests.append(("HEAD" if head else "GET", self.path))
                if state.request_observer is not None:
                    state.request_observer("HEAD" if head else "GET", self.path)
                path = urlsplit(self.path).path
                token = self.headers.get("Authorization", "")
                valid = token in ("Bearer fixture-first", "Bearer fixture-second")
                if path == "/api/whoami-v2":
                    if not valid:
                        self.response(401, b'{"error":"invalid credentials"}')
                    else:
                        account = "first" if token.endswith("first") else "second"
                        self.response(200, json.dumps({"name": account}).encode())
                    return
                if not state.approved or not valid:
                    status, error = state.denial
                    if state.denial_barrier is not None and "/resolve/" in path:
                        state.denial_barrier.wait()
                    self.response(status, error=error, head=head)
                    return
                if path.endswith("/auth-check"):
                    self.response(200, b"{}")
                elif path.startswith("/api/models/"):
                    self.response(
                        200,
                        json.dumps(
                            {"id": state.repository, "sha": state.revision}
                        ).encode(),
                    )
                elif "/resolve/" in path:
                    revision, _, filename = path.split("/resolve/", 1)[1].partition("/")
                    if unquote(revision) not in ("main", state.revision):
                        self.response(404, error="RevisionNotFound", head=head)
                    elif unquote(filename) not in state.files:
                        self.response(404, error="EntryNotFound", head=head)
                    elif not head and unquote(filename) in state.redirects:
                        self.send_response(302)
                        self.send_header("Location", state.redirects[unquote(filename)])
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                    else:
                        interval = None
                        if not head and self.headers.get("Range"):
                            match = re.fullmatch(
                                r"bytes=(\d+)-(\d+)", self.headers["Range"]
                            )
                            if match is None:
                                self.response(416)
                                return
                            interval = tuple(map(int, match.groups()))
                            state.range_requests.append((unquote(filename), *interval))
                        self.response(
                            200,
                            state.files[unquote(filename)],
                            metadata=True,
                            head=head,
                            hold=unquote(filename) == state.held_file,
                            interval=interval,
                        )
                else:
                    self.response(404, head=head)

            def do_GET(self):
                self.serve_request()

            def do_HEAD(self):
                self.serve_request(head=True)

        with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            state.endpoint = f"http://127.0.0.1:{server.server_port}"
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                yield state
            finally:
                state.release_body.set()
                server.shutdown()
                thread.join()
