# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import http.server
import threading
from contextlib import contextmanager

import pytest

from loom.model import ManifestError, ModelCard
from loom.model.fetch_source import (
    Document,
    FetchError,
    Source,
    SourceResolver,
    open_source,
)


def _selection(root):
    return ModelCard(
        {
            "schema_version": 1,
            "model": {"id": "fixture/model", "name": "Fixture"},
            "origins": {"library": root},
            "programs": {
                "p": {"origin": "@library/", "inputs": ["main.loom"], "libraries": []}
            },
            "services": {"s": {"kind": "text"}},
            "variants": {"v": {"services": {"s": {"programs": ["p"]}}}},
        }
    ).select()


@contextmanager
def _server():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/content")
                self.end_headers()
                return
            self.send_response(200)
            if self.path == "/broken-chunk":
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.send_header("Content-Length", "4")
                if self.path == "/duplicate-length":
                    self.send_header("Content-Length", "4")
                elif self.path == "/ambiguous-framing":
                    self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(b"a\r\nshort" if self.path == "/broken-chunk" else b"code")

        def log_message(self, *_):
            pass

    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


def test_http_redirect_retains_logical_coordinates():
    with _server() as base:
        source = Source(
            "@library/program.loom", {"kind": "url", "url": base + "/redirect"}
        )
        with open_source(source, offline=False, size=4) as (stream, size):
            assert stream.read() == b"code"
            assert size == 4
        assert source.origin == "@library/program.loom"
        assert source.transport["url"] == base + "/redirect"


def test_http_length_mismatch_fails_before_payload_is_exposed():
    with _server() as base:
        source = Source("x", {"kind": "url", "url": base + "/content"})
        with (
            pytest.raises(FetchError, match="does not match locked size"),
            open_source(source, offline=False, size=5),
        ):
            raise AssertionError("payload was exposed")


def test_offline_http_never_opens_network():
    source = Source("x", {"kind": "url", "url": "http://127.0.0.1:1/content"})
    with (
        pytest.raises(FetchError, match="--offline"),
        open_source(source, offline=True),
    ):
        raise AssertionError("network was opened")


@pytest.mark.parametrize(
    ("path", "diagnostic"),
    [
        ("/duplicate-length", "multiple Content-Length"),
        ("/ambiguous-framing", "ambiguous HTTP"),
        ("/broken-chunk", "HTTP transfer failed"),
    ],
)
def test_malformed_http_framing_fails(path, diagnostic):
    with _server() as base:
        source = Source("x", {"kind": "url", "url": base + path})
        with (
            pytest.raises(FetchError, match=diagnostic),
            open_source(source, offline=False) as (stream, _),
        ):
            stream.read()


def test_remote_card_cannot_name_host_local_sources(tmp_path):
    selected = _selection({"kind": "local", "path": str(tmp_path)})
    resolver = SourceResolver(
        Document("https://example.org/model.json", ""),
        selected.document["origins"],
        {},
        None,
        pin_revisions=False,
    )
    with pytest.raises(FetchError, match="host-local"):
        resolver.resolve("@library/main.loom", None)


def test_override_uses_local_bytes_without_hf_dependency_or_network(tmp_path):
    selected = _selection(
        {"kind": "hf", "repository": "gated/model", "revision": "main"}
    )
    (tmp_path / "main.loom").write_bytes(b"code")
    resolver = SourceResolver(
        Document("https://example.org/model.json", ""),
        selected.document["origins"],
        {"library": str(tmp_path)},
        None,
        pin_revisions=False,
    )
    source = resolver.resolve("@library/main.loom", None)
    with open_source(source, offline=True) as (stream, size):
        assert stream.read() == b"code"
        assert size == 4
    assert source.origin == "@library/main.loom"
    assert "resolved" not in resolver.origins["library"]


def test_missing_override_never_falls_back(tmp_path):
    selected = _selection({"kind": "url", "url": "https://example.org/"})
    resolver = SourceResolver(
        Document(str(tmp_path / "model.json"), ""),
        selected.document["origins"],
        {"library": str(tmp_path)},
        None,
        pin_revisions=False,
    )
    source = resolver.resolve("@library/main.loom", None)
    with pytest.raises(FileNotFoundError), open_source(source, offline=False):
        raise AssertionError("missing override was ignored")


def test_lock_update_cannot_invent_revision_through_mirror(tmp_path):
    selected = _selection({"kind": "hf", "repository": "org/model", "revision": "main"})
    resolver = SourceResolver(
        Document(str(tmp_path / "model.json"), ""),
        selected.document["origins"],
        {"library": str(tmp_path)},
        None,
        pin_revisions=True,
    )
    with pytest.raises(FetchError, match="floating Hugging Face revision"):
        resolver.resolve("@library/main.loom", None)


def test_unknown_alias_fails_before_io(tmp_path):
    selected = _selection({"kind": "url", "url": "https://example.org/"})
    with pytest.raises(ManifestError, match="unknown origin overrides"):
        SourceResolver(
            Document(str(tmp_path / "model.json"), ""),
            selected.document["origins"],
            {"typo": str(tmp_path)},
            None,
            pin_revisions=False,
        )
