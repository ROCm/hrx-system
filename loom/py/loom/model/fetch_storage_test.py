# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import base64
import hashlib
import http.server
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from pathlib import Path

import pytest

from loom.model import ManifestError
from loom.model.fetch_source import FetchError, Source
from loom.model.fetch_storage import ContentCache, installation_lock, share_file
from loom.model.http_test_server import serving_http
from loom.model.lock import FileIdentity


def _source(content):
    return Source(
        "program.loom", {"kind": "inline", "base64": base64.b64encode(content).decode()}
    )


def _identity(content):
    return FileIdentity(
        "program.loom", len(content), hashlib.sha256(content).hexdigest()
    )


def test_verified_blob_is_shared_by_multiple_installations(tmp_path):
    content = b"executable code"
    cache = ContentCache(tmp_path / "cache")
    acquired = cache.acquire(_source(content), _identity(content), offline=True)
    assert acquired.identity == _identity(content)
    first, second = tmp_path / "first", tmp_path / "second"
    share_file(acquired.path, first)
    share_file(acquired.path, second)
    assert os.path.samefile(first, second)
    assert first.read_bytes() == content
    assert list((cache.root / "temporary").iterdir()) == []


def test_local_file_is_not_copied_into_another_cache(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"weights")
    cache = ContentCache(tmp_path / "cache")
    acquired = cache.acquire(
        Source("weights", {"kind": "local", "path": str(source)}), None, offline=True
    )
    assert acquired.path == source
    assert not cache.root.exists()


@contextmanager
def _held_http_body():
    started, release = threading.Event(), threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "4")
            self.end_headers()
            started.set()
            release.wait()
            # Cancellation deliberately closes the client before release.
            with suppress(
                BrokenPipeError, ConnectionResetError, ConnectionAbortedError
            ):
                self.wfile.write(b"code")

    with (
        http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server,
        serving_http(server),
    ):
        try:
            yield f"http://127.0.0.1:{server.server_port}/code", started, release
        finally:
            release.set()


def test_cancel_releases_http_reader_and_removes_its_temporary_file(tmp_path):
    with _held_http_body() as (url, started, release):
        cache = ContentCache(tmp_path / "cache")
        source = Source("program.loom", {"kind": "url", "url": url})
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(
                    cache.acquire, source, _identity(b"code"), offline=False
                )
                future.add_done_callback(lambda _: started.set())
                started.wait()
                assert not future.done(), future.result()
                cache.cancel()
                with pytest.raises(FetchError, match="cancelled"):
                    future.result()
                assert not release.is_set()
                assert list((cache.root / "temporary").iterdir()) == []
                assert not cache.blob_path(_identity(b"code").sha256).exists()
            finally:
                cache.cancel()
                release.set()
        result = ContentCache(cache.root).acquire(
            source, _identity(b"code"), offline=False
        )
        assert result.path.read_bytes() == b"code"


def test_programmatic_caller_needs_no_process_entrypoint_guard(tmp_path):
    caller = tmp_path / "caller.py"
    caller.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "from loom.model.fetch_source import Source\n"
        "from loom.model.fetch_storage import ContentCache\n"
        "source = Source('code', {'kind': 'url', 'url': sys.argv[1]})\n"
        "result = ContentCache(Path(sys.argv[2])).acquire(source, None, offline=False)\n"
        "print(result.path.read_bytes().decode())\n"
    )
    with _held_http_body() as (url, _, release):
        release.set()
        result = subprocess.run(
            [sys.executable, str(caller), url, str(tmp_path / "cache")],
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "code"


def test_cancel_stops_document_fetch_without_publishing_metadata(tmp_path):
    with _held_http_body() as (url, started, release):
        cache = ContentCache(tmp_path / "cache")
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(cache.read_document, url, offline=False)
                future.add_done_callback(lambda _: started.set())
                started.wait()
                assert not future.done(), future.result()
                cache.cancel()
                with pytest.raises(FetchError, match="cancelled"):
                    future.result()
                assert not release.is_set()
                assert not cache.root.exists()
            finally:
                cache.cancel()
                release.set()


@pytest.mark.parametrize("content", [b"wrong contents", b"short", b"a longer payload"])
def test_bad_bytes_never_publish_a_cache_object(tmp_path, content):
    cache = ContentCache(tmp_path / "cache")
    with pytest.raises((ManifestError, FetchError)):
        cache.acquire(_source(content), _identity(b"expected bytes"), offline=True)
    assert not cache.blob_path(_identity(b"expected bytes").sha256).exists()
    assert list((cache.root / "temporary").iterdir()) == []


def test_offline_reuses_verified_blob_without_touching_url(tmp_path):
    cache = ContentCache(tmp_path / "cache")
    identity = _identity(b"code")
    cached = cache.acquire(_source(b"code"), identity, offline=True)
    acquired = cache.acquire(
        Source("program.loom", {"kind": "url", "url": "http://127.0.0.1:1/code"}),
        identity,
        offline=True,
    )
    assert acquired.path == cached.path


def test_cache_corruption_fails_instead_of_silently_refetching(tmp_path):
    cache = ContentCache(tmp_path / "cache")
    identity = _identity(b"code")
    acquired = cache.acquire(_source(b"code"), identity, offline=True)
    acquired.path.write_bytes(b"evil")
    with pytest.raises(ManifestError, match="integrity mismatch"):
        cache.acquire(_source(b"code"), identity, offline=True)


def test_installation_lock_has_exclusive_ownership_and_releases(tmp_path):
    with (
        installation_lock(tmp_path),
        pytest.raises(FetchError, match="another fetch"),
        installation_lock(tmp_path),
    ):
        raise AssertionError("second writer was admitted")
    with installation_lock(tmp_path):
        assert (tmp_path / ".loom-fetch.lock").is_file()


def test_share_never_replaces_an_existing_destination(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        share_file(source, destination)
    assert destination.read_bytes() == b"old"


def test_share_follows_relative_snapshot_links(tmp_path):
    blob = tmp_path / "blob"
    blob.write_bytes(b"weights")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "weights").symlink_to(Path("..") / "blob")
    output = tmp_path / "installed/package"
    output.mkdir(parents=True)
    share_file(snapshot / "weights", output / "weights")
    assert (output / "weights").read_bytes() == b"weights"
    assert os.path.samefile(blob, output / "weights")
    assert not (output / "weights").is_symlink()


def test_invalid_utf8_document_is_rejected(tmp_path):
    source = tmp_path / "model.json"
    source.write_bytes(b"\xff")
    with pytest.raises(ManifestError, match="UTF-8"):
        ContentCache(tmp_path / "cache").read_document(str(source), offline=True)
