# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import base64
import hashlib
import http.server
import lzma
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from pathlib import Path

import pytest

from loom.model import ManifestError
from loom.model.fetch_filesystem import share_file
from loom.model.fetch_source import FetchError, Source
from loom.model.fetch_storage import ContentCache, installation_lock
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


@pytest.mark.parametrize(
    "failure", ["truncated", "checksum", "padding", "overrun", "digest"]
)
def test_invalid_xz_never_publishes_decoded_bytes(tmp_path, failure):
    raw = b"module" * 1000
    frame = lzma.compress(raw)
    length = len(raw)
    digest = hashlib.sha256(raw).hexdigest()
    if failure == "truncated":
        frame = frame[:-1]
    elif failure == "checksum":
        frame = frame[:-9] + bytes([frame[-9] ^ 1]) + frame[-8:]
    elif failure == "padding":
        frame += b"\0"
    elif failure == "overrun":
        length -= 1
    else:
        digest = "0" * 64
    source = _source(frame)
    transform = {"kind": "xz", "version": 1}
    identity = FileIdentity(source.origin, length, digest, transform=transform)
    cache = ContentCache(tmp_path / "cache")
    with pytest.raises((FetchError, ManifestError), match=r"XZ|lock"):
        cache.acquire_transformed(
            source, transform, identity, offline=True, preserve_sources=False
        )
    assert not (cache.root / "blob").exists()
    assert not any((cache.root / "temporary").iterdir())


@contextmanager
def _held_http_body(*, prefix=b"", declared_size=True):
    started, release = threading.Event(), threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.send_response(200)
            if declared_size:
                self.send_header("Content-Length", str(len(prefix) + 4))
            self.end_headers()
            self.wfile.write(prefix)
            self.wfile.flush()
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


@pytest.mark.parametrize("transferred", [0, 1024 * 1024])
def test_progress_crosses_worker_pipe_before_held_download_and_cancellation(
    tmp_path, transferred
):
    observed = threading.Event()
    events = []

    def observe(event):
        events.append(event)
        if event.phase == "download" and event.completed_bytes == transferred:
            observed.set()

    content = b"a" * transferred + b"code"
    with _held_http_body(prefix=content[:-4]) as (url, _, release):
        cache = ContentCache(tmp_path / "cache", progress=observe)
        source = Source("program.loom", {"kind": "url", "url": url})
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(
                    cache.acquire,
                    source,
                    _identity(content),
                    offline=False,
                    progress=cache.progress.file("program.loom"),
                )
                future.add_done_callback(lambda _: observed.set())
                observed.wait()
                assert not future.done(), future.result()
                event = events[-1]
                assert event.phase == "download"
                assert event.path == "program.loom"
                assert (event.completed_bytes, event.total_bytes) == (
                    transferred,
                    len(content),
                )
                assert url not in repr(events)
                cache.cancel()
                with pytest.raises(FetchError, match="cancelled"):
                    future.result()
                assert not release.is_set()
                assert not cache.blob_path(_identity(content).sha256).exists()
                assert not list((cache.root / "temporary").iterdir())
            finally:
                cache.cancel()
                release.set()


def test_http_progress_does_not_invent_an_unknown_length(tmp_path):
    events = []
    with _held_http_body(declared_size=False) as (url, _, release):
        release.set()
        cache = ContentCache(tmp_path / "cache", progress=events.append)
        result = cache.acquire(
            Source("program.loom", {"kind": "url", "url": url}), None, offline=False
        )
        assert result.path.read_bytes() == b"code"
        download = [event for event in events if event.phase == "download"]
        assert [event.completed_bytes for event in download] == [0, 4]
        assert all(event.total_bytes is None for event in download)


@pytest.mark.parametrize("failure", [OSError, BrokenPipeError, EOFError])
def test_observer_failure_joins_transfer_before_returning_to_caller(tmp_path, failure):
    def fail(event):
        raise failure("progress sink closed")

    with _held_http_body() as (url, _, release):
        cache = ContentCache(tmp_path / "cache", progress=fail)
        with pytest.raises(failure, match="progress sink closed"):
            cache.acquire(
                Source("program.loom", {"kind": "url", "url": url}),
                _identity(b"code"),
                offline=False,
            )
        assert not release.is_set()
        assert not list((cache.root / "temporary").iterdir())
        assert not cache.blob_path(_identity(b"code").sha256).exists()


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


def test_cancelling_one_cache_preserves_another_callers_transfer(tmp_path):
    with _held_http_body() as first, _held_http_body() as second:
        first_url, first_started, first_release = first
        second_url, second_started, second_release = second
        first_cache = ContentCache(tmp_path / "first")
        second_cache = ContentCache(tmp_path / "second")
        with ThreadPoolExecutor(max_workers=2) as executor:
            try:
                first_result = executor.submit(
                    first_cache.read_document, first_url, offline=False
                )
                second_result = executor.submit(
                    second_cache.read_document, second_url, offline=False
                )
                first_result.add_done_callback(lambda _: first_started.set())
                second_result.add_done_callback(lambda _: second_started.set())
                first_started.wait()
                second_started.wait()
                assert not first_result.done(), first_result.result()
                assert not second_result.done(), second_result.result()
                first_cache.cancel()
                with pytest.raises(FetchError, match="cancelled"):
                    first_result.result()
                assert not second_result.done(), second_result.result()
                second_release.set()
                assert second_result.result().text == "code"
            finally:
                first_cache.cancel()
                second_cache.cancel()
                first_release.set()
                second_release.set()


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


def test_offline_editable_source_uses_and_verifies_its_local_receipt(tmp_path):
    cache = ContentCache(tmp_path / "cache")
    with _held_http_body() as (url, _, release):
        release.set()
        source = Source("program.loom", {"kind": "url", "url": url})
        acquired = cache.acquire(source, None, offline=False)
    offline = ContentCache(cache.root).acquire(source, None, offline=True)
    assert offline.path == acquired.path
    assert offline.identity == acquired.identity
    acquired.path.write_bytes(b"evil")
    with pytest.raises(ManifestError, match="integrity mismatch"):
        cache.acquire(source, None, offline=True)


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
