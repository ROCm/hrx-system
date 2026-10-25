# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib
import http.server
import json
import os
import re
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress

import pytest

from loom.model import ManifestError, ModelCard
from loom.model.fetch_source import FetchError, Source
from loom.model.fetch_storage import ContentCache
from loom.model.http_test_server import serving_http
from loom.model.lock import FileIdentity, ModelLock


def archive(tensors):
    metadata, data = {}, b""
    for name, dtype, shape, content in tensors:
        metadata[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(data), len(data) + len(content)],
        }
        data += content
    header = json.dumps(metadata, separators=(",", ":")).encode()
    return struct.pack("<Q", len(header)) + header + data


def transform(*names):
    return {
        "kind": "safetensors",
        "version": 1,
        "alignment": 4096,
        **({"tensors": list(names)} if names else {}),
    }


def local_source(tmp_path, content):
    path = tmp_path / "source"
    path.write_bytes(content)
    return Source("checkpoint", {"kind": "local", "path": str(path)})


@contextmanager
def serving(content, mode="normal"):
    requests = []
    started, release = threading.Event(), threading.Event()
    header_length = struct.unpack("<Q", content[:8])[0]

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            first, last = map(
                int, re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers["Range"]).groups()
            )
            requests.append((first, last - first + 1))
            body = content[first : last + 1]
            self.send_response(200 if mode == "whole" else 206)
            self.send_header("Content-Length", str(len(body)))
            self.send_header(
                "Content-Range",
                f"bytes {first + (mode == 'wrong')}-{last}/"
                + ("*" if mode == "unknown" else str(len(content))),
            )
            self.send_header(
                "ETag", '"changed"' if mode == "changed" and first else '"original"'
            )
            if mode == "duplicate":
                self.send_header("Content-Length", str(len(body)))
            if mode == "ambiguous":
                self.send_header("Transfer-Encoding", "chunked")
            if mode == "encoding":
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            if mode == "whole":
                return
            if mode == "held" and first >= 8 + header_length:
                started.set()
                release.wait()
            with suppress(
                BrokenPipeError, ConnectionResetError, ConnectionAbortedError
            ):
                self.wfile.write(body[:-1] if mode == "short" else body)

        def log_message(self, *_):
            pass

    with (
        http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server,
        serving_http(server),
    ):
        try:
            yield (
                Source(
                    "checkpoint",
                    {
                        "kind": "url",
                        "url": f"http://127.0.0.1:{server.server_port}/source",
                    },
                ),
                requests,
                started,
                release,
            )
        finally:
            release.set()


@pytest.mark.parametrize(
    ("dtype", "size"),
    [
        ("F4", 4),
        ("F6_E2M3", 6),
        ("F6_E3M2", 6),
        ("BOOL", 8),
        ("U8", 8),
        ("I8", 8),
        ("F8_E5M2", 8),
        ("F8_E4M3", 8),
        ("F8_E8M0", 8),
        ("F8_E4M3FNUZ", 8),
        ("F8_E5M2FNUZ", 8),
        ("I16", 16),
        ("U16", 16),
        ("F16", 16),
        ("BF16", 16),
        ("I32", 32),
        ("U32", 32),
        ("F32", 32),
        ("I64", 64),
        ("U64", 64),
        ("F64", 64),
        ("C64", 64),
    ],
)
def test_complete_dtype_family_preserves_bytes_and_metadata(tmp_path, dtype, size):
    content = bytes(range(size))
    source = local_source(tmp_path, archive([("tensor", dtype, [8], content)]))
    result = ContentCache(tmp_path / "cache").acquire_transformed(
        source, transform(), None, offline=True, preserve_sources=False
    )
    encoded = result.path.read_bytes()
    header_length = struct.unpack("<Q", encoded[:8])[0]
    assert (8 + header_length) % 4096 == 0
    assert json.loads(encoded[8 : 8 + header_length]) == {
        "tensor": {"dtype": dtype, "shape": [8], "data_offsets": [0, size]}
    }
    assert encoded[8 + header_length :] == content
    assert result.identity.sha256 == hashlib.sha256(encoded).hexdigest()


def test_empty_tensors_scalars_and_string_metadata_round_trip(tmp_path):
    header = json.dumps(
        {
            "": {"dtype": "I32", "shape": [0], "data_offsets": [0, 0]},
            "scalar": {"dtype": "I32", "shape": [], "data_offsets": [0, 4]},
            "__metadata__": {"": "", "note": "\u03bb\u0000"},
        }
    ).encode()
    source = local_source(tmp_path, struct.pack("<Q", len(header)) + header + b"abcd")
    cache = ContentCache(tmp_path / "cache")
    result = cache.acquire_transformed(
        source, transform(), None, offline=True, preserve_sources=False
    )
    encoded = result.path.read_bytes()
    length = struct.unpack("<Q", encoded[:8])[0]
    assert json.loads(encoded[8 : 8 + length]) == json.loads(header)


def test_ranges_skip_unselected_bytes_and_preservation_is_explicit(tmp_path):
    content = archive(
        [
            ("unused", "U8", [1024 * 1024], b"x" * (1024 * 1024)),
            ("bias", "I32", [1], struct.pack("<i", 2)),
        ]
    )
    header_length = struct.unpack("<Q", content[:8])[0]
    cache = ContentCache(tmp_path / "cache")
    recipe = transform("bias")
    with serving(content) as (source, requests, *_):
        acquired = cache.acquire_transformed(
            source, recipe, None, offline=False, preserve_sources=False
        )
        assert requests == [(0, 8), (8, header_length), (len(content) - 4, 4)]
        assert acquired.identity.size == 4100
        assert [path for path in (cache.root / "blob/sha256").iterdir()] == [
            acquired.path
        ]
        assert not (cache.root / "preserved").exists()
        before = len(requests)
        second = cache.acquire_transformed(
            source, recipe, acquired.identity, offline=True, preserve_sources=False
        )
        assert os.path.samefile(acquired.path, second.path)
        assert len(requests) == before
        retained = cache.acquire_transformed(
            source, recipe, acquired.identity, offline=False, preserve_sources=True
        )
        assert os.path.samefile(acquired.path, retained.path)
        receipt = json.loads(next((cache.root / "preserved").iterdir()).read_text())
        assert (
            sum(interval["length"] for interval in receipt["ranges"])
            == 12 + header_length
        )
        before = len(requests)
        cache.acquire_transformed(
            source, recipe, acquired.identity, offline=True, preserve_sources=True
        )
        assert len(requests) == before
        assert list((cache.root / "temporary").iterdir()) == []
    compact = local_source(
        tmp_path, archive([("bias", "I32", [1], struct.pack("<i", 2))])
    )
    other = ContentCache(tmp_path / "other").acquire_transformed(
        compact, recipe, acquired.identity, offline=True, preserve_sources=False
    )
    assert other.path.read_bytes() == acquired.path.read_bytes()


def test_full_repack_streams_one_payload_and_reuses_existing_inode(tmp_path):
    content = archive([("z", "U8", [2**21], b"x" * 2**21), ("a", "I32", [1], b"abcd")])
    cache = ContentCache(tmp_path / "cache")
    with serving(content) as (source, requests, *_):
        first = cache.acquire_transformed(
            source, transform(), None, offline=False, preserve_sources=False
        )
        link = tmp_path / "installation"
        os.link(first.path, link)
        second = cache.acquire_transformed(
            source, transform(), None, offline=False, preserve_sources=False
        )
        assert os.path.samefile(link, second.path)
        header_length = struct.unpack("<Q", content[:8])[0]
        assert requests[:3] == [
            (0, 8),
            (8, header_length),
            (8 + header_length, len(content) - 8 - header_length),
        ]


@pytest.mark.parametrize(
    "mode",
    [
        "whole",
        "wrong",
        "unknown",
        "duplicate",
        "ambiguous",
        "encoding",
        "short",
        "changed",
    ],
)
def test_invalid_range_never_publishes_output(tmp_path, mode):
    cache = ContentCache(tmp_path / "cache")
    with (
        serving(archive([("bias", "I32", [1], b"abcd")]), mode) as (source, *_),
        pytest.raises(FetchError),
    ):
        cache.acquire_transformed(
            source, transform(), None, offline=False, preserve_sources=True
        )
    assert not (cache.root / "blob").exists()
    assert not (cache.root / "preserved").exists()
    assert list((cache.root / "temporary").iterdir()) == []


def test_cancel_joins_range_writer_and_removes_owned_partial_files(tmp_path):
    cache = ContentCache(tmp_path / "cache")
    with (
        serving(archive([("bias", "I32", [1], b"abcd")]), "held") as (
            source,
            _,
            started,
            release,
        ),
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        future = executor.submit(
            cache.acquire_transformed,
            source,
            transform(),
            None,
            offline=False,
            preserve_sources=True,
        )
        future.add_done_callback(lambda _: started.set())
        started.wait()
        assert not future.done(), future.result()
        cache.cancel()
        with pytest.raises(FetchError, match="cancelled"):
            future.result()
        assert not release.is_set()
        assert list((cache.root / "temporary").iterdir()) == []
        assert not (cache.root / "blob").exists()


@pytest.mark.parametrize(
    ("dtype", "shape", "data"),
    [
        ("F4", [1], b"a"),
        ("F32", [1], b"a"),
        ("unknown", [1], b"a"),
        ("I32", [-1], b"a"),
        ("I32", [2**63, 8], b"a"),
    ],
)
def test_invalid_archive_fails_before_output(tmp_path, dtype, shape, data):
    cache = ContentCache(tmp_path / "cache")
    with pytest.raises(ManifestError):
        cache.acquire_transformed(
            local_source(tmp_path, archive([("tensor", dtype, shape, data)])),
            transform(),
            None,
            offline=True,
            preserve_sources=False,
        )
    assert not (cache.root / "blob").exists()


def test_output_lock_distinguishes_operations_and_does_not_hash_parent(tmp_path):
    content = archive([("bias", "I32", [1], b"abcd")])
    source = local_source(tmp_path, content)
    cache = ContentCache(tmp_path / "cache")
    result = cache.acquire_transformed(
        source, transform(), None, offline=True, preserve_sources=False
    )
    lock = ModelLock(
        ModelLock.snapshot(
            {},
            [
                FileIdentity(
                    "checkpoint", len(content), hashlib.sha256(content).hexdigest()
                ),
                result.identity,
            ],
        )
    )
    assert len(lock.files) == 2
    broken = FileIdentity(
        "checkpoint", result.identity.size, "0" * 64, transform=transform()
    )
    with pytest.raises(ManifestError, match="transformed parameter"):
        cache.acquire_transformed(
            source, transform(), broken, offline=True, preserve_sources=False
        )


def test_transform_requires_explicit_typed_parameter_ownership():
    document = {
        "schema_version": 1,
        "model": {"id": "test/test", "name": "test"},
        "programs": {"p": {"inputs": ["code.loom"], "libraries": []}},
        "services": {"s": {"kind": "text"}},
        "variants": {
            "v": {
                "services": {
                    "s": {
                        "programs": ["p"],
                        "parameters": {
                            "model": {
                                "files": [
                                    {
                                        "path": "weights.data",
                                        "format": "safetensors",
                                        "transform": {
                                            "kind": "safetensors",
                                            "version": 1,
                                        },
                                    }
                                ]
                            }
                        },
                    }
                }
            }
        },
    }
    card = ModelCard(document)
    assert card.select().files["weights.data"].transform["alignment"] == 8
    document["programs"]["p"]["inputs"] = document["variants"]["v"]["services"]["s"][
        "parameters"
    ]["model"]["files"]
    with pytest.raises(ManifestError, match="unknown fields"):
        ModelCard(document)
