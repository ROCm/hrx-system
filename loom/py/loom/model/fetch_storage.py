# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Verified content storage and exclusive installation publication."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from .fetch_hf import HuggingFace, hf_url_source
from .fetch_safetensors import repack
from .fetch_source import Document, FetchError, Source, document_source, open_source
from .fetch_transfer import run_transfer
from .lock import FileIdentity, file_digest, file_key
from .manifest import ManifestError, parse_json

# Document limits protect the public metadata boundary; payloads stream instead.
_DOCUMENT_LIMIT = 16 * 1024 * 1024
_CHUNK_SIZE = 1024 * 1024


def atomic_write(path: Path, content: bytes) -> None:
    """Replaces one caller-owned metadata file after its complete write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".loom-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def share_file(source: Path, destination: Path) -> str:
    """Creates an explicitly shared view; never silently allocates a full copy."""
    # Repository snapshots often contain relative symlinks. Some platforms link
    # the symlink inode even with follow_symlinks=True; sharing the resolved blob
    # preserves its bytes across a different destination directory layout.
    source = source.resolve(strict=True)
    try:
        os.link(source, destination, follow_symlinks=True)
        return "hardlink"
    except OSError as error:
        if error.errno not in (
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
        ):
            raise
    try:
        os.symlink(source.absolute(), destination)
        return "symlink"
    except OSError as error:
        raise FetchError(
            "this filesystem cannot share the selected file; use an output on "
            "the cache/source volume or explicitly select --materialize=copy"
        ) from error


@contextmanager
def installation_lock(directory: Path):
    """Holds an OS lock through publication, released automatically on process exit."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".loom-fetch.lock"
    if path.is_symlink():
        raise FetchError("installation lock must not be a symlink")
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            if not stream.seek(0, os.SEEK_END):
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise FetchError(
                    "another fetch owns this installation directory"
                ) from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise FetchError(
                    "another fetch owns this installation directory"
                ) from error
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class AcquiredFile:
    # Verified bytes, owned by the caller's local source or the shared cache.
    path: Path
    # Observed identity including its original logical source coordinate.
    identity: FileIdentity


def _copy_source(
    source: Source, identity: FileIdentity | None, offline: bool, temporary: Path
) -> tuple[int, str]:
    with open_source(
        source, offline=offline, size=identity.size if identity else None
    ) as (input_stream, transport_size):
        expected_size = identity.size if identity else transport_size
        digest = hashlib.sha256()
        size = 0
        with temporary.open("wb") as output:
            while True:
                limit = (
                    _CHUNK_SIZE
                    if expected_size is None
                    else min(_CHUNK_SIZE, expected_size - size + 1)
                )
                chunk = input_stream.read(limit)
                if not chunk:
                    break
                size += len(chunk)
                if expected_size is not None and size > expected_size:
                    raise FetchError("source exceeded its declared byte count")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    if expected_size is not None and size != expected_size:
        raise FetchError("source transfer ended before its declared byte count")
    digest = digest.hexdigest()
    if identity is not None and digest != identity.sha256:
        raise ManifestError("source SHA-256 does not match its lock")
    return size, digest


def _read_document_bytes(source: Source, offline: bool) -> bytes:
    with open_source(source, offline=offline) as (stream, size):
        if size is not None and size > _DOCUMENT_LIMIT:
            raise FetchError("model/lock document exceeds the 16 MiB metadata limit")
        content = stream.read(_DOCUMENT_LIMIT + 1)
        if len(content) > _DOCUMENT_LIMIT:
            raise FetchError("model/lock document exceeds the 16 MiB metadata limit")
        if size is not None and len(content) != size:
            raise FetchError("document transfer ended before its declared length")
        return content


class ContentCache:
    """Whole files are shared by verified content, not by model/preset name."""

    def __init__(self, root: Path, *, huggingface=None, wait_for_access=None):
        # Loom-owned document/content storage, independent of the Hub's cache.
        self.root = root.absolute()
        # Optional per-operation provider, constructed only when HF is needed.
        self._huggingface = huggingface
        # Human interaction belongs to the caller, including plain or visual modes.
        self.wait_for_access = wait_for_access
        # Shared operation lifetime across local hashing and every transfer provider.
        self.cancelled = (
            huggingface.cancelled if huggingface is not None else threading.Event()
        )

    @property
    def huggingface(self):
        if self._huggingface is None:
            self._huggingface = HuggingFace(
                wait_for_access=self.wait_for_access, cancelled=self.cancelled
            )
        return self._huggingface

    def _hf_url(self, location: str):
        endpoint = self._huggingface.endpoint if self._huggingface else None
        return hf_url_source(location, endpoint=endpoint)

    def cancel(self):
        """Requests cancellation of owned IO, hashing, and approval waits."""
        self.cancelled.set()

    def prepare_source(
        self, source: Source, identity: FileIdentity | None, *, offline: bool
    ) -> Source:
        """Reuses local content or checks selected repository access before transfer."""
        if source.transport["kind"] == "url":
            recognized = self._hf_url(source.transport["url"])
            if recognized is not None:
                transport, _ = recognized
                transport["revision"] = self.huggingface.resolve_revision(
                    transport, offline=offline
                )
                source = replace(source, transport=transport)
        if source.transport["kind"] != "hf":
            return source
        cached = self.blob_path(identity.sha256) if identity else None
        if cached is None or not cached.is_file():
            cached = self.huggingface.cached_file(source.transport)
        if cached is not None and cached.is_file():
            return replace(source, transport={"kind": "local", "path": str(cached)})
        if offline:
            raise FetchError("--offline Hugging Face source is not cached")
        metadata = self.huggingface.metadata(source.transport)
        if metadata.commit_hash != source.transport["revision"]:
            raise FetchError("Hugging Face file metadata changed the pinned revision")
        if identity is not None and metadata.size != identity.size:
            raise FetchError("Hugging Face file size does not match its lock")
        return source

    def blob_path(self, digest: str) -> Path:
        return self.root / "blob" / "sha256" / digest

    def find(self, identity: FileIdentity) -> Path | None:
        path = self.blob_path(identity.sha256)
        if not path.exists():
            return None
        identity.verify(path, cancelled=self.cancelled)
        return path

    def _publish(self, temporary: Path, identity: FileIdentity) -> Path:
        path = self.blob_path(identity.sha256)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(temporary, path)
        except FileExistsError:
            # Keep the existing shared inode: replacing it would leave two whole
            # copies alive when an installation still links the older inode.
            identity.verify(path, cancelled=self.cancelled)
        return path

    def acquire_transformed(
        self,
        source: Source,
        transform: dict,
        identity: FileIdentity | None,
        *,
        offline: bool,
        preserve_sources: bool,
    ) -> AcquiredFile:
        if self.cancelled.is_set():
            raise FetchError("model acquisition cancelled")
        key = hashlib.sha256(file_key(source.transport, transform).encode()).hexdigest()
        receipt = self.root / "preserved" / (key + ".json")
        cached = None
        if identity is not None:
            cached = self.find(identity)
            if cached is not None:
                if (
                    not preserve_sources
                    or source.transport["kind"] == "inline"
                    or (
                        source.transport["kind"] == "local"
                        and Path(source.transport["path"]).is_file()
                    )
                ):
                    return AcquiredFile(cached, identity)
                if receipt.is_file():
                    record = parse_json(receipt.read_text(encoding="utf-8"))
                    if record["output_sha256"] == identity.sha256:
                        for interval in record["ranges"]:
                            retained = FileIdentity(
                                source.origin, interval["length"], interval["sha256"]
                            )
                            retained.verify(
                                self.blob_path(retained.sha256),
                                cancelled=self.cancelled,
                            )
                        return AcquiredFile(cached, identity)
        source = self.prepare_source(source, None, offline=offline)
        kind = source.transport["kind"]
        if cached is not None and kind in ("local", "inline"):
            return AcquiredFile(cached, identity)
        if offline and kind not in ("local", "inline"):
            raise FetchError(
                "--offline requires a materialized result or local transform source"
            )
        temporary_root = self.root / "temporary"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            directory = Path(temporary)
            destination = directory / "result" if cached is None else None
            preserve = (
                directory / "source"
                if preserve_sources and kind in ("url", "hf")
                else None
            )
            result = (
                self.huggingface.repack(source, transform, destination, preserve)
                if kind == "hf"
                else run_transfer(
                    repack, (source, transform, destination, preserve), self.cancelled
                )
            )
            observed = FileIdentity(
                source.origin,
                result["size"],
                result["sha256"],
                source.resolved,
                transform,
            )
            if identity is not None and (observed.size, observed.sha256) != (
                identity.size,
                identity.sha256,
            ):
                raise ManifestError(
                    "transformed parameter bytes do not match their lock"
                )
            if self.cancelled.is_set():
                raise FetchError("model acquisition cancelled")
            if preserve is not None:
                records = []
                for interval in result["ranges"]:
                    retained = FileIdentity(
                        source.origin, interval["length"], interval["sha256"]
                    )
                    self._publish(preserve / interval["path"], retained)
                    records.append(
                        {
                            name: value
                            for name, value in interval.items()
                            if name != "path"
                        }
                    )
                atomic_write(
                    receipt,
                    (
                        json.dumps(
                            {
                                "source": source.transport,
                                "size": result["source_size"],
                                "transform": transform,
                                "output_sha256": observed.sha256,
                                "ranges": records,
                            },
                            indent=2,
                        )
                        + "\n"
                    ).encode(),
                )
            path = (
                cached if cached is not None else self._publish(destination, observed)
            )
            return AcquiredFile(path, observed)

    def read_document(self, location: str, *, offline: bool) -> Document:
        if self.cancelled.is_set():
            raise FetchError("model acquisition cancelled")
        source = document_source(location)
        remote = source.transport["kind"] == "url"
        key = hashlib.sha256(location.encode("utf-8")).hexdigest()
        cached = self.root / "document" / key
        if remote and offline and cached.is_file():
            # Cache the resolved document coordinate together with its bytes:
            # relative files and the sibling lock must keep the same snapshot.
            with cached.open("rb") as stream:
                encoded = stream.read(_DOCUMENT_LIMIT * 7 + 1)
            if len(encoded) > _DOCUMENT_LIMIT * 7:
                raise FetchError("cached model/lock document exceeds its size limit")
            try:
                record = parse_json(encoded.decode("utf-8"))
                if (
                    not isinstance(record, dict)
                    or set(record) != {"location", "text"}
                    or not all(isinstance(value, str) for value in record.values())
                    or len(record["text"].encode("utf-8")) > _DOCUMENT_LIMIT
                    or not Document(record["location"], "").remote
                ):
                    raise ValueError("invalid cached document")
                return Document(**record)
            except (UnicodeError, ManifestError, ValueError) as error:
                raise FetchError("cached model/lock document is invalid") from error
        recognized = self._hf_url(location) if remote else None
        if recognized is not None:
            transport, endpoint = recognized
            transport["revision"] = self.huggingface.resolve_revision(
                transport, offline=offline
            )
            path = self.huggingface.cached_file(transport)
            if path is None:
                if offline:
                    raise FetchError("--offline Hugging Face document is not cached")
                metadata = self.huggingface.metadata(transport)
                if metadata.commit_hash != transport["revision"]:
                    raise FetchError("Hugging Face changed the document revision")
                if metadata.size is None or not 0 <= metadata.size <= _DOCUMENT_LIMIT:
                    raise FetchError(
                        "model/lock document requires a declared size within 16 MiB"
                    )
                path = self.huggingface.download(transport, offline=False)
            source = document_source(str(path))
            location = self.huggingface.file_url(transport, endpoint=endpoint)
        content = (
            run_transfer(_read_document_bytes, (source, offline), self.cancelled)
            if source.transport["kind"] == "url" and not offline
            else _read_document_bytes(source, offline)
        )
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ManifestError("model/lock documents must use UTF-8") from error
        if remote and not offline:
            atomic_write(
                cached,
                json.dumps({"location": location, "text": text}).encode("utf-8"),
            )
        return Document(location, text)

    def acquire(
        self, source: Source, identity: FileIdentity | None, *, offline: bool
    ) -> AcquiredFile:
        if self.cancelled.is_set():
            raise FetchError("model acquisition cancelled")
        kind = source.transport["kind"]
        if kind in ("local", "hf"):
            path = (
                Path(source.transport["path"])
                if kind == "local"
                else self.huggingface.download(source.transport, offline=offline)
            )
            size, digest = file_digest(path, cancelled=self.cancelled)
            observed = FileIdentity(source.origin, size, digest, source.resolved)
            if identity is not None and (size, digest) != (
                identity.size,
                identity.sha256,
            ):
                raise ManifestError(f"integrity mismatch for local source {path}")
            return AcquiredFile(path, observed)
        if identity is not None:
            cached = self.find(identity)
            if cached is not None:
                return AcquiredFile(cached, identity)
        temporary_directory = self.root / "temporary"
        temporary_directory.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(dir=temporary_directory)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            arguments = (source, identity, offline, temporary)
            size, digest = (
                run_transfer(_copy_source, arguments, self.cancelled)
                if kind == "url" and not offline
                else _copy_source(*arguments)
            )
            observed = FileIdentity(source.origin, size, digest, source.resolved)
            path = self._publish(temporary, observed)
            return AcquiredFile(path, observed)
        finally:
            temporary.unlink(missing_ok=True)
