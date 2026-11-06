# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bounded telemetry for persistent libFuzzer campaigns."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TextIO

FAILURE_OUTPUT_LIMIT = 64 * 1024
_STREAM_CHUNK_SIZE = 16 * 1024
_PARSER_LINE_LIMIT = 8 * 1024
_HASH_CHUNK_SIZE = 1024 * 1024

_PROGRESS_PREFIX_RE = re.compile(rb"^#(\d+)\s+([A-Za-z]+)\s+")
_PROGRESS_INTEGER_FIELDS = {
    b"cov": "coverage_points",
    b"ft": "features",
    b"exec/s": "average_executions_per_second",
    b"rss": "peak_rss_mb",
}
_CORPUS_RE = re.compile(
    rb"(?:^|\s)corp:\s*(\d+)/(\d+(?:\.\d+)?)([KMGT]?b)(?:\s|$)",
    re.IGNORECASE,
)
_FINAL_STAT_RE = re.compile(rb"^stat::([a-z_]+):\s*([0-9]+(?:\.[0-9]+)?)\s*$")
_FINAL_STAT_FIELDS = {
    b"number_of_executed_units": ("executed_units", int),
    b"average_exec_per_sec": ("average_executions_per_second", int),
    b"new_units_added": ("new_units_added", int),
    b"slowest_unit_time_sec": ("slowest_unit_seconds", float),
    b"peak_rss_mb": ("peak_rss_mb", int),
}
_BYTE_MULTIPLIERS = {
    b"b": 1,
    b"kb": 1024,
    b"mb": 1024 * 1024,
    b"gb": 1024 * 1024 * 1024,
    b"tb": 1024 * 1024 * 1024 * 1024,
}
_OUTPUT_WRITE_LOCK = threading.Lock()


def telemetry_arguments(arguments: list[str]) -> list[str]:
    """Forces the bounded final statistics required by campaign reporting."""
    return [
        argument
        for argument in arguments
        if not argument.startswith("-print_final_stats=")
    ] + ["-print_final_stats=1"]


@dataclass(frozen=True)
class DirectorySnapshot:
    file_count: int
    byte_count: int

    def json(self) -> dict[str, int]:
        return {"files": self.file_count, "bytes": self.byte_count}


@dataclass(frozen=True)
class ExecutableIdentity:
    byte_count: int
    sha256: str

    def json(self) -> dict[str, int | str]:
        return {"bytes": self.byte_count, "sha256": self.sha256}


@dataclass(frozen=True)
class SourceIdentity:
    revision: str
    dirty: bool
    worktree_state_sha256: str | None

    def json(self) -> dict[str, bool | str | None]:
        return {
            "revision": self.revision,
            "dirty": self.dirty,
            "worktree_state_sha256": self.worktree_state_sha256,
        }


@dataclass
class LibFuzzerStatistics:
    executed_units: int | None = None
    average_executions_per_second: int | None = None
    new_units_added: int | None = None
    slowest_unit_seconds: float | None = None
    peak_rss_mb: int | None = None
    initial_coverage_points: int | None = None
    coverage_points: int | None = None
    initial_features: int | None = None
    features: int | None = None
    initial_corpus_units: int | None = None
    corpus_units: int | None = None
    initial_corpus_bytes: int | None = None
    corpus_bytes: int | None = None

    def json(self) -> dict[str, int | float | None]:
        return {
            "executed_units": self.executed_units,
            "average_executions_per_second": self.average_executions_per_second,
            "new_units_added": self.new_units_added,
            "slowest_unit_seconds": self.slowest_unit_seconds,
            "peak_rss_mb": self.peak_rss_mb,
            "initial_coverage_points": self.initial_coverage_points,
            "coverage_points": self.coverage_points,
            "coverage_points_added": _difference(
                self.initial_coverage_points, self.coverage_points
            ),
            "initial_features": self.initial_features,
            "features": self.features,
            "features_added": _difference(self.initial_features, self.features),
            "initial_corpus_units": self.initial_corpus_units,
            "corpus_units": self.corpus_units,
            "initial_corpus_bytes": self.initial_corpus_bytes,
            "corpus_bytes": self.corpus_bytes,
        }


class FuzzerOutputObserver:
    """Parses libFuzzer progress while retaining only a bounded output tail."""

    def __init__(self, tail_limit: int = FAILURE_OUTPUT_LIMIT):
        if tail_limit <= 0:
            raise ValueError("Fuzzer output tail limit must be positive")
        self.statistics = LibFuzzerStatistics()
        self.total_bytes = 0
        self._tail_limit = tail_limit
        self._tail = bytearray()
        self._line = bytearray()
        self._finalized = False

    @property
    def tail(self) -> bytes:
        return bytes(self._tail)

    @property
    def tail_truncated(self) -> bool:
        return self.total_bytes > len(self._tail)

    def consume(self, data: bytes) -> None:
        if self._finalized:
            raise RuntimeError("Cannot consume fuzzer output after finalization")
        if not data:
            return
        self.total_bytes += len(data)
        self._tail.extend(data)
        if len(self._tail) > self._tail_limit:
            del self._tail[: len(self._tail) - self._tail_limit]

        self._line.extend(data)
        while True:
            newline = self._line.find(b"\n")
            if newline < 0:
                break
            line = bytes(self._line[:newline]).rstrip(b"\r")
            del self._line[: newline + 1]
            self._parse_line(line)
        if len(self._line) > _PARSER_LINE_LIMIT:
            del self._line[: len(self._line) - _PARSER_LINE_LIMIT]

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        if self._line:
            self._parse_line(bytes(self._line).rstrip(b"\r"))
            self._line.clear()

    def _parse_line(self, line: bytes) -> None:
        final_match = _FINAL_STAT_RE.match(line)
        if final_match:
            field = _FINAL_STAT_FIELDS.get(final_match.group(1))
            if field is not None:
                field_name, field_type = field
                setattr(self.statistics, field_name, field_type(final_match.group(2)))
            return

        progress_match = _PROGRESS_PREFIX_RE.match(line)
        if not progress_match:
            return
        self.statistics.executed_units = int(progress_match.group(1))
        initial_progress = progress_match.group(2).upper() == b"INITED"
        for label, field_name in _PROGRESS_INTEGER_FIELDS.items():
            match = re.search(rb"(?:^|\s)" + re.escape(label) + rb":\s*(\d+)", line)
            if match:
                value = int(match.group(1))
                if field_name == "peak_rss_mb":
                    current = self.statistics.peak_rss_mb or 0
                    value = max(current, value)
                setattr(self.statistics, field_name, value)
                if initial_progress and field_name in ("coverage_points", "features"):
                    setattr(self.statistics, f"initial_{field_name}", value)
        corpus_match = _CORPUS_RE.search(line)
        if corpus_match:
            self.statistics.corpus_units = int(corpus_match.group(1))
            value = float(corpus_match.group(2))
            multiplier = _BYTE_MULTIPLIERS[corpus_match.group(3).lower()]
            self.statistics.corpus_bytes = int(value * multiplier)
            if initial_progress:
                self.statistics.initial_corpus_units = self.statistics.corpus_units
                self.statistics.initial_corpus_bytes = self.statistics.corpus_bytes


def snapshot_directory(path: Path) -> DirectorySnapshot:
    """Snapshots owned regular files without following directory symlinks."""
    file_count = 0
    byte_count = 0
    pending = [path]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                    continue
                entry_stat = entry.stat(follow_symlinks=False)
                file_count += 1
                byte_count += entry_stat.st_size
    return DirectorySnapshot(file_count=file_count, byte_count=byte_count)


def directory_change(
    before: DirectorySnapshot, after: DirectorySnapshot
) -> dict[str, object]:
    return {
        "before": before.json(),
        "after": after.json(),
        "delta": {
            "files": after.file_count - before.file_count,
            "bytes": after.byte_count - before.byte_count,
        },
    }


def _difference(initial_value: int | None, final_value: int | None) -> int | None:
    if initial_value is None or final_value is None:
        return None
    return final_value - initial_value


def executable_identity(path: Path) -> ExecutableIdentity:
    digest = hashlib.sha256()
    with path.open("rb") as executable_file:
        while data := executable_file.read(_HASH_CHUNK_SIZE):
            digest.update(data)
    return ExecutableIdentity(byte_count=path.stat().st_size, sha256=digest.hexdigest())


def source_identity(repository_root: Path) -> SourceIdentity:
    revision = _run_git(repository_root, ["rev-parse", "HEAD"]).decode().strip()
    status_output = _run_git(
        repository_root,
        ["status", "--porcelain=v1", "--untracked-files=all", "-z"],
    )
    if not status_output:
        return SourceIdentity(
            revision=revision,
            dirty=False,
            worktree_state_sha256=None,
        )

    digest = hashlib.sha256()
    digest.update(
        _run_git(repository_root, ["diff", "--binary", "--no-ext-diff", "HEAD", "--"])
    )
    digest.update(status_output)
    untracked_output = _run_git(
        repository_root,
        ["ls-files", "--others", "--exclude-standard", "-z"],
    )
    for encoded_path in sorted(filter(None, untracked_output.split(b"\0"))):
        digest.update(len(encoded_path).to_bytes(8, "little"))
        digest.update(encoded_path)
        file_path = repository_root / os.fsdecode(encoded_path)
        file_stat = file_path.lstat()
        digest.update(file_stat.st_mode.to_bytes(4, "little"))
        digest.update(file_stat.st_size.to_bytes(8, "little"))
        if stat.S_ISLNK(file_stat.st_mode):
            digest.update(os.fsencode(os.readlink(file_path)))
        elif stat.S_ISREG(file_stat.st_mode):
            with file_path.open("rb") as source_file:
                while data := source_file.read(_HASH_CHUNK_SIZE):
                    digest.update(data)
    return SourceIdentity(
        revision=revision,
        dirty=True,
        worktree_state_sha256=digest.hexdigest(),
    )


def observe_stream(
    stream: BinaryIO,
    observer: FuzzerOutputObserver,
    destination: TextIO | None,
) -> None:
    """Streams child output to the terminal while collecting bounded telemetry."""
    try:
        while data := stream.read(_STREAM_CHUNK_SIZE):
            observer.consume(data)
            if destination is None:
                continue
            with _OUTPUT_WRITE_LOCK:
                binary_destination = getattr(destination, "buffer", None)
                if binary_destination is not None:
                    binary_destination.write(data)
                else:
                    destination.write(data.decode("utf-8", errors="replace"))
                destination.flush()
    finally:
        observer.finalize()
        stream.close()


def write_failure_output(path: Path, observer: FuzzerOutputObserver) -> None:
    """Atomically replaces one target's bounded last-failure transcript."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = observer.tail
    if observer.tail_truncated:
        payload = b"... earlier output truncated ...\n" + payload
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_bytes(payload)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _run_git(repository_root: Path, arguments: list[str]) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout
