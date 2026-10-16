# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stages Bazel test outputs named by one build-event stream."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname


class BazelTestResultsError(RuntimeError):
    """Raised when current-invocation test evidence cannot be staged."""


@dataclass(frozen=True)
class BazelTestResultsCapture:
    """One Bazel test phase and its fresh result staging root."""

    build_event_path: Path
    output_dir: Path


def prepare_output_directory(output_dir: Path) -> None:
    """Creates an empty staging root without discarding prior evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise BazelTestResultsError(
            f"Bazel test result directory is not empty: {output_dir}"
        )
    (output_dir / ".events").mkdir()


def _source_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise BazelTestResultsError(f"test output is not a local file URI: {uri}")
    return Path(url2pathname(parsed.path))


def _staged_relative_path(source_path: Path) -> Path:
    parts = source_path.parts
    try:
        testlogs_index = len(parts) - 1 - tuple(reversed(parts)).index("testlogs")
    except ValueError as error:
        raise BazelTestResultsError(
            f"test output is outside a Bazel testlogs tree: {source_path}"
        ) from error
    suffix = parts[testlogs_index + 1 :]
    if not suffix or any(part in ("", ".", "..") for part in suffix):
        raise BazelTestResultsError(
            f"test output has an invalid testlogs-relative path: {source_path}"
        )
    return Path("bazel-testlogs", *suffix)


def stage(build_event_path: Path, output_dir: Path) -> int:
    """Copies current-invocation test.log/test.xml outputs into output_dir."""
    if not build_event_path.is_file():
        raise BazelTestResultsError(
            f"Bazel build-event file does not exist: {build_event_path}"
        )

    outputs: list[tuple[Path, Path]] = []
    with build_event_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise BazelTestResultsError(
                    f"invalid Bazel build event at {build_event_path}:{line_number}: "
                    f"{error.msg}"
                ) from error
            for output in event.get("testResult", {}).get("testActionOutput", ()):
                if output.get("name") not in ("test.log", "test.xml"):
                    continue
                uri = output.get("uri")
                if not isinstance(uri, str):
                    raise BazelTestResultsError(
                        f"test output has no URI at {build_event_path}:{line_number}"
                    )
                source_path = _source_path(uri)
                if not source_path.is_file():
                    raise BazelTestResultsError(
                        f"Bazel did not materialize test output: {source_path}"
                    )
                outputs.append(
                    (source_path, output_dir / _staged_relative_path(source_path))
                )

    staged_sources: dict[Path, Path] = {}
    for source_path, destination_path in outputs:
        previous_source = staged_sources.get(destination_path)
        if previous_source is not None:
            if previous_source != source_path:
                raise BazelTestResultsError(
                    "current invocation produced colliding test outputs: "
                    f"{previous_source} and {source_path}"
                )
            continue
        if destination_path.exists():
            raise BazelTestResultsError(
                "current test phases produced colliding test outputs: "
                f"{destination_path}"
            )
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        staged_sources[destination_path] = source_path
    return len(staged_sources)
