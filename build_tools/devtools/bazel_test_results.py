# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stages Bazel test outputs named by one build-event stream."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname


class BazelTestResultsError(RuntimeError):
    """Raised when current-invocation test evidence cannot be staged."""


@dataclass(frozen=True)
class BazelTestResultsCapture:
    """One Bazel test phase and its fresh result staging root."""

    build_event_path: Path
    output_dir: Path


@dataclass(frozen=True)
class BazelTestResultsStage:
    """Outputs and compact observation produced from one build-event stream."""

    output_count: int
    observation_path: Path
    observation_byte_length: int
    observation_seconds: float


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


def _nonnegative_integer(
    value: Any, *, field: str, line_number: int, default: int | None = None
) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise BazelTestResultsError(
            f"Bazel build event field {field} is not an integer at line {line_number}"
        )
    try:
        result = int(value)
    except ValueError as error:
        raise BazelTestResultsError(
            f"Bazel build event field {field} is not an integer at line {line_number}"
        ) from error
    if result < 0:
        raise BazelTestResultsError(
            f"Bazel build event field {field} is negative at line {line_number}"
        )
    return result


def _event_label(event: dict, event_kind: str, line_number: int) -> str:
    event_id = event.get("id")
    identity = event_id.get(event_kind) if isinstance(event_id, dict) else None
    label = identity.get("label") if isinstance(identity, dict) else None
    if not isinstance(label, str) or not label:
        raise BazelTestResultsError(
            f"Bazel {event_kind} event has no label at line {line_number}"
        )
    return label


def _configuration_id(event: dict, event_kind: str, line_number: int) -> str:
    event_id = event.get("id")
    identity = event_id.get(event_kind) if isinstance(event_id, dict) else None
    configuration = (
        identity.get("configuration") if isinstance(identity, dict) else None
    )
    configuration_id = (
        configuration.get("id") if isinstance(configuration, dict) else None
    )
    if not isinstance(configuration_id, str) or not configuration_id:
        raise BazelTestResultsError(
            f"Bazel {event_kind} event has no configuration at line {line_number}"
        )
    return configuration_id


def _label_set_digest(labels: set[str]) -> str:
    digest = hashlib.sha256()
    for label in sorted(labels):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _observation(
    *,
    phase: str,
    selected_labels: set[str],
    test_summaries: list[dict],
    failed_attempts: list[dict],
    finished: dict | None,
) -> dict:
    configuration_ids = sorted(
        {record["_configuration_id"] for record in (*test_summaries, *failed_attempts)}
    )
    configuration_indexes = {
        configuration_id: index
        for index, configuration_id in enumerate(configuration_ids)
    }
    for record in (*test_summaries, *failed_attempts):
        record["configuration"] = configuration_indexes[record.pop("_configuration_id")]
    test_summaries.sort(key=lambda record: (record["label"], record["configuration"]))
    failed_attempts.sort(
        key=lambda record: (
            record["label"],
            record["configuration"],
            record["run"],
            record["shard"],
            record["attempt"],
        )
    )
    invocation = {"complete": finished is not None}
    if finished is not None:
        exit_code = finished.get("exitCode")
        if not isinstance(exit_code, dict):
            exit_code = {}
        exit_code_name = exit_code.get("name")
        if exit_code_name is not None and not isinstance(exit_code_name, str):
            raise BazelTestResultsError(
                "Bazel build-finished exit-code name is not a string"
            )
        invocation.update(
            overall_success=bool(finished.get("overallSuccess", False)),
            exit_code=_nonnegative_integer(
                exit_code.get("code"),
                field="finished.exitCode.code",
                line_number=0,
                default=0,
            ),
            exit_code_name=exit_code_name or "",
        )
    return {
        "schema_version": 1,
        "phase": phase,
        "selection": {
            "test_count": len(selected_labels),
            "label_set_sha256": _label_set_digest(selected_labels),
        },
        "configurations": configuration_ids,
        "test_runs": {
            "fresh": sum(record["fresh_run_count"] for record in test_summaries),
            "cached": sum(record["cached_run_count"] for record in test_summaries),
        },
        "tests": test_summaries,
        "failed_attempts": failed_attempts,
        "invocation": invocation,
    }


def stage(build_event_path: Path, output_dir: Path) -> BazelTestResultsStage:
    """Stages test outputs and a compact observation into output_dir."""
    if not build_event_path.is_file():
        raise BazelTestResultsError(
            f"Bazel build-event file does not exist: {build_event_path}"
        )

    observation_started = time.monotonic()
    outputs: list[tuple[Path, Path]] = []
    selected_labels: set[str] = set()
    test_summaries = []
    failed_attempts = []
    finished = None
    with build_event_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise BazelTestResultsError(
                    f"invalid Bazel build event at {build_event_path}:{line_number}: "
                    f"{error.msg}"
                ) from error
            configured = event.get("configured")
            if isinstance(configured, dict) and "testSize" in configured:
                selected_labels.add(
                    _event_label(event, "targetConfigured", line_number)
                )
            summary = event.get("testSummary")
            if isinstance(summary, dict):
                status = summary.get("overallStatus")
                if not isinstance(status, str) or not status:
                    raise BazelTestResultsError(
                        f"Bazel testSummary has no status at line {line_number}"
                    )
                total_run_count = _nonnegative_integer(
                    summary.get("totalRunCount"),
                    field="testSummary.totalRunCount",
                    line_number=line_number,
                    default=0,
                )
                cached_run_count = _nonnegative_integer(
                    summary.get("totalNumCached"),
                    field="testSummary.totalNumCached",
                    line_number=line_number,
                    default=0,
                )
                if cached_run_count > total_run_count:
                    raise BazelTestResultsError(
                        "Bazel testSummary cached run count exceeds its total at "
                        f"line {line_number}"
                    )
                test_summaries.append(
                    {
                        "label": _event_label(event, "testSummary", line_number),
                        "_configuration_id": _configuration_id(
                            event, "testSummary", line_number
                        ),
                        "status": status,
                        "total_run_count": total_run_count,
                        "fresh_run_count": total_run_count - cached_run_count,
                        "cached_run_count": cached_run_count,
                        "attempt_count": _nonnegative_integer(
                            summary.get("attemptCount"),
                            field="testSummary.attemptCount",
                            line_number=line_number,
                            default=0,
                        ),
                        "duration_millis": _nonnegative_integer(
                            summary.get("totalRunDurationMillis"),
                            field="testSummary.totalRunDurationMillis",
                            line_number=line_number,
                            default=0,
                        ),
                    }
                )
            result = event.get("testResult")
            if isinstance(result, dict):
                status = result.get("status")
                if not isinstance(status, str) or not status:
                    raise BazelTestResultsError(
                        f"Bazel testResult has no status at line {line_number}"
                    )
                if status != "PASSED":
                    event_id = event.get("id", {}).get("testResult", {})
                    execution_info = result.get("executionInfo")
                    if not isinstance(execution_info, dict):
                        execution_info = {}
                    failed_attempts.append(
                        {
                            "label": _event_label(event, "testResult", line_number),
                            "_configuration_id": _configuration_id(
                                event, "testResult", line_number
                            ),
                            "run": _nonnegative_integer(
                                event_id.get("run"),
                                field="id.testResult.run",
                                line_number=line_number,
                                default=0,
                            ),
                            "shard": _nonnegative_integer(
                                event_id.get("shard"),
                                field="id.testResult.shard",
                                line_number=line_number,
                                default=0,
                            ),
                            "attempt": _nonnegative_integer(
                                event_id.get("attempt"),
                                field="id.testResult.attempt",
                                line_number=line_number,
                                default=0,
                            ),
                            "status": status,
                            "duration_millis": _nonnegative_integer(
                                result.get("testAttemptDurationMillis"),
                                field="testResult.testAttemptDurationMillis",
                                line_number=line_number,
                                default=0,
                            ),
                            "cached": bool(
                                result.get("cachedLocally", False)
                                or execution_info.get("cachedRemotely", False)
                            ),
                        }
                    )
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
            if "finished" in event:
                if finished is not None:
                    raise BazelTestResultsError(
                        "Bazel build-event stream contains multiple finished events"
                    )
                finished = event["finished"]
                if not isinstance(finished, dict):
                    raise BazelTestResultsError(
                        f"Bazel finished event is malformed at line {line_number}"
                    )

    observation = _observation(
        phase=output_dir.name,
        selected_labels=selected_labels,
        test_summaries=test_summaries,
        failed_attempts=failed_attempts,
        finished=finished,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    observation_path = output_dir / "bep-observation.json"
    observation_path.write_text(
        json.dumps(observation, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    observation_byte_length = observation_path.stat().st_size
    observation_seconds = time.monotonic() - observation_started

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
    return BazelTestResultsStage(
        output_count=len(staged_sources),
        observation_path=observation_path,
        observation_byte_length=observation_byte_length,
        observation_seconds=observation_seconds,
    )
