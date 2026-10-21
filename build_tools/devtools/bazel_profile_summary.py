# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Extracts stable CI cost evidence from one Bazel execution profile."""

from __future__ import annotations

import dataclasses
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "hrx.bazel-profile-summary.v1"
SLOW_DURATION_SECONDS = 10.0
MAX_RECORDED_SUSPECTS = 50
MAX_RENDERED_SUSPECTS = 20


class BazelProfileSummaryError(RuntimeError):
    """Raised when a requested Bazel profile cannot produce a valid summary."""


@dataclasses.dataclass(frozen=True)
class ActionAggregate:
    """Aggregate wall time for one Bazel action mnemonic."""

    # Bazel action mnemonic, such as TestRunner or CppCompile.
    mnemonic: str
    # Number of complete action-processing events with this mnemonic.
    count: int
    # Sum of action-processing durations, including parallel actions.
    duration_seconds: float


@dataclasses.dataclass(frozen=True)
class ActionTiming:
    """One action or critical-path component selected for inspection."""

    # Human-readable action or component description from the profile.
    name: str
    # Duration represented by this profile event.
    duration_seconds: float
    # Bazel action mnemonic when the event describes an action.
    mnemonic: str | None


@dataclasses.dataclass(frozen=True)
class TestTiming:
    """Execution cost attributed to one Bazel TestRunner action."""

    # Bazel test label named by the TestRunner action.
    label: str
    # Full TestRunner action duration, including setup and scheduling overhead.
    action_seconds: float
    # Nested test-process duration, or None when the profile did not record it.
    process_seconds: float | None
    # Action time outside recorded test processes, or None without process timing.
    overhead_seconds: float | None
    # Number of nested process events contributing to process_seconds.
    process_event_count: int
    # Whether this TestRunner action appears on Bazel's reported critical path.
    critical_path: bool


@dataclasses.dataclass(frozen=True)
class BazelProfileSummary:
    """Compact, machine-readable evidence extracted from one Bazel profile."""

    # Schema identifier for consumers retaining summaries across revisions.
    schema: str
    # Basename of the source profile without a runner-local directory.
    profile: str
    # Bazel version recorded by the profile, when present.
    bazel_version: str | None
    # Bazel invocation build identifier, when present.
    build_id: str | None
    # Profile creation timestamp, when present.
    date: str | None
    # Span from the first to the last recorded profile event.
    elapsed_seconds: float
    # Number of complete action-processing events.
    action_count: int
    # Sum of action-processing durations, including parallel actions.
    action_seconds: float
    # Per-mnemonic action counts and aggregate durations.
    action_aggregates: tuple[ActionAggregate, ...]
    # Number of profiled TestRunner actions.
    test_action_count: int
    # Sum of TestRunner action durations.
    test_action_seconds: float
    # Number of TestRunner actions with nested process timing.
    test_process_count: int
    # Sum of nested test-process durations.
    test_process_seconds: float
    # Number of tests whose process or action duration meets the suspect threshold.
    slow_test_count: int
    # Number of tests whose nested process duration meets the suspect threshold.
    slow_test_process_count: int
    # Number of tests whose full action duration meets the suspect threshold.
    slow_test_action_count: int
    # Longest tests meeting either suspect threshold, capped for artifact size.
    slow_tests: tuple[TestTiming, ...]
    # Number of non-test actions meeting the suspect threshold.
    slow_non_test_action_count: int
    # Longest non-test actions meeting the suspect threshold, capped for artifact size.
    slow_non_test_actions: tuple[ActionTiming, ...]
    # Bazel-reported critical-path components in dependency order.
    critical_path: tuple[ActionTiming, ...]


def _seconds(microseconds: int | float) -> float:
    return round(float(microseconds) / 1_000_000.0, 6)


def _event_number(event: dict[str, Any], field: str) -> float:
    value = event.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BazelProfileSummaryError(
            f"profile event has no numeric {field}: {event!r}"
        )
    return float(value)


def _event_name(event: dict[str, Any], fallback: str) -> str:
    value = event.get("name")
    if value is None or value == "":
        return fallback
    if not isinstance(value, str):
        raise BazelProfileSummaryError(
            f"profile event has a non-string name: {event!r}"
        )
    return value


def _action_mnemonic(event: dict[str, Any]) -> str:
    args = event.get("args", {})
    if not isinstance(args, dict):
        raise BazelProfileSummaryError(f"action event has non-object args: {event!r}")
    mnemonic = args.get("mnemonic", "<unknown>")
    if not isinstance(mnemonic, str):
        raise BazelProfileSummaryError(
            f"action event has a non-string mnemonic: {event!r}"
        )
    return mnemonic or "<unknown>"


def _is_complete_event(event: dict[str, Any]) -> bool:
    return event.get("ph") == "X"


def _is_process_event(event: dict[str, Any]) -> bool:
    category = event.get("cat", "")
    return event.get("name") == "subprocess.run" or (
        isinstance(category, str) and "process wall time" in category.lower()
    )


def _critical_action_names(events: list[dict[str, Any]]) -> frozenset[str]:
    names = []
    for event in events:
        if event.get("cat") != "critical path component":
            continue
        name = _event_name(event, "<unnamed critical-path component>")
        if name.startswith("action '") and name.endswith("'"):
            names.append(name.removeprefix("action '").removesuffix("'"))
    return frozenset(names)


def _profile_events(profile_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not profile_path.is_file():
        raise BazelProfileSummaryError(f"Bazel profile does not exist: {profile_path}")
    try:
        with gzip.open(profile_path, "rt", encoding="utf-8") as stream:
            profile = json.load(stream)
    except (EOFError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BazelProfileSummaryError(
            f"cannot read Bazel profile {profile_path}: {error}"
        ) from error
    if not isinstance(profile, dict):
        raise BazelProfileSummaryError(
            f"Bazel profile root is not an object: {profile_path}"
        )
    events = profile.get("traceEvents")
    if not isinstance(events, list) or not events:
        raise BazelProfileSummaryError(
            f"Bazel profile has no trace events: {profile_path}"
        )
    if not all(isinstance(event, dict) for event in events):
        raise BazelProfileSummaryError(
            f"Bazel profile contains a non-object trace event: {profile_path}"
        )
    return profile, events


def summarize(profile_path: Path) -> BazelProfileSummary:
    """Parses profile_path and returns bounded CI cost evidence."""
    profile, events = _profile_events(profile_path)
    spans = []
    for event in events:
        timestamp = event.get("ts")
        if timestamp is None:
            continue
        start = _event_number(event, "ts")
        spans.append(
            (start, start + (_event_number(event, "dur") if "dur" in event else 0.0))
        )
    if not spans:
        raise BazelProfileSummaryError(
            f"Bazel profile has no timed trace events: {profile_path}"
        )

    action_events = [
        event
        for event in events
        if event.get("cat") == "action processing" and _is_complete_event(event)
    ]
    process_events_by_thread: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if _is_complete_event(event) and _is_process_event(event):
            process_events_by_thread[event.get("tid")].append(event)

    aggregate_counts: dict[str, int] = defaultdict(int)
    aggregate_durations: dict[str, float] = defaultdict(float)
    action_seconds = 0.0
    test_timings = []
    non_test_actions = []
    critical_action_names = _critical_action_names(events)
    for event in action_events:
        duration_microseconds = _event_number(event, "dur")
        duration_seconds = _seconds(duration_microseconds)
        mnemonic = _action_mnemonic(event)
        name = _event_name(event, f"<unnamed {mnemonic} action>")
        aggregate_counts[mnemonic] += 1
        aggregate_durations[mnemonic] += duration_microseconds
        action_seconds += duration_microseconds
        if mnemonic != "TestRunner":
            non_test_actions.append(
                ActionTiming(
                    name=name,
                    duration_seconds=duration_seconds,
                    mnemonic=mnemonic,
                )
            )
            continue

        start = _event_number(event, "ts")
        end = start + duration_microseconds
        nested_process_events = [
            nested
            for nested in process_events_by_thread[event.get("tid")]
            if _event_number(nested, "ts") >= start
            and _event_number(nested, "ts") + _event_number(nested, "dur") <= end
        ]
        process_microseconds = sum(
            _event_number(nested, "dur") for nested in nested_process_events
        )
        if process_microseconds > duration_microseconds:
            raise BazelProfileSummaryError(
                f"nested process time exceeds TestRunner action time: {name}"
            )
        process_seconds = (
            _seconds(process_microseconds) if nested_process_events else None
        )
        overhead_seconds = (
            _seconds(duration_microseconds - process_microseconds)
            if nested_process_events
            else None
        )
        display = name.removeprefix("Testing ")
        test_timings.append(
            TestTiming(
                label=display,
                action_seconds=duration_seconds,
                process_seconds=process_seconds,
                overhead_seconds=overhead_seconds,
                process_event_count=len(nested_process_events),
                critical_path=name in critical_action_names,
            )
        )

    action_aggregates = tuple(
        ActionAggregate(
            mnemonic=mnemonic,
            count=aggregate_counts[mnemonic],
            duration_seconds=_seconds(aggregate_durations[mnemonic]),
        )
        for mnemonic in sorted(
            aggregate_counts,
            key=lambda value: (-aggregate_durations[value], value),
        )
    )
    slow_tests = [
        timing
        for timing in test_timings
        if timing.action_seconds >= SLOW_DURATION_SECONDS
        or (
            timing.process_seconds is not None
            and timing.process_seconds >= SLOW_DURATION_SECONDS
        )
    ]
    slow_tests.sort(
        key=lambda timing: (
            -max(timing.action_seconds, timing.process_seconds or 0.0),
            timing.label,
        )
    )
    slow_non_test_actions = [
        timing
        for timing in non_test_actions
        if timing.duration_seconds >= SLOW_DURATION_SECONDS
    ]
    slow_non_test_actions.sort(
        key=lambda timing: (-timing.duration_seconds, timing.name)
    )

    critical_path = []
    for event in events:
        if event.get("cat") != "critical path component":
            continue
        # Bazel emits zero-duration critical-path components as instant events.
        # They retain their position and name without a duration field.
        duration = 0.0 if event.get("ph") == "i" else _event_number(event, "dur")
        critical_path.append(
            (
                _event_number(event, "ts"),
                ActionTiming(
                    name=_event_name(event, "<unnamed critical-path component>"),
                    duration_seconds=_seconds(duration),
                    mnemonic=None,
                ),
            )
        )
    critical_path.sort(key=lambda item: item[0])

    other_data = profile.get("otherData", {})
    if not isinstance(other_data, dict):
        raise BazelProfileSummaryError(
            f"Bazel profile otherData is not an object: {profile_path}"
        )
    for field in ("bazel_version", "build_id", "date"):
        value = other_data.get(field)
        if value is not None and not isinstance(value, str):
            raise BazelProfileSummaryError(
                f"Bazel profile {field} is not a string: {profile_path}"
            )

    process_timings = [
        timing for timing in test_timings if timing.process_seconds is not None
    ]
    return BazelProfileSummary(
        schema=SCHEMA,
        profile=profile_path.name,
        bazel_version=other_data.get("bazel_version"),
        build_id=other_data.get("build_id"),
        date=other_data.get("date"),
        elapsed_seconds=_seconds(
            max(end for _, end in spans) - min(start for start, _ in spans)
        ),
        action_count=len(action_events),
        action_seconds=_seconds(action_seconds),
        action_aggregates=action_aggregates,
        test_action_count=len(test_timings),
        test_action_seconds=round(
            sum(timing.action_seconds for timing in test_timings), 6
        ),
        test_process_count=len(process_timings),
        test_process_seconds=round(
            sum(timing.process_seconds or 0.0 for timing in process_timings), 6
        ),
        slow_test_count=len(slow_tests),
        slow_test_process_count=sum(
            timing.process_seconds is not None
            and timing.process_seconds >= SLOW_DURATION_SECONDS
            for timing in test_timings
        ),
        slow_test_action_count=sum(
            timing.action_seconds >= SLOW_DURATION_SECONDS for timing in test_timings
        ),
        slow_tests=tuple(slow_tests[:MAX_RECORDED_SUSPECTS]),
        slow_non_test_action_count=len(slow_non_test_actions),
        slow_non_test_actions=tuple(slow_non_test_actions[:MAX_RECORDED_SUSPECTS]),
        critical_path=tuple(timing for _, timing in critical_path),
    )


def summary_path(profile_path: Path) -> Path:
    """Returns the machine-readable summary path beside profile_path."""
    name = profile_path.name
    if not name.endswith(".profile.gz"):
        raise BazelProfileSummaryError(
            f"Bazel profile name must end in .profile.gz: {profile_path}"
        )
    return profile_path.with_name(name.removesuffix(".profile.gz") + ".summary.json")


def write(summary: BazelProfileSummary, output_path: Path) -> None:
    """Writes summary as stable formatted JSON."""
    output_path.write_text(
        json.dumps(dataclasses.asdict(summary), indent=2) + "\n", encoding="utf-8"
    )


def _markdown_text(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "\\|")
        .replace("\n", " ")
    )


def _seconds_cell(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f} s"


def render_markdown(summary: BazelProfileSummary, title: str) -> str:
    """Renders one concise GitHub job-summary section."""
    lines = [
        f"### Bazel profile: {_markdown_text(title)}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Phase span | {summary.elapsed_seconds:.3f} s |",
        f"| Actions | {summary.action_count:,} |",
        f"| TestRunner actions | {summary.test_action_count:,} |",
        f"| Aggregate test-process work | {summary.test_process_seconds:.3f} s |",
        f"| Test processes at or above {SLOW_DURATION_SECONDS:g} s | "
        f"{summary.slow_test_process_count:,} |",
        f"| Test actions at or above {SLOW_DURATION_SECONDS:g} s | "
        f"{summary.slow_test_action_count:,} |",
        f"| Non-test actions at or above {SLOW_DURATION_SECONDS:g} s | "
        f"{summary.slow_non_test_action_count:,} |",
        "",
    ]
    if summary.slow_tests:
        lines.extend(
            [
                f"#### Test suspects (at or above {SLOW_DURATION_SECONDS:g} s)",
                "",
                "| Process | Action | Target |",
                "| ---: | ---: | --- |",
            ]
        )
        for timing in summary.slow_tests[:MAX_RENDERED_SUSPECTS]:
            critical = " **critical path**" if timing.critical_path else ""
            lines.append(
                f"| {_seconds_cell(timing.process_seconds)} | "
                f"{_seconds_cell(timing.action_seconds)} | "
                f"`{_markdown_text(timing.label)}`{critical} |"
            )
        if summary.slow_test_count > MAX_RENDERED_SUSPECTS:
            lines.append(
                f"|  |  | {summary.slow_test_count - MAX_RENDERED_SUSPECTS:,} "
                f"additional suspects; JSON retains the longest "
                f"{len(summary.slow_tests):,} and the total count |"
            )
        lines.append("")
    else:
        lines.extend(
            [
                f"No test process or action reached {SLOW_DURATION_SECONDS:g} s.",
                "",
            ]
        )

    if summary.slow_non_test_actions:
        lines.extend(
            [
                f"#### Non-test action suspects (at or above {SLOW_DURATION_SECONDS:g} s)",
                "",
                "| Duration | Mnemonic | Action |",
                "| ---: | --- | --- |",
            ]
        )
        for timing in summary.slow_non_test_actions[:MAX_RENDERED_SUSPECTS]:
            lines.append(
                f"| {_seconds_cell(timing.duration_seconds)} | "
                f"`{_markdown_text(timing.mnemonic or '<unknown>')}` | "
                f"{_markdown_text(timing.name)} |"
            )
        if summary.slow_non_test_action_count > MAX_RENDERED_SUSPECTS:
            lines.append(
                f"|  |  | {summary.slow_non_test_action_count - MAX_RENDERED_SUSPECTS:,} "
                f"additional suspects; JSON retains the longest "
                f"{len(summary.slow_non_test_actions):,} and the total count |"
            )
        lines.append("")

    if summary.critical_path:
        lines.extend(
            [
                "#### Largest critical-path components",
                "",
                "| Duration | Component |",
                "| ---: | --- |",
            ]
        )
        for timing in sorted(
            summary.critical_path,
            key=lambda value: (-value.duration_seconds, value.name),
        )[:MAX_RENDERED_SUSPECTS]:
            lines.append(
                f"| {_seconds_cell(timing.duration_seconds)} | "
                f"{_markdown_text(timing.name)} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_console(summary: BazelProfileSummary, title: str) -> str:
    """Renders a compact log summary for local and CI diagnosis."""
    lines = [
        f"[profile] {title}: {summary.elapsed_seconds:.3f}s phase, "
        f"{summary.test_action_count} test actions, "
        f"{summary.test_process_seconds:.3f}s process work, "
        f"{summary.slow_test_count} test suspect(s), "
        f"{summary.slow_non_test_action_count} non-test action suspect(s)"
    ]
    for timing in summary.slow_tests[:MAX_RENDERED_SUSPECTS]:
        process = (
            f"{timing.process_seconds:.3f}s process"
            if timing.process_seconds is not None
            else "process timing unavailable"
        )
        lines.append(
            f"  [test] {process}, {timing.action_seconds:.3f}s action {timing.label}"
        )
    for timing in summary.slow_non_test_actions[:MAX_RENDERED_SUSPECTS]:
        lines.append(
            f"  [action] {timing.duration_seconds:.3f}s "
            f"{timing.mnemonic or '<unknown>'} {timing.name}"
        )
    return "\n".join(lines)
