# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reports recent GitHub Actions lane and step cost distributions."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import statistics
import sys
import urllib.parse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from build_tools.ci import github_api

DEFAULT_REPOSITORY = "ROCm/hrx-system"
DEFAULT_WORKFLOW = "ci_iree.yml"
DEFAULT_RUN_COUNT = 10
DEFAULT_TOP_COUNT = 20
MAX_RUN_COUNT = 100
SCHEMA = "hrx.github-actions-cost-history.v1"


@dataclasses.dataclass(frozen=True)
class Distribution:
    """Bounded descriptive statistics for one duration population."""

    count: int
    median_seconds: float
    p95_seconds: float
    maximum_seconds: float


@dataclasses.dataclass(frozen=True)
class LaneSummary:
    """Recent queue and successful execution cost for one stable job name."""

    name: str
    sample_count: int
    success_count: int
    queue: Distribution
    successful_duration: Distribution | None


@dataclasses.dataclass(frozen=True)
class StepSummary:
    """Recent successful execution cost for one stable job step."""

    lane_name: str
    step_name: str
    duration: Distribution


@dataclasses.dataclass(frozen=True)
class HistorySummary:
    """Cost evidence collected from a bounded set of workflow runs."""

    schema: str
    repository: str
    workflow: str
    event: str
    run_ids: tuple[int, ...]
    lanes: tuple[LaneSummary, ...]
    steps: tuple[StepSummary, ...]


@dataclasses.dataclass(frozen=True)
class _StepSample:
    name: str
    duration_seconds: float


@dataclasses.dataclass(frozen=True)
class _LaneSample:
    name: str
    conclusion: str
    queue_seconds: float
    duration_seconds: float
    steps: tuple[_StepSample, ...]


def _parse_timestamp(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _duration_seconds(start_value: Any, end_value: Any) -> float | None:
    start = _parse_timestamp(start_value)
    end = _parse_timestamp(end_value)
    if start is None or end is None:
        return None
    seconds = (end - start).total_seconds()
    return seconds if seconds >= 0 else None


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty duration population")
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]


def _distribution(values: Sequence[float]) -> Distribution:
    if not values:
        raise ValueError("cannot summarize an empty duration population")
    return Distribution(
        count=len(values),
        median_seconds=round(float(statistics.median(values)), 3),
        p95_seconds=round(_nearest_rank(values, 0.95), 3),
        maximum_seconds=round(max(values), 3),
    )


def _lane_sample(job: Mapping[str, Any]) -> _LaneSample | None:
    conclusion = job.get("conclusion")
    if conclusion == "skipped" or not isinstance(conclusion, str):
        return None
    queue_seconds = _duration_seconds(job.get("created_at"), job.get("started_at"))
    duration_seconds = _duration_seconds(job.get("started_at"), job.get("completed_at"))
    if queue_seconds is None or duration_seconds is None:
        return None

    steps = []
    if conclusion == "success":
        for step in job.get("steps", []):
            if not isinstance(step, dict) or step.get("conclusion") != "success":
                continue
            step_duration = _duration_seconds(
                step.get("started_at"), step.get("completed_at")
            )
            step_name = step.get("name")
            if step_duration is None or not isinstance(step_name, str) or not step_name:
                continue
            steps.append(_StepSample(step_name, step_duration))
    name = job.get("name")
    if not isinstance(name, str) or not name:
        return None
    return _LaneSample(
        name,
        conclusion,
        queue_seconds,
        duration_seconds,
        tuple(steps),
    )


def summarize(
    *,
    repository: str,
    workflow: str,
    event: str,
    run_ids: Sequence[int],
    jobs: Sequence[Mapping[str, Any]],
) -> HistorySummary:
    """Summarizes stable lane and step identities across completed runs."""
    samples = [sample for job in jobs if (sample := _lane_sample(job)) is not None]
    samples_by_lane: dict[str, list[_LaneSample]] = defaultdict(list)
    for sample in samples:
        samples_by_lane[sample.name].append(sample)

    lanes = []
    step_durations: dict[tuple[str, str], list[float]] = defaultdict(list)
    for lane_name, lane_samples in samples_by_lane.items():
        successful_samples = [
            sample for sample in lane_samples if sample.conclusion == "success"
        ]
        lanes.append(
            LaneSummary(
                name=lane_name,
                sample_count=len(lane_samples),
                success_count=len(successful_samples),
                queue=_distribution([sample.queue_seconds for sample in lane_samples]),
                successful_duration=(
                    _distribution(
                        [sample.duration_seconds for sample in successful_samples]
                    )
                    if successful_samples
                    else None
                ),
            )
        )
        for sample in successful_samples:
            for step in sample.steps:
                step_durations[(lane_name, step.name)].append(step.duration_seconds)

    lanes.sort(
        key=lambda lane: (
            -(
                lane.successful_duration.median_seconds
                if lane.successful_duration is not None
                else -1.0
            ),
            lane.name,
        )
    )
    steps = [
        StepSummary(lane_name, step_name, _distribution(durations))
        for (lane_name, step_name), durations in step_durations.items()
    ]
    steps.sort(
        key=lambda step: (
            -step.duration.median_seconds,
            -step.duration.p95_seconds,
            step.lane_name,
            step.step_name,
        )
    )
    return HistorySummary(
        schema=SCHEMA,
        repository=repository,
        workflow=workflow,
        event=event,
        run_ids=tuple(run_ids),
        lanes=tuple(lanes),
        steps=tuple(steps),
    )


def collect(
    client: github_api.RepositoryClient,
    *,
    workflow: str,
    event: str,
    run_count: int,
) -> HistorySummary:
    """Collects a bounded history without downloading logs or artifacts."""
    if run_count < 1 or run_count > MAX_RUN_COUNT:
        raise ValueError(f"run count must be between 1 and {MAX_RUN_COUNT}")
    workflow_component = urllib.parse.quote(workflow, safe="")
    query = urllib.parse.urlencode(
        {"event": event, "status": "completed", "per_page": run_count}
    )
    payload = client.json(f"/actions/workflows/{workflow_component}/runs?{query}")
    runs = payload.get("workflow_runs") if isinstance(payload, dict) else None
    if not isinstance(runs, list):
        raise github_api.GitHubAPIError("GitHub response has no workflow_runs list")

    run_ids = []
    jobs = []
    for run in runs[:run_count]:
        if not isinstance(run, dict) or not isinstance(run.get("id"), int):
            raise github_api.GitHubAPIError("GitHub returned an invalid workflow run")
        run_id = int(run["id"])
        run_ids.append(run_id)
        run_jobs = client.pages(f"/actions/runs/{run_id}/jobs?filter=latest", "jobs")
        jobs.extend(job for job in run_jobs if isinstance(job, dict))

    return summarize(
        repository=client.repository,
        workflow=workflow,
        event=event,
        run_ids=run_ids,
        jobs=jobs,
    )


def _duration_text(seconds: float) -> str:
    rounded = int(round(seconds))
    if rounded < 60:
        return f"{rounded}s"
    minutes, remaining_seconds = divmod(rounded, 60)
    if minutes < 60:
        return f"{minutes}m{remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h{remaining_minutes:02d}m"


def render(summary: HistorySummary, *, top_count: int) -> str:
    """Renders a compact report sorted by recurring median cost."""
    if top_count < 1:
        raise ValueError("top count must be positive")
    lines = [
        f"GitHub Actions cost history: {summary.repository} / {summary.workflow}",
        f"event={summary.event}; completed runs={len(summary.run_ids)}; "
        f"run IDs={','.join(str(run_id) for run_id in summary.run_ids)}",
        "",
        "Lanes (successful duration; queue includes every non-skipped sample)",
        "samples  green   queue median/p95/max   duration median/p95/max  lane",
    ]
    for lane in summary.lanes[:top_count]:
        queue = lane.queue
        if lane.successful_duration is None:
            duration_text = "—"
        else:
            duration = lane.successful_duration
            duration_text = (
                f"{_duration_text(duration.median_seconds)}/"
                f"{_duration_text(duration.p95_seconds)}/"
                f"{_duration_text(duration.maximum_seconds)}"
            )
        lines.append(
            f"{lane.sample_count:>7}  {lane.success_count:>3}/"
            f"{lane.sample_count:<3}   "
            f"{_duration_text(queue.median_seconds):>6}/"
            f"{_duration_text(queue.p95_seconds)}/"
            f"{_duration_text(queue.maximum_seconds):<6}   "
            f"{duration_text:<22}  {lane.name}"
        )

    lines.extend(
        [
            "",
            "Steps (successful jobs)",
            "samples  duration median/p95/max  lane — step",
        ]
    )
    for step in summary.steps[:top_count]:
        duration = step.duration
        lines.append(
            f"{duration.count:>7}  "
            f"{_duration_text(duration.median_seconds):>6}/"
            f"{_duration_text(duration.p95_seconds)}/"
            f"{_duration_text(duration.maximum_seconds):<6}  "
            f"{step.lane_name} — {step.step_name}"
        )
    return "\n".join(lines)


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workflow",
        nargs="?",
        default=DEFAULT_WORKFLOW,
        help="workflow file name or numeric workflow ID (default: %(default)s)",
    )
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPOSITORY),
        help="GitHub owner/repository (default: %(default)s)",
    )
    parser.add_argument(
        "--event",
        default="push",
        help="GitHub workflow event to compare (default: %(default)s)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUN_COUNT,
        help=f"completed runs to inspect, at most {MAX_RUN_COUNT} (default: %(default)s)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_COUNT,
        help="maximum lanes and steps to print (default: %(default)s)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the complete machine-readable summary",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    try:
        arguments = parse_arguments(argv)
        client = github_api.RepositoryClient(
            arguments.repository, github_api.resolve_token()
        )
        summary = collect(
            client,
            workflow=arguments.workflow,
            event=arguments.event,
            run_count=arguments.runs,
        )
        if arguments.json:
            print(json.dumps(dataclasses.asdict(summary), indent=2))
        else:
            print(render(summary, top_count=arguments.top))
        return 0
    except (OSError, ValueError, github_api.GitHubAPIError) as exc:
        print(f"GitHub Actions history: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
