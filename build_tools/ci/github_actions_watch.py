# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reports high-signal state for one Loom pull request or workflow run."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from build_tools.ci import github_actions_triage as triage
from build_tools.ci import github_api

DEFAULT_REPOSITORY = "ROCm/hrx-system"
WORKFLOW_FILE = "ci_iree.yml"
POLL_SECONDS = 60
HEARTBEAT_SECONDS = 5 * 60
LIVE_JOB_DETAIL_LIMIT = 4

EXIT_CLEAN = 0
EXIT_FAILURE = 1
EXIT_TOOL_ERROR = 2
EXIT_INCOMPLETE = 3
EXIT_CLASSIFIED_NONREPOSITORY = 4
EXIT_INTERRUPTED = 130

HELP_EPILOG = """examples:
  python -m build_tools.ci.github_actions_watch 2202
  python -m build_tools.ci.github_actions_watch 2202 --watch
  python -m build_tools.ci.github_actions_watch run:39682481914 --json

agent watch behavior:
  --watch polls every 60 seconds and writes as soon as a meaningful state
  changes, including a newly failed job. In a long-running exec session, retain
  the session ID and wait on that same session for output. The wait returns on
  emitted output, while the watcher stays alive until the workflow is terminal.
  Investigate a new FAIL or UNKNOWN finding immediately, then resume waiting on
  the same session; completed-job evidence is cached and is not downloaded
  again. A five-minute heartbeat keeps a long unchanged queue observable.

exit status:
  0  clean or proven obsolete
  1  terminal repository or unknown failure
  2  command or GitHub API error
  3  workflow still active (snapshot mode)
  4  terminal infrastructure or explicitly tracked failure
  130 interrupted by the operator
"""


class WatchError(RuntimeError):
    """Raised when the requested GitHub state cannot be observed safely."""


@dataclasses.dataclass(frozen=True)
class Target:
    kind: str
    number: int
    repository: str | None = None


@dataclasses.dataclass(frozen=True)
class Selection:
    label: str
    run: Mapping[str, Any] | None
    obsolete: bool
    detail: str = ""


@dataclasses.dataclass(frozen=True)
class LiveJob:
    state: str
    name: str
    url: str
    age_seconds: int
    detail: str


@dataclasses.dataclass(frozen=True)
class Snapshot:
    label: str
    run_id: int | None
    run_url: str | None
    run_status: str
    run_conclusion: str | None
    obsolete: bool
    disposition: str
    terminal: bool
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    live_jobs: tuple[LiveJob, ...]
    finding_groups: tuple[triage.FindingGroup, ...]
    warnings: tuple[str, ...]
    detail: str = ""


class GitHubClient:
    """Repository-scoped access to the GitHub Actions API."""

    def __init__(
        self,
        repository: str,
        token: str,
        *,
        get_json: Callable[[str, str], Any] = github_api.get_json,
        get_bytes: Callable[[str, str], bytes] = github_api.get_bytes,
    ):
        if not re.fullmatch(r"[^/]+/[^/]+", repository):
            raise WatchError(f"invalid GitHub repository {repository!r}")
        self.repository = repository
        self.token = token
        self._get_json = get_json
        self._get_bytes = get_bytes

    @property
    def api_root(self) -> str:
        return f"https://api.github.com/repos/{self.repository}"

    def _url(self, path_or_url: str) -> str:
        if path_or_url.startswith("/"):
            return self.api_root + path_or_url
        parsed = urllib.parse.urlsplit(path_or_url)
        if (
            parsed.scheme == "https"
            and parsed.hostname == "api.github.com"
            and parsed.path.startswith(f"/repos/{self.repository}/")
        ):
            return path_or_url
        raise WatchError(f"refusing unexpected GitHub API URL {path_or_url!r}")

    def json(self, path_or_url: str) -> Any:
        return self._get_json(self._url(path_or_url), self.token)

    def bytes(self, path_or_url: str) -> bytes:
        return self._get_bytes(self._url(path_or_url), self.token)

    def pages(self, path_or_url: str, field: str | None) -> list[Any]:
        url = self._url(path_or_url)
        results = []
        page = 1
        while True:
            parsed = urllib.parse.urlsplit(url)
            query = dict(urllib.parse.parse_qsl(parsed.query))
            query.update({"per_page": "100", "page": str(page)})
            page_url = urllib.parse.urlunsplit(
                parsed._replace(query=urllib.parse.urlencode(query))
            )
            payload = self._get_json(page_url, self.token)
            if field is None:
                page_results = payload
            elif isinstance(payload, dict):
                page_results = payload.get(field)
            else:
                page_results = None
            if not isinstance(page_results, list):
                raise WatchError(f"GitHub response has no {field or 'list'} page")
            results.extend(page_results)
            if len(page_results) < 100:
                return results
            page += 1


class EvidenceLoader:
    """Loads each completed job's immutable evidence at most once."""

    def __init__(
        self,
        client: GitHubClient,
        open_issues: frozenset[str],
    ):
        self.client = client
        self.open_issues = open_issues
        self._cache: dict[int, triage.JobFinding] = {}

    def load(self, job: Mapping[str, Any]) -> tuple[triage.JobFinding, str | None]:
        job_id = int(job["id"])
        cached = self._cache.get(job_id)
        if cached is not None:
            return cached, None
        try:
            check_run_url = str(job["check_run_url"]).rstrip("/")
            annotations = self.client.pages(check_run_url + "/annotations", None)
            messages = [
                str(annotation.get("message", ""))
                for annotation in annotations
                if isinstance(annotation, dict)
            ]
            finding = triage.classify_failure(
                job,
                messages,
                "",
                open_issues=self.open_issues,
            )
            if finding.category == triage.UNKNOWN:
                log = self.client.bytes(f"/actions/jobs/{job_id}/logs").decode(
                    "utf-8", errors="replace"
                )
                finding = triage.classify_failure(
                    job,
                    messages,
                    log,
                    open_issues=self.open_issues,
                )
        except (KeyError, OSError, ValueError, WatchError) as exc:
            finding = triage.Finding(
                triage.UNKNOWN,
                "evidence_unavailable",
                f"job {job_id}",
                "completed failure evidence could not be loaded",
            )
            return self._job_finding(job, finding), str(exc)
        result = self._job_finding(job, finding)
        self._cache[job_id] = result
        return result, None

    @staticmethod
    def _job_finding(
        job: Mapping[str, Any], finding: triage.Finding
    ) -> triage.JobFinding:
        return triage.JobFinding(
            int(job["id"]),
            str(job.get("name", f"job {job['id']}")),
            str(job.get("html_url", "")),
            finding,
        )


def parse_target(value: str) -> Target:
    url_match = re.fullmatch(
        r"https://github\.com/([^/]+/[^/]+)/(pull|actions/runs)/(\d+)/?",
        value,
    )
    if url_match:
        repository, kind, number = url_match.groups()
        return Target("pr" if kind == "pull" else "run", int(number), repository)
    prefix_match = re.fullmatch(r"(pr|run):(\d+)", value)
    if prefix_match:
        kind, number = prefix_match.groups()
        return Target(kind, int(number))
    if value.isdigit():
        return Target("pr", int(value))
    raise WatchError(
        "target must be a PR number, pr:NUMBER, run:NUMBER, or GitHub PR/run URL"
    )


def resolve_token(environment: Mapping[str, str] = os.environ) -> str:
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        token = environment.get(name, "").strip()
        if token:
            return token
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as exc:
        raise WatchError("no GitHub token and gh is unavailable") from exc
    token = result.stdout.strip()
    if result.returncode or not token:
        raise WatchError("set GH_TOKEN/GITHUB_TOKEN or authenticate gh")
    return token


def open_tracked_issues() -> frozenset[str]:
    command = ["br", "list"]
    for issue in triage.TRACKED_ISSUES:
        command.extend(("--id", issue))
    command.extend(("--format", "csv", "--fields", "id,status"))
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return frozenset()
    if result.returncode:
        return frozenset()
    rows = csv.DictReader(result.stdout.splitlines())
    return frozenset(row["id"] for row in rows if row.get("status") == "open")


def _run_created_key(run: Mapping[str, Any]) -> tuple[str, int]:
    return (str(run.get("created_at", "")), int(run.get("id", 0)))


def select_run(client: GitHubClient, target: Target) -> Selection:
    if target.kind == "run":
        run = client.json(f"/actions/runs/{target.number}")
        if not isinstance(run, dict):
            raise WatchError("GitHub returned an invalid workflow run")
        if run.get("path") != f".github/workflows/{WORKFLOW_FILE}":
            raise WatchError(f"run {target.number} is not a Loom workflow")
        return Selection(f"run {target.number}", run, False)

    pull = client.json(f"/pulls/{target.number}")
    if not isinstance(pull, dict) or not isinstance(pull.get("head"), dict):
        raise WatchError("GitHub returned an invalid pull request")
    head_sha = str(pull["head"].get("sha", ""))
    if not head_sha:
        raise WatchError("pull request has no head revision")
    query = urllib.parse.urlencode({"event": "pull_request", "head_sha": head_sha})
    runs = client.pages(
        f"/actions/workflows/{WORKFLOW_FILE}/runs?{query}", "workflow_runs"
    )
    candidates = [
        run
        for run in runs
        if isinstance(run, dict)
        and run.get("event") == "pull_request"
        and run.get("head_sha") == head_sha
    ]
    obsolete = pull.get("state") != "open"
    label = f"PR #{target.number}"
    if not candidates:
        detail = f"no Loom workflow exists yet for {head_sha[:12]}"
        if obsolete:
            detail = "pull request is closed and has no current Loom workflow"
        return Selection(label, None, obsolete, detail)
    return Selection(label, max(candidates, key=_run_created_key), obsolete)


def _parse_timestamp(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def _age_seconds(value: Any, now: dt.datetime) -> int:
    timestamp = _parse_timestamp(value)
    if timestamp is None:
        return 0
    return max(0, int((now - timestamp).total_seconds()))


def _live_job(job: Mapping[str, Any], now: dt.datetime) -> LiveJob:
    steps = job.get("steps", [])
    current_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and step.get("status") == "in_progress"
        ),
        None,
    )
    if not job.get("runner_id") and not steps:
        state = "waiting"
        detail = "no runner assigned"
        age_from = job.get("created_at") or job.get("started_at")
    elif job.get("status") == "in_progress":
        state = "running"
        detail = (
            str(current_step.get("name"))
            if current_step is not None
            else "runner setup"
        )
        age_from = (
            current_step.get("started_at")
            if current_step is not None
            else job.get("started_at")
        )
    else:
        state = "waiting"
        detail = str(job.get("status", "pending"))
        age_from = job.get("created_at") or job.get("started_at")
    return LiveJob(
        state,
        str(job.get("name", f"job {job.get('id', '?')}")),
        str(job.get("html_url", "")),
        _age_seconds(age_from, now),
        detail,
    )


def collect_snapshot(
    client: GitHubClient,
    selection: Selection,
    evidence_loader: EvidenceLoader,
    *,
    now: dt.datetime | None = None,
) -> Snapshot:
    now = now or dt.datetime.now(dt.UTC)
    if selection.run is None:
        return Snapshot(
            selection.label,
            None,
            None,
            "waiting",
            None,
            selection.obsolete,
            "obsolete" if selection.obsolete else "in_progress",
            selection.obsolete,
            0,
            0,
            0,
            (),
            (),
            (),
            selection.detail,
        )

    run = selection.run
    run_id = int(run["id"])
    jobs = client.pages(f"/actions/runs/{run_id}/jobs?filter=latest", "jobs")
    job_findings = []
    warnings = []
    for job in jobs:
        if not isinstance(job, dict) or job.get("conclusion") not in (
            "failure",
            "timed_out",
        ):
            continue
        job_finding, error = evidence_loader.load(job)
        job_findings.append(job_finding)
        if error:
            warnings.append(f"job {job_finding.job_id}: {error}")

    live_jobs = tuple(
        _live_job(job, now)
        for job in jobs
        if isinstance(job, dict) and job.get("status") != "completed"
    )
    run_status = str(run.get("status", "unknown"))
    terminal = run_status == "completed" or selection.obsolete
    finding_disposition = triage.failure_disposition(job_findings)
    if selection.obsolete:
        disposition = "obsolete"
    elif finding_disposition is not None:
        disposition = finding_disposition
    elif not terminal:
        disposition = "in_progress"
    elif run.get("conclusion") == "success":
        disposition = "clean"
    else:
        disposition = triage.UNKNOWN

    return Snapshot(
        selection.label,
        run_id,
        str(run.get("html_url", "")),
        run_status,
        str(run["conclusion"]) if run.get("conclusion") is not None else None,
        selection.obsolete,
        disposition,
        terminal,
        len(jobs),
        sum(
            1
            for job in jobs
            if isinstance(job, dict) and job.get("status") == "completed"
        ),
        len(job_findings),
        live_jobs,
        tuple(triage.group_findings(job_findings)),
        tuple(warnings),
    )


def _duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h{remaining_minutes:02d}m"


def _displayed_live_jobs(live_jobs: Sequence[LiveJob]) -> tuple[LiveJob, ...]:
    return tuple(
        sorted(
            live_jobs,
            key=lambda job: (-job.age_seconds, job.state, job.name),
        )[:LIVE_JOB_DETAIL_LIMIT]
    )


def _live_state_summary(live_jobs: Sequence[LiveJob]) -> str:
    counts = {
        state: sum(job.state == state for job in live_jobs)
        for state in sorted({job.state for job in live_jobs})
    }
    return ", ".join(f"{count} {state}" for state, count in counts.items())


def render_snapshot(snapshot: Snapshot) -> str:
    run_identity = f"run {snapshot.run_id}" if snapshot.run_id else "no run"
    lines = [
        f"Loom {snapshot.label} · {run_identity} · {snapshot.disposition}",
    ]
    if snapshot.run_url:
        lines.append(snapshot.run_url)
    if snapshot.detail:
        lines.append(snapshot.detail)
    if snapshot.total_jobs:
        active_detail = ""
        if snapshot.live_jobs:
            active_detail = f" ({_live_state_summary(snapshot.live_jobs)})"
        lines.append(
            f"{snapshot.completed_jobs}/{snapshot.total_jobs} jobs complete; "
            f"{snapshot.failed_jobs} failed; {len(snapshot.live_jobs)} active"
            f"{active_detail}"
        )
    displayed_live_jobs = _displayed_live_jobs(snapshot.live_jobs)
    for live_job in displayed_live_jobs:
        lines.append(
            f"{live_job.state.upper():7} {_duration(live_job.age_seconds):>6}  "
            f"{live_job.name} — {live_job.detail}"
        )
        if live_job.url:
            lines.append(f"         {live_job.url}")
    omitted_live_jobs = len(snapshot.live_jobs) - len(displayed_live_jobs)
    if omitted_live_jobs:
        lines.append(
            f"         {omitted_live_jobs} more active job"
            f"{'s' if omitted_live_jobs != 1 else ''}"
        )
    labels = {
        triage.REPOSITORY: "FAIL",
        triage.UNKNOWN: "UNKNOWN",
        triage.INFRASTRUCTURE: "INFRA",
    }
    for group in snapshot.finding_groups:
        finding = group.finding
        issue = f" [{finding.issue}]" if finding.issue else ""
        lines.append(
            f"{labels.get(finding.category, finding.category.upper()):7} "
            f"{finding.summary}{issue} ({len(group.jobs)} job"
            f"{'s' if len(group.jobs) != 1 else ''})"
        )
        for job_finding in group.jobs:
            lines.append(f"         {job_finding.job_name}")
            if job_finding.job_url:
                lines.append(f"         {job_finding.job_url}")
    for warning in snapshot.warnings:
        lines.append(f"WARNING  {warning}")
    return "\n".join(lines)


def snapshot_json(snapshot: Snapshot) -> dict[str, Any]:
    return {
        "label": snapshot.label,
        "run_id": snapshot.run_id,
        "run_url": snapshot.run_url,
        "run_status": snapshot.run_status,
        "run_conclusion": snapshot.run_conclusion,
        "obsolete": snapshot.obsolete,
        "disposition": snapshot.disposition,
        "terminal": snapshot.terminal,
        "job_counts": {
            "total": snapshot.total_jobs,
            "completed": snapshot.completed_jobs,
            "failed": snapshot.failed_jobs,
            "active": len(snapshot.live_jobs),
        },
        "live_jobs": [dataclasses.asdict(job) for job in snapshot.live_jobs],
        "findings": [
            {
                **dataclasses.asdict(group.finding),
                "jobs": [
                    {
                        "id": job.job_id,
                        "name": job.job_name,
                        "url": job.job_url,
                    }
                    for job in group.jobs
                ],
            }
            for group in snapshot.finding_groups
        ],
        "warnings": list(snapshot.warnings),
        "detail": snapshot.detail,
    }


def _state_key(snapshot: Snapshot) -> str:
    payload = snapshot_json(snapshot)
    payload["live_job_states"] = _live_state_summary(snapshot.live_jobs)
    payload["live_jobs"] = [
        dataclasses.asdict(job) for job in _displayed_live_jobs(snapshot.live_jobs)
    ]
    for live_job in payload["live_jobs"]:
        live_job.pop("age_seconds", None)
    return json.dumps(payload, sort_keys=True)


def snapshot_exit_code(snapshot: Snapshot) -> int:
    if not snapshot.terminal:
        return EXIT_INCOMPLETE
    if snapshot.disposition in (triage.REPOSITORY, triage.UNKNOWN):
        return EXIT_FAILURE
    if snapshot.disposition == triage.INFRASTRUCTURE:
        return EXIT_CLASSIFIED_NONREPOSITORY
    return EXIT_CLEAN


def observe(
    client: GitHubClient,
    target: Target,
    *,
    watch: bool,
    json_output: bool,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    evidence_loader = EvidenceLoader(client, open_tracked_issues())
    previous_state = None
    last_output_at = 0.0
    while True:
        selection = select_run(client, target)
        snapshot = collect_snapshot(client, selection, evidence_loader)
        state = _state_key(snapshot)
        observed_at = monotonic()
        if state != previous_state or observed_at - last_output_at >= HEARTBEAT_SECONDS:
            if previous_state is not None and not json_output:
                print()
            if json_output:
                print(json.dumps(snapshot_json(snapshot), sort_keys=True))
            else:
                print(render_snapshot(snapshot))
            sys.stdout.flush()
            previous_state = state
            last_output_at = observed_at
        if not watch or snapshot.terminal:
            return snapshot_exit_code(snapshot)
        sleep(POLL_SECONDS)


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_EPILOG,
    )
    parser.add_argument(
        "target",
        help="PR number, pr:NUMBER, run:NUMBER, or a GitHub PR/workflow-run URL",
    )
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPOSITORY),
        help="Repository for shorthand targets (default: %(default)s)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Poll until the selected current run is terminal",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object per reported state",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    try:
        arguments = parse_arguments(argv)
        target = parse_target(arguments.target)
        repository = target.repository or arguments.repository
        client = GitHubClient(repository, resolve_token())
        return observe(
            client,
            target,
            watch=arguments.watch,
            json_output=arguments.json,
        )
    except (OSError, ValueError, WatchError) as exc:
        print(f"GitHub Actions watch: {exc}", file=sys.stderr)
        return EXIT_TOOL_ERROR
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
