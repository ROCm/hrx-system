# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Conservatively classifies completed GitHub Actions job failures."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

INFRASTRUCTURE = "infrastructure"
REPOSITORY = "repository"
UNKNOWN = "unknown"
TRACKED_ISSUES = (
    "hrx-78tqc",
    "hrx-b1gxf3",
    "hrx-b8oh4d",
    "hrx-m2tne4",
    "hrx-vvf9c",
)

RUNNER_LOST_MESSAGE = (
    "The self-hosted runner lost communication with the server. Verify the "
    "machine is running and has a healthy network connection. Anything in your "
    "workflow that terminates the runner process, starves it for CPU/Memory, or "
    "blocks its network access can cause this error."
)


@dataclass(frozen=True)
class Finding:
    """One classified failure identity."""

    category: str
    code: str
    subject: str
    summary: str
    issue: str | None = None

    @property
    def identity(self) -> tuple[str, str, str, str | None]:
        return (self.category, self.code, self.subject, self.issue)


@dataclass(frozen=True)
class JobFinding:
    """A finding and the GitHub job that produced it."""

    job_id: int
    job_name: str
    job_url: str
    finding: Finding

    @property
    def identity(self) -> tuple[str, str, str, str, str | None]:
        """Returns an observation identity scoped to one stable CI lane."""
        return (self.job_name, *self.finding.identity)


@dataclass(frozen=True)
class FindingGroup:
    """Jobs that produced the same stable failure identity."""

    finding: Finding
    jobs: tuple[JobFinding, ...]


def _finding(
    category: str,
    code: str,
    subject: str,
    summary: str,
    *,
    issue: str | None = None,
) -> Finding:
    return Finding(category, code, subject, summary, issue)


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _step_names(job: Mapping[str, Any], conclusion: str) -> list[str]:
    return [
        str(step.get("name", ""))
        for step in job.get("steps", [])
        if step.get("conclusion") == conclusion
    ]


def _tracked_finding(
    *,
    open_issues: set[str] | frozenset[str],
    issue: str,
    tracked_category: str,
    code: str,
    subject: str,
    summary: str,
) -> Finding:
    if issue in open_issues:
        return _finding(
            tracked_category,
            code,
            subject,
            summary,
            issue=issue,
        )
    return _finding(
        REPOSITORY,
        code,
        subject,
        f"{summary}; owning issue {issue} is not confirmed open",
        issue=issue,
    )


def classify_failure(
    job: Mapping[str, Any],
    annotation_messages: Sequence[str],
    log: str,
    *,
    open_issues: set[str] | frozenset[str] = frozenset(),
) -> Finding:
    """Classifies one failed job without inferring beyond exact evidence."""

    failed_steps = _step_names(job, "failure")
    skipped_steps = _step_names(job, "skipped")
    messages = sorted(set(annotation_messages))

    if (
        failed_steps == ["Initialize containers"]
        and "Checkout repository" in skipped_steps
        and any(
            message.startswith("Executing the custom container implementation failed.")
            for message in messages
        )
    ):
        return _finding(
            INFRASTRUCTURE,
            "container_initialization",
            "self-hosted container provider",
            "custom container initialization failed before checkout",
        )

    if not job.get("steps") and messages == [RUNNER_LOST_MESSAGE]:
        started_at = job.get("started_at")
        completed_at = job.get("completed_at")
        if isinstance(started_at, str) and isinstance(completed_at, str):
            duration_seconds = (
                _parse_time(completed_at) - _parse_time(started_at)
            ).total_seconds()
            if 599 <= duration_seconds <= 601:
                return _finding(
                    INFRASTRUCTURE,
                    "runner_lost_before_steps",
                    "self-hosted runner provider",
                    "GitHub lost the runner before any step was recorded",
                )

    if (
        failed_steps == ["Install Bazel host tools"]
        and "https://github.com/bazelbuild/buildtools/releases/" in log
        and "socket.gaierror: [Errno -2] Name or service not known" in log
    ):
        return _finding(
            INFRASTRUCTURE,
            "github_asset_dns_failure",
            "runner network",
            "DNS resolution failed while downloading a pinned Bazel host tool",
        )

    if (
        failed_steps == ["Install Clang sanitizer runtimes"]
        and "https://ppa.launchpadcontent.net/git-core/ppa/" in log
        and log.count("503  Service Unavailable") >= 2
        and "E: Failed to fetch" in log
    ):
        return _finding(
            INFRASTRUCTURE,
            "launchpad_service_unavailable",
            "external package provider",
            "the pinned package source returned repeated HTTP 503 responses",
        )

    if (
        failed_steps == ["Fetch ROCm toolchain"]
        and "HTTPSConnectionPool(host='files.pythonhosted.org'" in log
        and "ReadTimeoutError" in log
    ):
        return _finding(
            INFRASTRUCTURE,
            "python_package_read_timeout",
            "external package provider",
            "the Python package host timed out while fetching the ROCm toolchain",
        )

    if (
        failed_steps == ["Run IREE Bazel CI"]
        and log.count("class java.io.IOException Connect timed out") >= 2
        and "https://github.com/bazelbuild/" in log
        and "Error in download_and_extract" in log
    ):
        return _finding(
            INFRASTRUCTURE,
            "bazel_dependency_connect_timeout",
            "runner network",
            "multiple pinned Bazel dependencies timed out at GitHub",
        )

    if (
        "Windows" in str(job.get("name", ""))
        and failed_steps == ["Run IREE Bazel CI"]
        and "UnrecoverableHttpException Unknown host:" in log
        and "Error in download_and_extract:" in log
    ):
        return _tracked_finding(
            open_issues=open_issues,
            issue="hrx-m2tne4",
            tracked_category=INFRASTRUCTURE,
            code="windows_bazel_dependency_dns_failure",
            subject="Windows Bazel dependency downloads",
            summary="the Windows runner could not resolve a Bazel dependency host",
        )

    if (
        "profile event has no numeric dur:" in log
        and "'ph': 'i'" in log
        and "Summarize Bazel profile" in log
    ):
        return _finding(
            REPOSITORY,
            "bazel_instant_profile_event",
            "Bazel profile summarizer",
            "CI rejected a valid instant Bazel profile event",
        )

    if (
        "hardware completion epoch" in log
        and "exceeds host publication epoch" in log
        and "after submission quiescence" in log
    ):
        return _tracked_finding(
            open_issues=open_issues,
            issue="hrx-78tqc",
            tracked_category=REPOSITORY,
            code="amdgpu_unowned_completion_epoch",
            subject="AMDGPU host queue completion ownership",
            summary="the AMDGPU queue observed an unowned completion epoch",
        )

    if (
        "ColdStateTransitionsPreserveWarmWaveformProgress/0" in log
        and "loom_serve_device_memory_statistics(device).retained.pending_bytes" in log
        and "Which is: 12288" in log
    ):
        return _tracked_finding(
            open_issues=open_issues,
            issue="hrx-b1gxf3",
            tracked_category=REPOSITORY,
            code="speech_cold_retained_memory_accounting",
            subject="ColdStateTransitionsPreserveWarmWaveformProgress/0",
            summary="cold speech retained-memory accounting observed zero pending bytes",
        )

    if (
        "Windows" in str(job.get("name", ""))
        and "An Application Control policy has blocked this file" in log
        and re.search(r"(?:error:\s*|\[WinError\s+)4551\b", log, re.IGNORECASE)
    ):
        return _tracked_finding(
            open_issues=open_issues,
            issue="hrx-vvf9c",
            tracked_category=INFRASTRUCTURE,
            code="windows_application_control_denial",
            subject="Windows Application Control",
            summary="Windows policy denied a generated executable before launch",
        )

    if (
        "Windows" in str(job.get("name", ""))
        and failed_steps == ["Run IREE Bazel CI"]
        and "error executing IreeGenerate command" in log
        and "return io.open(self, mode, buffering, encoding, errors, newline)" in log
        and re.search(
            r"OSError: \[Errno 9\] Bad file descriptor: "
            r"['\"]bazel-out[/\\]",
            log,
        )
    ):
        return _tracked_finding(
            open_issues=open_issues,
            issue="hrx-b8oh4d",
            tracked_category=REPOSITORY,
            code="windows_generator_output_open_ebadf",
            subject="Windows Bazel generator output creation",
            summary="a Windows Python generator could not open its declared output",
        )

    undefined_symbol_match = re.search(
        r"(?:lld-link|ld\.lld): error: undefined symbol: (\S+)", log
    )
    if undefined_symbol_match:
        symbol = undefined_symbol_match.group(1)
        return _finding(
            REPOSITORY,
            "link_undefined_symbol",
            symbol,
            f"the linked program is missing symbol {symbol}",
        )

    if (
        "CMake executable name 'loom-check' is ambiguous" in log
        and "loom_target_arch_amd_xdna_aie2p_loom-check" in log
        and "loom_tools_loom-check_loom-check" in log
    ):
        return _finding(
            REPOSITORY,
            "duplicate_loom_check_executable",
            "loom-check CMake target",
            "two CMake targets emitted the same loom-check executable name",
        )

    xpass_match = re.search(
        r'"record":"([^"]+)","diagnostic":"([^"]+)",'
        r'"outcome":"xpass","policy":"strict"',
        log,
    )
    if xpass_match:
        record, diagnostic = xpass_match.groups()
        return _finding(
            REPOSITORY,
            "strict_xpass",
            f"{record} ({diagnostic})",
            f"strict expected failure {record} unexpectedly passed",
        )

    loom_test_match = re.search(r"FAIL (loom/\S+\.loom-test) :: case (\d+)", log)
    if loom_test_match:
        test_path, case_number = loom_test_match.groups()
        subject = f"{test_path} case {case_number}"
        return _finding(
            REPOSITORY,
            "loom_golden_mismatch",
            subject,
            f"{subject} disagreed with its exact expected output",
        )

    bazel_test_match = re.search(r"FAIL: (//\S+)", log)
    if bazel_test_match:
        target = bazel_test_match.group(1)
        return _finding(
            REPOSITORY,
            "bazel_test_failure",
            target,
            f"repository-owned Bazel test {target} failed",
        )

    for line in log.splitlines():
        ctest_match = re.search(r"\s\d+ - ([^ ]+) \(Failed\)", line)
        if ctest_match:
            target = ctest_match.group(1)
            return _finding(
                REPOSITORY,
                "ctest_failure",
                target,
                f"repository-owned CTest {target} failed",
            )

    failed_step_subject = f"job {job.get('id', '?')}"
    if failed_steps:
        failed_step_subject += f" ({', '.join(failed_steps)})"
    return _finding(
        UNKNOWN,
        "unknown_failure",
        failed_step_subject,
        "no reviewed exact failure signature matched",
    )


def group_findings(findings: Iterable[JobFinding]) -> list[FindingGroup]:
    """Groups only findings whose complete stable identities agree."""

    groups: dict[tuple[str, str, str, str, str | None], list[JobFinding]] = defaultdict(
        list
    )
    for job_finding in findings:
        groups[job_finding.identity].append(job_finding)
    return [
        FindingGroup(group[0].finding, tuple(group))
        for _, group in sorted(
            groups.items(),
            key=lambda item: (
                category_precedence(item[0][1]),
                item[0][2],
                item[0][3],
                item[0][0],
            ),
        )
    ]


def category_precedence(category: str) -> int:
    return {
        REPOSITORY: 0,
        UNKNOWN: 1,
        INFRASTRUCTURE: 2,
    }.get(category, 1)


def failure_disposition(findings: Iterable[JobFinding]) -> str | None:
    """Returns the highest-priority failure category, if any."""

    categories = {job_finding.finding.category for job_finding in findings}
    if not categories:
        return None
    return min(categories, key=category_precedence)
