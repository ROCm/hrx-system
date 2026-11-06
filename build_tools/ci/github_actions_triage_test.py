# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import unittest

from build_tools.ci import github_actions_triage as triage


def job(
    *,
    job_id: int = 7,
    job_name: str = "Bazel / Linux x86-64 / Clang / ASAN",
    failed_step: str | None = "Run IREE Bazel CI",
    skipped_steps: tuple[str, ...] = (),
    steps: bool = True,
) -> dict:
    result = {
        "id": job_id,
        "name": job_name,
        "html_url": f"https://github.example/jobs/{job_id}",
        "started_at": "2026-10-21T12:00:00Z",
        "completed_at": "2026-10-21T12:10:00Z",
        "steps": [],
    }
    if steps:
        if failed_step:
            result["steps"].append({"name": failed_step, "conclusion": "failure"})
        result["steps"].extend(
            {"name": name, "conclusion": "skipped"} for name in skipped_steps
        )
    return result


class GitHubActionsTriageTest(unittest.TestCase):
    def classify(
        self,
        log: str,
        *,
        input_job: dict | None = None,
        annotations: tuple[str, ...] = (),
        open_issues: frozenset[str] = frozenset(),
    ) -> triage.Finding:
        return triage.classify_failure(
            input_job or job(),
            annotations,
            log,
            open_issues=open_issues,
        )

    def test_generated_container_failure_requires_precheckout_structure(self):
        input_job = job(
            failed_step="Initialize containers",
            skipped_steps=("Checkout repository",),
        )
        annotation = "Executing the custom container implementation failed. provider"
        finding = self.classify(
            "",
            input_job=input_job,
            annotations=(annotation,),
        )
        self.assertEqual(finding.category, triage.INFRASTRUCTURE)
        self.assertEqual(finding.code, "container_initialization")

        input_job["steps"][1]["conclusion"] = "success"
        self.assertEqual(
            self.classify("", input_job=input_job, annotations=(annotation,)).category,
            triage.UNKNOWN,
        )

    def test_runner_loss_requires_exact_annotation_and_duration(self):
        input_job = job(steps=False)
        finding = self.classify(
            "",
            input_job=input_job,
            annotations=(triage.RUNNER_LOST_MESSAGE,),
        )
        self.assertEqual(finding.code, "runner_lost_before_steps")
        input_job["completed_at"] = "2026-10-21T12:11:00Z"
        self.assertEqual(
            self.classify(
                "",
                input_job=input_job,
                annotations=(triage.RUNNER_LOST_MESSAGE,),
            ).category,
            triage.UNKNOWN,
        )

    def test_external_download_failures_require_named_boundary_and_step(self):
        cases = (
            (
                "Install Bazel host tools",
                "https://github.com/bazelbuild/buildtools/releases/x "
                "socket.gaierror: [Errno -2] Name or service not known",
                "github_asset_dns_failure",
            ),
            (
                "Install Clang sanitizer runtimes",
                "https://ppa.launchpadcontent.net/git-core/ppa/ "
                "503  Service Unavailable 503  Service Unavailable "
                "E: Failed to fetch",
                "launchpad_service_unavailable",
            ),
            (
                "Fetch ROCm toolchain",
                "HTTPSConnectionPool(host='files.pythonhosted.org' ReadTimeoutError",
                "python_package_read_timeout",
            ),
            (
                "Run IREE Bazel CI",
                "class java.io.IOException Connect timed out "
                "class java.io.IOException Connect timed out "
                "https://github.com/bazelbuild/rules_cc Error in download_and_extract",
                "bazel_dependency_connect_timeout",
            ),
        )
        for failed_step, log, code in cases:
            with self.subTest(code=code):
                finding = self.classify(log, input_job=job(failed_step=failed_step))
                self.assertEqual(finding.category, triage.INFRASTRUCTURE)
                self.assertEqual(finding.code, code)

    def test_repository_failure_identities_are_stable_and_specific(self):
        cases = (
            (
                "FAIL loom/example.loom-test :: case 14",
                "loom_golden_mismatch",
                "loom/example.loom-test case 14",
            ),
            (
                "FAIL: //runtime/example:test (see log)",
                "bazel_test_failure",
                "//runtime/example:test",
            ),
            (
                "  19 - runtime/example_test (Failed)",
                "ctest_failure",
                "runtime/example_test",
            ),
            (
                '{"record":"@wide_values","diagnostic":"BACKEND/005",'
                '"outcome":"xpass","policy":"strict"}',
                "strict_xpass",
                "@wide_values (BACKEND/005)",
            ),
            (
                "lld-link: error: undefined symbol: loom_cfg_forward_empty_blocks",
                "link_undefined_symbol",
                "loom_cfg_forward_empty_blocks",
            ),
        )
        for log, code, subject in cases:
            with self.subTest(code=code):
                finding = self.classify(log)
                self.assertEqual(finding.category, triage.REPOSITORY)
                self.assertEqual(finding.code, code)
                self.assertEqual(finding.subject, subject)

    def test_identified_regressions_remain_repository_failures(self):
        logs = (
            "profile event has no numeric dur: {'ph': 'i'} Summarize Bazel profile",
            "CMake executable name 'loom-check' is ambiguous "
            "loom_target_arch_amd_xdna_aie2p_loom-check "
            "loom_tools_loom-check_loom-check",
        )
        for log in logs:
            with self.subTest(log=log):
                self.assertEqual(self.classify(log).category, triage.REPOSITORY)

    def test_tracked_patterns_require_an_open_owning_issue(self):
        cases = (
            (
                "hardware completion epoch 2 exceeds host publication epoch 1 "
                "after submission quiescence",
                "hrx-78tqc",
                triage.KNOWN_PRODUCT,
            ),
            (
                "ColdStateTransitionsPreserveWarmWaveformProgress/0 "
                "loom_serve_device_memory_statistics(device).retained.pending_bytes "
                "Which is: 12288",
                "hrx-b1gxf3",
                triage.KNOWN_FLAKE,
            ),
        )
        for log, issue, category in cases:
            with self.subTest(issue=issue):
                unconfirmed = self.classify(log)
                self.assertEqual(unconfirmed.category, triage.REPOSITORY)
                self.assertEqual(unconfirmed.issue, issue)
                confirmed = self.classify(log, open_issues=frozenset({issue}))
                self.assertEqual(confirmed.category, category)
                self.assertEqual(confirmed.issue, issue)

    def test_windows_application_control_precedes_reported_test_failure(self):
        log = (
            "FAIL: //loom/example:test (see log)\n"
            "CreateProcessW failed (error: 4551)\n"
            "An Application Control policy has blocked this file."
        )
        input_job = job(job_name="Bazel / Windows x86-64 / Clang CL / Fastbuild")
        unconfirmed = self.classify(log, input_job=input_job)
        self.assertEqual(unconfirmed.category, triage.REPOSITORY)
        self.assertEqual(unconfirmed.issue, "hrx-vvf9c")

        confirmed = self.classify(
            log,
            input_job=input_job,
            open_issues=frozenset({"hrx-vvf9c"}),
        )
        self.assertEqual(confirmed.category, triage.INFRASTRUCTURE)
        self.assertEqual(confirmed.code, "windows_application_control_denial")
        self.assertEqual(confirmed.issue, "hrx-vvf9c")

    def test_windows_bazel_dns_failure_requires_owning_issue_and_boundary(self):
        log = (
            "Download failed: UnrecoverableHttpException Unknown host: github.com\n"
            "Error in download_and_extract: java.io.IOException: Unknown host: "
            "github.com"
        )
        input_job = job(
            job_name="Bazel / Windows x86-64 / Clang CL / Fastbuild",
            failed_step="Run IREE Bazel CI",
        )
        unconfirmed = self.classify(log, input_job=input_job)
        self.assertEqual(unconfirmed.category, triage.REPOSITORY)
        self.assertEqual(unconfirmed.issue, "hrx-m2tne4")

        confirmed = self.classify(
            log,
            input_job=input_job,
            open_issues=frozenset({"hrx-m2tne4"}),
        )
        self.assertEqual(confirmed.category, triage.INFRASTRUCTURE)
        self.assertEqual(confirmed.code, "windows_bazel_dependency_dns_failure")
        self.assertEqual(confirmed.issue, "hrx-m2tne4")

        linux_job = job(job_name="Bazel / Linux x86-64 / Clang / Fastbuild")
        self.assertEqual(
            self.classify(
                log,
                input_job=linux_job,
                open_issues=frozenset({"hrx-m2tne4"}),
            ).category,
            triage.UNKNOWN,
        )

    def test_unknown_failures_do_not_share_an_identity_across_jobs(self):
        first = self.classify("opaque", input_job=job(job_id=10))
        second = self.classify("opaque", input_job=job(job_id=11))
        self.assertNotEqual(first.identity, second.identity)

    def test_repository_step_timeout_is_not_infrastructure(self):
        finding = self.classify(
            "The job running on runner X has exceeded the maximum execution time."
        )
        self.assertEqual(finding.category, triage.UNKNOWN)

    def test_grouping_and_precedence_preserve_repository_failure(self):
        shared = triage.Finding(
            triage.REPOSITORY,
            "loom_golden_mismatch",
            "loom/example.loom-test case 2",
            "golden mismatch",
        )
        infrastructure = triage.Finding(
            triage.INFRASTRUCTURE,
            "container_initialization",
            "container provider",
            "container failed",
        )
        findings = [
            triage.JobFinding(1, "Linux", "https://example/1", shared),
            triage.JobFinding(2, "Windows", "https://example/2", shared),
            triage.JobFinding(3, "GPU", "https://example/3", infrastructure),
        ]
        groups = triage.group_findings(findings)
        self.assertEqual([len(group.jobs) for group in groups], [2, 1])
        self.assertEqual(
            triage.failure_disposition(findings),
            triage.REPOSITORY,
        )


if __name__ == "__main__":
    unittest.main()
