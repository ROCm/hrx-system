# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import dataclasses
import datetime as dt
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from build_tools.ci import github_actions_triage as triage
from build_tools.ci import github_actions_watch as watch


class FakeClient:
    def __init__(self):
        self.json_results = {}
        self.page_results = {}
        self.byte_results = {}
        self.byte_requests = []

    def json(self, path):
        return self.json_results[path]

    def pages(self, path, field):
        del field
        return self.page_results[path]

    def bytes(self, path):
        self.byte_requests.append(path)
        result = self.byte_results[path]
        if isinstance(result, Exception):
            raise result
        return result


def run(run_id=20, *, status="completed", conclusion="success"):
    return {
        "id": run_id,
        "event": "pull_request",
        "head_sha": "a" * 40,
        "path": ".github/workflows/ci_iree.yml",
        "created_at": f"2026-10-21T12:{run_id:02d}:00Z",
        "status": status,
        "conclusion": conclusion,
        "html_url": f"https://github.example/runs/{run_id}",
    }


def completed_job(job_id, *, name="Linux", conclusion="success"):
    return {
        "id": job_id,
        "name": name,
        "html_url": f"https://github.example/jobs/{job_id}",
        "check_run_url": (
            f"https://api.github.com/repos/ROCm/hrx-system/check-runs/{job_id}"
        ),
        "created_at": "2026-10-21T12:00:00Z",
        "started_at": "2026-10-21T12:01:00Z",
        "completed_at": "2026-10-21T12:02:00Z",
        "status": "completed",
        "conclusion": conclusion,
        "runner_id": 1,
        "steps": [
            {
                "name": "Run IREE Bazel CI",
                "status": "completed",
                "conclusion": conclusion,
                "started_at": "2026-10-21T12:01:00Z",
                "completed_at": "2026-10-21T12:02:00Z",
            }
        ],
    }


class GitHubActionsWatchTest(unittest.TestCase):
    def test_target_accepts_shorthand_and_urls(self):
        self.assertEqual(watch.parse_target("2198"), watch.Target("pr", 2198))
        self.assertEqual(
            watch.parse_target("run:39665280602"),
            watch.Target("run", 39665280602),
        )
        self.assertEqual(
            watch.parse_target("https://github.com/ROCm/hrx-system/pull/2198"),
            watch.Target("pr", 2198, "ROCm/hrx-system"),
        )
        self.assertEqual(
            watch.parse_target(
                "https://github.com/ROCm/hrx-system/actions/runs/39665280602"
            ),
            watch.Target("run", 39665280602, "ROCm/hrx-system"),
        )
        with self.assertRaises(watch.WatchError):
            watch.parse_target("maybe")

    def test_help_describes_agent_wait_contract(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as exit_context:
            watch.parse_arguments(("--help",))

        normalized_output = " ".join(output.getvalue().split())
        self.assertEqual(exit_context.exception.code, 0)
        self.assertIn("retain the session ID", normalized_output)
        self.assertIn("newly failed job", normalized_output)
        self.assertIn("130 interrupted by the operator", normalized_output)

    def test_repository_client_pages_without_leaking_scope(self):
        requested_urls = []

        def get_json(url, _token):
            requested_urls.append(url)
            page = int(dict(watch.urllib.parse.parse_qsl(url.split("?", 1)[1]))["page"])
            return {"jobs": list(range(100 if page == 1 else 1))}

        client = watch.GitHubClient(
            "ROCm/hrx-system",
            "secret",
            get_json=get_json,
        )
        results = client.pages("/actions/runs/1/jobs?filter=latest", "jobs")
        self.assertEqual(len(results), 101)
        self.assertIn("filter=latest", requested_urls[0])
        with self.assertRaises(watch.WatchError):
            client.json("https://api.github.com/repos/other/project/actions/runs/1")

    def test_pull_request_selects_latest_exact_head_run(self):
        client = FakeClient()
        client.json_results["/pulls/9"] = {
            "state": "open",
            "head": {"sha": "a" * 40},
        }
        query = watch.urllib.parse.urlencode(
            {"event": "pull_request", "head_sha": "a" * 40}
        )
        path = f"/actions/workflows/{watch.WORKFLOW_FILE}/runs?{query}"
        wrong_head = run(30)
        wrong_head["head_sha"] = "b" * 40
        client.page_results[path] = [run(10), wrong_head, run(20)]

        selection = watch.select_run(client, watch.Target("pr", 9))

        self.assertEqual(selection.run["id"], 20)
        self.assertFalse(selection.obsolete)

    def test_closed_pull_without_run_is_proven_obsolete(self):
        client = FakeClient()
        client.json_results["/pulls/9"] = {
            "state": "closed",
            "head": {"sha": "a" * 40},
        }
        query = watch.urllib.parse.urlencode(
            {"event": "pull_request", "head_sha": "a" * 40}
        )
        client.page_results[
            f"/actions/workflows/{watch.WORKFLOW_FILE}/runs?{query}"
        ] = []

        selection = watch.select_run(client, watch.Target("pr", 9))

        self.assertTrue(selection.obsolete)
        self.assertIsNone(selection.run)

    def test_structural_infrastructure_failure_does_not_download_log(self):
        client = FakeClient()
        input_job = completed_job(7, conclusion="failure")
        input_job["steps"] = [
            {
                "name": "Initialize containers",
                "status": "completed",
                "conclusion": "failure",
            },
            {
                "name": "Checkout repository",
                "status": "completed",
                "conclusion": "skipped",
            },
        ]
        annotations_path = input_job["check_run_url"] + "/annotations"
        client.page_results[annotations_path] = [
            {
                "message": (
                    "Executing the custom container implementation failed. provider"
                )
            }
        ]
        loader = watch.EvidenceLoader(client, frozenset())

        result, error = loader.load(input_job)

        self.assertIsNone(error)
        self.assertEqual(result.finding.category, triage.INFRASTRUCTURE)
        self.assertEqual(client.byte_requests, [])

    def test_log_failures_group_while_live_jobs_remain_distinct(self):
        client = FakeClient()
        workflow_run = run(20, status="queued", conclusion=None)
        selection = watch.Selection("PR #9", workflow_run, False)
        first_failure = completed_job(1, name="Linux", conclusion="failure")
        second_failure = completed_job(2, name="Windows", conclusion="failure")
        queued = {
            "id": 3,
            "name": "NPU4",
            "html_url": "https://github.example/jobs/3",
            "created_at": "2026-10-21T12:00:00Z",
            "started_at": "2026-10-21T12:00:00Z",
            "completed_at": None,
            "status": "queued",
            "conclusion": None,
            "runner_id": None,
            "steps": [],
        }
        running = {
            "id": 4,
            "name": "Windows Clang CL",
            "html_url": "https://github.example/jobs/4",
            "created_at": "2026-10-21T12:10:00Z",
            "started_at": "2026-10-21T12:10:00Z",
            "completed_at": None,
            "status": "in_progress",
            "conclusion": None,
            "runner_id": 4,
            "steps": [
                {
                    "name": "Run IREE Bazel CI",
                    "status": "in_progress",
                    "conclusion": None,
                    "started_at": "2026-10-21T12:20:00Z",
                    "completed_at": None,
                }
            ],
        }
        jobs_path = "/actions/runs/20/jobs?filter=latest"
        client.page_results[jobs_path] = [
            first_failure,
            second_failure,
            queued,
            running,
        ]
        for input_job in (first_failure, second_failure):
            client.page_results[input_job["check_run_url"] + "/annotations"] = []
            client.byte_results[f"/actions/jobs/{input_job['id']}/logs"] = (
                b"FAIL loom/example.loom-test :: case 14"
            )
        loader = watch.EvidenceLoader(client, frozenset())

        snapshot = watch.collect_snapshot(
            client,
            selection,
            loader,
            now=dt.datetime(2026, 10, 21, 13, 0, tzinfo=dt.UTC),
        )

        self.assertEqual(snapshot.disposition, triage.REPOSITORY)
        self.assertFalse(snapshot.terminal)
        self.assertEqual(len(snapshot.finding_groups), 2)
        self.assertEqual([len(group.jobs) for group in snapshot.finding_groups], [1, 1])
        self.assertEqual(
            [(job.state, job.detail) for job in snapshot.live_jobs],
            [
                ("waiting", "no runner assigned"),
                ("running", "Run IREE Bazel CI"),
            ],
        )
        self.assertEqual(snapshot.live_jobs[0].age_seconds, 3600)
        self.assertEqual(snapshot.live_jobs[1].age_seconds, 2400)
        rendered = watch.render_snapshot(snapshot)
        self.assertIn("WAITING", rendered)
        self.assertIn("FAIL", rendered)
        self.assertEqual(watch.snapshot_exit_code(snapshot), watch.EXIT_INCOMPLETE)

    def test_evidence_fetch_error_stays_unknown_and_is_retried(self):
        client = FakeClient()
        input_job = completed_job(7, conclusion="failure")
        annotations_path = input_job["check_run_url"] + "/annotations"
        client.page_results[annotations_path] = []
        client.byte_results["/actions/jobs/7/logs"] = OSError("log unavailable")
        loader = watch.EvidenceLoader(client, frozenset())

        first, first_error = loader.load(input_job)
        second, second_error = loader.load(input_job)

        self.assertEqual(first.finding.category, triage.UNKNOWN)
        self.assertEqual(second.finding.category, triage.UNKNOWN)
        self.assertEqual(first_error, "log unavailable")
        self.assertEqual(second_error, "log unavailable")
        self.assertEqual(client.byte_requests, ["/actions/jobs/7/logs"] * 2)

    def test_terminal_disposition_exit_codes(self):
        base = watch.Snapshot(
            "run 1",
            1,
            "https://example/run",
            "completed",
            "success",
            False,
            "clean",
            True,
            1,
            1,
            0,
            (),
            (),
            (),
        )
        self.assertEqual(watch.snapshot_exit_code(base), watch.EXIT_CLEAN)
        for disposition in (triage.REPOSITORY, triage.UNKNOWN):
            with self.subTest(disposition=disposition):
                self.assertEqual(
                    watch.snapshot_exit_code(
                        dataclasses.replace(base, disposition=disposition)
                    ),
                    watch.EXIT_FAILURE,
                )
        self.assertEqual(
            watch.snapshot_exit_code(
                dataclasses.replace(base, disposition=triage.INFRASTRUCTURE)
            ),
            watch.EXIT_CLASSIFIED_NONREPOSITORY,
        )

    def test_json_preserves_every_grouped_job_link(self):
        finding = triage.Finding(
            triage.REPOSITORY,
            "bazel_test_failure",
            "//example:test",
            "test failed",
        )
        jobs = tuple(
            triage.JobFinding(
                index, f"job {index}", f"https://example/{index}", finding
            )
            for index in range(3)
        )
        snapshot = watch.Snapshot(
            "run 1",
            1,
            "https://example/run",
            "completed",
            "failure",
            False,
            triage.REPOSITORY,
            True,
            3,
            3,
            3,
            (),
            (triage.FindingGroup(finding, jobs),),
            (),
        )

        payload = watch.snapshot_json(snapshot)

        self.assertEqual(
            [job["url"] for job in payload["findings"][0]["jobs"]],
            [f"https://example/{index}" for index in range(3)],
        )
        json.dumps(payload)

    def test_human_output_limits_live_detail_while_json_preserves_it(self):
        live_jobs = tuple(
            watch.LiveJob(
                "waiting",
                f"job {index}",
                f"https://example/{index}",
                index,
                "no runner assigned",
            )
            for index in range(6)
        )
        snapshot = watch.Snapshot(
            "run 1",
            1,
            "https://example/run",
            "queued",
            None,
            False,
            "in_progress",
            False,
            6,
            0,
            0,
            live_jobs,
            (),
            (),
        )

        rendered = watch.render_snapshot(snapshot)

        self.assertIn("6 active (6 waiting)", rendered)
        self.assertIn("2 more active jobs", rendered)
        self.assertNotIn("job 0 ", rendered)
        self.assertNotIn("job 1 ", rendered)
        self.assertEqual(len(watch.snapshot_json(snapshot)["live_jobs"]), 6)

    def test_watch_polls_until_terminal_and_reports_state_changes(self):
        active = watch.Snapshot(
            "PR #9",
            20,
            "https://example/run",
            "in_progress",
            None,
            False,
            "in_progress",
            False,
            2,
            1,
            0,
            (),
            (),
            (),
        )
        completed = dataclasses.replace(
            active,
            run_status="completed",
            run_conclusion="success",
            disposition="clean",
            terminal=True,
            completed_jobs=2,
        )
        output = io.StringIO()
        sleeps = []
        monotonic_values = iter((0.0, 60.0, 120.0))
        selection = watch.Selection("PR #9", run(), False)
        with (
            mock.patch.object(watch, "open_tracked_issues", return_value=frozenset()),
            mock.patch.object(watch, "select_run", return_value=selection),
            mock.patch.object(
                watch,
                "collect_snapshot",
                side_effect=(active, active, completed),
            ),
            redirect_stdout(output),
        ):
            exit_code = watch.observe(
                FakeClient(),
                watch.Target("pr", 9),
                watch=True,
                json_output=False,
                sleep=sleeps.append,
                monotonic=lambda: next(monotonic_values),
            )

        self.assertEqual(exit_code, watch.EXIT_CLEAN)
        self.assertEqual(sleeps, [watch.POLL_SECONDS, watch.POLL_SECONDS])
        self.assertEqual(output.getvalue().count("Loom PR #9"), 2)

    def test_interrupt_exits_without_a_traceback(self):
        error_output = io.StringIO()
        with (
            mock.patch.object(watch, "resolve_token", return_value="secret"),
            mock.patch.object(watch, "observe", side_effect=KeyboardInterrupt),
            redirect_stderr(error_output),
        ):
            exit_code = watch.main(("9", "--watch"))

        self.assertEqual(exit_code, watch.EXIT_INTERRUPTED)
        self.assertEqual(error_output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
