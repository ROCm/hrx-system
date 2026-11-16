# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import unittest
import urllib.parse
from datetime import UTC, datetime, timedelta

from build_tools.ci import github_actions_history as history


def timestamp(offset_seconds: int) -> str:
    value = datetime(2026, 11, 15, 12, 0, tzinfo=UTC) + timedelta(
        seconds=offset_seconds
    )
    return value.isoformat().replace("+00:00", "Z")


def job(
    name: str,
    *,
    created_offset: int,
    started_offset: int,
    completed_offset: int,
    conclusion: str = "success",
    step_seconds: int = 0,
) -> dict:
    return {
        "name": name,
        "conclusion": conclusion,
        "created_at": timestamp(created_offset),
        "started_at": timestamp(started_offset),
        "completed_at": timestamp(completed_offset),
        "steps": [
            {
                "name": "Build",
                "conclusion": conclusion,
                "started_at": timestamp(started_offset),
                "completed_at": timestamp(started_offset + step_seconds),
            }
        ],
    }


class FakeClient:
    repository = "ROCm/hrx-system"

    def __init__(self):
        self.json_path = ""
        self.page_paths = []

    def json(self, path):
        self.json_path = path
        return {"workflow_runs": [{"id": 30}, {"id": 20}]}

    def pages(self, path, field):
        self.page_paths.append((path, field))
        return [
            job(
                "Linux",
                created_offset=0,
                started_offset=10,
                completed_offset=40,
                step_seconds=40,
            )
        ]


class GitHubActionsHistoryTest(unittest.TestCase):
    def test_summarizes_reliability_wait_and_successful_work_separately(self):
        jobs = [
            job(
                "Linux",
                created_offset=0,
                started_offset=10,
                completed_offset=110,
                step_seconds=80,
            ),
            job(
                "Linux",
                created_offset=0,
                started_offset=30,
                completed_offset=150,
                step_seconds=90,
            ),
            job(
                "Linux",
                created_offset=0,
                started_offset=20,
                completed_offset=80,
                conclusion="failure",
            ),
            job(
                "Skipped",
                created_offset=0,
                started_offset=0,
                completed_offset=0,
                conclusion="skipped",
            ),
        ]

        summary = history.summarize(
            repository="ROCm/hrx-system",
            workflow="ci_iree.yml",
            event="push",
            run_ids=[30, 20, 10],
            jobs=jobs,
        )

        self.assertEqual(summary.run_ids, (30, 20, 10))
        self.assertEqual(len(summary.lanes), 1)
        lane = summary.lanes[0]
        self.assertEqual(
            (lane.name, lane.sample_count, lane.success_count), ("Linux", 3, 2)
        )
        self.assertEqual(lane.queue, history.Distribution(3, 20.0, 30.0, 30.0))
        self.assertEqual(
            lane.successful_duration,
            history.Distribution(2, 110.0, 120.0, 120.0),
        )
        self.assertEqual(len(summary.steps), 1)
        self.assertEqual(
            summary.steps[0].duration,
            history.Distribution(2, 85.0, 90.0, 90.0),
        )

    def test_rejects_incomplete_and_negative_timestamp_samples(self):
        incomplete = job(
            "Incomplete",
            created_offset=0,
            started_offset=10,
            completed_offset=20,
        )
        incomplete["completed_at"] = None
        negative = job(
            "Negative",
            created_offset=30,
            started_offset=10,
            completed_offset=20,
        )

        summary = history.summarize(
            repository="ROCm/hrx-system",
            workflow="ci_iree.yml",
            event="push",
            run_ids=[30],
            jobs=[incomplete, negative],
        )

        self.assertEqual(summary.lanes, ())
        self.assertEqual(summary.steps, ())

    def test_collect_bounds_queries_and_requests_latest_job_attempts(self):
        client = FakeClient()

        summary = history.collect(
            client, workflow="ci_iree.yml", event="push", run_count=2
        )

        parsed_query = dict(urllib.parse.parse_qsl(client.json_path.split("?", 1)[1]))
        self.assertEqual(
            parsed_query,
            {"event": "push", "status": "completed", "per_page": "2"},
        )
        self.assertEqual(
            client.page_paths,
            [
                ("/actions/runs/30/jobs?filter=latest", "jobs"),
                ("/actions/runs/20/jobs?filter=latest", "jobs"),
            ],
        )
        self.assertEqual(summary.run_ids, (30, 20))
        with self.assertRaises(ValueError):
            history.collect(client, workflow="ci_iree.yml", event="push", run_count=0)

    def test_render_and_json_preserve_evidence(self):
        summary = history.summarize(
            repository="ROCm/hrx-system",
            workflow="ci_iree.yml",
            event="push",
            run_ids=[30],
            jobs=[
                job(
                    "Linux",
                    created_offset=0,
                    started_offset=10,
                    completed_offset=100,
                    step_seconds=40,
                )
            ],
        )

        rendered = history.render(summary, top_count=10)
        serialized = json.loads(json.dumps(history.dataclasses.asdict(summary)))

        self.assertIn("1/1", rendered)
        self.assertIn("Linux — Build", rendered)
        self.assertIn("1m30s", rendered)
        self.assertEqual(serialized["schema"], history.SCHEMA)
        self.assertEqual(serialized["lanes"][0]["success_count"], 1)


if __name__ == "__main__":
    unittest.main()
