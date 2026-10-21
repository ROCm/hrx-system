# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from build_tools.devtools import bazel_profile_summary


class BazelProfileSummaryTest(unittest.TestCase):
    def write_profile(self, path: Path, events: list[dict]) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(
                {
                    "otherData": {
                        "bazel_version": "release test",
                        "build_id": "build-id",
                        "date": "2026-10-20T00:00:00Z",
                    },
                    "traceEvents": events,
                },
                stream,
            )

    def test_separates_test_process_action_and_critical_path_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "test-example.profile.gz"
            self.write_profile(
                profile_path,
                [
                    {
                        "name": "thread_name",
                        "ph": "M",
                        "pid": 1,
                        "tid": 7,
                        "args": {"name": "worker"},
                    },
                    {
                        "cat": "build phase marker",
                        "name": "Launch Blaze",
                        "ph": "X",
                        "ts": -1_000_000,
                        "dur": 1_000_000,
                    },
                    {
                        "cat": "action processing",
                        "name": "Testing //example:slow_test",
                        "ph": "X",
                        "ts": 1_000_000,
                        "dur": 12_500_000,
                        "tid": 7,
                        "args": {"mnemonic": "TestRunner"},
                    },
                    {
                        "cat": "Unknown event",
                        "name": "subprocess.run",
                        "ph": "X",
                        "ts": 2_000_000,
                        "dur": 10_250_000,
                        "tid": 7,
                    },
                    {
                        "cat": "remote action execution",
                        "name": "fabric result wait",
                        "ph": "X",
                        "ts": 12_300_000,
                        "dur": 500_000,
                        "tid": 7,
                    },
                    {
                        "cat": "critical path component",
                        "name": "action 'Testing //example:slow_test'",
                        "ph": "X",
                        "ts": 1_000_000,
                        "dur": 12_500_000,
                    },
                    {
                        "cat": "action processing",
                        "name": "",
                        "ph": "X",
                        "ts": 14_000_000,
                        "dur": 15_000_000,
                        "tid": 8,
                        "args": {"mnemonic": "IreeGenerate"},
                    },
                    {
                        "cat": "critical path component",
                        "name": "action 'Generating example'",
                        "ph": "X",
                        "ts": 14_000_000,
                        "dur": 15_000_000,
                    },
                    {
                        "cat": "action processing",
                        "name": "Compiling example.cc",
                        "ph": "X",
                        "ts": 29_000_000,
                        "dur": 2_000_000,
                        "tid": 9,
                        "args": {"mnemonic": "CppCompile"},
                    },
                ],
            )

            summary = bazel_profile_summary.summarize(profile_path)

            self.assertEqual(summary.schema, bazel_profile_summary.SCHEMA)
            self.assertEqual(summary.profile, profile_path.name)
            self.assertEqual(summary.elapsed_seconds, 32.0)
            self.assertEqual(summary.action_count, 3)
            self.assertEqual(summary.action_seconds, 29.5)
            self.assertEqual(summary.test_action_count, 1)
            self.assertEqual(summary.test_action_seconds, 12.5)
            self.assertEqual(summary.test_execution_seconds, 10.25)
            self.assertEqual(summary.test_process_count, 1)
            self.assertEqual(summary.test_process_seconds, 10.25)
            self.assertEqual(summary.test_remote_result_count, 0)
            self.assertEqual(summary.test_action_fallback_count, 0)
            self.assertEqual(summary.test_admission_count, 0)
            self.assertEqual(summary.slow_test_count, 1)
            self.assertEqual(summary.slow_test_action_count, 1)
            self.assertEqual(
                summary.slow_tests[0],
                bazel_profile_summary.TestTiming(
                    label="//example:slow_test",
                    action_seconds=12.5,
                    execution_seconds=10.25,
                    execution_source=bazel_profile_summary.EXECUTION_SOURCE_PROCESS,
                    admission_seconds=None,
                    critical_path=True,
                ),
            )
            self.assertEqual(summary.slow_non_test_action_count, 1)
            self.assertEqual(
                summary.slow_non_test_actions[0].name,
                "<unnamed IreeGenerate action>",
            )
            self.assertEqual(
                [aggregate.mnemonic for aggregate in summary.action_aggregates],
                ["IreeGenerate", "TestRunner", "CppCompile"],
            )
            self.assertEqual(len(summary.critical_path), 2)

            markdown = bazel_profile_summary.render_markdown(summary, "Test example")
            self.assertIn("Observed test execution | 10.250 s", markdown)
            self.assertIn("Process observations | 1 / 10.250 s", markdown)
            self.assertIn("`//example:slow_test` **critical path**", markdown)
            self.assertIn("&lt;unnamed IreeGenerate action&gt;", markdown)
            console = bazel_profile_summary.render_console(summary, "Test example")
            self.assertIn("10.250s process, 12.500s action", console)

    def test_retains_instant_critical_path_components(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "instant.profile.gz"
            self.write_profile(
                profile_path,
                [
                    {
                        "cat": "critical path component",
                        "name": "action 'Writing script header-cmd.bat'",
                        "ph": "i",
                        "ts": 415978,
                        "pid": 1,
                        "args": {"tid": 70},
                        "tid": 0,
                    },
                    {
                        "cat": "critical path component",
                        "name": "action 'Testing //example:queue_test'",
                        "ph": "X",
                        "ts": 416000,
                        "dur": 100000,
                    },
                ],
            )
            summary = bazel_profile_summary.summarize(profile_path)
            self.assertEqual(
                summary.critical_path,
                (
                    bazel_profile_summary.ActionTiming(
                        name="action 'Writing script header-cmd.bat'",
                        duration_seconds=0.0,
                        mnemonic=None,
                    ),
                    bazel_profile_summary.ActionTiming(
                        name="action 'Testing //example:queue_test'",
                        duration_seconds=0.1,
                        mnemonic=None,
                    ),
                ),
            )
            self.assertIn(
                "Writing script header-cmd.bat",
                bazel_profile_summary.render_markdown(summary, "Native queues"),
            )

    def test_complete_critical_path_components_require_a_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "malformed.profile.gz"
            self.write_profile(
                profile_path,
                [{"cat": "critical path component", "ph": "X", "ts": 0}],
            )
            with self.assertRaisesRegex(
                bazel_profile_summary.BazelProfileSummaryError, "no numeric dur"
            ):
                bazel_profile_summary.summarize(profile_path)

    def test_uses_remote_result_instead_of_resource_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "test-example.profile.gz"
            self.write_profile(
                profile_path,
                [
                    {
                        "cat": "action processing",
                        "name": "Testing //example:queued_test",
                        "ph": "X",
                        "ts": 1_000_000,
                        "dur": 12_500_000,
                        "tid": 1,
                        "args": {"mnemonic": "TestRunner"},
                    },
                    {
                        "cat": "remote action execution",
                        "name": "fabric request and admission",
                        "ph": "X",
                        "ts": 1_100_000,
                        "dur": 10_000_000,
                        "tid": 1,
                    },
                    {
                        "cat": "remote action execution",
                        "name": "fabric result wait",
                        "ph": "X",
                        "ts": 11_200_000,
                        "dur": 1_800_000,
                        "tid": 1,
                    },
                    {
                        "cat": "action processing",
                        "name": "Generating queued data",
                        "ph": "X",
                        "ts": 14_000_000,
                        "dur": 15_000_000,
                        "tid": 2,
                        "args": {"mnemonic": "IreeGenerate"},
                    },
                    {
                        "cat": "remote action execution",
                        "name": "fabric request and admission",
                        "ph": "X",
                        "ts": 14_100_000,
                        "dur": 13_000_000,
                        "tid": 2,
                    },
                    {
                        "cat": "remote action execution",
                        "name": "fabric result wait",
                        "ph": "X",
                        "ts": 27_200_000,
                        "dur": 1_000_000,
                        "tid": 2,
                    },
                ],
            )

            summary = bazel_profile_summary.summarize(profile_path)

            self.assertEqual(summary.test_action_seconds, 12.5)
            self.assertEqual(summary.test_execution_seconds, 1.8)
            self.assertEqual(summary.test_process_count, 0)
            self.assertEqual(summary.test_remote_result_count, 1)
            self.assertEqual(summary.test_remote_result_seconds, 1.8)
            self.assertEqual(summary.test_action_fallback_count, 0)
            self.assertEqual(summary.test_admission_count, 1)
            self.assertEqual(summary.test_admission_seconds, 10.0)
            self.assertEqual(summary.slow_test_count, 0)
            self.assertEqual(summary.slow_test_action_count, 1)
            self.assertEqual(summary.slow_non_test_action_count, 0)
            self.assertEqual(summary.slow_non_test_raw_action_count, 1)

            markdown = bazel_profile_summary.render_markdown(summary, "Queued")
            self.assertIn("Remote-result observations | 1 / 1.800 s", markdown)
            self.assertIn("Test resource admission | 1 / 10.000 s", markdown)
            self.assertIn("No observed test execution reached 10 s", markdown)
            console = bazel_profile_summary.render_console(summary, "Queued")
            self.assertIn("0 test execution suspect(s)", console)
            self.assertIn("1 slow test action(s)", console)

    def test_falls_back_to_action_when_execution_timing_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "test-example.profile.gz"
            self.write_profile(
                profile_path,
                [
                    {
                        "cat": "action processing",
                        "name": "Testing //example:queued_test",
                        "ph": "X",
                        "ts": 0,
                        "dur": 11_000_000,
                        "tid": 1,
                        "args": {"mnemonic": "TestRunner"},
                    }
                ],
            )

            summary = bazel_profile_summary.summarize(profile_path)

            self.assertEqual(summary.test_process_count, 0)
            self.assertEqual(summary.test_process_seconds, 0.0)
            self.assertEqual(summary.test_action_fallback_count, 1)
            self.assertEqual(summary.test_action_fallback_seconds, 11.0)
            self.assertEqual(summary.slow_test_action_count, 1)
            self.assertEqual(summary.slow_test_count, 1)
            self.assertEqual(
                summary.slow_tests[0].execution_source,
                bazel_profile_summary.EXECUTION_SOURCE_ACTION_FALLBACK,
            )
            self.assertIsNone(summary.slow_tests[0].admission_seconds)
            self.assertIn(
                "11.000s action fallback, 11.000s action",
                bazel_profile_summary.render_console(summary, "Test example"),
            )

    def test_writes_stable_summary_beside_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "build-example.profile.gz"
            self.write_profile(
                profile_path,
                [
                    {
                        "cat": "action processing",
                        "name": "Compiling example.cc",
                        "ph": "X",
                        "ts": 0,
                        "dur": 1_000_000,
                        "args": {"mnemonic": "CppCompile"},
                    }
                ],
            )
            summary = bazel_profile_summary.summarize(profile_path)
            output_path = bazel_profile_summary.summary_path(profile_path)

            bazel_profile_summary.write(summary, output_path)

            payload = json.loads(output_path.read_text())
            self.assertEqual(payload["schema"], bazel_profile_summary.SCHEMA)
            self.assertEqual(payload["profile"], profile_path.name)
            self.assertEqual(payload["action_aggregates"][0]["mnemonic"], "CppCompile")

    def test_rejects_missing_or_malformed_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.profile.gz"
            with self.assertRaisesRegex(
                bazel_profile_summary.BazelProfileSummaryError, "does not exist"
            ):
                bazel_profile_summary.summarize(missing)

            malformed = root / "malformed.profile.gz"
            with gzip.open(malformed, "wt", encoding="utf-8") as stream:
                json.dump({"traceEvents": []}, stream)
            with self.assertRaisesRegex(
                bazel_profile_summary.BazelProfileSummaryError, "no trace events"
            ):
                bazel_profile_summary.summarize(malformed)


if __name__ == "__main__":
    unittest.main()
