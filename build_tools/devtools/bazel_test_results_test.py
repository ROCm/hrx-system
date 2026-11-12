# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from build_tools.devtools import bazel_test_results


def test_result_event(outputs: list[dict], *, status: str = "PASSED") -> dict:
    return {
        "id": {
            "testResult": {
                "label": "//libamdf:current_test",
                "configuration": {"id": "configuration"},
                "run": 1,
                "shard": 1,
                "attempt": 1,
            }
        },
        "testResult": {
            "status": status,
            "testAttemptDurationMillis": "1",
            "testActionOutput": outputs,
        },
    }


class BazelTestResultsTest(unittest.TestCase):
    def test_stages_only_outputs_named_by_current_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "current"
            bazel_test_results.prepare_output_directory(output_dir)
            testlogs = root / "bazel-out" / "config" / "testlogs"
            current = testlogs / "libamdf" / "current_test"
            stale = testlogs / "libamdf" / "stale_test"
            current.mkdir(parents=True)
            stale.mkdir(parents=True)
            (current / "test.log").write_text("current log")
            (current / "test.xml").write_text("<current/>")
            (current / "outputs.zip").write_text("ignored")
            (stale / "test.log").write_text("stale log")
            build_events = output_dir / ".events" / "test.json"
            build_events.write_text(
                json.dumps(
                    test_result_event(
                        [
                            {
                                "name": "test.log",
                                "uri": (current / "test.log").as_uri(),
                            },
                            {
                                "name": "test.xml",
                                "uri": (current / "test.xml").as_uri(),
                            },
                            {
                                "name": "test.outputs/outputs.zip",
                                "uri": (current / "outputs.zip").as_uri(),
                            },
                        ]
                    )
                )
                + "\n"
            )

            result = bazel_test_results.stage(build_events, output_dir)
            self.assertEqual(result.output_count, 2)
            staged = output_dir / "bazel-testlogs" / "libamdf" / "current_test"
            self.assertEqual((staged / "test.log").read_text(), "current log")
            self.assertEqual((staged / "test.xml").read_text(), "<current/>")
            self.assertFalse(
                (output_dir / "bazel-testlogs" / "libamdf" / "stale_test").exists()
            )

    def test_rejects_colliding_configurations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "current"
            bazel_test_results.prepare_output_directory(output_dir)
            sources = []
            for configuration in ("one", "two"):
                source = (
                    root
                    / "bazel-out"
                    / configuration
                    / "testlogs"
                    / "libamdf"
                    / "same_test"
                    / "test.log"
                )
                source.parent.mkdir(parents=True)
                source.write_text(configuration)
                sources.append(source)
            build_events = output_dir / ".events" / "test.json"
            build_events.write_text(
                "\n".join(
                    json.dumps(
                        test_result_event(
                            [{"name": "test.log", "uri": source.as_uri()}]
                        )
                    )
                    for source in sources
                )
                + "\n"
            )

            with self.assertRaisesRegex(
                bazel_test_results.BazelTestResultsError, "colliding test outputs"
            ):
                bazel_test_results.stage(build_events, output_dir)

    def test_rejects_unmaterialized_remote_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "current"
            bazel_test_results.prepare_output_directory(output_dir)
            build_events = output_dir / ".events" / "test.json"
            build_events.write_text(
                json.dumps(
                    test_result_event(
                        [
                            {
                                "name": "test.log",
                                "uri": "bytestream://cache/output",
                            }
                        ]
                    )
                )
                + "\n"
            )

            with self.assertRaisesRegex(
                bazel_test_results.BazelTestResultsError, "not a local file URI"
            ):
                bazel_test_results.stage(build_events, output_dir)

    def test_rejects_output_from_an_earlier_test_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "current"
            bazel_test_results.prepare_output_directory(output_dir)
            source = (
                root
                / "bazel-out"
                / "config"
                / "testlogs"
                / "libamdf"
                / "same_test"
                / "test.log"
            )
            source.parent.mkdir(parents=True)
            source.write_text("second phase")
            destination = (
                output_dir / "bazel-testlogs" / "libamdf" / "same_test" / "test.log"
            )
            destination.parent.mkdir(parents=True)
            destination.write_text("first phase")
            build_events = output_dir / ".events" / "second.json"
            build_events.write_text(
                json.dumps(
                    test_result_event([{"name": "test.log", "uri": source.as_uri()}])
                )
                + "\n"
            )

            with self.assertRaisesRegex(
                bazel_test_results.BazelTestResultsError,
                "test phases produced colliding test outputs",
            ):
                bazel_test_results.stage(build_events, output_dir)
            self.assertEqual(destination.read_text(), "first phase")

    def test_refuses_populated_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "current"
            output_dir.mkdir()
            (output_dir / "stale").write_text("stale")

            with self.assertRaisesRegex(
                bazel_test_results.BazelTestResultsError, "is not empty"
            ):
                bazel_test_results.prepare_output_directory(output_dir)

    def test_writes_compact_fresh_cached_and_failed_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "native-phase"
            output_dir.mkdir()
            build_events = root / "events.json"
            failed_log = (
                root
                / "bazel-out"
                / "config"
                / "testlogs"
                / "z"
                / "timeout"
                / "test.log"
            )
            failed_log.parent.mkdir(parents=True)
            failed_log.write_text("timeout")
            events = [
                {
                    "id": {"targetConfigured": {"label": "//z:timeout"}},
                    "configured": {"testSize": "SMALL"},
                },
                {
                    "id": {"targetConfigured": {"label": "//a:cached"}},
                    "configured": {"testSize": "MEDIUM"},
                },
                {
                    "id": {
                        "testSummary": {
                            "label": "//a:cached",
                            "configuration": {"id": "configuration-a"},
                        }
                    },
                    "testSummary": {
                        "overallStatus": "PASSED",
                        "totalRunCount": 1,
                        "totalNumCached": 1,
                        "attemptCount": 1,
                        "totalRunDurationMillis": "35",
                    },
                },
                {
                    "id": {
                        "testResult": {
                            "label": "//z:timeout",
                            "configuration": {"id": "configuration-z"},
                            "run": 2,
                            "shard": 1,
                            "attempt": 2,
                        }
                    },
                    "testResult": {
                        "status": "TIMEOUT",
                        "testAttemptDurationMillis": "1234",
                        "testActionOutput": [
                            {
                                "name": "test.log",
                                "uri": failed_log.as_uri(),
                            }
                        ],
                    },
                },
                {
                    "id": {
                        "testSummary": {
                            "label": "//z:timeout",
                            "configuration": {"id": "configuration-z"},
                        }
                    },
                    "testSummary": {
                        "overallStatus": "TIMEOUT",
                        "totalRunCount": 2,
                        "attemptCount": 2,
                        "totalRunDurationMillis": "1234",
                    },
                },
                {
                    "id": {"buildFinished": {}},
                    "finished": {
                        "overallSuccess": False,
                        "exitCode": {"name": "TESTS_FAILED", "code": 3},
                    },
                },
            ]
            build_events.write_text(
                "".join(json.dumps(event) + "\n" for event in events)
            )

            result = bazel_test_results.stage(build_events, output_dir)
            observation_bytes = result.observation_path.read_bytes()
            observation = json.loads(observation_bytes)
            expected_digest = hashlib.sha256(b"//a:cached\0//z:timeout\0").hexdigest()
            self.assertEqual(result.output_count, 1)
            self.assertEqual(result.observation_byte_length, len(observation_bytes))
            self.assertGreaterEqual(result.observation_seconds, 0)
            self.assertEqual(observation["schema_version"], 1)
            self.assertEqual(observation["phase"], "native-phase")
            self.assertEqual(
                observation["selection"],
                {"test_count": 2, "label_set_sha256": expected_digest},
            )
            self.assertEqual(
                observation["configurations"],
                ["configuration-a", "configuration-z"],
            )
            self.assertEqual(observation["test_runs"], {"fresh": 2, "cached": 1})
            self.assertEqual(
                [test["label"] for test in observation["tests"]],
                ["//a:cached", "//z:timeout"],
            )
            self.assertEqual(
                observation["failed_attempts"],
                [
                    {
                        "attempt": 2,
                        "cached": False,
                        "configuration": 1,
                        "duration_millis": 1234,
                        "label": "//z:timeout",
                        "run": 2,
                        "shard": 1,
                        "status": "TIMEOUT",
                    }
                ],
            )
            self.assertEqual(
                observation["invocation"],
                {
                    "complete": True,
                    "overall_success": False,
                    "exit_code": 3,
                    "exit_code_name": "TESTS_FAILED",
                },
            )
            self.assertNotIn(str(root).encode(), observation_bytes)

    def test_rejects_impossible_cached_run_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_events = root / "events.json"
            build_events.write_text(
                json.dumps(
                    {
                        "id": {
                            "testSummary": {
                                "label": "//example:test",
                                "configuration": {"id": "configuration"},
                            }
                        },
                        "testSummary": {
                            "overallStatus": "PASSED",
                            "totalRunCount": 1,
                            "totalNumCached": 2,
                        },
                    }
                )
                + "\n"
            )

            with self.assertRaisesRegex(
                bazel_test_results.BazelTestResultsError,
                "cached run count exceeds its total",
            ):
                bazel_test_results.stage(build_events, root / "output")


if __name__ == "__main__":
    unittest.main()
