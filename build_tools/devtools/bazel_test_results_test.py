# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from build_tools.devtools import bazel_test_results


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
                    {
                        "testResult": {
                            "testActionOutput": [
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
                        }
                    }
                )
                + "\n"
            )

            self.assertEqual(bazel_test_results.stage(build_events, output_dir), 2)
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
                        {
                            "testResult": {
                                "testActionOutput": [
                                    {"name": "test.log", "uri": source.as_uri()}
                                ]
                            }
                        }
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
                    {
                        "testResult": {
                            "testActionOutput": [
                                {
                                    "name": "test.log",
                                    "uri": "bytestream://cache/output",
                                }
                            ]
                        }
                    }
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
                    {
                        "testResult": {
                            "testActionOutput": [
                                {"name": "test.log", "uri": source.as_uri()}
                            ]
                        }
                    }
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


if __name__ == "__main__":
    unittest.main()
