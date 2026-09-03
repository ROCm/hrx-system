# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import execution


class ExecutionUnitTest(unittest.TestCase):
    def test_substitution_preserves_unknown_braces(self):
        with tempfile.TemporaryDirectory() as directory:
            tool_path = Path(directory) / "tool"
            substituter = execution._Substituter(
                {"tmp": directory},
                {"fixture": execution.ToolCommand(executable=str(tool_path))},
            )
            self.assertEqual(
                substituter.substitute("{tmp}/file {json: true} {tool:fixture}"),
                f"{directory}/file {{json: true}} {tool_path}",
            )

    def test_substitution_rejects_multi_argument_tool_command(self):
        substituter = execution._Substituter(
            {},
            {
                "fixture": execution.ToolCommand(
                    executable=sys.executable,
                    arguments=("/path/to/fixture.py",),
                )
            },
        )
        with self.assertRaisesRegex(execution.SchemaError, "multi-argument command"):
            substituter.substitute("{tool:fixture}")

    def test_runner_launches_tool_command_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "cases": [
                            {
                                "name": "command prefix",
                                "run": {
                                    "tool": "fixture",
                                    "args": ["manifest argument"],
                                },
                                "stdout": {"contains": ["manifest argument"]},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            runner = execution.ExecutionRunner(
                tools={
                    "fixture": execution.ToolCommand(
                        executable=sys.executable,
                        arguments=(
                            "-c",
                            "import sys; print(sys.argv[1])",
                        ),
                    )
                }
            )

            self.assertEqual(
                runner.run_manifest(manifest_path), execution.RunSummary(case_count=1)
            )

    def test_parse_tool_bindings_appends_fixed_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "fixture"
            executable.touch()

            tools = execution.parse_tool_bindings(
                [f"fixture={executable}"],
                ["fixture=first", "fixture=second=value"],
            )

            self.assertEqual(
                tools["fixture"],
                execution.ToolCommand(
                    executable=str(executable),
                    arguments=("first", "second=value"),
                ),
            )

    def test_write_text_preserves_utf8_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.txt"
            text = "héllo\nworld\n"
            runner = execution.ExecutionRunner(tools={})

            runner._run_write_step(
                "fixture case", "write fixture", {"path": str(path), "text": text}
            )

            self.assertEqual(path.read_bytes(), text.encode("utf-8"))

    def test_contains_ordering(self):
        runner = execution.ExecutionRunner(tools={})
        runner._check_contains(
            "case", "step", "stdout", "alpha\nbeta\ngamma\n", ["alpha", "gamma"]
        )
        with self.assertRaises(execution.CaseFailure):
            runner._check_contains(
                "case", "step", "stdout", "alpha\nbeta\ngamma\n", ["gamma", "alpha"]
            )

    def test_non_empty_stream(self):
        runner = execution.ExecutionRunner(tools={})
        runner._check_stream(
            "case",
            "step",
            ["fixture"],
            "stdout",
            b"generated output\n",
            {"non_empty": True},
        )
        with self.assertRaisesRegex(execution.CaseFailure, "expected non-empty stdout"):
            runner._check_stream(
                "case",
                "step",
                ["fixture"],
                "stdout",
                b"",
                {"non_empty": True},
            )

    def test_stream_rejects_conflicting_empty_expectations(self):
        runner = execution.ExecutionRunner(tools={})
        with self.assertRaisesRegex(
            execution.SchemaError, "'empty' and 'non_empty' are mutually exclusive"
        ):
            runner._check_stream(
                "case",
                "step",
                ["fixture"],
                "stdout",
                b"",
                {"empty": True, "non_empty": True},
            )

    def test_sanitizer_env_resolves_suppressions_runfile(self):
        with tempfile.TemporaryDirectory() as directory:
            runfiles_dir = Path(directory)
            workspace_name = "test_workspace"
            suppression_path = (
                runfiles_dir
                / workspace_name
                / "build_tools"
                / "sanitizer"
                / "lsan_suppressions_vulkan.txt"
            )
            suppression_path.parent.mkdir(parents=True)
            suppression_path.write_text("# test\n", encoding="utf-8")

            env = {
                "LSAN_OPTIONS": (
                    "verbosity=1" + os.pathsep + "suppressions=build_tools/sanitizer/"
                    "lsan_suppressions_vulkan.txt"
                ),
            }
            with mock.patch.dict(
                os.environ,
                {
                    "RUNFILES_DIR": str(runfiles_dir),
                    "TEST_WORKSPACE": workspace_name,
                },
            ):
                execution._resolve_sanitizer_env(env)

            self.assertEqual(
                env["LSAN_OPTIONS"],
                "verbosity=1" + os.pathsep + f"suppressions={suppression_path}",
            )


if __name__ == "__main__":
    unittest.main()
