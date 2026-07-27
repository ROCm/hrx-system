# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

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
                {"fixture": tool_path},
            )
            self.assertEqual(
                substituter.substitute("{tmp}/file {json: true} {tool:fixture}"),
                f"{directory}/file {{json: true}} {tool_path}",
            )

    def test_contains_ordering(self):
        runner = execution.ExecutionRunner(tools={})
        runner._check_contains(
            "case", "step", "stdout", "alpha\nbeta\ngamma\n", ["alpha", "gamma"]
        )
        with self.assertRaises(execution.CaseFailure):
            runner._check_contains(
                "case", "step", "stdout", "alpha\nbeta\ngamma\n", ["gamma", "alpha"]
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
