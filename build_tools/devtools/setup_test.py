# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from build_tools.devtools.command_plan import CommandStep, WriteFileStep
from build_tools.devtools.environment import ToolEnvironment, ToolMode
from build_tools.devtools.setup import setup_plan


class SetupPlanTest(unittest.TestCase):
    def test_system_mode_does_not_create_venv_or_aliases_by_default(self):
        plan = setup_plan("bazel", ToolEnvironment(ToolMode.SYSTEM, None), None)

        commands = [step for step in plan.steps if isinstance(step, CommandStep)]
        self.assertFalse(
            any("-m" in step.argv and "venv" in step.argv for step in commands)
        )
        self.assertFalse(any(isinstance(step, WriteFileStep) for step in plan.steps))

    def test_system_mode_can_write_explicit_alias_dir(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            alias_dir = Path(temporary_directory) / "aliases"
            plan = setup_plan(
                "bazel", ToolEnvironment(ToolMode.SYSTEM, None), alias_dir
            )

            self.assertTrue(any(isinstance(step, WriteFileStep) for step in plan.steps))
            self.assertIn(str(alias_dir), plan.describe())

    def test_venv_mode_schedules_python_tool_install(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            venv_root = Path(temporary_directory) / "venv"
            plan = setup_plan("bazel", ToolEnvironment(ToolMode.VENV, venv_root), None)

            commands = [step for step in plan.steps if isinstance(step, CommandStep)]
            self.assertTrue(any("-m venv" in step.describe() for step in commands))
            self.assertTrue(
                any("requirements-dev.lock.txt" in step.describe() for step in commands)
            )
            self.assertTrue(
                any(
                    "requirements-analysis.lock.txt" in step.describe()
                    for step in commands
                )
            )
            self.assertTrue(
                any("--only-binary=:all:" in step.describe() for step in commands)
            )
            self.assertTrue(
                any("--group bazel" in step.describe() for step in commands)
            )


if __name__ == "__main__":
    unittest.main()
