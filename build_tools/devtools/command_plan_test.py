# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.devtools import command_plan
from build_tools.devtools.command_plan import (
    CommandPlan,
    CommandStep,
    ExecCommandStep,
    OptionalCheckCommandStep,
    WriteFileStep,
)


class CommandPlanTest(unittest.TestCase):
    def test_dry_run_does_not_write_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "out.txt"
            plan = CommandPlan([WriteFileStep(output_path, "hello\n")])

            self.assertEqual(plan.run(dry_run=True), 0)
            self.assertFalse(output_path.exists())

    def test_write_file_step_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "out.txt"
            plan = CommandPlan([WriteFileStep(output_path, "hello\n")])

            self.assertEqual(plan.run(), 0)
            self.assertEqual(plan.run(), 0)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "hello\n")

    def test_write_file_step_describes_non_write_label(self):
        step = WriteFileStep(
            Path("lefthook-local.yml"),
            "hello\n",
            label="select bazel hook policy with ci profile",
        )

        self.assertIn(
            "# select bazel hook policy with ci profile: write lefthook-local.yml",
            step.describe(),
        )

    def test_command_step_describes_argv_not_shell_blob(self):
        step = CommandStep(["python", "dev.py", "bazel", "setup"], cwd=Path.cwd())

        self.assertIn("python dev.py bazel setup", step.describe())

    def test_command_step_is_quiet_by_default(self):
        plan = CommandPlan(
            [
                CommandStep(
                    [sys.executable, "-c", ""],
                    cwd=Path.cwd(),
                    label="quiet command",
                )
            ]
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(plan.run(), 0)

        self.assertEqual(output.getvalue(), "")

    def test_verbose_command_step_prints_label_and_command(self):
        plan = CommandPlan(
            [
                CommandStep(
                    [sys.executable, "-c", ""],
                    cwd=Path.cwd(),
                    label="loud command",
                )
            ]
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(plan.run(verbose=True), 0)

        self.assertIn("dev.py: loud command", output.getvalue())
        self.assertIn(sys.executable, output.getvalue())

    def test_announced_command_step_prints_label_without_command(self):
        plan = CommandPlan(
            [
                CommandStep(
                    [sys.executable, "-c", ""],
                    cwd=Path.cwd(),
                    label="starting long command",
                    announce=True,
                )
            ]
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(plan.run(), 0)

        self.assertEqual(output.getvalue(), "dev.py: starting long command\n")

    def test_windows_command_step_resolves_executable_from_child_path(self):
        child_path = "C:/source/.venv/Scripts;C:/Windows/System32"
        resolved_executable = "C:/source/.venv/Scripts/lefthook.exe"
        step = CommandStep(
            ["lefthook", "version"],
            cwd=Path.cwd(),
            env={"PATH": child_path},
        )
        completed_process = subprocess.CompletedProcess(step.argv, 0)

        with (
            mock.patch.object(command_plan, "is_windows", return_value=True),
            mock.patch.object(
                command_plan.shutil,
                "which",
                return_value=resolved_executable,
            ) as mock_which,
            mock.patch.object(
                subprocess, "run", return_value=completed_process
            ) as mock_run,
        ):
            self.assertEqual(step.run(), 0)

        mock_which.assert_called_once_with("lefthook", path=child_path)
        mock_run.assert_called_once_with(
            [resolved_executable, "version"],
            cwd=step.cwd,
            env=step.env,
        )

    def test_exec_command_step_replaces_current_process(self):
        step = ExecCommandStep(["tool", "arg"], cwd=Path.cwd(), env={"PATH": "test"})

        with (
            mock.patch.object(command_plan, "is_windows", return_value=False),
            mock.patch.object(os, "chdir") as mock_chdir,
            mock.patch.object(
                os, "execvpe", side_effect=OSError("missing")
            ) as mock_exec,
        ):
            self.assertEqual(step.run(), 127)

        mock_chdir.assert_called_once_with(Path.cwd())
        mock_exec.assert_called_once_with("tool", ["tool", "arg"], {"PATH": "test"})

    def test_exec_command_step_waits_for_child_on_windows(self):
        step = ExecCommandStep(["tool", "arg"], cwd=Path.cwd(), env={"PATH": "test"})
        completed_process = subprocess.CompletedProcess(step.argv, 7)

        with (
            mock.patch.object(command_plan, "is_windows", return_value=True),
            mock.patch.object(
                subprocess, "run", return_value=completed_process
            ) as mock_run,
            mock.patch.object(os, "execvpe") as mock_exec,
        ):
            self.assertEqual(step.run(), 7)

        mock_run.assert_called_once_with(step.argv, cwd=step.cwd, env=step.env)
        mock_exec.assert_not_called()

    def test_windows_exec_command_step_propagates_real_child_status(self):
        step = ExecCommandStep(
            [sys.executable, "-c", "raise SystemExit(7)"], cwd=Path.cwd()
        )

        with mock.patch.object(command_plan, "is_windows", return_value=True):
            self.assertEqual(step.run(), 7)

    def test_command_step_description_uses_windows_cmd_syntax(self):
        original_path = os.environ.get("PATH", "")
        step = CommandStep(
            ["python.exe", "dev.py", "bazel", "setup"],
            cwd=Path("C:/source tree"),
            env={"PATH": "C:/source tree/.venv/Scripts" + os.pathsep + original_path},
        )

        with mock.patch.object(command_plan, "is_windows", return_value=True):
            description = step.describe()

        self.assertIn('set "PATH=C:/source tree/.venv/Scripts;%PATH%"', description)
        expected_cwd = subprocess.list2cmdline([str(step.cwd)])
        self.assertIn(f"cd /d {expected_cwd} &&", description)
        self.assertNotIn("$PATH", description)

    def test_optional_check_step_warns_without_failing(self):
        plan = CommandPlan(
            [
                OptionalCheckCommandStep(
                    ["definitely-not-an-iree-tool"],
                    cwd=Path.cwd(),
                    label="missing optional tool",
                )
            ]
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(plan.run(), 0)

        self.assertIn("warning", output.getvalue())

    def test_required_check_step_reports_missing_command(self):
        step = command_plan.CheckCommandStep(
            ["definitely-not-an-iree-tool"], cwd=Path.cwd()
        )
        error = io.StringIO()

        with contextlib.redirect_stderr(error):
            self.assertEqual(step.run(), 127)

        self.assertIn("failed to run", error.getvalue())

    def test_optional_check_step_reports_hint(self):
        plan = CommandPlan(
            [
                OptionalCheckCommandStep(
                    ["definitely-not-an-iree-tool"],
                    cwd=Path.cwd(),
                    hint="run setup",
                    label="missing optional tool",
                )
            ]
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(plan.run(), 0)

        self.assertIn("dev.py: hint: run setup", output.getvalue())


if __name__ == "__main__":
    unittest.main()
