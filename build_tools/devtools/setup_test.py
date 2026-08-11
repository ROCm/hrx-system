# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from build_tools.devtools.command_plan import CommandStep, WriteFileStep
from build_tools.devtools.environment import REPO_ROOT, ToolEnvironment, ToolMode
from build_tools.devtools.setup import common_setup_plan, setup_plan


def locked_versions(path: Path) -> dict[str, str]:
    versions = {}
    for line in path.read_text().splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^ ]+)", line)
        if match:
            normalized_name = re.sub(r"[-_.]+", "-", match.group(1).lower())
            versions[normalized_name] = match.group(2)
    return versions


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
            plan = setup_plan(
                "bazel",
                ToolEnvironment(ToolMode.VENV, venv_root),
                None,
                platform_name="linux",
            )

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
            install_index = next(
                index
                for index, step in enumerate(commands)
                if any(
                    argument.endswith("requirements-dev.lock.txt")
                    for argument in step.argv
                )
            )
            probe_index, probe = next(
                (index, step)
                for index, step in enumerate(commands)
                if step.label == "check Lefthook CLI compatibility"
            )
            self.assertLess(install_index, probe_index)
            self.assertIn("--file", probe.argv)
            self.assertNotIn("--files", probe.argv)
            self.assertEqual(probe.argv[probe.argv.index("--file") + 1], "dev.py")
            self.assertIn("__iree_cli_compatibility_probe__", probe.argv)

    def test_windows_venv_does_not_install_unusable_semgrep_package(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            venv_root = Path(temporary_directory) / "venv"
            plan = setup_plan(
                "bazel",
                ToolEnvironment(ToolMode.VENV, venv_root),
                None,
                platform_name="win32",
            )

            description = plan.describe()
            self.assertIn("requirements-dev.lock.txt", description)
            self.assertNotIn("requirements-analysis.lock.txt", description)

    def test_optional_docs_setup_uses_locked_python_and_standalone_tools(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            venv_root = Path(temporary_directory) / "venv"
            plan = common_setup_plan(
                ToolEnvironment(ToolMode.VENV, venv_root),
                include_docs=True,
                platform_name="linux",
            )

            description = plan.describe()
            self.assertIn("loom/docs/requirements.lock.txt", description)
            self.assertEqual(description.count("--group docs"), 2)
            self.assertIn("--check", description)
            python_requirements_step = next(
                step
                for step in plan.steps
                if isinstance(step, CommandStep)
                and any(
                    argument.endswith("requirements-analysis.lock.txt")
                    for argument in step.argv
                )
            )
            self.assertTrue(
                any(
                    argument.endswith("loom/docs/requirements.lock.txt")
                    for argument in python_requirements_step.argv
                )
            )

    def test_windows_optional_docs_setup_skips_analysis_environment(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            venv_root = Path(temporary_directory) / "venv"
            plan = common_setup_plan(
                ToolEnvironment(ToolMode.VENV, venv_root),
                include_docs=True,
                platform_name="win32",
            )

            description = plan.describe()
            self.assertIn("loom/docs/requirements.lock.txt", description)
            self.assertNotIn("requirements-analysis.lock.txt", description)
            self.assertIn("--group docs", description)

    def test_optional_docs_setup_rejects_system_toolchain(self):
        with self.assertRaisesRegex(
            ValueError, "documentation setup requires a managed tool environment"
        ):
            common_setup_plan(ToolEnvironment(ToolMode.SYSTEM, None), include_docs=True)

    def test_optional_docs_lock_matches_shared_analysis_packages(self):
        analysis_versions = locked_versions(
            REPO_ROOT / "requirements-analysis.lock.txt"
        )
        docs_versions = locked_versions(
            REPO_ROOT / "loom" / "docs" / "requirements.lock.txt"
        )

        shared_names = analysis_versions.keys() & docs_versions.keys()
        self.assertTrue(shared_names)
        self.assertEqual(
            {name: analysis_versions[name] for name in shared_names},
            {name: docs_versions[name] for name in shared_names},
        )


if __name__ == "__main__":
    unittest.main()
