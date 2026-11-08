# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests doctor planning for checkouts, linked worktrees, and source archives."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.devtools.doctor import LANE_TOOLS, doctor_plan
from build_tools.devtools.environment import REPO_ROOT, ToolEnvironment, ToolMode


class DoctorPlanTest(unittest.TestCase):
    def test_bazel_version_matches_repository_baseline(self):
        expected_version = (REPO_ROOT / ".bazelversion").read_text().strip()
        bazel_check = next(
            check for check in LANE_TOOLS["bazel"] if check[0] == "bazel"
        )
        self.assertRegex(f"bazel {expected_version}", bazel_check[2])

    def test_git_status_requires_checkout_metadata(self):
        for layout in ["archive", "checkout", "worktree"]:
            with (
                self.subTest(layout=layout),
                tempfile.TemporaryDirectory() as directory,
            ):
                source_root = Path(directory)
                if layout == "checkout":
                    (source_root / ".git").mkdir()
                elif layout == "worktree":
                    (source_root / ".git").write_text("gitdir: ../repo.git")
                with mock.patch("build_tools.devtools.doctor.REPO_ROOT", source_root):
                    plan = doctor_plan("bazel", ToolEnvironment(ToolMode.SYSTEM, None))

                commands = [step.argv for step in plan.steps]
                self.assertEqual(
                    ["git", "status", "--short"] in commands, layout != "archive"
                )
                self.assertTrue(
                    any(Path(command[0]).stem == "bazel" for command in commands)
                )


if __name__ == "__main__":
    unittest.main()
