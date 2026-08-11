# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.devtools import project_presubmit


class ProjectPresubmitTest(unittest.TestCase):
    def test_common_arguments_expose_project_hygiene_phase(self):
        parser = argparse.ArgumentParser()
        project_presubmit.add_common_arguments(parser, project_name="example")

        args = parser.parse_args(
            [
                "--fix",
                "--hygiene",
                "--lane",
                "cmake",
                "--files-from",
                "paths.txt",
            ]
        )

        self.assertTrue(args.fix)
        self.assertFalse(args.check)
        self.assertTrue(args.hygiene)
        self.assertFalse(args.tests)
        self.assertEqual(args.lane, "cmake")
        self.assertEqual(args.files_from, "paths.txt")
        self.assertIn("Apply and stage", parser.format_help())

    def test_cmake_build_dir_uses_environment_override(self):
        with mock.patch.dict(
            os.environ,
            {project_presubmit.CMAKE_BUILD_DIR_ENV: "/tmp/iree-cmake-build"},
        ):
            self.assertEqual(
                project_presubmit.cmake_build_dir(Path("/repo")),
                Path("/tmp/iree-cmake-build"),
            )

    def test_validate_rejects_incomplete_ninja_build_tree(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            build_dir = Path(temporary_directory)
            (build_dir / "CMakeCache.txt").write_text(
                "CMAKE_GENERATOR:INTERNAL=Ninja\n",
                encoding="utf-8",
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertFalse(
                    project_presubmit.validate_cmake_build_tree("runtime", build_dir)
                )

            self.assertIn("mixed generator state", output.getvalue())

    def test_build_and_resolve_bazel_executable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo_root = Path(temporary_directory)
            executable_path = repo_root / "bazel-bin/example/tool"
            executable_path.parent.mkdir(parents=True)
            executable_path.write_text("tool\n", encoding="utf-8")
            executable_path.chmod(0o755)

            with (
                mock.patch.object(
                    project_presubmit, "run_command", return_value=True
                ) as run_command,
                mock.patch.object(
                    project_presubmit.bazel_dev,
                    "resolve_bazel_output_path",
                    return_value=executable_path,
                ) as resolve_bazel_output_path,
            ):
                result = project_presubmit.build_and_resolve_executable(
                    "example",
                    repo_root,
                    lane="bazel",
                    bazel_target="//example:tool",
                    cmake_target="example::tool",
                    bazel_args=("--config=presubmit",),
                )

            self.assertEqual(result, executable_path)
            run_command.assert_called_once_with(
                "example",
                [
                    "bazel",
                    "build",
                    "--config=presubmit",
                    "//example:tool",
                ],
                "Build Bazel executable //example:tool",
                cwd=repo_root,
            )
            resolve_bazel_output_path.assert_called_once_with(
                bazel="bazel",
                target="//example:tool",
                bazel_args=["--config=presubmit"],
                cwd=repo_root,
                env=None,
            )

    def test_build_and_resolve_cmake_executable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo_root = Path(temporary_directory)
            build_dir = repo_root / "build"
            executable_path = build_dir / "bin/tool"
            executable_path.parent.mkdir(parents=True)
            executable_path.write_text("tool\n", encoding="utf-8")
            executable_path.chmod(0o755)
            target = project_presubmit.cmake_file_api.CMakeExecutableTarget(
                name="example_tool",
                path=executable_path,
            )

            with (
                mock.patch.object(
                    project_presubmit,
                    "cmake_build_dir",
                    return_value=build_dir,
                ),
                mock.patch.object(
                    project_presubmit,
                    "validate_cmake_build_tree",
                    return_value=True,
                ) as validate_cmake_build_tree,
                mock.patch.object(
                    project_presubmit.cmake_file_api,
                    "resolve_target_name",
                    return_value="example_tool",
                ) as resolve_target_name,
                mock.patch.object(
                    project_presubmit, "run_command", return_value=True
                ) as run_command,
                mock.patch.object(
                    project_presubmit.cmake_file_api,
                    "resolve_executable",
                    return_value=target,
                ) as resolve_executable,
            ):
                result = project_presubmit.build_and_resolve_executable(
                    "example",
                    repo_root,
                    lane="cmake",
                    bazel_target="//example:tool",
                    cmake_target="example::tool",
                )

            self.assertEqual(result, executable_path)
            validate_cmake_build_tree.assert_called_once_with("example", build_dir)
            resolve_target_name.assert_called_once_with(build_dir, "example::tool")
            run_command.assert_called_once_with(
                "example",
                [
                    "cmake",
                    "--build",
                    str(build_dir),
                    "--target",
                    "example_tool",
                    "--parallel",
                ],
                "Build CMake executable example::tool",
                cwd=repo_root,
            )
            resolve_executable.assert_called_once_with(build_dir, "example::tool")

    def test_build_and_resolve_rejects_missing_bazel_output(self):
        output = io.StringIO()
        with (
            mock.patch.object(project_presubmit, "run_command", return_value=True),
            mock.patch.object(
                project_presubmit.bazel_dev,
                "resolve_bazel_output_path",
                return_value=None,
            ),
            contextlib.redirect_stderr(output),
        ):
            result = project_presubmit.build_and_resolve_executable(
                "example",
                Path("/repo"),
                lane="bazel",
                bazel_target="//example:tool",
                cmake_target="example::tool",
            )

        self.assertIsNone(result)
        self.assertIn("has no executable output", output.getvalue())

    def test_stage_changed_paths_stages_only_existing_or_tracked_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo_root = Path(temporary_directory)
            subprocess_commands = (
                ["git", "init", "--quiet"],
                ["git", "config", "user.name", "Presubmit Test"],
                ["git", "config", "user.email", "presubmit@example.com"],
            )
            for command in subprocess_commands:
                project_presubmit.subprocess.run(
                    command,
                    cwd=repo_root,
                    check=True,
                    stdout=project_presubmit.subprocess.PIPE,
                    stderr=project_presubmit.subprocess.PIPE,
                )

            deleted_path = repo_root / "generated/deleted.txt"
            deleted_path.parent.mkdir()
            deleted_path.write_text("tracked\n", encoding="utf-8")
            unrelated_path = repo_root / "unrelated.txt"
            unrelated_path.write_text("original\n", encoding="utf-8")
            project_presubmit.subprocess.run(
                ["git", "add", "--", "generated/deleted.txt", "unrelated.txt"],
                cwd=repo_root,
                check=True,
            )
            project_presubmit.subprocess.run(
                ["git", "commit", "--quiet", "-m", "baseline"],
                cwd=repo_root,
                check=True,
            )

            deleted_path.unlink()
            new_path = repo_root / "generated/new.txt"
            new_path.write_text("generated\n", encoding="utf-8")
            obsolete_untracked_path = repo_root / "generated/obsolete.txt"
            obsolete_untracked_path.write_text("obsolete\n", encoding="utf-8")
            obsolete_untracked_path.unlink()
            unrelated_path.write_text("modified\n", encoding="utf-8")

            self.assertTrue(
                project_presubmit.stage_changed_paths(
                    "example",
                    repo_root,
                    (
                        "generated/deleted.txt",
                        "generated/new.txt",
                        "generated/obsolete.txt",
                    ),
                )
            )

            staged = project_presubmit.subprocess.run(
                ["git", "diff", "--cached", "--name-status", "--no-renames"],
                cwd=repo_root,
                check=True,
                stdout=project_presubmit.subprocess.PIPE,
                text=True,
            ).stdout
            self.assertEqual(
                staged,
                "D\tgenerated/deleted.txt\nA\tgenerated/new.txt\n",
            )
            unstaged = project_presubmit.subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=repo_root,
                check=True,
                stdout=project_presubmit.subprocess.PIPE,
                text=True,
            ).stdout
            self.assertEqual(unstaged, "unrelated.txt\n")

    def test_stage_changed_paths_reports_git_query_failure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = io.StringIO()
            with contextlib.redirect_stderr(output):
                self.assertFalse(
                    project_presubmit.stage_changed_paths(
                        "example",
                        Path(temporary_directory),
                        ("generated/file.txt",),
                    )
                )

        self.assertIn("failed to query tracked source updates", output.getvalue())


if __name__ == "__main__":
    unittest.main()
