# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.devtools import environment


class EnvironmentTest(unittest.TestCase):
    def test_windows_bazel_shell_follows_git_for_windows_install(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            git_root = Path(temporary_directory) / "Git"
            git_executable = git_root / "cmd/git.exe"
            bash_executable = git_root / "bin/bash.exe"
            git_executable.parent.mkdir(parents=True)
            bash_executable.parent.mkdir(parents=True)
            git_executable.write_bytes(b"")
            bash_executable.write_bytes(b"")

            result = environment.find_windows_bazel_sh(
                {}, platform_name="nt", git_executable=str(git_executable)
            )

            self.assertEqual(result, str(bash_executable))

    def test_windows_bazel_shell_preserves_explicit_override(self):
        result = environment.find_windows_bazel_sh(
            {environment.BAZEL_SH_ENV: "C:/tools/bash.exe"},
            platform_name="nt",
        )

        self.assertEqual(result, "C:/tools/bash.exe")

    def test_windows_bazel_shell_uses_space_free_short_path(self):
        long_path = "C:/Program Files/Git/bin/bash.exe"
        short_path = "C:/PROGRA~1/Git/bin/bash.exe"
        with mock.patch.object(
            environment,
            "_read_windows_short_path",
            return_value=short_path,
        ) as read_short_path:
            result = environment.find_windows_bazel_sh(
                {environment.BAZEL_SH_ENV: long_path},
                platform_name="nt",
            )

        self.assertEqual(result, short_path)
        read_short_path.assert_called_once_with(long_path)

    def test_windows_bazel_shell_aliases_installation_when_short_names_are_disabled(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            git_root = temporary_path / "Program Files" / "Git"
            bash_executable = git_root / "usr/bin/bash.exe"
            bash_executable.parent.mkdir(parents=True)
            bash_executable.write_bytes(b"")
            alias_root = temporary_path / "aliases"
            created_junctions: list[tuple[Path, Path]] = []

            def create_test_junction(link: Path, target: Path) -> None:
                created_junctions.append((link, target))
                link.symlink_to(target, target_is_directory=True)

            result = environment.bazel_compatible_windows_shell_path(
                str(bash_executable),
                platform_name="nt",
                short_path_reader=lambda _path: None,
                alias_root=alias_root,
                junction_creator=create_test_junction,
            )

            result_path = Path(result)
            self.assertFalse(any(character.isspace() for character in result))
            self.assertTrue(result_path.samefile(bash_executable))
            self.assertEqual(len(created_junctions), 1)
            self.assertEqual(created_junctions[0][1], git_root)

            repeated_result = environment.bazel_compatible_windows_shell_path(
                str(bash_executable),
                platform_name="nt",
                short_path_reader=lambda _path: None,
                alias_root=alias_root,
                junction_creator=create_test_junction,
            )

            self.assertEqual(repeated_result, result)
            self.assertEqual(len(created_junctions), 1)

    def test_non_windows_host_does_not_search_for_bazel_shell(self):
        with mock.patch.object(environment.shutil, "which") as which:
            result = environment.find_windows_bazel_sh({}, platform_name="posix")

        self.assertIsNone(result)
        which.assert_not_called()

    def test_tool_environment_exports_discovered_bazel_shell(self):
        tool_environment = environment.ToolEnvironment(
            environment.ToolMode.SYSTEM, None
        )
        with mock.patch.object(
            environment,
            "find_windows_bazel_sh",
            return_value="C:/Program Files/Git/bin/bash.exe",
        ):
            result = tool_environment.path_env({"PATH": "C:/Windows"})

        self.assertEqual(
            result[environment.BAZEL_SH_ENV],
            "C:/Program Files/Git/bin/bash.exe",
        )


if __name__ == "__main__":
    unittest.main()
