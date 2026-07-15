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
