# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import unittest

from build_tools.devtools import hooks


class HooksTest(unittest.TestCase):
    def test_hook_content_uses_posix_shell_quoting(self):
        content = hooks.hook_content(
            "bazel",
            "paranoid",
            r"C:\Program Files\Python\python.exe",
        )

        self.assertIn("'C:/Program Files/Python/python.exe'", content)
        self.assertNotIn(r"C:\Program Files\Python", content)
        self.assertIn((hooks.REPO_ROOT / "dev.py").as_posix(), content)
        self.assertIn("'{1}'", content)

    def test_hook_content_persists_bazel_policy(self):
        content = hooks.hook_content(
            "bazel",
            "paranoid",
            "/usr/bin/python3",
            bazel_configs=("remote-execution", "local-tests"),
        )

        self.assertIn("--bazel-config=remote-execution", content)
        self.assertIn("--bazel-config=local-tests", content)

    def test_cmake_hook_rejects_bazel_policy(self):
        with self.assertRaisesRegex(ValueError, "require the Bazel hook lane"):
            hooks.hook_content(
                "cmake",
                "default",
                "/usr/bin/python3",
                bazel_configs=("remote-execution",),
            )


if __name__ == "__main__":
    unittest.main()
