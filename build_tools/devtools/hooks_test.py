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


if __name__ == "__main__":
    unittest.main()
