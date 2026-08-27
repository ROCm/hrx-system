# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path

from build_tools.devtools import smoke_test_lib


class RunCommandTest(unittest.TestCase):
    def test_successful_command_output_is_suppressed(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            smoke_test_lib.run_command(
                Path.cwd(),
                [
                    sys.executable,
                    "-c",
                    "print('successful ' + 'child output')",
                ],
            )

        self.assertIn("smoke:", stdout.getvalue())
        self.assertNotIn("successful child output", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_failed_command_output_is_reported(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        child = (
            "import sys; "
            "print('failed child stdout'); "
            "print('failed child stderr', file=sys.stderr); "
            "raise SystemExit(7)"
        )

        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(subprocess.CalledProcessError) as raised,
        ):
            smoke_test_lib.run_command(
                Path.cwd(),
                [sys.executable, "-c", child],
            )

        self.assertEqual(raised.exception.returncode, 7)
        self.assertIn("smoke:", stdout.getvalue())
        self.assertIn("failed child stdout", stderr.getvalue())
        self.assertIn("failed child stderr", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
