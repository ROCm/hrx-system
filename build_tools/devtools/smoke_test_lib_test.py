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
from unittest import mock

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


class RunSmokeTest(unittest.TestCase):
    def test_runs_against_live_repository_without_copying(self):
        scenario_runner = mock.Mock()
        with (
            mock.patch.object(smoke_test_lib, "parse_arguments"),
            mock.patch.object(
                smoke_test_lib,
                "repository_status",
                side_effect=["candidate\n", "candidate\n"],
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = smoke_test_lib.run_smoke(
                description="test smoke",
                scenario_runner=scenario_runner,
            )

        self.assertEqual(result, 0)
        scenario_runner.assert_called_once_with(smoke_test_lib.REPO_ROOT)

    def test_fails_when_dry_run_mutates_repository(self):
        with (
            mock.patch.object(smoke_test_lib, "parse_arguments"),
            mock.patch.object(
                smoke_test_lib,
                "repository_status",
                side_effect=["candidate\n", "candidate\n?? generated.txt\n"],
            ),
            self.assertRaisesRegex(RuntimeError, "generated.txt"),
        ):
            smoke_test_lib.run_smoke(
                description="test smoke",
                scenario_runner=lambda _repo_root: None,
            )


if __name__ == "__main__":
    unittest.main()
