# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.cmake.test_environment import (
    CONFIGURATION,
    CTEST_COMMAND,
    REPO_ROOT,
    configure_project,
    install_project,
    run_command,
    test_project,
)

FIXTURE = REPO_ROOT / "build_tools/cmake/testdata/test_results"


class CMakeTestResultsTest(unittest.TestCase):
    def check_results(self, test_directory, scratch_root):
        tests = json.loads(test_project(test_directory, "--show-only=json-v1"))["tests"]
        reports = {}
        for test in tests:
            environment = next(
                prop["value"]
                for prop in test["properties"]
                if prop["name"] == "ENVIRONMENT"
            )
            environment = dict(entry.split("=", 1) for entry in environment)
            report = Path(environment["XML_OUTPUT_FILE"])
            self.assertTrue(report.is_absolute())
            self.assertTrue(report.is_relative_to(scratch_root))
            self.assertEqual(report.parent, Path(environment["TEST_TMPDIR"]))
            self.assertNotIn(report, reports)
            reports[report] = test["name"]
        self.assertEqual(len(reports), 6)

        # A second parallel run must replace old output, including the failed
        # invocation's report, without aliasing another invocation's scratch.
        for _ in range(2):
            output = run_command(
                CTEST_COMMAND,
                "--test-dir",
                str(test_directory),
                "--build-config",
                CONFIGURATION,
                "--parallel",
                "4",
                "--output-on-failure",
                "--no-tests=error",
                expect_failure=True,
            )
            self.assertIn("1 tests failed out of 6", output)
            self.assertIn("Intentional fixture failure", output)
            for report, name in reports.items():
                self.assertEqual(report.read_text(), name + "\n")

    def test_results_follow_invocations_and_relocated_scratch(self):
        with tempfile.TemporaryDirectory(prefix="cmake test results ") as temporary:
            root = Path(temporary)
            source, build, prefix = (
                root / name for name in ("source", "build", "install")
            )
            shutil.copytree(FIXTURE, source)
            configure_project(source, build)
            self.check_results(build, build / "test_tmpdir")
            install_project(build, prefix)
            relocated = root / "relocated install"
            prefix.rename(relocated)
            source.rename(root / "retired source")
            build.rename(root / "retired build")
            tests = relocated / "share/tests"
            with mock.patch.dict(os.environ, {"HRX_TEST_TMPDIR": ""}):
                self.check_results(tests, tests / "tmp")
            scratch = root / "custom scratch"
            with mock.patch.dict(os.environ, {"HRX_TEST_TMPDIR": str(scratch)}):
                self.check_results(tests, scratch)


if __name__ == "__main__":
    unittest.main()
