# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from build_tools.cmake.test_environment import configured_cmake_arguments
from build_tools.devtools import ctest as ctest_dev
from build_tools.devtools.environment import REPO_ROOT

FIXTURE_SOURCE_DIR = REPO_ROOT / "build_tools/cmake/testdata/test_metadata"
CMAKE_COMMAND = os.environ["IREE_TEST_CMAKE_COMMAND"]
CTEST_COMMAND = os.environ["IREE_TEST_CTEST_COMMAND"]


class CTestIntegrationTest(unittest.TestCase):
    def test_selected_runner_builds_only_the_selected_closure(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            build_dir = Path(temporary_dir) / "build"
            subprocess.run(
                [
                    CMAKE_COMMAND,
                    "-S",
                    str(FIXTURE_SOURCE_DIR),
                    "-B",
                    str(build_dir),
                    *configured_cmake_arguments(),
                    f"-DIREE_REPO_ROOT={REPO_ROOT}",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            step = ctest_dev.CTestBuildAndRunStep(
                cmake=CMAKE_COMMAND,
                ctest=CTEST_COMMAND,
                build_dir=build_dir,
                arguments=["-R", "^source-only$"],
                cwd=REPO_ROOT,
            )
            self.assertEqual(step.run(), 0)
            self.assertFalse((build_dir / "host.built").exists())

            step = ctest_dev.CTestBuildAndRunStep(
                cmake=CMAKE_COMMAND,
                ctest=CTEST_COMMAND,
                build_dir=build_dir,
                arguments=["-R", "^host$"],
                cwd=REPO_ROOT,
            )
            self.assertEqual(step.run(), 0)
            self.assertTrue((build_dir / "host.built").is_file())
            self.assertFalse((build_dir / "benchmark.built").exists())

            step = ctest_dev.CTestBuildAndRunStep(
                cmake=CMAKE_COMMAND,
                ctest=CTEST_COMMAND,
                build_dir=build_dir,
                arguments=["-R", "^fixture-required$"],
                cwd=REPO_ROOT,
            )
            self.assertEqual(step.run(), 0)
            for test_name in (
                "fixture-setup",
                "fixture-required",
                "fixture-cleanup",
            ):
                self.assertTrue((build_dir / f"{test_name}.built").is_file())

            step = ctest_dev.CTestBuildAndRunStep(
                cmake=CMAKE_COMMAND,
                ctest=CTEST_COMMAND,
                build_dir=build_dir,
                arguments=["-R", "^tool-backed$"],
                cwd=REPO_ROOT,
            )
            self.assertEqual(step.run(), 0)
            self.assertTrue((build_dir / "tool.built").is_file())


if __name__ == "__main__":
    unittest.main()
