# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from build_tools.cmake.test_environment import configured_cmake_arguments

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_SOURCE_DIR = Path(__file__).resolve().parent / "testdata/data_dependencies"
CMAKE_COMMAND = os.environ["IREE_TEST_CMAKE_COMMAND"]


def configure_fixture(build_dir: Path, *cmake_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            CMAKE_COMMAND,
            "-S",
            str(FIXTURE_SOURCE_DIR),
            "-B",
            str(build_dir),
            *configured_cmake_arguments(),
            f"-DIREE_REPO_ROOT={REPO_ROOT}",
            *cmake_args,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


class CMakeDataDependenciesTest(unittest.TestCase):
    def test_resolves_later_target_and_copies_file_data(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            build_dir = Path(temporary_dir) / "build"
            configure_result = configure_fixture(build_dir)
            self.assertEqual(
                configure_result.returncode,
                0,
                msg=configure_result.stdout,
            )

            build_result = subprocess.run(
                [
                    CMAKE_COMMAND,
                    "--build",
                    str(build_dir),
                    "--target",
                    "data_consumer",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(build_result.returncode, 0, msg=build_result.stdout)
            self.assertEqual(
                (build_dir / "fixture.txt").read_text(encoding="utf-8"),
                "fixture data\n",
            )
            self.assertEqual(
                (build_dir / "generated.txt").read_text(encoding="utf-8"),
                "fixture data\n",
            )
            self.assertTrue(build_dir.joinpath("tool-built.marker").is_file())

    def test_rejects_missing_target_with_consumer_context(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            build_dir = Path(temporary_dir) / "build"
            configure_result = configure_fixture(
                build_dir,
                "-DIREE_TEST_DECLARE_TOOL=OFF",
            )

            self.assertNotEqual(configure_result.returncode, 0)
            self.assertIn(
                "IREE target data_consumer depends on missing target: fixture::tool",
                configure_result.stdout,
            )


if __name__ == "__main__":
    unittest.main()
