# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from build_tools.cmake.test_environment import configured_cmake_arguments

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_SOURCE_DIR = Path(__file__).resolve().parent / "testdata/test_metadata"
CMAKE_COMMAND = os.environ["IREE_TEST_CMAKE_COMMAND"]
CTEST_COMMAND = os.environ["IREE_TEST_CTEST_COMMAND"]


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


class CMakeTestMetadataTest(unittest.TestCase):
    def test_emits_exact_roots_for_common_test_shapes(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            build_dir = Path(temporary_dir) / "build"
            configure_result = configure_fixture(build_dir)
            self.assertEqual(
                configure_result.returncode,
                0,
                msg=configure_result.stdout,
            )

            ctest_result = subprocess.run(
                [
                    CTEST_COMMAND,
                    "--test-dir",
                    str(build_dir),
                    "--show-only=json-v1",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            test_model = json.loads(ctest_result.stdout)
            tests = {test["name"]: test for test in test_model["tests"]}
            build_target_catalog = json.loads(
                (build_dir / "iree_ctest_build_targets.json").read_text()
            )

            expected_roots = {
                "host": ["host_root"],
                "benchmark": ["benchmark_root"],
                "runtime-resource": ["runtime_resource_root"],
                "manual": ["manual_root"],
                "fixture-setup": ["fixture_setup_root"],
                "fixture-required": ["fixture_required_root"],
                "fixture-cleanup": ["fixture_cleanup_root"],
                "source-only": [],
                "tool-backed": ["tool_backed_root"],
            }
            self.assertEqual(
                build_target_catalog,
                {
                    "kind": "ireeCtestBuildTargets",
                    "version": 1,
                    "tests": expected_roots,
                },
            )
            self.assertEqual(set(tests), set(expected_roots))

            labels = {
                test_name: next(
                    prop["value"]
                    for prop in test["properties"]
                    if prop["name"] == "LABELS"
                )
                for test_name, test in tests.items()
                if any(prop["name"] == "LABELS" for prop in test["properties"])
            }
            self.assertEqual(labels["benchmark"], ["benchmark"])
            self.assertEqual(
                labels["runtime-resource"],
                ["runtime-resource=amd-gpu"],
            )
            self.assertEqual(labels["manual"], ["manual"])

    def test_rejects_repository_test_without_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            configure_result = configure_fixture(
                Path(temporary_dir) / "build",
                "-DIREE_TEST_ADD_UNMANAGED_TEST=ON",
            )

            self.assertNotEqual(configure_result.returncode, 0)
            self.assertIn(
                "repository CTest test is missing IREE_BUILD_TARGETS metadata: "
                "unmanaged",
                configure_result.stdout,
            )

    def test_rejects_missing_build_root_with_test_context(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            configure_result = configure_fixture(
                Path(temporary_dir) / "build",
                "-DIREE_TEST_ADD_MISSING_ROOT=ON",
            )

            self.assertNotEqual(configure_result.returncode, 0)
            self.assertIn(
                "CTest test missing-root has missing IREE_BUILD_TARGETS target: "
                "missing_root",
                configure_result.stdout,
            )

    def test_rejects_non_buildable_roots_with_test_context(self):
        invalid_roots = [
            (
                "-DIREE_TEST_ADD_ALIAS_ROOT=ON",
                "CTest test alias-root has non-buildable alias "
                "IREE_BUILD_TARGETS target: alias_root",
            ),
            (
                "-DIREE_TEST_ADD_IMPORTED_ROOT=ON",
                "CTest test imported-root has non-buildable imported "
                "IREE_BUILD_TARGETS target: imported_root",
            ),
            (
                "-DIREE_TEST_ADD_INTERFACE_ROOT=ON",
                "CTest test interface-root has non-buildable interface library "
                "IREE_BUILD_TARGETS target: interface_root",
            ),
        ]
        for cmake_arg, expected_message in invalid_roots:
            with self.subTest(cmake_arg=cmake_arg):
                with tempfile.TemporaryDirectory() as temporary_dir:
                    configure_result = configure_fixture(
                        Path(temporary_dir) / "build",
                        cmake_arg,
                    )

                    self.assertNotEqual(configure_result.returncode, 0)
                    self.assertIn(
                        expected_message,
                        " ".join(configure_result.stdout.split()),
                    )


if __name__ == "__main__":
    unittest.main()
