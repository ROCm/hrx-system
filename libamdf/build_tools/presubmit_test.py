# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def load_presubmit_module():
    presubmit_path = Path(__file__).with_name("presubmit.py")
    spec = importlib.util.spec_from_file_location("libamdf_presubmit", presubmit_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {presubmit_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LibamdfPresubmitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.presubmit = load_presubmit_module()

    def test_bazel_tests_explicitly_enable_libamdf(self):
        command = self.presubmit.bazel_test_command()

        self.assertEqual(command[0], self.presubmit.sys.executable)
        self.assertEqual(command[1], str(self.presubmit.REPO_ROOT / "dev.py"))
        self.assertEqual(command[2:4], ["bazel", "test"])
        self.assertEqual(command.count("--config=presubmit"), 0)
        self.assertIn("--//libamdf/config:enabled=true", command)
        self.assertIn("//libamdf/...", command)

    def test_cmake_build_names_public_library_targets(self):
        build_dir = Path("build/libamdf")
        command = self.presubmit.cmake_build_command(build_dir)

        self.assertEqual(command[-4:], ["cmake", "build", "amdf", "amdf_static"])

    def test_cmake_tests_forward_selection_filters(self):
        build_dir = Path("build/libamdf")
        with mock.patch.multiple(
            self.presubmit,
            CMAKE_TEST_REGEX="TARGET_A",
            CMAKE_TEST_LABEL_EXCLUDE_REGEX="TARGET_B",
        ):
            command = self.presubmit.cmake_test_command(build_dir)

        self.assertEqual(
            command[-6:],
            ["cmake", "test", "-R", "TARGET_A", "-LE", "TARGET_B"],
        )

    def test_libamdf_change_runs_tests(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_list = Path(temporary_directory) / "files.txt"
            file_list.write_text("libamdf/include/amdf/amdf.h\n", encoding="utf-8")
            self.assertTrue(self.presubmit.should_run_tests(str(file_list)))

    def test_unrelated_change_skips_tests(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_list = Path(temporary_directory) / "files.txt"
            file_list.write_text("docs/unrelated.md\n", encoding="utf-8")
            self.assertFalse(self.presubmit.should_run_tests(str(file_list)))


if __name__ == "__main__":
    unittest.main()
