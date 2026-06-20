# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def load_presubmit_module():
    presubmit_path = Path(__file__).with_name("presubmit.py")
    spec = importlib.util.spec_from_file_location("id4_presubmit", presubmit_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {presubmit_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Id4PresubmitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.presubmit = load_presubmit_module()

    def test_bazel_build_and_test_cover_id4_package(self):
        self.assertEqual(
            self.presubmit.bazel_build_command(),
            ["bazel", "build", "//experimental/id4/..."],
        )
        self.assertEqual(
            self.presubmit.bazel_test_command(),
            ["bazel", "test", "--config=presubmit", "//experimental/id4/..."],
        )

    def test_cmake_tests_select_id4_ctest_names(self):
        self.assertEqual(self.presubmit.CMAKE_TEST_REGEX, "^experimental/id4/")


if __name__ == "__main__":
    unittest.main()
