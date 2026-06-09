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
    spec = importlib.util.spec_from_file_location("loom_presubmit", presubmit_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {presubmit_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LoomPresubmitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.presubmit = load_presubmit_module()

    def test_bazel_tests_exclude_runtime_resource_requirements(self):
        command = self.presubmit.bazel_test_command()

        self.assertEqual(command[:3], ["bazel", "test", "--config=presubmit"])
        self.assertEqual(command[-1], "//loom/...")

        tag_filter = next(
            arg for arg in command if arg.startswith("--test_tag_filters=")
        )
        self.assertIn("-iree-run-requirement=runtime.resource.amd_gpu", tag_filter)
        self.assertIn(
            "-iree-run-requirement=runtime.resource.vulkan_device", tag_filter
        )
        self.assertNotIn("loom.resource", tag_filter)

    def test_cmake_tests_exclude_runtime_resource_labels(self):
        self.assertEqual(
            self.presubmit.CTEST_RESOURCE_LABEL_EXCLUDE_REGEX,
            "runtime-resource=",
        )


if __name__ == "__main__":
    unittest.main()
