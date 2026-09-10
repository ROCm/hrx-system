# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest import mock


def load_presubmit_module():
    presubmit_path = Path(__file__).with_name("presubmit.py")
    spec = importlib.util.spec_from_file_location("libhrx_presubmit", presubmit_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {presubmit_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LibhrxPresubmitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.presubmit = load_presubmit_module()

    def test_bazel_tests_exclude_runtime_resource_requirements(self):
        command = self.presubmit.bazel_test_command()

        self.assertEqual(command[:3], ["bazel", "test", "--config=presubmit"])
        self.assertEqual(command[-1], "//libhrx/...")
        tag_filter = next(
            arg for arg in command if arg.startswith("--test_tag_filters=")
        )
        self.assertIn("-iree-run-requirement=runtime.resource.amd_gpu", tag_filter)

    def test_cmake_tests_exclude_runtime_resource_requirements(self):
        with (
            mock.patch.object(
                self.presubmit.project_presubmit,
                "validate_cmake_build_tree",
                return_value=True,
            ),
            mock.patch.object(
                self.presubmit.project_presubmit, "run_command", return_value=True
            ) as run_command,
        ):
            self.assertTrue(self.presubmit.run_cmake_tests())

        command = run_command.call_args.args[1]
        self.assertEqual(command[0], "ctest")
        self.assertEqual(command[command.index("-R") + 1], "^libhrx/")
        self.assertEqual(command[command.index("-LE") + 1], "runtime-resource=")


if __name__ == "__main__":
    unittest.main()
