# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def load_target_config_module():
    module_path = Path(__file__).with_name("target_config.py")
    spec = importlib.util.spec_from_file_location(
        "loom_amdgpu_target_config_test_subject",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TargetConfigMaintenanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target_config = load_target_config_module()

    def test_checked_in_file_set_owns_bazel_and_cmake_fragments(self):
        with (
            mock.patch.object(
                self.target_config,
                "render_bzl",
                return_value="bzl contents\n",
            ),
            mock.patch.object(
                self.target_config,
                "render_cmake",
                return_value="cmake contents\n",
            ),
        ):
            generated_file_set = self.target_config.checked_in_file_set(
                SimpleNamespace()
            )

        self.assertEqual(
            generated_file_set.output_paths,
            (
                "loom/build_tools/amdgpu/target_config.bzl",
                "loom/build_tools/amdgpu/target_config.cmake",
            ),
        )
        self.assertEqual(generated_file_set.obsolete_paths, ())

    def test_main_requires_an_explicit_maintenance_mode(self):
        with self.assertRaises(SystemExit) as context:
            self.target_config.main([])

        self.assertEqual(context.exception.code, 2)

    def test_main_selects_read_only_check_mode(self):
        with mock.patch.object(
            self.target_config,
            "maintain_checked_in_files",
            return_value=SimpleNamespace(ok=True),
        ) as maintain_checked_in_files:
            result = self.target_config.main(["--check"])

        self.assertEqual(result, 0)
        maintain_checked_in_files.assert_called_once_with("check")

    def test_main_selects_update_mode(self):
        with mock.patch.object(
            self.target_config,
            "maintain_checked_in_files",
            return_value=SimpleNamespace(ok=True),
        ) as maintain_checked_in_files:
            result = self.target_config.main(["--in-place"])

        self.assertEqual(result, 0)
        maintain_checked_in_files.assert_called_once_with("update")


if __name__ == "__main__":
    unittest.main()
