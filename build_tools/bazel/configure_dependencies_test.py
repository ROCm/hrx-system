# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.bazel import configure as bazel_configure
from build_tools.bazel import configure_dependencies
from build_tools.bazel.configure_api import as_bazel_path, bazelrc_line


class ConfigureDependenciesTest(unittest.TestCase):
    def make_rocm_root(self, temporary_directory: str) -> Path:
        root = Path(temporary_directory) / "rocm"
        (root / "include").mkdir(parents=True)
        return root

    def generate(self, arguments: list[str]) -> str:
        return bazel_configure.generate_config(
            bazel_configure.parse_arguments(arguments)
        )

    def assert_rocm_path(self, config: str, root: Path) -> None:
        self.assertIn(
            bazelrc_line(
                "common",
                "--repo_env=IREE_ROCM_PATH=" + as_bazel_path(root),
            ),
            config,
        )

    def test_rocm_root_selects_package_headers_and_device_toolchain(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = self.make_rocm_root(temporary_directory)
            config = self.generate(
                [
                    "--//runtime/config/hal:drivers=amdgpu,task",
                    f"--repo_env=IREE_ROCM_PATH={root}",
                ]
            )

        self.assertIn("common --repo_env=IREE_DEPENDENCY_MODE=pinned\n", config)
        self.assertIn("common --repo_env=IREE_ROCM_DEPENDENCY_MODE=package\n", config)
        self.assertIn(
            "common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm\n", config
        )
        self.assert_rocm_path(config, root)

    def test_explicit_rocm_mode_overrides_the_path_default(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = self.make_rocm_root(temporary_directory)
            config = self.generate(
                [
                    "--//runtime/config/hal:drivers=amdgpu,task",
                    "--repo_env=IREE_ROCM_DEPENDENCY_MODE=pinned",
                    f"--repo_env=IREE_ROCM_PATH={root}",
                ]
            )

        self.assertIn("common --repo_env=IREE_ROCM_DEPENDENCY_MODE=pinned\n", config)
        self.assert_rocm_path(config, root)

    def test_legacy_amdgpu_uses_a_valid_environment_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = self.make_rocm_root(temporary_directory)
            with mock.patch.dict(
                configure_dependencies.os.environ,
                {"IREE_ROCM_PATH": str(root)},
                clear=True,
            ):
                config = self.generate(["-DIREE_HAL_DRIVER_AMDGPU=ON"])

        self.assertIn(
            "common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm\n", config
        )
        self.assert_rocm_path(config, root)

    def test_pinned_legacy_amdgpu_needs_no_rocm_root(self):
        with mock.patch.dict(configure_dependencies.os.environ, {}, clear=True):
            config = self.generate(["-DIREE_HAL_DRIVER_AMDGPU=ON"])

        self.assertIn("common --repo_env=IREE_DEPENDENCY_MODE=pinned\n", config)
        self.assertIn(
            "common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=none\n", config
        )
        self.assertNotIn("IREE_ROCM_PATH", config)

    def test_package_mode_requires_a_rocm_root(self):
        with mock.patch.dict(configure_dependencies.os.environ, {}, clear=True):
            with self.assertRaisesRegex(SystemExit, "package dependency mode"):
                self.generate(
                    [
                        "-DIREE_HAL_DRIVER_AMDGPU=ON",
                        "-DIREE_ROCM_DEPENDENCY_MODE=package",
                    ]
                )

    def test_dependency_modes_reject_unknown_values(self):
        for option in ("IREE_DEPENDENCY_MODE", "IREE_ROCM_DEPENDENCY_MODE"):
            with (
                self.subTest(option=option),
                self.assertRaisesRegex(SystemExit, option),
            ):
                self.generate([f"-D{option}=banana"])


if __name__ == "__main__":
    unittest.main()
