# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

CONFIGURE_BAZEL_PATH = Path(__file__).with_name("configure.py")


def load_configure_bazel_module():
    spec = importlib.util.spec_from_file_location(
        "configure_bazel", CONFIGURE_BAZEL_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {CONFIGURE_BAZEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ConfigureBazelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configure_bazel = load_configure_bazel_module()

    def make_rocm_root(self, temporary_directory: str) -> Path:
        rocm_root = Path(temporary_directory) / "rocm"
        (rocm_root / "include").mkdir(parents=True)
        return rocm_root

    def test_portable_project_options_configure_amdgpu(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            rocm_root = self.make_rocm_root(temporary_directory)

            args = self.configure_bazel.parse_arguments(
                [
                    "-DIREE_HAL_DRIVER_AMDGPU=ON",
                    f"-DIREE_ROCM_PATH={rocm_root}",
                ]
            )
            config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//runtime/config/hal:drivers=local-sync,local-task,null,amdgpu",
            config,
        )
        self.assertIn("common --repo_env=IREE_DEPENDENCY_MODE=pinned", config)
        self.assertIn("common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm", config)
        self.assertIn(f"common --repo_env=IREE_ROCM_PATH={rocm_root}", config)
        self.assertNotIn("--deleted_packages", config)

    def test_native_bazel_options_configure_amdgpu(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            rocm_root = self.make_rocm_root(temporary_directory)

            args = self.configure_bazel.parse_arguments(
                [
                    "--//runtime/config/hal:drivers=amdgpu,local-sync,local-task,null",
                    f"--repo_env=IREE_ROCM_PATH={rocm_root}",
                ]
            )
            config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//runtime/config/hal:drivers=local-sync,local-task,null,amdgpu",
            config,
        )
        self.assertIn("common --repo_env=IREE_DEPENDENCY_MODE=pinned", config)
        self.assertIn("common --repo_env=IREE_ROCM_DEPENDENCY_MODE=package", config)
        self.assertIn("common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm", config)
        self.assertIn(f"common --repo_env=IREE_ROCM_PATH={rocm_root}", config)
        self.assertNotIn("--deleted_packages", config)

    def test_rocm_dependency_mode_overrides_rocm_path_default(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            rocm_root = self.make_rocm_root(temporary_directory)

            args = self.configure_bazel.parse_arguments(
                [
                    "--//runtime/config/hal:drivers=amdgpu,local-sync,local-task,null",
                    "--repo_env=IREE_ROCM_DEPENDENCY_MODE=pinned",
                    f"--repo_env=IREE_ROCM_PATH={rocm_root}",
                ]
            )
            config = self.configure_bazel.generate_config(args)

        self.assertIn("common --repo_env=IREE_DEPENDENCY_MODE=pinned", config)
        self.assertIn("common --repo_env=IREE_ROCM_DEPENDENCY_MODE=pinned", config)
        self.assertIn("common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm", config)
        self.assertIn(f"common --repo_env=IREE_ROCM_PATH={rocm_root}", config)

    def test_portable_project_options_configure_hip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            rocm_root = self.make_rocm_root(temporary_directory)

            args = self.configure_bazel.parse_arguments(
                [
                    "-DIREE_HAL_DRIVER_HIP=ON",
                    f"-DIREE_ROCM_PATH={rocm_root}",
                ]
            )
            config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//runtime/config/hal:drivers=local-sync,local-task,null,hip",
            config,
        )
        self.assertIn("common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=none", config)
        self.assertIn(f"common --repo_env=IREE_ROCM_PATH={rocm_root}", config)
        self.assertNotIn("--deleted_packages", config)

    def test_portable_project_options_configure_amdgpu_and_hip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            rocm_root = self.make_rocm_root(temporary_directory)

            args = self.configure_bazel.parse_arguments(
                [
                    "-DIREE_HAL_DRIVER_AMDGPU=ON",
                    "-DIREE_HAL_DRIVER_HIP=ON",
                    f"-DIREE_ROCM_PATH={rocm_root}",
                ]
            )
            config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//runtime/config/hal:drivers=local-sync,local-task,null,amdgpu,hip",
            config,
        )
        self.assertIn("common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm", config)
        self.assertIn(f"common --repo_env=IREE_ROCM_PATH={rocm_root}", config)
        self.assertNotIn("--deleted_packages", config)

    def test_native_bazel_options_configure_hip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            rocm_root = self.make_rocm_root(temporary_directory)

            args = self.configure_bazel.parse_arguments(
                [
                    "--//runtime/config/hal:drivers=hip,local-sync,local-task,null",
                    f"--repo_env=IREE_ROCM_PATH={rocm_root}",
                ]
            )
            config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//runtime/config/hal:drivers=local-sync,local-task,null,hip",
            config,
        )
        self.assertIn("common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=none", config)
        self.assertIn(f"common --repo_env=IREE_ROCM_PATH={rocm_root}", config)
        self.assertNotIn("--deleted_packages", config)

    def test_portable_project_options_configure_webgpu(self):
        args = self.configure_bazel.parse_arguments(["-DIREE_HAL_DRIVER_WEBGPU=ON"])
        config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//runtime/config/hal:drivers=local-sync,local-task,null,webgpu",
            config,
        )
        self.assertIn("common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=none", config)
        self.assertNotIn("runtime/src/iree/hal/drivers/webgpu,", config)
        self.assertNotIn("runtime/src/iree/hal/drivers/webgpu/registration", config)

    def test_portable_project_options_configure_vulkan(self):
        args = self.configure_bazel.parse_arguments(["-DIREE_HAL_DRIVER_VULKAN=ON"])
        config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//runtime/config/hal:drivers=local-sync,local-task,null,vulkan",
            config,
        )
        self.assertIn("common --repo_env=IREE_DEPENDENCY_MODE=pinned", config)
        self.assertIn("common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=none", config)
        self.assertNotIn("IREE_ROCM_PATH", config)

    def test_environment_rocm_path_configures_amdgpu(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            rocm_root = self.make_rocm_root(temporary_directory)
            args = self.configure_bazel.parse_arguments(["-DIREE_HAL_DRIVER_AMDGPU=ON"])

            with mock.patch.dict(
                self.configure_bazel.os.environ,
                {"IREE_ROCM_PATH": str(rocm_root)},
                clear=True,
            ):
                config = self.configure_bazel.generate_config(args)

        self.assertIn("common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm", config)
        self.assertIn(f"common --repo_env=IREE_ROCM_PATH={rocm_root}", config)

    def test_environment_rocm_path_configures_hip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            rocm_root = self.make_rocm_root(temporary_directory)
            args = self.configure_bazel.parse_arguments(["-DIREE_HAL_DRIVER_HIP=ON"])

            with mock.patch.dict(
                self.configure_bazel.os.environ,
                {"IREE_ROCM_PATH": str(rocm_root)},
                clear=True,
            ):
                config = self.configure_bazel.generate_config(args)

        self.assertIn("common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=none", config)
        self.assertIn(f"common --repo_env=IREE_ROCM_PATH={rocm_root}", config)

    def test_pinned_rocm_driver_without_rocm_path_uses_no_device_toolchain(self):
        for driver_define in (
            "IREE_HAL_DRIVER_AMDGPU",
            "IREE_HAL_DRIVER_HIP",
        ):
            with self.subTest(driver_define=driver_define):
                args = self.configure_bazel.parse_arguments([f"-D{driver_define}=ON"])

                with mock.patch.dict(self.configure_bazel.os.environ, {}, clear=True):
                    config = self.configure_bazel.generate_config(args)

                self.assertIn("common --repo_env=IREE_DEPENDENCY_MODE=pinned", config)
                self.assertIn(
                    "common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=none",
                    config,
                )
                self.assertNotIn("IREE_ROCM_PATH", config)

    def test_package_rocm_driver_without_rocm_path_fails(self):
        args = self.configure_bazel.parse_arguments(
            [
                "-DIREE_HAL_DRIVER_AMDGPU=ON",
                "-DIREE_ROCM_DEPENDENCY_MODE=package",
            ]
        )

        with mock.patch.dict(self.configure_bazel.os.environ, {}, clear=True):
            with self.assertRaisesRegex(SystemExit, "package dependency mode"):
                self.configure_bazel.generate_config(args)

    def test_unknown_dependency_mode_fails(self):
        args = self.configure_bazel.parse_arguments(
            [
                "-DIREE_DEPENDENCY_MODE=banana",
            ]
        )

        with self.assertRaisesRegex(SystemExit, "IREE_DEPENDENCY_MODE"):
            self.configure_bazel.generate_config(args)

    def test_default_loom_scope_is_dependency_satisfied(self):
        args = self.configure_bazel.parse_arguments([])
        config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//loom/config/target:enable=amdgpu,iree_vm,llvmir,spirv,x86",
            config,
        )
        self.assertIn("build --//loom/config/execute:enable=iree_hal,iree_vm", config)
        self.assertIn("build --//loom/config/emit:enable=", config)

    def test_portable_loom_target_option_configures_target_scope(self):
        args = self.configure_bazel.parse_arguments(["-DLOOM_TARGET_WASM=ON"])
        config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//loom/config/target:enable=amdgpu,iree_vm,llvmir,spirv,wasm,x86",
            config,
        )
        self.assertIn("build --//loom/config/execute:enable=iree_hal,iree_vm", config)
        self.assertIn("build --//loom/config/emit:enable=", config)
        self.assertIn("common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=none", config)
        self.assertNotIn("IREE_ROCM_PATH", config)

    def test_portable_loom_target_option_removes_default_target(self):
        args = self.configure_bazel.parse_arguments(["-DLOOM_TARGET_AMDGPU=OFF"])
        config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//loom/config/target:enable=iree_vm,llvmir,spirv,x86",
            config,
        )
        self.assertIn("build --//loom/config/execute:enable=iree_hal,iree_vm", config)
        self.assertIn("build --//loom/config/emit:enable=", config)

    def test_portable_loom_execute_option_configures_execute_scope(self):
        args = self.configure_bazel.parse_arguments(["-DLOOM_EXECUTE_IREE_HAL=OFF"])
        config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//loom/config/target:enable=amdgpu,iree_vm,llvmir,spirv,x86",
            config,
        )
        self.assertIn("build --//loom/config/execute:enable=iree_vm", config)
        self.assertIn("build --//loom/config/emit:enable=", config)

    def test_portable_loom_llvmir_option_configures_explicit_emitter_scope(self):
        args = self.configure_bazel.parse_arguments(["-DLOOM_EMIT_LLVMIR=ON"])
        config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//loom/config/target:enable=amdgpu,iree_vm,llvmir,spirv,x86",
            config,
        )
        self.assertIn("build --//loom/config/execute:enable=iree_hal,iree_vm", config)
        self.assertIn("build --//loom/config/emit:enable=llvmir", config)

    def test_native_loom_target_execute_and_emit_options_configure_scope(self):
        args = self.configure_bazel.parse_arguments(
            [
                "--//loom/config/target:enable=amdgpu,iree_vm,spirv",
                "--//loom/config/execute:enable=iree_hal,iree_vm",
                "--//loom/config/emit:enable=llvmir",
            ]
        )
        config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//loom/config/target:enable=amdgpu,iree_vm,spirv",
            config,
        )
        self.assertIn("build --//loom/config/execute:enable=iree_hal,iree_vm", config)
        self.assertIn("build --//loom/config/emit:enable=llvmir", config)

    def test_portable_and_native_loom_target_options_conflict(self):
        args = self.configure_bazel.parse_arguments(
            [
                "-DLOOM_TARGET_AMDGPU=ON",
                "--//loom/config/target:enable=iree_vm",
            ]
        )

        with self.assertRaisesRegex(SystemExit, "Do not mix portable"):
            self.configure_bazel.generate_config(args)

    def test_portable_and_native_loom_execute_options_conflict(self):
        args = self.configure_bazel.parse_arguments(
            [
                "-DLOOM_EXECUTE_IREE_HAL=ON",
                "--//loom/config/execute:enable=iree_vm",
            ]
        )

        with self.assertRaisesRegex(SystemExit, "Do not mix portable"):
            self.configure_bazel.generate_config(args)

    def test_removed_driver_dialect_fails(self):
        args = self.configure_bazel.parse_arguments(["--enable-driver=amdgpu"])

        with self.assertRaisesRegex(SystemExit, "was removed"):
            self.configure_bazel.generate_config(args)

    def test_unknown_portable_define_fails(self):
        args = self.configure_bazel.parse_arguments(["-DCMAKE_BUILD_TYPE=Debug"])

        with self.assertRaisesRegex(SystemExit, "Unsupported Bazel configure option"):
            self.configure_bazel.generate_config(args)


if __name__ == "__main__":
    unittest.main()
