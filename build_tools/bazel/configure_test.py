# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path, PureWindowsPath
from unittest import mock

from build_tools.bazel.configure_api import as_bazel_path

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

    def test_direct_script_entrypoint_loads_project_contributors(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "configured.bazelrc"
            result = subprocess.run(
                [
                    sys.executable,
                    str(CONFIGURE_BAZEL_PATH),
                    "--output",
                    str(output),
                    "-DLOOM_TARGET_ALL_CPU=ON",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            config = output.read_text(encoding="utf-8")

        self.assertIn("Loom target selection: cpu", result.stdout)
        self.assertIn("# Loom target selection: cpu", config)
        self.assertIn("build --//loom/config/arch:enable=vm,x86", config)

    def test_mixed_project_request_renders_each_owner(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            rocm_root = Path(temporary_directory) / "rocm"
            (rocm_root / "include").mkdir(parents=True)
            config = self.configure_bazel.generate_config(
                self.configure_bazel.parse_arguments(
                    [
                        "-DIREE_HAL_DRIVER_AMDGPU=ON",
                        "-DIREE_HAL_DRIVER_AMD_XDNA=OFF",
                        "-DAMDF_FAMILY_CDNA=OFF",
                        "-DLOOM_TARGET_ALL_CPU=ON",
                        "-DLOOM_SERVE_IMAGE_INPUT=OFF",
                        "-DIREE_ENABLE_VULKAN=OFF",
                        f"-DIREE_ROCM_PATH={rocm_root}",
                    ]
                )
            )

        for expected in (
            "build --//runtime/config/hal:drivers=task,amdgpu,amd-gpu",
            "build --//libamdf/config:enabled=true",
            "build --//libamdf/config:families=rdna,xdna",
            "build --//build_tools/vulkan/config:enabled=false",
            "build --//loom/config/arch:enable=vm,x86",
            "build --//loom/config/serve:image_input=false",
            "common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm",
        ):
            self.assertIn(expected + "\n", config)

    def test_bazel_paths_use_forward_slashes_on_windows(self):
        self.assertEqual(
            as_bazel_path(PureWindowsPath(r"C:\home\runner\rocm-root")),
            "C:/home/runner/rocm-root",
        )

    def test_non_windows_host_does_not_probe_windows_capabilities(self):
        symbolic_link_probe = mock.Mock(return_value=False)

        self.configure_bazel.require_windows_bazel_host(
            platform_name="linux",
            symbolic_link_probe=symbolic_link_probe,
        )

        symbolic_link_probe.assert_not_called()

    def test_windows_host_accepts_required_capabilities(self):
        self.configure_bazel.require_windows_bazel_host(
            platform_name="win32",
            symbolic_link_probe=lambda: True,
        )

    def test_windows_host_reports_missing_symbolic_link_capability(self):
        with self.assertRaisesRegex(SystemExit, "Symbolic-link creation") as context:
            self.configure_bazel.require_windows_bazel_host(
                platform_name="win32",
                symbolic_link_probe=lambda: False,
            )

        message = str(context.exception)
        self.assertIn("Developer Mode", message)
        self.assertIn("Create symbolic links", message)
        self.assertIn("base image", message)

    def test_api_options_have_equivalent_native_and_portable_spellings(self):
        portable = self.configure_bazel.generate_config(
            self.configure_bazel.parse_arguments(
                ["-DIREE_ENABLE_VULKAN=ON", "-DIREE_ENABLE_D3D12=OFF"]
            )
        )
        native = self.configure_bazel.generate_config(
            self.configure_bazel.parse_arguments(
                [
                    "--//build_tools/vulkan/config:enabled=true",
                    "--//build_tools/d3d12/config:enabled=false",
                ]
            )
        )

        self.assertEqual(portable, native)
        self.assertIn("build --//build_tools/vulkan/config:enabled=true\n", portable)
        self.assertIn("build --//build_tools/d3d12/config:enabled=false\n", portable)

    def test_api_options_reject_mixed_or_invalid_spellings(self):
        for api in ("vulkan", "d3d12"):
            portable = f"-DIREE_ENABLE_{api.upper()}"
            native = f"--//build_tools/{api}/config:enabled"
            for arguments in (
                [portable + "=ON", native + "=false"],
                [portable + "=unknown"],
                [native + "=unknown"],
                [native],
            ):
                with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                    self.configure_bazel.generate_config(
                        self.configure_bazel.parse_arguments(arguments)
                    )

    def test_hal_driver_selection_does_not_mutate_api_selection(self):
        config = self.configure_bazel.generate_config(
            self.configure_bazel.parse_arguments(
                ["-DIREE_HAL_DRIVER_VULKAN=ON", "-DIREE_ENABLE_VULKAN=OFF"]
            )
        )

        self.assertIn("build --//runtime/config/hal:drivers=task,vulkan", config)
        self.assertIn("build --//build_tools/vulkan/config:enabled=false\n", config)

    def test_native_runtime_and_loom_target_lists_remain_independent(self):
        config = self.configure_bazel.generate_config(
            self.configure_bazel.parse_arguments(
                [
                    "--//runtime/src/iree/hal/drivers/amdgpu:targets=gfx942,gfx125X-all",
                    "--//loom/config/arch/amdgpu:targets=gfx942,gfx125X-all",
                ]
            )
        )

        self.assertIn(
            "build --//runtime/src/iree/hal/drivers/amdgpu:targets=gfx125X-all,gfx942\n",
            config,
        )
        self.assertIn(
            "build --//loom/config/arch/amdgpu:targets=gfx125X-all,gfx942\n",
            config,
        )

    def test_unknown_options_fail_at_the_public_boundary(self):
        for arguments in (
            ["-DCMAKE_BUILD_TYPE=Debug"],
            ["--//unknown/config:flag=true"],
            ["--repo_env=UNPUBLISHED_SETTING=value"],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                self.configure_bazel.generate_config(
                    self.configure_bazel.parse_arguments(arguments)
                )


if __name__ == "__main__":
    unittest.main()
