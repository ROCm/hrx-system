# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import unittest

from build_tools.bazel import configure as bazel_configure


class RuntimeConfigureTest(unittest.TestCase):
    def generate(self, arguments: list[str]) -> str:
        return bazel_configure.generate_config(
            bazel_configure.parse_arguments(arguments)
        )

    def test_amd_umbrella_defaults_and_explicit_opt_outs(self):
        for arguments, drivers, provider in (
            ([], "task,amd-gpu,amd-xdna", True),
            (["-DIREE_HAL_DRIVER_AMD_XDNA=OFF"], "task,amd-gpu", True),
            (["-DIREE_HAL_DRIVER_AMD_GPU=OFF"], "task,amd-xdna", True),
            (["-DIREE_HAL_DRIVER_AMD=OFF"], "task", False),
            (["--//runtime/config/hal:drivers=task"], "task", False),
        ):
            with self.subTest(arguments=arguments):
                config = self.generate(arguments)
                self.assertIn(
                    "build --//runtime/config/hal:drivers=" + drivers + "\n", config
                )
                self.assertIn(
                    "build --//libamdf/config:enabled=" + str(provider).lower() + "\n",
                    config,
                )

    def test_amd_engine_overrides_are_order_independent(self):
        for umbrella in (False, True):
            for gpu in (None, False, True):
                for xdna in (None, False, True):
                    defines = [f"-DIREE_HAL_DRIVER_AMD={'ON' if umbrella else 'OFF'}"]
                    expected = []
                    for engine, enabled in (("GPU", gpu), ("XDNA", xdna)):
                        if enabled is not None:
                            defines.append(
                                f"-DIREE_HAL_DRIVER_AMD_{engine}="
                                + ("ON" if enabled else "OFF")
                            )
                        effective = umbrella if enabled is None else enabled
                        if effective:
                            expected.append("amd-" + engine.lower())
                    for arguments in (defines, list(reversed(defines))):
                        with self.subTest(arguments=arguments):
                            config = self.generate(arguments)
                            self.assertIn(
                                "build --//runtime/config/hal:drivers="
                                + ",".join(["task", *expected])
                                + "\n",
                                config,
                            )
                            self.assertIn(
                                "build --//libamdf/config:enabled="
                                + str(bool(expected)).lower()
                                + "\n",
                                config,
                            )

    def test_native_driver_list_is_complete_and_requires_libamdf(self):
        for selection, expected in (
            ("amd", "amd-gpu,amd-xdna"),
            ("amd-gpu", "amd-gpu"),
            ("amd-xdna", "amd-xdna"),
            ("amd-gpu,amd-xdna", "amd-gpu,amd-xdna"),
        ):
            with self.subTest(selection=selection):
                config = self.generate(["--//runtime/config/hal:drivers=" + selection])
                self.assertIn(
                    "build --//runtime/config/hal:drivers=" + expected + "\n", config
                )
                self.assertIn("build --//libamdf/config:enabled=true\n", config)

    def test_disabled_amd_engines_do_not_require_libamdf(self):
        config = self.generate(
            [
                "-DIREE_HAL_DRIVER_AMD=ON",
                "-DIREE_HAL_DRIVER_AMD_GPU=OFF",
                "-DIREE_HAL_DRIVER_AMD_XDNA=OFF",
                "-DAMDF_BUILD=OFF",
            ]
        )
        self.assertIn("build --//runtime/config/hal:drivers=task\n", config)
        self.assertIn("build --//libamdf/config:enabled=false\n", config)

    def test_independent_driver_and_rdma_options_reach_native_settings(self):
        config = self.generate(
            [
                "-DIREE_HAL_DRIVER_VULKAN=ON",
                "-DIREE_HAL_DRIVER_WEBGPU=ON",
                "-DIREE_NET_RDMA=ON",
            ]
        )
        self.assertIn(
            "build --//runtime/config/hal:drivers=task,vulkan,webgpu,amd-gpu,amd-xdna\n",
            config,
        )
        self.assertIn("build --//runtime/config/net:rdma=true\n", config)

    def test_portable_and_native_runtime_spellings_conflict(self):
        for arguments in (
            ["-DIREE_HAL_DRIVER_VULKAN=ON", "--//runtime/config/hal:drivers=task"],
            ["-DIREE_NET_RDMA=ON", "--//runtime/config/net:rdma=false"],
        ):
            with (
                self.subTest(arguments=arguments),
                self.assertRaisesRegex(SystemExit, "Do not mix portable"),
            ):
                self.generate(arguments)

    def test_removed_driver_dialect_fails_with_replacement(self):
        with self.assertRaisesRegex(SystemExit, "was removed"):
            self.generate(["--enable-driver=amdgpu"])


if __name__ == "__main__":
    unittest.main()
