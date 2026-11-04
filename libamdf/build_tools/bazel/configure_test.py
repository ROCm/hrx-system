# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import unittest

from build_tools.bazel import configure as bazel_configure
from libamdf.build_tools.bazel import configure as libamdf_configure


class LibamdfConfigureTest(unittest.TestCase):
    def generate(self, arguments: list[str]) -> str:
        return bazel_configure.generate_config(
            bazel_configure.parse_arguments(arguments)
        )

    def test_library_can_be_enabled_without_an_amd_hal_engine(self):
        config = self.generate(["-DIREE_HAL_DRIVER_AMD=OFF", "-DAMDF_BUILD=ON"])

        self.assertIn("build --//libamdf/config:enabled=true\n", config)
        self.assertIn("build --//runtime/config/hal:drivers=task\n", config)

    def test_every_family_subset_is_preserved_with_native_amd_hal(self):
        families = libamdf_configure.FAMILIES
        for mask in range(1 << len(families)):
            selected = [
                family for index, family in enumerate(families) if mask & (1 << index)
            ]
            arguments = ["-DIREE_HAL_DRIVER_AMD=ON"] + [
                f"-DAMDF_FAMILY_{family.upper()}="
                + ("ON" if family in selected else "OFF")
                for family in families
            ]
            with self.subTest(families=selected):
                config = self.generate(arguments)
                self.assertIn(
                    "build --//libamdf/config:families=" + ",".join(selected) + "\n",
                    config,
                )

    def test_native_library_and_family_options_are_independent(self):
        config = self.generate(
            [
                "-DIREE_HAL_DRIVER_AMD=OFF",
                "--//libamdf/config:enabled=true",
                "--//libamdf/config:families=cdna,xdna",
            ]
        )

        self.assertIn("build --//libamdf/config:enabled=true\n", config)
        self.assertIn("build --//libamdf/config:families=cdna,xdna\n", config)

    def test_explicit_disable_conflicts_with_native_amd_hal(self):
        with self.assertRaisesRegex(SystemExit, "conflicts with"):
            self.generate(["-DIREE_HAL_DRIVER_AMD=ON", "-DAMDF_BUILD=OFF"])

    def test_family_options_reject_unknown_values_and_mixed_spellings(self):
        for arguments in (
            ["--//libamdf/config:families=banana"],
            ["-DAMDF_FAMILY_CDNA=OFF", "--//libamdf/config:families=rdna"],
            ["-DAMDF_BUILD=ON", "--//libamdf/config:enabled=true"],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                self.generate(arguments)


if __name__ == "__main__":
    unittest.main()
