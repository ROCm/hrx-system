# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.devtools import install


class InstallTest(unittest.TestCase):
    def test_windows_amd64_assets_are_pinned_executables(self):
        with (
            mock.patch.object(install.platform, "system", return_value="Windows"),
            mock.patch.object(install.platform, "machine", return_value="AMD64"),
        ):
            platform_key = install.host_platform_key()

        self.assertEqual(platform_key, "windows-amd64")
        for tool in install.TOOLS.values():
            asset = tool.assets[platform_key]
            self.assertTrue(asset.binary_name.endswith(".exe"))
            self.assertIn("-windows-amd64.exe", asset.url)
            self.assertEqual(len(asset.sha256), 64)

    def test_windows_arm64_assets_are_available(self):
        with (
            mock.patch.object(install.platform, "system", return_value="Windows"),
            mock.patch.object(install.platform, "machine", return_value="ARM64"),
        ):
            platform_key = install.host_platform_key()

        self.assertEqual(platform_key, "windows-arm64")
        for tool in install.TOOLS.values():
            self.assertIn(platform_key, tool.assets)

    def test_windows_install_names_have_executable_suffixes(self):
        self.assertEqual(
            install.install_name("bazelisk", system="Windows"), "bazelisk.exe"
        )
        self.assertEqual(install.install_name("bazel", system="Windows"), "bazel.exe")
        self.assertEqual(
            install.install_name("buildifier", system="Windows"), "buildifier.exe"
        )
        self.assertEqual(install.install_name("tool.exe", system="Windows"), "tool.exe")

    def test_windows_venv_bin_dir_uses_scripts(self):
        self.assertEqual(
            install.venv_bin_dir(Path("venv"), system="Windows"),
            Path("venv") / "Scripts",
        )

    def test_windows_alias_is_a_copy(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            bin_dir = Path(temporary_directory)
            primary_path = bin_dir / "bazelisk.exe"
            alias_path = bin_dir / "bazel.exe"
            contents = b"pinned executable"
            primary_path.write_bytes(contents)
            expected_sha256 = hashlib.sha256(contents).hexdigest()

            install.install_alias(
                primary_path,
                alias_path,
                expected_sha256,
                system="Windows",
            )

            self.assertFalse(alias_path.is_symlink())
            self.assertEqual(alias_path.read_bytes(), contents)
            self.assertTrue(
                install.check_alias(primary_path, alias_path, expected_sha256)
            )

    def test_unresolvable_alias_is_invalid(self):
        primary_path = mock.Mock(spec=Path)
        alias_path = mock.Mock(spec=Path)
        primary_path.__eq__ = mock.Mock(return_value=False)
        alias_path.is_symlink.return_value = True
        alias_path.resolve.side_effect = OSError("link cannot be followed")

        self.assertFalse(install.check_alias(primary_path, alias_path, "unused"))


if __name__ == "__main__":
    unittest.main()
