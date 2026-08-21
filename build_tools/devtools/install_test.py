# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import hashlib
import io
import os
import tarfile
import tempfile
import unittest
import zipfile
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
        for tool in (tool for tool in install.TOOLS.values() if tool.default):
            asset = tool.assets[platform_key]
            self.assertIsNotNone(asset.binary_name)
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
        for tool in (tool for tool in install.TOOLS.values() if tool.default):
            self.assertIn(platform_key, tool.assets)

    def test_docs_group_selects_documentation_build_tools(self):
        args = argparse.Namespace(list=False, group=["docs"], tools=[])

        self.assertEqual(
            set(install.selected_tools(args)),
            {"bazelisk", "doxygen"},
        )

    def test_extracts_zip_archive_files_with_recorded_hashes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "tool.zip"
            executable_contents = b"executable"
            companion_contents = b"companion"
            with zipfile.ZipFile(archive_path, mode="w") as archive:
                archive.writestr("release/bin/tool", executable_contents)
                archive.writestr("release/bin/companion", companion_contents)
            asset = install.ToolAsset(
                url="https://example.invalid/tool.zip",
                sha256=install.file_sha256(archive_path) or "",
                archive_format="zip",
                archive_files=(
                    install.ArchiveFile(
                        member_name="release/bin/tool",
                        install_name="tool",
                        sha256=hashlib.sha256(executable_contents).hexdigest(),
                        executable=True,
                    ),
                    install.ArchiveFile(
                        member_name="release/bin/companion",
                        install_name="companion",
                        sha256=hashlib.sha256(companion_contents).hexdigest(),
                    ),
                ),
            )

            install.extract_archive(asset, archive_path, root)

            self.assertEqual((root / "tool").read_bytes(), executable_contents)
            self.assertEqual((root / "companion").read_bytes(), companion_contents)
            if os.name != "nt":
                self.assertTrue((root / "tool").stat().st_mode & 0o100)

    def test_extracts_tar_archive_file_with_recorded_hash(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "tool.tar.gz"
            contents = b"executable"
            archive_info = tarfile.TarInfo("release/bin/tool")
            archive_info.size = len(contents)
            with tarfile.open(archive_path, mode="w:gz") as archive:
                archive.addfile(archive_info, io.BytesIO(contents))
            asset = install.ToolAsset(
                url="https://example.invalid/tool.tar.gz",
                sha256=install.file_sha256(archive_path) or "",
                archive_format="tar.gz",
                archive_files=(
                    install.ArchiveFile(
                        member_name="release/bin/tool",
                        install_name="tool",
                        sha256=hashlib.sha256(contents).hexdigest(),
                        executable=True,
                    ),
                ),
            )

            install.extract_archive(asset, archive_path, root)

            self.assertEqual((root / "tool").read_bytes(), contents)
            if os.name != "nt":
                self.assertTrue((root / "tool").stat().st_mode & 0o100)

    def test_archive_install_names_cannot_escape_bin_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "tool.zip"
            contents = b"executable"
            with zipfile.ZipFile(archive_path, mode="w") as archive:
                archive.writestr("release/bin/tool", contents)
            asset = install.ToolAsset(
                url="https://example.invalid/tool.zip",
                sha256=install.file_sha256(archive_path) or "",
                archive_format="zip",
                archive_files=(
                    install.ArchiveFile(
                        member_name="release/bin/tool",
                        install_name="../tool",
                        sha256=hashlib.sha256(contents).hexdigest(),
                    ),
                ),
            )

            with self.assertRaisesRegex(RuntimeError, "must be a file name"):
                install.extract_archive(asset, archive_path, root)

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
