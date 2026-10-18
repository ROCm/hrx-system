# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from build_tools.ci import windows_stl_asan


class WindowsStlAsanTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.tools = self.root / "Program Files (x86)" / "MSVC" / "14.44.35207"
        self.output = self.root / "output"
        self.catalog_path = self.root / "catalog.json"
        self.payload = {
            "url": "https://download.visualstudio.microsoft.com/example.vsix",
            "sha256": "",
        }
        self.package = {
            "id": "Microsoft.VC.14.44.17.14.ASAN.X64.base",
            "version": "14.44.35226",
            "payloads": [self.payload],
        }
        self.catalog = {"packages": [self.package]}

    def archive(self, version="14.44.35207") -> bytes:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr(
                f"Contents/VC/Tools/MSVC/{version}/lib/x64/stl_asan.lib", b"library"
            )
            archive.writestr("Contents/unneeded.dll", b"unneeded")
        data = output.getvalue()
        self.payload["sha256"] = hashlib.sha256(data).hexdigest()
        return data

    def resolve(self) -> Path:
        self.catalog_path.write_text(json.dumps(self.catalog), encoding="utf-8")
        return windows_stl_asan.resolve_stl_asan(
            self.tools, self.catalog_path, self.output
        )

    def test_installed_companion_needs_neither_catalog_nor_network(self):
        library = self.tools / "lib" / "x64" / "stl_asan.lib"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"installed")
        with mock.patch.object(windows_stl_asan.urllib.request, "urlopen") as fetch:
            result = windows_stl_asan.resolve_stl_asan(
                self.tools, self.catalog_path, self.output
            )
        fetch.assert_not_called()
        self.assertEqual(result, self.output / self.tools.name / "x64")
        self.assertEqual((result / "stl_asan.lib").read_bytes(), b"installed")
        self.assertEqual(
            [path.name for path in self.output.rglob("*") if path.is_file()],
            ["stl_asan.lib"],
        )

    def test_verified_package_extracts_only_exact_companion(self):
        data = self.archive()
        # Another installed toolset's package is not a candidate.
        self.catalog["packages"].append(
            {"id": "Microsoft.VC.14.43.17.13.ASAN.X64.base"}
        )
        with mock.patch.object(
            windows_stl_asan.urllib.request, "urlopen", return_value=io.BytesIO(data)
        ):
            result = self.resolve()
        self.assertEqual((result / "stl_asan.lib").read_bytes(), b"library")
        self.assertEqual(
            [path.name for path in self.output.rglob("*") if path.is_file()],
            ["stl_asan.lib"],
        )
        self.assertFalse(self.tools.exists())

    def test_wrong_digest_preserves_previous_output(self):
        data = self.archive()
        library = self.output / self.tools.name / "x64" / "stl_asan.lib"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"previous")
        self.payload["sha256"] = "0" * 64
        with mock.patch.object(
            windows_stl_asan.urllib.request, "urlopen", return_value=io.BytesIO(data)
        ):
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.resolve()
        self.assertEqual(library.read_bytes(), b"previous")
        self.assertEqual(list(library.parent.iterdir()), [library])

    def test_wrong_tools_version_is_not_substituted(self):
        data = self.archive(version="14.44.99999")
        with mock.patch.object(
            windows_stl_asan.urllib.request, "urlopen", return_value=io.BytesIO(data)
        ):
            with self.assertRaisesRegex(KeyError, "14.44.35207"):
                self.resolve()
        self.assertFalse(list(self.output.rglob("*.lib")))

    def test_missing_or_ambiguous_package_fails_before_fetching(self):
        for packages in ([], [self.package, self.package]):
            with self.subTest(count=len(packages)):
                self.catalog["packages"] = packages
                with mock.patch.object(
                    windows_stl_asan.urllib.request, "urlopen"
                ) as fetch:
                    with self.assertRaisesRegex(ValueError, "Expected one"):
                        self.resolve()
                fetch.assert_not_called()

    def test_untrusted_payload_location_fails_before_fetching(self):
        for url in (
            "http://download.visualstudio.microsoft.com/example.vsix",
            "https://example.invalid/example.vsix",
        ):
            with self.subTest(url=url):
                self.payload["url"] = url
                with mock.patch.object(
                    windows_stl_asan.urllib.request, "urlopen"
                ) as fetch:
                    with self.assertRaisesRegex(ValueError, "Unexpected"):
                        self.resolve()
                fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
