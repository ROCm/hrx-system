# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path, PurePosixPath

from build_tools.bazel.python_runtime.compact_stdlib import (
    is_archive_member,
    write_standard_library_archive,
)


class CompactStandardLibraryTest(unittest.TestCase):
    def test_filters_runtime_cache_and_ambient_packages(self) -> None:
        self.assertTrue(is_archive_member(PurePosixPath("email/__init__.py")))
        self.assertFalse(
            is_archive_member(PurePosixPath("email/__pycache__/__init__.pyc"))
        )
        self.assertFalse(
            is_archive_member(PurePosixPath("site-packages/ambient/__init__.py"))
        )
        self.assertFalse(is_archive_member(PurePosixPath("ssl.py")))
        self.assertFalse(is_archive_member(PurePosixPath("module.pyo")))

    def test_writes_stable_standard_library_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            library_directory = root / "Lib"
            members = {
                "encodings/__init__.py": "encoding = True\n",
                "email/__init__.py": "email = True\n",
                "importlib/__init__.py": "importer = True\n",
                "site.py": "site = True\n",
                "site-packages/ambient.py": "ambient = True\n",
                "ssl.py": "ssl = True\n",
            }
            for relative_path, contents in members.items():
                path = library_directory / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")

            first_archive = root / "first.zip"
            second_archive = root / "second.zip"
            write_standard_library_archive(library_directory, first_archive)
            write_standard_library_archive(library_directory, second_archive)

            self.assertEqual(first_archive.read_bytes(), second_archive.read_bytes())
            with zipfile.ZipFile(first_archive) as archive:
                self.assertEqual(
                    archive.namelist(),
                    [
                        "email/__init__.py",
                        "encodings/__init__.py",
                        "importlib/__init__.py",
                        "site.py",
                    ],
                )
                for archive_info in archive.infolist():
                    self.assertEqual(archive_info.date_time, (1980, 1, 1, 0, 0, 0))
                    self.assertEqual(archive_info.compress_type, zipfile.ZIP_STORED)


if __name__ == "__main__":
    unittest.main()
