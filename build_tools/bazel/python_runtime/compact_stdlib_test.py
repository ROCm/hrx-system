# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import ntpath
import tempfile
import unittest
import zipfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import patch

from build_tools.bazel.python_runtime.compact_stdlib import (
    is_archive_member,
    write_standard_library_archive,
)
from build_tools.bazel.python_runtime.windows_runtime import (
    _configure_runtime,
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
                "site.py": "def main():\n    execsitecustomize()\n",
                "site-packages/ambient.py": "ambient = True\n",
                "ssl.py": "ssl = True\n",
            }
            for relative_path, contents in members.items():
                path = library_directory / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents.encode("utf-8"))

            first_archive = root / "first.zip"
            second_archive = root / "second.zip"
            runtime_module_path = root / "windows_runtime.py"
            runtime_module_path.write_bytes(b"runtime = True\n")
            write_standard_library_archive(
                library_directory, runtime_module_path, first_archive
            )
            write_standard_library_archive(
                library_directory, runtime_module_path, second_archive
            )

            self.assertEqual(first_archive.read_bytes(), second_archive.read_bytes())
            with zipfile.ZipFile(first_archive) as archive:
                self.assertEqual(
                    archive.namelist(),
                    [
                        "_hrx_windows_runtime.py",
                        "email/__init__.py",
                        "encodings/__init__.py",
                        "importlib/__init__.py",
                        "site.py",
                    ],
                )
                self.assertEqual(
                    archive.read("_hrx_windows_runtime.py"), b"runtime = True\n"
                )
                self.assertNotIn("sitecustomize.py", archive.namelist())
                site_contents = archive.read("site.py")
                self.assertIn(b"import _hrx_windows_runtime", site_contents)
                self.assertLess(
                    site_contents.index(b"import _hrx_windows_runtime"),
                    site_contents.index(b"execsitecustomize()"),
                )
                for archive_info in archive.infolist():
                    self.assertEqual(archive_info.date_time, (1980, 1, 1, 0, 0, 0))
                    self.assertEqual(archive_info.compress_type, zipfile.ZIP_STORED)

    def test_configures_runtime_identity_and_tcl_paths(self) -> None:
        cases = (
            (r"\\?\C:\runtime\python312.zip", r"C:\runtime"),
            (
                r"\\?\UNC\server\share\runtime\python312.zip",
                r"\\server\share\runtime",
            ),
        )
        for archive_path, runtime_root in cases:
            with self.subTest(archive_path=archive_path):
                system = SimpleNamespace(
                    base_prefix=r"C:\host-python",
                    base_exec_prefix=r"C:\host-python",
                )
                environment = {}
                with patch(
                    "build_tools.bazel.python_runtime.windows_runtime.os.path.realpath",
                    return_value=archive_path,
                ):
                    _configure_runtime(
                        archive_path + r"\_hrx_windows_runtime.py",
                        system,
                        environment,
                    )

                self.assertEqual(system.base_prefix, runtime_root)
                self.assertEqual(system.base_exec_prefix, runtime_root)
                self.assertEqual(
                    environment["TCL_LIBRARY"],
                    ntpath.join(runtime_root, "tcl", "tcl8.6"),
                )
                self.assertEqual(
                    environment["TK_LIBRARY"],
                    ntpath.join(runtime_root, "tcl", "tk8.6"),
                )


if __name__ == "__main__":
    unittest.main()
