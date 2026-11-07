# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path, PurePosixPath

from build_tools.bazel.python_runtime.compact_stdlib import (
    is_archive_member,
    write_site_initialization_template,
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
                self.assertNotIn("sitecustomize.py", archive.namelist())
                self.assertEqual(
                    archive.read("site.py"),
                    members["site.py"].encode("utf-8"),
                )
                for archive_info in archive.infolist():
                    self.assertEqual(archive_info.date_time, (1980, 1, 1, 0, 0, 0))
                    self.assertEqual(archive_info.compress_type, zipfile.ZIP_STORED)

    def test_extends_rules_python_site_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            upstream_template = root / "upstream.py"
            upstream_template.write_bytes(
                b"_fixup_sys_base_executable()\n"
                b"_fixup_stdlib_paths()\n\n"
                b"COVERAGE_SETUP = _setup_sys_path()\n"
            )
            output_template = root / "compact.py"
            write_site_initialization_template(upstream_template, output_template)

            contents = output_template.read_text(encoding="utf-8")
            compile(contents, str(output_template), "exec")
            self.assertEqual(contents.count("_fixup_stdlib_paths()"), 1)
            self.assertEqual(contents.count("sys.path.insert(0, dlls_directory)"), 1)
            self.assertEqual(contents.count("os.add_dll_directory("), 1)
            self.assertIn("_get_windows_path_with_unc_prefix(", contents)
            self.assertIn('os.path.join(runtime_root, "DLLs")', contents)
            self.assertIn("_COMPACT_RUNTIME_DLL_DIRECTORY_HANDLE", contents)
            self.assertIn('os.environ["TCL_LIBRARY"]', contents)
            self.assertIn('os.environ["TK_LIBRARY"]', contents)
            self.assertLess(
                contents.index("_fixup_stdlib_paths()"),
                contents.index("sys.path.insert(0, dlls_directory)"),
            )
            self.assertLess(
                contents.index("sys.path.insert(0, dlls_directory)"),
                contents.index("os.add_dll_directory("),
            )
            self.assertLess(
                contents.index("os.add_dll_directory("),
                contents.index('os.environ["TCL_LIBRARY"]'),
            )
            self.assertLess(
                contents.index('os.environ["TK_LIBRARY"]'),
                contents.index("_COMPACT_RUNTIME_DLL_DIRECTORY_HANDLE"),
            )
            self.assertLess(
                contents.index("_COMPACT_RUNTIME_DLL_DIRECTORY_HANDLE"),
                contents.index("COVERAGE_SETUP = _setup_sys_path()"),
            )


if __name__ == "__main__":
    unittest.main()
