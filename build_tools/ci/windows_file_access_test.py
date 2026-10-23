# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from build_tools.ci import windows_diagnostics as diagnostics

# This dependency holds a real Win32 handle until the test explicitly releases
# it. Read/write sharing matches a readable file whose deletion is blocked.
FILE_HOLDER = r"""
import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.CreateFileW.argtypes = [
    wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
    wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
]
kernel.CreateFileW.restype = wintypes.HANDLE
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.CloseHandle.restype = wintypes.BOOL
handle = kernel.CreateFileW(
    sys.argv[1], 0x80000000, int(sys.argv[2]), None, 3,
    0x02000000 if Path(sys.argv[1]).is_dir() else 0, None
)
if handle == ctypes.c_void_p(-1).value:
    raise ctypes.WinError(ctypes.get_last_error())
try:
    print("ready", flush=True)
    sys.stdin.readline()
finally:
    if not kernel.CloseHandle(handle):
        raise ctypes.WinError(ctypes.get_last_error())
"""


@unittest.skipUnless(os.name == "nt", "requires native Windows file sharing")
class WindowsFileAccessTest(unittest.TestCase):
    def capture_records(self, root: Path, output: str, label: str) -> list[dict]:
        artifacts = root / label
        artifacts.mkdir()
        (artifacts / "output.log").write_text(output, encoding="utf-8")
        diagnostics.capture_failure(
            artifacts, root, os.environ, None, started_at=dt.datetime.now(dt.UTC)
        )
        manifest = json.loads((artifacts / "manifest.json").read_text())
        self.assertNotIn("capture_error", manifest)
        tool = next(
            entry
            for entry in manifest["tools"]
            if "windows-file-access" in entry["output"]
        )
        text = (artifacts / tool["output"]).read_text(encoding="utf-8")
        self.assertEqual(tool.get("returncode"), 0, (tool, text))
        return [json.loads(line) for line in text.splitlines()]

    def capture(self, root: Path, path: Path, label: str) -> tuple[dict, dict]:
        observations = self.capture_records(
            root,
            f"ERROR: Couldn't delete action output directory: {path} (Permission denied)\n",
            label,
        )
        access = observations[0]["file_access"]
        self.assertEqual(access["path"], str(path))
        users = observations[1]["resource_users"] if len(observations) > 1 else {}
        return access, users

    def test_live_resource_owner_and_access_recover_after_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "action stderr \N{GREEK SMALL LETTER LAMDA}"
            contents = b"compiler diagnostic\r\n"
            path.write_bytes(contents)
            for sharing, expected_read in ((3, 0), (0, 32)):
                with self.subTest(sharing=sharing):
                    child = subprocess.Popen(
                        [sys.executable, "-c", FILE_HOLDER, str(path), str(sharing)],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    try:
                        self.assertEqual(child.stdout.readline(), "ready\n")
                        access, users = self.capture(root, path, f"held-{sharing}")
                        self.assertEqual(access["read"]["winerror"], expected_read)
                        self.assertEqual(access["delete"]["winerror"], 32)
                        for operation in (
                            "start_session",
                            "register_resources",
                            "get_list",
                            "end_session",
                        ):
                            self.assertEqual(users[operation]["winerror"], 0, users)
                        owner = next(
                            entry
                            for entry in users["processes"]
                            if entry["pid"] == child.pid
                        )
                        self.assertGreater(owner["start_time_100ns"], 0)
                        self.assertTrue(owner["application_name"])
                    finally:
                        _, errors = child.communicate("release\n")
                        self.assertEqual(child.returncode, 0, errors)
                    access, users = self.capture(root, path, f"released-{sharing}")
                    self.assertEqual(access["read"]["winerror"], 0)
                    self.assertEqual(access["delete"]["winerror"], 0)
                    self.assertEqual(users["get_list"]["winerror"], 0, users)
                    self.assertNotIn(
                        child.pid, [entry["pid"] for entry in users["processes"]]
                    )
                    self.assertEqual(path.read_bytes(), contents)

    def test_missing_path_retains_native_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "absent"
            access, users = self.capture(root, path, "missing")
            self.assertEqual(access["read"]["winerror"], 2)
            self.assertEqual(access["delete"]["winerror"], 2)
            self.assertEqual(users, {})
            self.assertFalse(path.exists())

    def test_ctest_cleanup_records_locked_child_and_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / "test_tmpdir"
            path = tree / "nested test" / "locked \N{GREEK SMALL LETTER LAMDA}.txt"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"retained test output")
            child = subprocess.Popen(
                [sys.executable, "-c", FILE_HOLDER, str(path), "3"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(child.stdout.readline(), "ready\n")
                with self.assertRaises(OSError) as failure:
                    shutil.rmtree(tree)
                self.assertEqual(failure.exception.winerror, 32)
                records = self.capture_records(
                    root, f"Error removing directory {tree}\r\n", "ctest-held"
                )
                access = {
                    entry["file_access"]["path"]: entry["file_access"]
                    for entry in records
                    if "file_access" in entry
                }
                self.assertEqual(access[str(tree)]["delete"]["winerror"], 0)
                self.assertEqual(access[str(path)]["delete"]["winerror"], 32)
                users = next(
                    entry["resource_users"]
                    for entry in records
                    if "resource_users" in entry
                )
                self.assertEqual(users["paths"], [str(path)])
                self.assertIn(child.pid, [entry["pid"] for entry in users["processes"]])
                self.assertEqual(path.read_bytes(), b"retained test output")
            finally:
                _, errors = child.communicate("release\n")
                self.assertEqual(child.returncode, 0, errors)
            records = self.capture_records(
                root, f"Error removing directory {tree}\n", "ctest-released"
            )
            for entry in records:
                self.assertNotIn("resource_users", entry)
                self.assertEqual(entry["file_access"]["delete"]["winerror"], 0)
            self.assertEqual(path.read_bytes(), b"retained test output")

    def test_ctest_cleanup_records_locked_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / "test_tmpdir"
            tree.mkdir()
            child = subprocess.Popen(
                [sys.executable, "-c", FILE_HOLDER, str(tree), "3"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(child.stdout.readline(), "ready\n")
                with self.assertRaises(OSError) as failure:
                    tree.rmdir()
                self.assertEqual(failure.exception.winerror, 32)
                records = self.capture_records(
                    root, f"Error removing directory {tree}\n", "ctest-directory"
                )
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["file_access"]["path"], str(tree))
                self.assertEqual(records[0]["file_access"]["delete"]["winerror"], 32)
                self.assertTrue(tree.exists())
            finally:
                _, errors = child.communicate("release\n")
                self.assertEqual(child.returncode, 0, errors)

    def test_ctest_cleanup_observation_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / "test_tmpdir"
            tree.mkdir()
            for index in range(1024):
                (tree / f"test_{index}").mkdir()
            records = self.capture_records(
                root, f"Error removing directory {tree}\n", "ctest-large-tree"
            )
            self.assertEqual(sum("file_access" in entry for entry in records), 1024)
            self.assertEqual(
                records[-1], {"tree_limit": {"root": str(tree), "paths": 1024}}
            )
            self.assertEqual(len(list(tree.iterdir())), 1024)

    def test_ctest_cleanup_does_not_traverse_directory_junctions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.mkdir()
            (outside / "retained.txt").write_bytes(b"outside the failed tree")
            tree = root / "test_tmpdir"
            tree.mkdir()
            junction = tree / "directory junction"
            subprocess.run(
                ["cmd", "/d", "/c", "mklink", "/J", str(junction), str(outside)],
                check=True,
                capture_output=True,
            )
            for path in (tree, junction):
                with self.subTest(path=path):
                    records = self.capture_records(
                        root,
                        f"Error removing directory {path}\n",
                        "capture-" + path.name,
                    )
                    observed = [entry["file_access"]["path"] for entry in records]
                    self.assertEqual(
                        observed,
                        [str(tree), str(junction)] if path == tree else [str(junction)],
                    )
            self.assertEqual(
                (outside / "retained.txt").read_bytes(), b"outside the failed tree"
            )


if __name__ == "__main__":
    unittest.main()
