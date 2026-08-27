#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests for bazel_to_cmake.py."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bazel_to_cmake


class BazelToCMakeTest(unittest.TestCase):
    def test_rewrites_generated_cmake_with_lf_line_endings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_dir = root / "pkg"
            build_dir.mkdir()
            (build_dir / "BUILD.bazel").write_text(
                "package()\n", encoding="utf-8", newline="\n"
            )

            with (
                mock.patch.object(bazel_to_cmake, "repo_root", str(root)),
                mock.patch.object(bazel_to_cmake, "repo_cfg", object()),
                mock.patch.object(
                    bazel_to_cmake.bazel_to_cmake_converter,
                    "convert_build_file",
                    return_value="\nconverted_rule()\n",
                ),
            ):
                first_status = bazel_to_cmake.convert_directory(
                    str(build_dir),
                    write_files=True,
                    print_generated_content=False,
                    verbosity=0,
                )
                cmake_path = build_dir / "CMakeLists.txt"
                lf_content = cmake_path.read_bytes()
                cmake_path.write_bytes(lf_content.replace(b"\n", b"\r\n"))
                second_status = bazel_to_cmake.convert_directory(
                    str(build_dir),
                    write_files=True,
                    print_generated_content=False,
                    verbosity=0,
                )

            self.assertEqual(first_status, bazel_to_cmake.Status.UPDATED)
            self.assertEqual(second_status, bazel_to_cmake.Status.UPDATED)
            self.assertNotIn(b"\r\n", cmake_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
