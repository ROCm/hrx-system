# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clang-tidy", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    args, unittest_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *unittest_args]
    return args


def source_path(test_file: str, relative_path: str) -> Path:
    return Path(test_file).resolve().with_name(relative_path)


def run_clang_tidy(
    *,
    clang_tidy: Path,
    plugin: Path,
    checks: str,
    source: Path,
    compiler_args: list[str] | None = None,
) -> str:
    if compiler_args is None:
        compiler_args = ["-std=c11"]
    completed = subprocess.run(
        [
            str(clang_tidy),
            f"--load={plugin}",
            f"--checks={checks}",
            str(source),
            "--",
            *compiler_args,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(output)
    return output


def run_clang_tidy_fix(
    *,
    clang_tidy: Path,
    plugin: Path,
    checks: str,
    source: Path,
    compiler_args: list[str] | None = None,
) -> tuple[str, str]:
    if compiler_args is None:
        compiler_args = ["-std=c11"]
    with tempfile.TemporaryDirectory() as temp_dir:
        fixed_source = Path(temp_dir) / source.name
        shutil.copy2(source, fixed_source)
        completed = subprocess.run(
            [
                str(clang_tidy),
                f"--load={plugin}",
                f"--checks={checks}",
                "--fix",
                str(fixed_source),
                "--",
                *compiler_args,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output = completed.stdout + completed.stderr
        if completed.returncode != 0:
            raise RuntimeError(output)
        return output, fixed_source.read_text()


class ClangTidyAssertions(unittest.TestCase):
    def assertContainsAll(self, output: str, expected_strings: list[str]) -> None:
        for expected in expected_strings:
            with self.subTest(expected=expected):
                self.assertIn(expected, output)

    def assertContainsNone(self, output: str, absent_strings: list[str]) -> None:
        for absent in absent_strings:
            with self.subTest(absent=absent):
                self.assertNotIn(absent, output)
