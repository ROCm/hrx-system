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
    parser.add_argument("--clangxx", type=Path)
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


def run_clang_tidy_fix_and_compile(
    *,
    clang_tidy: Path,
    clangxx: Path | None,
    plugin: Path,
    checks: str,
    source: Path,
    companion_files: list[Path] | None = None,
    compiler_args: list[str] | None = None,
    clang_tidy_args: list[str] | None = None,
) -> tuple[str, str, dict[str, str]]:
    if clangxx is None:
        raise ValueError("--clangxx is required when compiling fixed source")
    if companion_files is None:
        companion_files = []
    if compiler_args is None:
        compiler_args = ["-std=c++20"]
    if clang_tidy_args is None:
        clang_tidy_args = []
    with tempfile.TemporaryDirectory() as temp_dir:
        temporary_directory = Path(temp_dir)
        fixed_source = temporary_directory / source.name
        shutil.copy2(source, fixed_source)
        fixed_companions: dict[str, Path] = {}
        for companion_file in companion_files:
            fixed_companion = temporary_directory / companion_file.name
            shutil.copy2(companion_file, fixed_companion)
            fixed_companions[companion_file.name] = fixed_companion
        tidy = subprocess.run(
            [
                str(clang_tidy),
                f"--load={plugin}",
                f"--checks={checks}",
                *clang_tidy_args,
                "--fix",
                str(fixed_source),
                "--",
                *compiler_args,
                f"-I{temporary_directory}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output = tidy.stdout + tidy.stderr
        if tidy.returncode != 0:
            raise RuntimeError(output)
        compile_result = subprocess.run(
            [
                str(clangxx),
                *compiler_args,
                f"-I{temporary_directory}",
                "-fsyntax-only",
                str(fixed_source),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if compile_result.returncode != 0:
            raise RuntimeError(
                "fixed source failed to compile:\n"
                + compile_result.stdout
                + compile_result.stderr
                + "\nfixed source:\n"
                + fixed_source.read_text()
            )
        return (
            output,
            fixed_source.read_text(),
            {
                name: companion.read_text()
                for name, companion in fixed_companions.items()
            },
        )


class ClangTidyAssertions(unittest.TestCase):
    def assertContainsAll(self, output: str, expected_strings: list[str]) -> None:
        for expected in expected_strings:
            with self.subTest(expected=expected):
                self.assertIn(expected, output)

    def assertContainsNone(self, output: str, absent_strings: list[str]) -> None:
        for absent in absent_strings:
            with self.subTest(absent=absent):
                self.assertNotIn(absent, output)
