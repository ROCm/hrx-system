#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime project presubmit entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from build_tools.devtools import bazel_test_selection, project_presubmit

PROJECT_NAME = "runtime"
PROJECT_ROOT = "runtime/"
CMAKE_TEST_REGEX = "^iree/"
CMAKE_TEST_EXCLUDE_REGEX = "^iree/hal/driver/task/executable/elf/elf_module_test$"
GLOBAL_CMAKE_TRIGGERS = (
    "BUILD.bazel",
    "MODULE.bazel",
    ".bazelrc",
    ".bazel_to_cmake.cfg.py",
    "requirements",
)
BAZEL_TEST_EXCLUDES = (
    "-//runtime/src/iree/hal/driver/task/executable/elf:elf_module_test",
)


BAZEL_TESTS = bazel_test_selection.TestSuite(
    name=PROJECT_NAME,
    targets=(
        "//runtime/...",
        *BAZEL_TEST_EXCLUDES,
    ),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run runtime project presubmit.")
    project_presubmit.add_common_arguments(parser, project_name=PROJECT_NAME)
    return parser.parse_args()


def run_command(command: list[str], description: str) -> bool:
    return project_presubmit.run_command(
        PROJECT_NAME, command, description, cwd=REPO_ROOT
    )


def is_global_trigger(path: str) -> bool:
    if "build_tools" in Path(path).parts:
        return True
    if path.startswith("requirements") and path.endswith(".txt"):
        return True
    return any(
        path == trigger or path.startswith(trigger) for trigger in GLOBAL_CMAKE_TRIGGERS
    )


def selected_files(files_from: str | None) -> list[str]:
    if not files_from:
        return []
    with open(files_from, encoding="utf-8") as file_list:
        return [line.strip() for line in file_list if line.strip()]


def should_run_cmake_tests(files_from: str | None) -> bool:
    paths = selected_files(files_from)
    if not paths:
        return files_from is None
    return any(
        path.startswith(PROJECT_ROOT) or is_global_trigger(path) for path in paths
    )


def run_bazel_tests(files_from: str | None = None) -> bool:
    return BAZEL_TESTS.run(
        REPO_ROOT,
        paths=selected_files(files_from) if files_from is not None else None,
    )


def run_cmake_tests() -> bool:
    build_dir = project_presubmit.cmake_build_dir(REPO_ROOT)
    if not project_presubmit.validate_cmake_build_tree(PROJECT_NAME, build_dir):
        return False
    if not run_command(
        ["cmake", "--build", str(build_dir), "--parallel"],
        "CMake build",
    ):
        return False
    return run_command(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "--output-on-failure",
            "-R",
            CMAKE_TEST_REGEX,
            "-E",
            CMAKE_TEST_EXCLUDE_REGEX,
        ],
        "CTest tests",
    )


def main() -> int:
    args = parse_arguments()
    if not args.tests:
        return 0
    if args.lane == "bazel":
        ok = run_bazel_tests(args.files_from)
    elif args.lane == "cmake":
        ok = not should_run_cmake_tests(args.files_from) or run_cmake_tests()
    else:
        raise ValueError(f"unknown lane: {args.lane}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
