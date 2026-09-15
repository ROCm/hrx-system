#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""libamdf project presubmit entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from build_tools.devtools import project_presubmit

PROJECT_NAME = "libamdf"
PROJECT_ROOT = "libamdf/"
CMAKE_TEST_REGEX = "^libamdf/"
CMAKE_TEST_LABEL_EXCLUDE_REGEX = "manual|runtime-resource="
BAZEL_RESOURCE_TAG_EXCLUDES = (
    "-iree-run-requirement=libamdf.resource.amd_gpu",
    "-iree-run-requirement=libamdf.resource.xdna",
)
GLOBAL_TEST_TRIGGERS = (
    "BUILD.bazel",
    "MODULE.bazel",
    ".bazelrc",
    ".bazel_to_cmake.cfg.py",
    "requirements",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run libamdf project presubmit.")
    project_presubmit.add_common_arguments(parser, project_name=PROJECT_NAME)
    return parser.parse_args()


def run_command(command: list[str], description: str) -> bool:
    return project_presubmit.run_command(
        PROJECT_NAME, command, description, cwd=REPO_ROOT
    )


def dev_command(*arguments: str) -> list[str]:
    return [sys.executable, str(REPO_ROOT / "dev.py"), *arguments]


def is_global_trigger(path: str) -> bool:
    if "build_tools" in Path(path).parts:
        return True
    if path.startswith("requirements") and path.endswith(".txt"):
        return True
    return any(
        path == trigger or path.startswith(trigger) for trigger in GLOBAL_TEST_TRIGGERS
    )


def selected_files(files_from: str | None) -> list[str]:
    if not files_from:
        return []
    with open(files_from, encoding="utf-8") as file_list:
        return [line.strip() for line in file_list if line.strip()]


def should_run_tests(files_from: str | None) -> bool:
    paths = selected_files(files_from)
    if not paths:
        return files_from is None
    return any(
        path.startswith(PROJECT_ROOT) or is_global_trigger(path) for path in paths
    )


def bazel_test_command() -> list[str]:
    return dev_command(
        "bazel",
        "test",
        "--//libamdf/config:enabled=true",
        "--test_tag_filters=" + ",".join(BAZEL_RESOURCE_TAG_EXCLUDES),
        "//libamdf/...",
    )


def cmake_build_command(build_dir: Path) -> list[str]:
    return dev_command(
        "--cmake-build-dir",
        str(build_dir),
        "cmake",
        "build",
        "amdf",
        "amdf_static",
    )


def cmake_test_command(build_dir: Path) -> list[str]:
    return dev_command(
        "--cmake-build-dir",
        str(build_dir),
        "cmake",
        "test",
        "-R",
        CMAKE_TEST_REGEX,
        "-LE",
        CMAKE_TEST_LABEL_EXCLUDE_REGEX,
    )


def run_bazel_tests() -> bool:
    return run_command(bazel_test_command(), "Bazel tests")


def run_cmake_tests() -> bool:
    build_dir = project_presubmit.cmake_build_dir(REPO_ROOT)
    if not project_presubmit.validate_cmake_build_tree(PROJECT_NAME, build_dir):
        return False
    if project_presubmit.cmake_cache_value(build_dir, "AMDF_BUILD") != "ON":
        print(
            "libamdf presubmit: CMake build tree has AMDF_BUILD disabled; "
            "reconfigure with `python dev.py cmake configure --fresh "
            "-DAMDF_BUILD=ON`"
        )
        return False
    if not run_command(
        cmake_build_command(build_dir),
        "CMake build",
    ):
        return False
    return run_command(
        cmake_test_command(build_dir),
        "CTest tests",
    )


def main() -> int:
    args = parse_arguments()
    if not args.tests:
        return 0
    if not should_run_tests(args.files_from):
        print("libamdf presubmit: no libamdf-affecting files")
        return 0
    if args.lane == "bazel":
        ok = run_bazel_tests()
    elif args.lane == "cmake":
        ok = run_cmake_tests()
    else:
        raise ValueError(f"unknown lane: {args.lane}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
