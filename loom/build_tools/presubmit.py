#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom project presubmit entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from build_tools.devtools import project_presubmit
from build_tools.devtools.source_lock import NonEmptyTrackedFileSnapshot

PROJECT_NAME = "loom"
PROJECT_ROOT = "loom/"
CMAKE_TEST_REGEX = "^loom/"
GLOBAL_TEST_TRIGGERS = (
    "BUILD.bazel",
    "MODULE.bazel",
    ".bazelrc",
    ".bazel_to_cmake.cfg.py",
    "requirements",
)
RESOURCE_TEST_TAG_FILTERS = (
    "-iree-run-requirement=runtime.resource.amd_gpu",
    "-iree-run-requirement=runtime.resource.nvidia_gpu",
    "-iree-run-requirement=runtime.resource.vulkan_device",
    "-iree-run-requirement=runtime.resource.webgpu_device",
)
CTEST_RESOURCE_LABEL_EXCLUDE_REGEX = "runtime-resource="
CI_LOOM_TARGETS = "amdgpu,iree_vm,llvmir,spirv,x86"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Loom project presubmit.")
    mutation = parser.add_mutually_exclusive_group()
    mutation.add_argument("--fix", action="store_true", help="Accepted for symmetry.")
    mutation.add_argument("--check", action="store_true", help="Accepted for symmetry.")
    parser.add_argument(
        "--lane",
        choices=("bazel", "cmake"),
        default="bazel",
        help="Build-system lane used for tests. Defaults to bazel.",
    )
    parser.add_argument("--tests", action="store_true", help="Run Loom tests.")
    parser.add_argument(
        "--files-from",
        help="Path to a newline-separated repo-relative changed-file list.",
    )
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
    return [
        "bazel",
        "test",
        "--config=presubmit",
        f"--//loom/config/target:enable={CI_LOOM_TARGETS}",
        "--test_tag_filters=" + ",".join(RESOURCE_TEST_TAG_FILTERS),
        "//loom/...",
    ]


def run_bazel_tests() -> bool:
    return run_command(
        bazel_test_command(),
        "Bazel tests",
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
            "-LE",
            CTEST_RESOURCE_LABEL_EXCLUDE_REGEX,
        ],
        "CTest tests",
    )


def run_presubmit(args: argparse.Namespace) -> int:
    if not args.tests:
        return 0
    if not should_run_tests(args.files_from):
        print("loom presubmit: no Loom-affecting files")
        return 0
    if args.lane == "bazel":
        ok = run_bazel_tests()
    elif args.lane == "cmake":
        ok = run_cmake_tests()
    else:
        raise ValueError(f"unknown lane: {args.lane}")
    return 0 if ok else 1


def main() -> int:
    args = parse_arguments()
    snapshot = NonEmptyTrackedFileSnapshot.capture_tracked_package_initializers(
        REPO_ROOT
    )
    result = run_presubmit(args)
    if not snapshot.verify(REPO_ROOT):
        result = 1
    return result


if __name__ == "__main__":
    sys.exit(main())
