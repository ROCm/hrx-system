#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Smoke tests for the cross-lane dev.py command surface."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from build_tools.devtools import smoke_test_lib
except ModuleNotFoundError:
    import smoke_test_lib

CI_DRY_RUN_COMMANDS = (
    ("iree-bazel-cpu",),
    ("iree-bazel-cpu-asan",),
    ("iree-bazel-cpu-msan",),
    ("iree-bazel-cpu-tsan",),
    ("iree-bazel-cpu-ubsan",),
    ("iree-bazel-cpu-sanitizers",),
    ("iree-bazel-amdgpu",),
    ("iree-bazel-amdgpu-asan",),
    ("iree-bazel-amdgpu-msan",),
    ("iree-bazel-amdgpu-tsan",),
    ("iree-bazel-amdgpu-ubsan",),
    ("iree-bazel-amdgpu-sanitizers",),
    ("iree-bazel-vulkan",),
    ("iree-bazel-vulkan-asan",),
    ("iree-bazel-vulkan-msan",),
    ("iree-bazel-vulkan-tsan",),
    ("iree-bazel-vulkan-ubsan",),
    ("iree-bazel-vulkan-sanitizers",),
    ("iree-cmake-cpu",),
    ("iree-cmake-cpu-asan",),
    ("iree-cmake-cpu-msan",),
    ("iree-cmake-cpu-tsan",),
    ("iree-cmake-cpu-ubsan",),
    ("iree-cmake-cpu-sanitizers",),
    ("iree-cmake-sanitizer-smoke",),
    ("iree-cmake-amdgpu",),
    ("iree-cmake-amdgpu-asan",),
    ("iree-cmake-amdgpu-msan",),
    ("iree-cmake-amdgpu-tsan",),
    ("iree-cmake-amdgpu-ubsan",),
    ("iree-cmake-amdgpu-sanitizers",),
    ("iree-cmake-vulkan",),
    ("iree-cmake-vulkan-asan",),
    ("iree-cmake-vulkan-msan",),
    ("iree-cmake-vulkan-tsan",),
    ("iree-cmake-vulkan-ubsan",),
    ("iree-cmake-vulkan-sanitizers",),
)


def run_dry_run_scenario(checkout: Path) -> None:
    tool_root = checkout.parent / "external-tools"
    smoke_test_lib.run_dev_command(checkout, ["--dry-run", "bazel", "setup"])
    smoke_test_lib.run_dev_command(
        checkout, ["--dry-run", "bazel", "setup", "--system"]
    )
    smoke_test_lib.run_dev_command(
        checkout, ["--dry-run", "bazel", "setup", "--tool-root", str(tool_root)]
    )
    smoke_test_lib.run_dev_command(
        checkout,
        ["--dry-run", "bazel", "configure", "-DIREE_HAL_DRIVER_AMDGPU=OFF"],
    )
    smoke_test_lib.run_dev_command(checkout, ["bazel", "build", "-n", "--config=asan"])
    smoke_test_lib.run_dev_command(
        checkout,
        ["--dry-run", "bazel", "query", "kind(cc_library, //runtime/...)"],
    )
    smoke_test_lib.run_dev_command(
        checkout,
        ["--dry-run", "bazel", "cquery", "--output=files", "//runtime/..."],
    )
    smoke_test_lib.run_dev_command(
        checkout, ["--dry-run", "bazel", "info", "execution_root"]
    )
    smoke_test_lib.run_dev_command(
        checkout,
        [
            "--dry-run",
            "bazel",
            "run",
            "//runtime/src/iree/base:allocator_benchmark",
            "--",
            "--help",
        ],
    )
    smoke_test_lib.run_dev_command(
        checkout,
        ["--dry-run", "bazel", "try", "-e", "int main() { return 0; }"],
    )
    smoke_test_lib.run_dev_command(
        checkout,
        [
            "--dry-run",
            "bazel",
            "compile-commands",
            "//runtime/src/iree/base/...",
        ],
    )
    smoke_test_lib.run_dev_command(
        checkout,
        [
            "--dry-run",
            "bazel",
            "fuzz",
            "//runtime/src/iree/tokenizer:special_tokens_fuzz",
            "--",
            "-max_total_time=1",
        ],
    )
    smoke_test_lib.run_dev_command(
        checkout,
        [
            "--dry-run",
            "bazel",
            "fuzz",
            "//runtime/src/iree/tokenizer/...",
            "--",
            "-max_total_time=1",
        ],
    )
    smoke_test_lib.run_dev_command(
        checkout,
        [
            "--dry-run",
            "--cmake-build-dir",
            "build/smoke-cmake",
            "cmake",
            "configure",
            "-DIREE_HAL_DRIVER_AMDGPU=OFF",
            "-DLIBHRX_BUILD=OFF",
        ],
    )
    smoke_test_lib.run_dev_command(
        checkout,
        [
            "cmake",
            "build",
            "-n",
            "--cmake-build-dir",
            "build/smoke-cmake",
            "hrx",
        ],
    )
    smoke_test_lib.run_dev_command(
        checkout,
        [
            "--dry-run",
            "--cmake-build-dir",
            "build/smoke-cmake",
            "cmake",
            "compile-commands",
        ],
    )
    smoke_test_lib.run_dev_command(
        checkout,
        [
            "--dry-run",
            "--cmake-build-dir",
            "build/smoke-cmake",
            "cmake",
            "run",
            "iree::tools::iree-run-module",
            "--",
            "--help",
        ],
    )
    smoke_test_lib.run_dev_command(
        checkout,
        [
            "--dry-run",
            "--cmake-build-dir",
            "build/smoke-cmake",
            "cmake",
            "try",
            "-e",
            "int main() { return 0; }",
        ],
    )
    smoke_test_lib.run_dev_command(
        checkout,
        [
            "--dry-run",
            "--cmake-build-dir",
            "build/smoke-cmake",
            "cmake",
            "fuzz",
            "iree::tokenizer::special_tokens_fuzz",
            "--",
            "-max_total_time=1",
        ],
    )
    smoke_test_lib.run_dev_command(
        checkout, ["--dry-run", "bazel", "hook", "--profile", "ci"]
    )
    smoke_test_lib.run_dev_command(
        checkout, ["--dry-run", "cmake", "hook", "--profile", "paranoid"]
    )
    smoke_test_lib.run_dev_command(
        checkout, ["--dry-run", "bazel", "precommit", "--profile", "default"]
    )
    smoke_test_lib.run_dev_command(
        checkout, ["--dry-run", "cmake", "precommit", "--profile", "ci"]
    )
    smoke_test_lib.run_dev_command(
        checkout, ["--dry-run", "bazel", "presubmit", "--profile", "paranoid"]
    )
    smoke_test_lib.run_dev_command(
        checkout, ["--dry-run", "cmake", "presubmit", "--profile", "default"]
    )
    for command in CI_DRY_RUN_COMMANDS:
        smoke_test_lib.run_command(
            checkout,
            [sys.executable, "build_tools/devtools/ci.py", *command, "--dry-run"],
        )
    smoke_test_lib.assert_absent(checkout / ".bazelrc.configured")
    smoke_test_lib.assert_absent(checkout / ".venv")
    smoke_test_lib.assert_absent(checkout / ".iree")
    smoke_test_lib.assert_absent(checkout / ".tmp/iree-cmake-try")
    smoke_test_lib.assert_absent(checkout / "lefthook-local.yml")
    smoke_test_lib.assert_absent(tool_root)


def main() -> int:
    return smoke_test_lib.run_smoke(
        description="Run dev.py CLI smoke tests.",
        scenario_runner=run_dry_run_scenario,
    )


if __name__ == "__main__":
    sys.exit(main())
