#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Smoke tests for the developer command launch boundaries."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from build_tools.devtools import smoke_test_lib
except ModuleNotFoundError:
    import smoke_test_lib


def file_contents(path: Path) -> bytes | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise RuntimeError(f"expected smoke state path to be a file: {path}")
    return path.read_bytes()


def run_dry_run_scenario(repo_root: Path) -> None:
    local_state_paths = (
        repo_root / ".bazelrc.configured",
        repo_root / "lefthook-local.yml",
        repo_root / ".iree/cmake_build_dir",
    )
    local_state = {path: file_contents(path) for path in local_state_paths}

    # Exercise the direct dev.py router and a plan containing file writes.
    smoke_test_lib.run_dev_command(
        repo_root, ["--dry-run", "bazel", "hook", "--profile", "ci"]
    )
    # Exercise one checked-in POSIX wrapper for each build-system lane. Unit
    # tests verify the complete wrapper-to-command mapping.
    smoke_test_lib.run_bin_wrapper(
        repo_root, "iree-bazel-build", ["-n", "--config=asan"]
    )
    smoke_test_lib.run_bin_wrapper(
        repo_root, "iree-cmake-build", ["-n", "hrx::hrx", "--parallel", "8"]
    )
    # Exercise the standalone CI entry point. Its command matrix is covered by
    # ci_test; the smoke only proves the process boundary and dry-run contract.
    smoke_test_lib.run_command(
        repo_root,
        [
            sys.executable,
            "build_tools/devtools/ci.py",
            "iree-bazel-cpu",
            "--dry-run",
        ],
    )

    for path, expected_contents in local_state.items():
        if file_contents(path) != expected_contents:
            raise RuntimeError(f"dry-run smoke changed local state file: {path}")


def main() -> int:
    return smoke_test_lib.run_smoke(
        description="Run developer command smoke tests.",
        scenario_runner=run_dry_run_scenario,
    )


if __name__ == "__main__":
    sys.exit(main())
