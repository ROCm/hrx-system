#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Smoke tests for checked-in iree-cmake-* wrappers."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from build_tools.devtools import smoke_test_lib
except ModuleNotFoundError:
    import smoke_test_lib

CMAKE_WRAPPERS = (
    "iree-cmake-dev",
    "iree-cmake-configure",
    "iree-cmake-build",
    "iree-cmake-test",
    "iree-cmake-run",
    "iree-cmake-try",
    "iree-cmake-fuzz",
)


def run_dry_run_scenario(checkout: Path) -> None:
    for wrapper_name in CMAKE_WRAPPERS:
        smoke_test_lib.run_bin_wrapper(checkout, wrapper_name, ["--help"])
        smoke_test_lib.run_bin_wrapper(checkout, wrapper_name, ["--agents-md"])

    smoke_test_lib.run_bin_wrapper(
        checkout,
        "iree-cmake-configure",
        ["-n", "-DIREE_HAL_DRIVER_AMDGPU=OFF", "-DLIBHRX_BUILD=OFF"],
    )
    smoke_test_lib.run_bin_wrapper(
        checkout, "iree-cmake-build", ["-n", "hrx::hrx", "--parallel", "8"]
    )
    smoke_test_lib.run_bin_wrapper(checkout, "iree-cmake-test", ["-n", "-R", "hrx"])
    smoke_test_lib.run_bin_wrapper(
        checkout,
        "iree-cmake-run",
        ["-n", "iree::tools::iree-run-module", "--", "--help"],
    )
    smoke_test_lib.run_bin_wrapper(
        checkout,
        "iree-cmake-try",
        ["-n", "-e", "int main() { return 0; }"],
    )
    smoke_test_lib.run_bin_wrapper(
        checkout,
        "iree-cmake-fuzz",
        ["-n", "iree::tokenizer::special_tokens_fuzz", "--", "-max_total_time=1"],
    )
    smoke_test_lib.assert_absent(checkout / ".bazelrc.configured")
    smoke_test_lib.assert_absent(checkout / ".venv")
    smoke_test_lib.assert_absent(checkout / ".iree")
    smoke_test_lib.assert_absent(checkout / ".iree-bazel-try")
    smoke_test_lib.assert_absent(checkout / ".iree-cmake-try")
    smoke_test_lib.assert_absent(checkout / "lefthook-local.yml")


def main() -> int:
    return smoke_test_lib.run_smoke(
        description="Run iree-cmake-* wrapper smoke tests.",
        scenario_runner=run_dry_run_scenario,
    )


if __name__ == "__main__":
    sys.exit(main())
