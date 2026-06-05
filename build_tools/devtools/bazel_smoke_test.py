#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Smoke tests for checked-in iree-bazel-* wrappers."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from build_tools.devtools import smoke_test_lib
except ModuleNotFoundError:
    import smoke_test_lib

BAZEL_WRAPPERS = (
    "iree-bazel-dev",
    "iree-bazel-configure",
    "iree-bazel-build",
    "iree-bazel-test",
    "iree-bazel-query",
    "iree-bazel-cquery",
    "iree-bazel-info",
    "iree-bazel-run",
    "iree-bazel-try",
    "iree-bazel-fuzz",
)


def run_dry_run_scenario(checkout: Path) -> None:
    for wrapper_name in BAZEL_WRAPPERS:
        smoke_test_lib.run_bin_wrapper(checkout, wrapper_name, ["--help"])
        smoke_test_lib.run_bin_wrapper(checkout, wrapper_name, ["--agents-md"])

    smoke_test_lib.run_bin_wrapper(
        checkout, "iree-bazel-build", ["-n", "--config=asan"]
    )
    smoke_test_lib.run_bin_wrapper(
        checkout, "iree-bazel-build", ["-n", "--", "--dry-run"]
    )
    smoke_test_lib.assert_absent(checkout / ".bazelrc.configured")
    smoke_test_lib.assert_absent(checkout / ".venv")
    smoke_test_lib.assert_absent(checkout / ".iree-bazel-try")
    smoke_test_lib.assert_absent(checkout / "lefthook-local.yml")


def main() -> int:
    return smoke_test_lib.run_smoke(
        description="Run iree-bazel-* wrapper smoke tests.",
        scenario_runner=run_dry_run_scenario,
    )


if __name__ == "__main__":
    sys.exit(main())
