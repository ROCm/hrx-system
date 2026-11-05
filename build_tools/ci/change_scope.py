# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Selects a compiler-only test scope from a complete revision difference."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

DOCUMENTATION_PATHS = {
    ".github/workflows/build_loom_sdk.yml",
    ".github/workflows/build_loom_sdk_artifacts.yml",
    ".github/workflows/ci_iree.yml",
    "build_tools/ci/change_scope.py",
    "build_tools/ci/loom_wheel.py",
    "dev.py",
    "requirements-analysis.lock.txt",
    "requirements-dev.lock.txt",
}
DOCUMENTATION_PREFIXES = (
    "build_tools/devtools/",
    "loom/binding/c/",
    "loom/docs/",
    "loom/py/loom/",
    "loom/src/loom/editor/textmate/",
)


def loom_only(paths: list[str]) -> bool:
    # Build helpers and capability settings can affect consumers outside Loom.
    shared_paths = ("loom/BUILD.bazel", "loom/CMakeLists.txt")
    shared_prefixes = ("loom/build_tools/", "loom/config/", "loom/requirements/")
    return bool(paths) and all(
        path.startswith("loom/")
        and path not in shared_paths
        and not path.startswith(shared_prefixes)
        for path in paths
    )


def loom_docs_affected(paths: list[str]) -> bool:
    return any(
        path in DOCUMENTATION_PATHS or path.startswith(DOCUMENTATION_PREFIXES)
        for path in paths
    )


def changed_paths(repository: Path, base: str) -> list[str]:
    # Checkout's shallow history may omit the base of a multi-commit push.
    probe = subprocess.run(
        ["git", "cat-file", "--batch-check"],
        input=f"{base}\n",
        cwd=repository,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    if probe.stdout.rstrip() == f"{base} missing":
        subprocess.run(
            ["git", "fetch", "--no-tags", "--depth=1", "origin", base],
            cwd=repository,
            check=True,
        )
    result = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", "-z", base, "HEAD", "--"],
        cwd=repository,
        stdout=subprocess.PIPE,
        check=True,
    )
    # Both sides of renames participate, including paths removed from runtime.
    return [
        path.decode("utf-8", errors="surrogateescape")
        for path in result.stdout.split(b"\0")
        if path
    ]


def select_loom_only(repository: Path, base: str) -> bool:
    # Dispatches and initial branch pushes have no comparison; retain full CI.
    return bool(base.strip("0")) and loom_only(changed_paths(repository, base))


def select_loom_docs(repository: Path, base: str) -> bool:
    # Explicit dispatches have no comparison and are expected to produce the
    # complete SDK and documentation artifacts.
    return not bool(base.strip("0")) or loom_docs_affected(
        changed_paths(repository, base)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="", help="Base commit from the CI event.")
    arguments = parser.parse_args()
    base = arguments.base
    if base and not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", base):
        parser.error("--base must be a full commit object ID")
    comparison_available = bool(base.strip("0"))
    paths = changed_paths(Path.cwd(), base) if comparison_available else []
    loom_scoped = comparison_available and loom_only(paths)
    documentation_scoped = not comparison_available or loom_docs_affected(paths)
    print(f"loom_only={str(loom_scoped).lower()}")
    print(f"loom_docs={str(documentation_scoped).lower()}")


if __name__ == "__main__":
    main()
