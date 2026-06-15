#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom Python linting orchestrator.

Runs code generators, ruff (lint + format), and mypy on the loom Python
package. Used as a pre-commit hook and runnable standalone:

    python loom/build_tools/linters/loom_lint.py

Exit code is 0 only if all steps pass and no files were modified.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, REPO_ROOT)

from build_tools.devtools.source_lock import (
    NonEmptyTrackedFileSnapshot,
    source_mutation_lock,
)


def _run(description: str, cmd: list[str], **kwargs: object) -> bool:
    """Run a command, print status, return True on success."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode == 0:
        print(f"  PASS  {description}")
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                print(f"        {line}")
        return True
    print(f"  FAIL  {description}")
    for line in (result.stdout + result.stderr).strip().splitlines():
        print(f"        {line}")
    return False


def _run_lint() -> int:
    ok = True

    print("loom-lint: generators")
    ok &= _run(
        "python builder stubs",
        [
            sys.executable,
            "loom/py/loom/gen/run.py",
            "builders_pyi",
            "--in-place",
        ],
        cwd=REPO_ROOT,
    )
    ok &= _run(
        "package init files",
        [
            sys.executable,
            "loom/py/loom/gen/run.py",
            "package_inits",
            "--in-place",
        ],
        cwd=REPO_ROOT,
    )
    ok &= _run(
        "c errors",
        [sys.executable, "loom/py/loom/gen/run.py", "c_errors", "--check"],
        cwd=REPO_ROOT,
    )
    ok &= _run(
        "c tables",
        [sys.executable, "loom/py/loom/gen/run.py", "c_tables", "--check"],
        cwd=REPO_ROOT,
    )
    ok &= _run(
        "low descriptors",
        [sys.executable, "loom/py/loom/gen/run.py", "low_descriptors"],
        cwd=REPO_ROOT,
    )
    ok &= _run(
        "amdgpu target config",
        [sys.executable, "loom/build_tools/amdgpu/target_config.py", "--check"],
        cwd=REPO_ROOT,
    )
    ok &= _run(
        "x86 target profiles",
        [
            sys.executable,
            "loom/py/loom/gen/run.py",
            "x86_target_profiles",
            "--check",
        ],
        cwd=REPO_ROOT,
    )
    ok &= _run(
        "textmate",
        [sys.executable, "loom/py/loom/gen/run.py", "textmate"],
        cwd=REPO_ROOT,
    )
    ok &= _run(
        "source invariants",
        [sys.executable, "loom/build_tools/linters/loom_source_lint.py"],
        cwd=REPO_ROOT,
    )

    print("loom-lint: ruff")
    ok &= _run(
        "format",
        ["ruff", "format", "--cache-dir", ".ruff_cache", "loom/py/loom/"],
        cwd=REPO_ROOT,
    )
    ok &= _run(
        "lint",
        ["ruff", "check", "--fix", "--cache-dir", ".ruff_cache", "loom/py/loom/"],
        cwd=REPO_ROOT,
    )
    ok &= _run(
        "package init files after fixups",
        [
            sys.executable,
            "loom/py/loom/gen/run.py",
            "package_inits",
            "--check",
        ],
        cwd=REPO_ROOT,
    )

    print("loom-lint: mypy")
    ok &= _run(
        "type-check", ["mypy", "loom/"], cwd=os.path.join(REPO_ROOT, "loom", "py")
    )

    return 0 if ok else 1


def main() -> int:
    repo_root = Path(REPO_ROOT)
    with source_mutation_lock(repo_root, "loom-lint"):
        snapshot = NonEmptyTrackedFileSnapshot.capture_tracked_package_initializers(
            repo_root
        )
        result = _run_lint()
        if not snapshot.verify(repo_root):
            result = 1
        return result


if __name__ == "__main__":
    sys.exit(main())
