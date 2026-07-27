# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared bootstrap helpers for root-invocable Loom generators."""

from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root(start: str | Path | None = None) -> Path:
    """Finds the repository root containing both Loom Python and C trees."""
    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "loom" / "py" / "loom").is_dir() and (candidate / "loom" / "src" / "loom").is_dir():
            return candidate
    raise RuntimeError(f"could not find Loom repository root from {current}")


def ensure_runtime_py_on_path(start: str | Path | None = None) -> Path:
    """Finds the repo root and makes `import loom` work for direct scripts."""
    repo_root = find_repo_root(start)
    runtime_py = str(repo_root / "loom" / "py")
    if runtime_py not in sys.path:
        sys.path.insert(0, runtime_py)
    return repo_root


REPO_ROOT = find_repo_root(__file__)
