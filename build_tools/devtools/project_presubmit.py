# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared helpers for project-local presubmit entry points."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CMAKE_BUILD_DIR_ENV = "IREE_CMAKE_BUILD_DIR"


def cmake_build_dir(repo_root: Path) -> Path:
    configured_build_dir = os.environ.get(CMAKE_BUILD_DIR_ENV)
    if configured_build_dir:
        return Path(configured_build_dir)
    return repo_root.parent / "builds" / repo_root.name


def cmake_cache_value(build_dir: Path, key: str) -> str | None:
    cache_path = build_dir / "CMakeCache.txt"
    with open(cache_path, encoding="utf-8") as cache_file:
        for line in cache_file:
            name_and_type, separator, value = line.partition("=")
            if not separator:
                continue
            name, _, _type = name_and_type.partition(":")
            if name == key:
                return value.strip()
    return None


def validate_cmake_build_tree(project_name: str, build_dir: Path) -> bool:
    if not (build_dir / "CMakeCache.txt").is_file():
        print(
            f"{project_name} presubmit: CMake build tree is not configured; "
            "run `python dev.py cmake configure --fresh -GNinja` first"
        )
        return False
    generator = cmake_cache_value(build_dir, "CMAKE_GENERATOR")
    if generator == "Unix Makefiles":
        print(
            f"{project_name} presubmit: CMake build tree uses Unix Makefiles; "
            "reconfigure a Ninja build tree with "
            "`python dev.py cmake configure --fresh -GNinja`"
        )
        return False
    if (
        generator
        and generator.startswith("Ninja")
        and not (build_dir / "build.ninja").is_file()
    ):
        print(
            f"{project_name} presubmit: CMake build tree is incomplete or has "
            "mixed generator state; remove the stale build tree and rerun "
            "`python dev.py cmake configure --fresh -GNinja`"
        )
        return False
    return True


def run_command(
    project_name: str, command: list[str], description: str, *, cwd: Path
) -> bool:
    print(f"{project_name} presubmit: {description}")
    print("  " + " ".join(command))
    sys.stdout.flush()
    result = subprocess.run(command, cwd=cwd)
    if result.returncode == 0:
        return True
    print(
        f"{project_name} presubmit: {description} failed with exit code "
        f"{result.returncode}"
    )
    return False
