# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared helpers for project-local presubmit entry points."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from build_tools.devtools import bazel as bazel_dev
from build_tools.devtools import cmake_file_api

CMAKE_BUILD_DIR_ENV = "IREE_CMAKE_BUILD_DIR"


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    project_name: str,
) -> None:
    """Adds the phases and inputs understood by every project presubmit."""
    mutation = parser.add_mutually_exclusive_group()
    mutation.add_argument(
        "--fix",
        action="store_true",
        help="Apply and stage supported project-local source maintenance.",
    )
    mutation.add_argument(
        "--check",
        action="store_true",
        help="Check project-local source maintenance without writing files.",
    )
    parser.add_argument(
        "--lane",
        choices=("bazel", "cmake"),
        default="bazel",
        help="Build-system lane used for tests. Defaults to bazel.",
    )
    parser.add_argument(
        "--hygiene",
        action="store_true",
        help=f"Run cheap {project_name} invariant checks.",
    )
    parser.add_argument(
        "--tests",
        action="store_true",
        help=f"Run {project_name} tests.",
    )
    parser.add_argument(
        "--files-from",
        help="Path to a newline-separated repo-relative changed-file list.",
    )


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


def build_and_resolve_executable(
    project_name: str,
    repo_root: Path,
    *,
    lane: str,
    bazel_target: str,
    cmake_target: str,
    bazel_args: Sequence[str] = (),
) -> Path | None:
    """Builds an executable target and returns its resolved artifact path."""
    executable_path: Path | None = None
    if lane == "bazel":
        bazel_args = tuple(bazel_args)
        if not run_command(
            project_name,
            ["bazel", "build", *bazel_args, bazel_target],
            f"Build Bazel executable {bazel_target}",
            cwd=repo_root,
        ):
            return None
        executable_path = bazel_dev.resolve_bazel_output_path(
            bazel="bazel",
            target=bazel_target,
            bazel_args=list(bazel_args),
            cwd=repo_root,
            env=None,
        )
        if executable_path is None:
            print(
                f"{project_name} presubmit: Bazel target {bazel_target} "
                "has no executable output",
                file=sys.stderr,
            )
            return None
    elif lane == "cmake":
        build_dir = cmake_build_dir(repo_root)
        if not validate_cmake_build_tree(project_name, build_dir):
            return None
        try:
            resolved_target = cmake_file_api.resolve_target_name(
                build_dir, cmake_target
            )
        except cmake_file_api.FileApiError as exc:
            print(f"{project_name} presubmit: {exc}", file=sys.stderr)
            return None
        if not run_command(
            project_name,
            [
                "cmake",
                "--build",
                str(build_dir),
                "--target",
                resolved_target,
                "--parallel",
            ],
            f"Build CMake executable {cmake_target}",
            cwd=repo_root,
        ):
            return None
        try:
            executable_path = cmake_file_api.resolve_executable(
                build_dir, cmake_target
            ).path
        except cmake_file_api.FileApiError as exc:
            print(f"{project_name} presubmit: {exc}", file=sys.stderr)
            return None
    else:
        raise ValueError(f"unknown lane: {lane}")

    if not executable_path.is_file():
        print(
            f"{project_name} presubmit: built executable is missing: {executable_path}",
            file=sys.stderr,
        )
        return None
    if not os.access(executable_path, os.X_OK):
        print(
            f"{project_name} presubmit: built output is not executable: "
            f"{executable_path}",
            file=sys.stderr,
        )
        return None
    return executable_path


def stage_changed_paths(
    project_name: str,
    repo_root: Path,
    paths: Sequence[str],
) -> bool:
    """Stages only named paths that still exist or are tracked deletions."""
    exact_paths = tuple(sorted(set(paths)))
    if not exact_paths:
        return True

    tracked_result = subprocess.run(
        ["git", "--literal-pathspecs", "ls-files", "-z", "--", *exact_paths],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if tracked_result.returncode != 0:
        print(
            f"{project_name} presubmit: failed to query tracked source updates",
            file=sys.stderr,
        )
        if tracked_result.stderr:
            print(
                tracked_result.stderr.decode("utf-8", errors="replace").rstrip(),
                file=sys.stderr,
            )
        return False

    tracked_paths = {path for path in tracked_result.stdout.split(b"\0") if path}
    stage_paths = [
        path
        for path in exact_paths
        if path.encode("utf-8") in tracked_paths
        or (repo_root / path).is_file()
        or (repo_root / path).is_symlink()
    ]
    if not stage_paths:
        return True
    return run_command(
        project_name,
        ["git", "--literal-pathspecs", "add", "--", *stage_paths],
        "Stage project source updates",
        cwd=repo_root,
    )
