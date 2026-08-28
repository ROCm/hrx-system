# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared helpers for devtools smoke tests."""

from __future__ import annotations

import argparse
import difflib
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

try:
    from build_tools.devtools import environment
except ModuleNotFoundError:
    import environment


def find_repo_root() -> Path:
    bazel_workspace_directory = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if bazel_workspace_directory:
        return Path(bazel_workspace_directory).resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "dev.py").is_file() and (cwd / "build_tools/devtools").is_dir():
        return cwd
    return Path(__file__).resolve().parents[2]


REPO_ROOT = find_repo_root()


def parse_arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    return parser.parse_args()


def repository_status() -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO_ROOT,
        text=True,
    )


def assert_repository_status(expected: str) -> None:
    actual = repository_status()
    if actual == expected:
        return
    difference = "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile="before smoke",
            tofile="after smoke",
        )
    )
    raise RuntimeError(f"dry-run smoke changed repository state:\n{difference}")


def run_dev_command(repo_root: Path, args: list[str]) -> None:
    command = [sys.executable, "dev.py", *args]
    run_command(repo_root, command, env=smoke_python_environment())


def smoke_python_environment(*, remove_python_override: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONSAFEPATH", None)
    if remove_python_override:
        env.pop("PYTHON", None)
    return env


def run_command(
    working_directory: Path,
    command: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    print("smoke:", " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=working_directory,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        if result.stdout:
            print(
                result.stdout,
                file=sys.stderr,
                end="" if result.stdout.endswith("\n") else "\n",
            )
        result.check_returncode()


def run_bin_wrapper(repo_root: Path, wrapper_name: str, args: list[str]) -> None:
    env = smoke_python_environment(remove_python_override=True)
    wrapper_path = repo_root / "build_tools/bin" / wrapper_name
    command = [str(wrapper_path), *args]
    if os.name == "nt":
        bazel_sh = environment.find_windows_bazel_sh(env)
        if not bazel_sh:
            raise RuntimeError(
                "Git Bash is required to run the build_tools/bin wrappers on "
                "Windows; install Git for Windows or set BAZEL_SH"
            )
        # Generated Windows aliases invoke the selected tool-environment Python.
        # Mirror that contract when exercising the POSIX source wrappers.
        env[environment.BAZEL_SH_ENV] = bazel_sh
        env["PYTHON"] = sys.executable
        command = [bazel_sh, str(wrapper_path), *args]
    run_command(
        repo_root,
        command,
        env=env,
    )


def run_smoke(
    *,
    description: str,
    scenario_runner: Callable[[Path], None],
) -> int:
    parse_arguments(description)
    status = repository_status()
    try:
        scenario_runner(REPO_ROOT)
    finally:
        assert_repository_status(status)
    print(f"smoke: passed in {REPO_ROOT}")
    return 0
