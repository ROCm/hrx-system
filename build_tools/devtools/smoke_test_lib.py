# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared helpers for devtools smoke tests."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path


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
    parser.add_argument(
        "--from-working-tree",
        action="store_true",
        help="Copy tracked and untracked working-tree files instead of cloning HEAD.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the temporary checkout for inspection.",
    )
    parser.add_argument(
        "--scenario",
        choices=("dry-run",),
        default="dry-run",
        help="Smoke scenario to run.",
    )
    return parser.parse_args()


def git_paths(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [path.decode("utf-8") for path in result.stdout.split(b"\0") if path]


def copy_working_tree(destination: Path) -> None:
    paths = git_paths("ls-files", "-z")
    paths += git_paths("ls-files", "-z", "--others", "--exclude-standard")
    for relative_path in paths:
        source_path = REPO_ROOT / relative_path
        if not source_path.is_file():
            continue
        destination_path = destination / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
    subprocess.run(["git", "init"], cwd=destination, check=True, stdout=subprocess.PIPE)


def clone_head(destination: Path) -> None:
    subprocess.run(
        ["git", "clone", "--local", "--no-hardlinks", str(REPO_ROOT), str(destination)],
        cwd=REPO_ROOT,
        check=True,
    )


def run_dev_command(checkout: Path, args: list[str]) -> None:
    command = [sys.executable, "dev.py", *args]
    run_command(checkout, command, env=smoke_python_environment())


def smoke_python_environment(*, remove_python_override: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONSAFEPATH", None)
    if remove_python_override:
        env.pop("PYTHON", None)
    return env


def run_command(
    checkout: Path,
    command: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    print("smoke:", " ".join(command), flush=True)
    subprocess.run(command, cwd=checkout, env=env, check=True)


def run_bin_wrapper(checkout: Path, wrapper_name: str, args: list[str]) -> None:
    env = smoke_python_environment(remove_python_override=True)
    run_command(
        checkout,
        [str(checkout / "build_tools/bin" / wrapper_name), *args],
        env=env,
    )


def assert_absent(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"dry-run smoke unexpectedly created {path}")


def run_smoke(
    *,
    description: str,
    scenario_runner: Callable[[Path], None],
) -> int:
    args = parse_arguments(description)
    if args.keep:
        temporary_root = Path(tempfile.mkdtemp(prefix="iree-x-dev-smoke-"))
        checkout = temporary_root / "checkout"
        _prepare_checkout(args, checkout)
        _run_scenario(args, checkout, scenario_runner)
        print(f"smoke: passed in {checkout}")
        print(f"smoke: kept {temporary_root}")
        return 0

    with tempfile.TemporaryDirectory(prefix="iree-x-dev-smoke-") as temporary_name:
        temporary_root = Path(temporary_name)
        checkout = temporary_root / "checkout"
        _prepare_checkout(args, checkout)
        _run_scenario(args, checkout, scenario_runner)
        print(f"smoke: passed in {checkout}")
        return 0


def _prepare_checkout(args: argparse.Namespace, checkout: Path) -> None:
    if args.from_working_tree:
        checkout.mkdir()
        copy_working_tree(checkout)
    else:
        clone_head(checkout)


def _run_scenario(
    args: argparse.Namespace,
    checkout: Path,
    scenario_runner: Callable[[Path], None],
) -> None:
    if args.scenario == "dry-run":
        scenario_runner(checkout)
        return
    raise ValueError(f"unknown smoke scenario: {args.scenario}")
