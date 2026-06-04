# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Presubmit and fix command planning."""

from __future__ import annotations

from pathlib import Path

from build_tools.devtools.command_plan import CommandPlan, CommandStep
from build_tools.devtools.environment import REPO_ROOT, ToolEnvironment

CMAKE_BUILD_DIR_ENV = "IREE_CMAKE_BUILD_DIR"
PROFILES = ("default", "paranoid", "ci")
PRESUBMIT_DEFAULT_PROFILE = "ci"
PRECOMMIT_DEFAULT_PROFILES = {
    "bazel": "paranoid",
    "cmake": "default",
}
TESTING_PROFILES = ("paranoid", "ci")


def build_system_name(lane: str) -> str:
    if lane == "bazel":
        return "Bazel"
    if lane == "cmake":
        return "CMake"
    raise ValueError(f"unknown lane: {lane}")


def precommit_default_profile(lane: str) -> str:
    try:
        return PRECOMMIT_DEFAULT_PROFILES[lane]
    except KeyError:
        raise ValueError(f"unknown lane: {lane}") from None


def profile_runs_tests(profile: str) -> bool:
    return profile in TESTING_PROFILES


def precommit_should_autofix(
    profile: str, commit: bool, staged: bool, paths: list[str] | None
) -> bool:
    return profile_runs_tests(profile) and (commit or staged or bool(paths))


def presubmit_plan(
    lane: str,
    tool_env: ToolEnvironment,
    profile: str,
    cmake_build_dir: Path | None = None,
    verbose: bool = False,
    project_tests: bool = True,
) -> CommandPlan:
    env = tool_env.path_env()
    label_name = build_system_name(lane)
    if cmake_build_dir is not None:
        env[CMAKE_BUILD_DIR_ENV] = str(cmake_build_dir)
    plan = CommandPlan()
    if lane == "bazel":
        command = [
            tool_env.python,
            str(REPO_ROOT / "build_tools/lefthook/presubmit.py"),
            "--lane",
            lane,
            "--profile",
            profile,
            "--check",
            "--all",
        ]
        if verbose:
            command.append("--verbose")
        if not project_tests:
            command.append("--no-project-tests")
        plan.add(
            CommandStep(
                command,
                cwd=REPO_ROOT,
                env=env,
                label=f"run {label_name} presubmit",
            )
        )
    elif lane == "cmake":
        command = [
            tool_env.python,
            str(REPO_ROOT / "build_tools/lefthook/presubmit.py"),
            "--lane",
            lane,
            "--profile",
            profile,
            "--check",
            "--all",
        ]
        if verbose:
            command.append("--verbose")
        if not project_tests:
            command.append("--no-project-tests")
        plan.add(
            CommandStep(
                command,
                cwd=REPO_ROOT,
                env=env,
                label=f"run {label_name} presubmit",
            )
        )
    else:
        raise ValueError(f"unknown lane: {lane}")
    return plan


def precommit_plan(
    lane: str,
    tool_env: ToolEnvironment,
    profile: str,
    base: str | None = None,
    commit: bool = False,
    cmake_build_dir: Path | None = None,
    staged: bool = False,
    paths: list[str] | None = None,
    verbose: bool = False,
) -> CommandPlan:
    if lane not in ("bazel", "cmake"):
        raise ValueError(f"unknown lane: {lane}")

    env = tool_env.path_env()
    label_name = build_system_name(lane)
    if cmake_build_dir is not None:
        env[CMAKE_BUILD_DIR_ENV] = str(cmake_build_dir)
    input_args: list[str] = []
    if paths:
        input_args += paths
    elif base is not None:
        input_args += ["--base", base]
    elif commit:
        input_args.append("--commit")
    elif staged:
        input_args.append("--staged")
    else:
        input_args.append("--changed")

    plan = CommandPlan()
    if precommit_should_autofix(profile, commit, staged, paths):
        command = [
            tool_env.python,
            str(REPO_ROOT / "build_tools/lefthook/presubmit.py"),
            "--lane",
            lane,
            "--fix",
            *input_args,
            "--profile",
            profile,
        ]
        if verbose:
            command.append("--verbose")
        plan.add(
            CommandStep(
                command,
                cwd=REPO_ROOT,
                env=env,
                label=f"apply {label_name} precommit fixups",
            )
        )

    command = [
        tool_env.python,
        str(REPO_ROOT / "build_tools/lefthook/presubmit.py"),
        "--lane",
        lane,
        "--check",
        *input_args,
        "--profile",
        profile,
    ]
    if verbose:
        command.append("--verbose")
    plan.add(
        CommandStep(
            command,
            cwd=REPO_ROOT,
            env=env,
            label=f"run {label_name} precommit",
        )
    )
    return plan


def fix_plan(tool_env: ToolEnvironment, verbose: bool = False) -> CommandPlan:
    env = tool_env.path_env()
    command = [
        tool_env.python,
        str(REPO_ROOT / "build_tools/lefthook/presubmit.py"),
        "--fix",
        "--staged",
        "--hygiene",
    ]
    if verbose:
        command.append("--verbose")
    return CommandPlan(
        [
            CommandStep(
                command,
                cwd=REPO_ROOT,
                env=env,
                label="apply staged mechanical fixups",
            )
        ]
    )
