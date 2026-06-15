# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Doctor command planning."""

from __future__ import annotations

from build_tools.devtools.command_plan import (
    CheckCommandStep,
    CommandPlan,
    CommandStep,
    OptionalCheckCommandStep,
)
from build_tools.devtools.environment import REPO_ROOT, ToolEnvironment

COMMON_TOOLS = (
    ("lefthook", "version", r"\b2\.1\.9\b"),
    ("ruff", "--version", r"\b0\.15\.15\b"),
    ("clang-format", "--version", r"\b22\.1\.3\b"),
)
SEMGREP_WARNING_FILTER = "ignore:pkg_resources is deprecated as an API:UserWarning"
OPTIONAL_COMMON_TOOLS = (
    (
        "semgrep",
        ("--disable-version-check", "--version"),
        r"\b1\.96\.0\b",
        "run python dev.py bazel setup --venv to install the managed tool environment",
        SEMGREP_WARNING_FILTER,
    ),
    ("clang-tidy", ("--version",), None, None, None),
    ("clang-apply-replacements", ("--version",), None, None, None),
)
LANE_TOOLS = {
    "bazel": (
        ("bazel", "--version", r"\b9\.1\.0\b"),
        ("buildifier", "--version", r"\b8\.5\.1\b"),
    ),
    "cmake": (
        ("cmake", "--version", None),
        ("ctest", "--version", None),
    ),
}


def doctor_plan(lane: str | None, tool_env: ToolEnvironment) -> CommandPlan:
    env = tool_env.path_env()
    plan = CommandPlan()
    plan.add(
        CommandStep(
            [tool_env.python, "--version"],
            cwd=REPO_ROOT,
            env=env,
            label="check Python",
        )
    )
    for tool, version_arg, expected_pattern in COMMON_TOOLS:
        plan.add(
            CheckCommandStep(
                [tool_env.tool(tool), version_arg],
                cwd=REPO_ROOT,
                env=env,
                expected_pattern=expected_pattern,
                label=f"check {tool}",
            )
        )
    for optional_tool in OPTIONAL_COMMON_TOOLS:
        tool, version_args, expected_pattern, hint, python_warnings = optional_tool
        step_env = env
        if python_warnings:
            step_env = {**env, "PYTHONWARNINGS": python_warnings}
        plan.add(
            OptionalCheckCommandStep(
                [tool_env.tool(tool), *version_args],
                cwd=REPO_ROOT,
                env=step_env,
                expected_pattern=expected_pattern,
                hint=hint,
                label=f"check optional {tool}",
            )
        )
    if lane:
        for tool, version_arg, expected_pattern in LANE_TOOLS[lane]:
            plan.add(
                CheckCommandStep(
                    [tool_env.tool(tool), version_arg],
                    cwd=REPO_ROOT,
                    env=env,
                    expected_pattern=expected_pattern,
                    label=f"check {tool}",
                )
            )
    plan.add(
        CommandStep(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            env=env,
            label="check Git worktree status",
        )
    )
    return plan
