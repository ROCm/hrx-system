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
OPTIONAL_COMMON_TOOLS = (
    ("semgrep", "--version", r"\b1\.164\.0\b"),
    ("clang-tidy", "--version", None),
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
    for tool, version_arg, expected_pattern in OPTIONAL_COMMON_TOOLS:
        plan.add(
            OptionalCheckCommandStep(
                [tool_env.tool(tool), version_arg],
                cwd=REPO_ROOT,
                env=env,
                expected_pattern=expected_pattern,
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
