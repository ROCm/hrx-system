# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Git hook command planning."""

from __future__ import annotations

from build_tools.devtools.command_plan import (
    CommandPlan,
    CommandStep,
    WriteFileStep,
    quote_command,
)
from build_tools.devtools.environment import REPO_ROOT, ToolEnvironment


def lefthook_cli_compatibility_probe(tool_env: ToolEnvironment) -> CommandStep:
    # Select a deliberately absent command so Lefthook parses the real
    # invocation and repository config without executing presubmit work.
    return CommandStep(
        [
            tool_env.tool("lefthook"),
            "run",
            "pre-commit",
            "--file",
            "dev.py",
            "--command",
            "__iree_cli_compatibility_probe__",
            "--no-auto-install",
            "--no-tty",
        ],
        cwd=REPO_ROOT,
        env=tool_env.path_env(),
        label="check Lefthook CLI compatibility",
    )


def hook_content(lane: str, profile: str, python_executable: str) -> str:
    if lane not in ("bazel", "cmake"):
        raise ValueError(f"unknown lane: {lane}")
    lane_name = {
        "bazel": "Bazel",
        "cmake": "CMake",
    }[lane]
    precommit_command = quote_command(
        [
            python_executable,
            str(REPO_ROOT / "dev.py"),
            lane,
            "precommit",
            "--profile",
            profile,
            "--commit",
            "--verbose",
        ]
    )
    commit_message_command = quote_command(
        [
            python_executable,
            str(REPO_ROOT / "build_tools/lefthook/commit_msg.py"),
            "{1}",
        ]
    )
    return f"""# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Local {lane_name}-lane hook policy.
# Installed by `python dev.py {lane} hook --profile {profile}`.
# Test-bearing commit-scope precommit profiles apply fixups before validation.
# Tool output streams live so long-running builds remain observable.

pre-commit:
  commands:
    precommit:
      run: >-
        {precommit_command}

commit-msg:
  commands:
    commit-message:
      run: >-
        {commit_message_command}
"""


def hook_plan(
    lane: str, tool_env: ToolEnvironment, verify: bool, profile: str
) -> CommandPlan:
    plan = CommandPlan()
    plan.add(
        WriteFileStep(
            path=REPO_ROOT / "lefthook-local.yml",
            content=hook_content(lane, profile, tool_env.python),
            label=f"select {lane} hook policy with {profile} profile",
        )
    )
    env = tool_env.path_env()
    plan.add(
        CommandStep(
            [tool_env.tool("lefthook"), "install"],
            cwd=REPO_ROOT,
            env=env,
            label="install Lefthook Git hooks",
        )
    )
    if verify:
        plan.add(
            CommandStep(
                [tool_env.tool("lefthook"), "run", "pre-commit", "--file", "dev.py"],
                cwd=REPO_ROOT,
                env=env,
                label="verify selected hook policy",
            )
        )
    return plan
