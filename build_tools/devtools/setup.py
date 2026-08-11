# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Setup command planning."""

from __future__ import annotations

import sys
from pathlib import Path

from build_tools.devtools import hooks
from build_tools.devtools.aliases import alias_steps
from build_tools.devtools.command_plan import (
    CommandPlan,
    CommandStep,
    EnsureDirectoryStep,
)
from build_tools.devtools.environment import REPO_ROOT, ToolEnvironment, ToolMode


def common_setup_plan(
    tool_env: ToolEnvironment,
    *,
    include_docs: bool = False,
    platform_name: str | None = None,
) -> CommandPlan:
    plan = CommandPlan()
    if tool_env.mode == ToolMode.SYSTEM:
        if include_docs:
            raise ValueError("documentation setup requires a managed tool environment")
        plan.add(
            CommandStep(
                [sys.executable, str(REPO_ROOT / "dev.py"), "doctor", "--system"],
                cwd=REPO_ROOT,
                label="check system tools",
            )
        )
        return plan

    if tool_env.root is None or tool_env.bin_dir is None:
        raise ValueError("managed setup requires a tool environment root")

    plan.add(
        CommandStep(
            [sys.executable, "-m", "venv", str(tool_env.root)],
            cwd=REPO_ROOT,
            label=f"create {tool_env.mode.value} tool environment",
        )
    )
    plan.add(
        CommandStep(
            [
                tool_env.python,
                "-m",
                "pip",
                "install",
                "--require-hashes",
                "-r",
                str(REPO_ROOT / "requirements-dev.lock.txt"),
            ],
            cwd=REPO_ROOT,
            label="install Python developer tools",
        )
    )
    plan.add(hooks.lefthook_cli_compatibility_probe(tool_env))
    platform_name = sys.platform if platform_name is None else platform_name
    if platform_name != "win32":
        requirements = [
            "-r",
            str(REPO_ROOT / "requirements-analysis.lock.txt"),
        ]
        if include_docs:
            requirements.extend(
                [
                    "-r",
                    str(REPO_ROOT / "loom" / "docs" / "requirements.lock.txt"),
                ]
            )
        plan.add(
            CommandStep(
                [
                    tool_env.python,
                    "-m",
                    "pip",
                    "install",
                    "--require-hashes",
                    "--only-binary=:all:",
                    *requirements,
                ],
                cwd=REPO_ROOT,
                label=(
                    "install static-analysis and documentation Python tools"
                    if include_docs
                    else "install static-analysis tools"
                ),
            )
        )
    elif include_docs:
        plan.add(
            CommandStep(
                [
                    tool_env.python,
                    "-m",
                    "pip",
                    "install",
                    "--require-hashes",
                    "--only-binary=:all:",
                    "-r",
                    str(REPO_ROOT / "loom" / "docs" / "requirements.lock.txt"),
                ],
                cwd=REPO_ROOT,
                label="install documentation Python tools",
            )
        )
    if include_docs:
        plan.add(
            CommandStep(
                [
                    tool_env.python,
                    str(REPO_ROOT / "build_tools/devtools/install.py"),
                    "--group",
                    "docs",
                    "--bin-dir",
                    str(tool_env.bin_dir),
                ],
                cwd=REPO_ROOT,
                label="install documentation standalone tools",
            )
        )
        plan.add(
            CommandStep(
                [
                    tool_env.python,
                    str(REPO_ROOT / "build_tools/devtools/install.py"),
                    "--group",
                    "docs",
                    "--bin-dir",
                    str(tool_env.bin_dir),
                    "--check",
                ],
                cwd=REPO_ROOT,
                label="check documentation standalone tools",
            )
        )
    return plan


def setup_plan(
    lane: str,
    tool_env: ToolEnvironment,
    alias_dir: Path | None,
    *,
    platform_name: str | None = None,
) -> CommandPlan:
    if tool_env.mode == ToolMode.SYSTEM:
        plan = CommandPlan()
        plan.add(
            CommandStep(
                [sys.executable, str(REPO_ROOT / "dev.py"), lane, "doctor", "--system"],
                cwd=REPO_ROOT,
                label=f"check {lane} system tools",
            )
        )
        if alias_dir:
            plan.add(EnsureDirectoryStep(alias_dir))
            plan.extend(alias_steps(lane, alias_dir, sys.executable))
        return plan

    plan = common_setup_plan(tool_env, platform_name=platform_name)
    if tool_env.root is None or tool_env.bin_dir is None:
        raise ValueError("managed setup requires a tool environment root")
    plan.add(
        CommandStep(
            [
                tool_env.python,
                str(REPO_ROOT / "build_tools/devtools/install.py"),
                "--group",
                lane,
                "--bin-dir",
                str(tool_env.bin_dir),
            ],
            cwd=REPO_ROOT,
            label=f"install {lane} standalone tools",
        )
    )
    plan.add(
        CommandStep(
            [
                tool_env.python,
                str(REPO_ROOT / "build_tools/devtools/install.py"),
                "--group",
                lane,
                "--bin-dir",
                str(tool_env.bin_dir),
                "--check",
            ],
            cwd=REPO_ROOT,
            label=f"check {lane} standalone tools",
        )
    )
    target_alias_dir = alias_dir or tool_env.bin_dir
    plan.add(EnsureDirectoryStep(target_alias_dir))
    plan.extend(alias_steps(lane, target_alias_dir, tool_env.python))
    return plan
