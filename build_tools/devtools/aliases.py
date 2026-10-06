# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generated developer command aliases."""

from __future__ import annotations

import os
from pathlib import Path

from build_tools.devtools.command_plan import WriteFileStep
from build_tools.devtools.environment import REPO_ROOT, script_name

BAZEL_ALIASES = {
    "iree-bazel-dev": ["bazel"],
    "iree-bazel-configure": ["bazel", "configure"],
    "iree-bazel-build": ["bazel", "build"],
    "iree-bazel-test": ["bazel", "test"],
    "iree-bazel-query": ["bazel", "query"],
    "iree-bazel-cquery": ["bazel", "cquery"],
    "iree-bazel-aquery": ["bazel", "aquery"],
    "iree-bazel-info": ["bazel", "info"],
    "iree-bazel-shutdown": ["bazel", "shutdown"],
    "iree-bazel-run": ["bazel", "run"],
    "iree-bazel-try": ["bazel", "try"],
    "iree-bazel-fuzz": ["bazel", "fuzz"],
}

CMAKE_ALIASES = {
    "iree-cmake-dev": ["cmake"],
    "iree-cmake-configure": ["cmake", "configure"],
    "iree-cmake-build": ["cmake", "build"],
    "iree-cmake-test": ["cmake", "test"],
    "iree-cmake-run": ["cmake", "run"],
    "iree-cmake-try": ["cmake", "try"],
    "iree-cmake-fuzz": ["cmake", "fuzz"],
}


def alias_map_for_lane(lane: str) -> dict[str, list[str]]:
    if lane == "bazel":
        return BAZEL_ALIASES
    if lane == "cmake":
        return CMAKE_ALIASES
    raise ValueError(f"unknown lane: {lane}")


def alias_steps(
    lane: str, alias_dir: Path, python_executable: str
) -> list[WriteFileStep]:
    steps = []
    for name, args in alias_map_for_lane(lane).items():
        alias_path = alias_dir / script_name(name)
        if os.name == "nt":
            content = windows_alias_content(python_executable, args)
            executable = False
        else:
            content = posix_alias_content(python_executable, args)
            executable = True
        steps.append(
            WriteFileStep(
                path=alias_path,
                content=content,
                executable=executable,
                label=f"write {name} alias",
            )
        )
    return steps


def posix_alias_content(python_executable: str, args: list[str]) -> str:
    quoted_args = " ".join(_single_quote(arg) for arg in args)
    return (
        "#!/usr/bin/env sh\n"
        "set -eu\n"
        f"REPO_ROOT={_single_quote(str(REPO_ROOT))}\n"
        'CALLER_ROOT=$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || :)\n'
        'if [ -n "$CALLER_ROOT" ] && [ -f "$CALLER_ROOT/dev.py" ]; then\n'
        "  REPO_ROOT=$CALLER_ROOT\n"
        "fi\n"
        f'exec {_single_quote(python_executable)} "$REPO_ROOT/dev.py" {quoted_args} "$@"\n'
    )


def windows_alias_content(python_executable: str, args: list[str]) -> str:
    quoted_args = " ".join(_cmd_quote(arg) for arg in args)
    return (
        "@echo off\r\n"
        f'set "REPO_ROOT={REPO_ROOT}"\r\n'
        'set "CALLER_ROOT="\r\n'
        'for /f "delims=" %%I in (\'git -C "%CD%" rev-parse --show-toplevel 2^>nul\') do set "CALLER_ROOT=%%I"\r\n'
        'if defined CALLER_ROOT if exist "%CALLER_ROOT%\\dev.py" set "REPO_ROOT=%CALLER_ROOT%"\r\n'
        f'{_cmd_quote(python_executable)} "%REPO_ROOT%\\dev.py" {quoted_args} %*\r\n'
    )


def _single_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _cmd_quote(value: str) -> str:
    return '"' + value.replace('"', '\\"') + '"'
