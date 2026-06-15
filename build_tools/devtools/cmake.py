# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CMake command helpers for dev.py."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from build_tools.devtools import cmake_file_api, cmake_fuzz, cmake_try
from build_tools.devtools.command_plan import (
    CommandPlan,
    CommandStep,
    EnsureDirectoryStep,
    WriteFileStep,
    quote_command,
)
from build_tools.devtools.environment import LOCAL_TMP_ROOT, REPO_ROOT, ToolEnvironment

CMAKE_BUILD_DIR_ENV = "IREE_CMAKE_BUILD_DIR"
CMAKE_STATE_DIR = LOCAL_TMP_ROOT / "iree"
CMAKE_BUILD_DIR_FILE = CMAKE_STATE_DIR / "cmake_build_dir"
DEFAULT_CMAKE_BUILD_DIR = REPO_ROOT / "build" / "cmake"


@dataclass(frozen=True)
class CMakeRunCommand:
    target: str
    print_path: bool = False
    program_args: list[str] = field(default_factory=list)
    run_cwd: Path = field(default_factory=Path.cwd)


@dataclass(frozen=True)
class CMakeCompileCommandsCommand:
    output: Path | None = None
    run_cwd: Path = field(default_factory=Path.cwd)


def split_program_args(arguments: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in arguments:
        return arguments, []
    separator_index = arguments.index("--")
    return arguments[:separator_index], arguments[separator_index + 1 :]


def parse_run_args(
    arguments: list[str],
    *,
    run_cwd: Path | None = None,
) -> CMakeRunCommand:
    tool_args, program_args = split_program_args(arguments)
    target = ""
    print_path = False
    for arg in tool_args:
        if arg in ("-p", "--print-path", "--print_path"):
            print_path = True
        elif not target:
            target = arg
        else:
            raise ValueError(
                f"unexpected cmake run argument {arg!r}; program args go after --"
            )
    if not target:
        raise ValueError("Target is required for cmake run")
    return CMakeRunCommand(
        target=target,
        print_path=print_path,
        program_args=program_args,
        run_cwd=run_cwd or Path.cwd(),
    )


def parse_compile_commands_args(
    arguments: list[str],
    *,
    run_cwd: Path | None = None,
) -> CMakeCompileCommandsCommand:
    output = None
    command_cwd = run_cwd or Path.cwd()

    index = 0
    while index < len(arguments):
        arg = arguments[index]
        if arg in ("-o", "--output"):
            index += 1
            if index >= len(arguments):
                raise ValueError(f"{arg} requires a path")
            output = Path(arguments[index])
        elif arg.startswith("--output="):
            output = Path(arg.split("=", 1)[1])
        else:
            raise ValueError(f"unexpected cmake compile-commands argument {arg!r}")
        index += 1

    if output is not None and not output.is_absolute():
        output = command_cwd / output

    return CMakeCompileCommandsCommand(
        output=output,
        run_cwd=command_cwd,
    )


def normalize_build_dir(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def read_configured_build_dir(state_file: Path = CMAKE_BUILD_DIR_FILE) -> Path | None:
    try:
        recorded_path = state_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if not recorded_path:
        return None
    return normalize_build_dir(Path(recorded_path))


def build_dir(
    configured_build_dir: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    state_file: Path = CMAKE_BUILD_DIR_FILE,
) -> Path:
    if configured_build_dir is not None:
        return normalize_build_dir(configured_build_dir)
    environ = os.environ if environ is None else environ
    env_build_dir = environ.get(CMAKE_BUILD_DIR_ENV)
    if env_build_dir:
        return normalize_build_dir(Path(env_build_dir).expanduser())
    recorded_build_dir = read_configured_build_dir(state_file)
    if recorded_build_dir is not None:
        return recorded_build_dir
    return DEFAULT_CMAKE_BUILD_DIR


def build_args(
    backend_args: list[str],
    *,
    configured_build_dir: Path | None = None,
) -> list[str]:
    requested_build_dir = build_dir(configured_build_dir)
    target_names = []
    raw_args = []
    for index, arg in enumerate(backend_args):
        if arg.startswith("-"):
            raw_args = backend_args[index:]
            break
        target_names.append(arg)
    else:
        raw_args = []

    cmake_args = []
    for target_name in target_names:
        target_name = cmake_file_api.resolve_target_name(
            requested_build_dir,
            target_name,
        )
        cmake_args.extend(["--target", target_name])
    cmake_args.extend(raw_args)
    return cmake_args


def configure_plan(
    tool_env: ToolEnvironment,
    *,
    configured_build_dir: Path | None,
    backend_args: list[str],
) -> CommandPlan:
    requested_build_dir = build_dir(configured_build_dir)
    codemodel_query_path = cmake_file_api.codemodel_query_path(requested_build_dir)
    return CommandPlan(
        [
            EnsureDirectoryStep(codemodel_query_path.parent),
            WriteFileStep(
                codemodel_query_path,
                "",
                label="request CMake File API codemodel",
            ),
            CommandStep(
                [
                    tool_env.tool("cmake"),
                    "-S",
                    str(REPO_ROOT),
                    "-B",
                    str(requested_build_dir),
                    *backend_args,
                ],
                cwd=REPO_ROOT,
                env=tool_env.path_env(),
                label="configure cmake",
            ),
            EnsureDirectoryStep(CMAKE_STATE_DIR),
            WriteFileStep(
                CMAKE_BUILD_DIR_FILE,
                str(requested_build_dir.resolve()) + "\n",
                label="record CMake build directory",
            ),
        ]
    )


@dataclass(frozen=True)
class CMakeRunStep:
    command: CMakeRunCommand
    build_dir: Path
    env: dict[str, str] | None = None

    def describe(self) -> str:
        lines = [
            f"# cmake run {self.command.target}",
            f"# resolve executable with CMake File API from {self.build_dir}",
        ]
        if self.command.print_path:
            lines.append("# print built executable path")
        else:
            lines.append(
                "exec "
                + quote_command(["<built executable>", *self.command.program_args])
            )
        return "\n".join(lines)

    def run(self, verbose: bool = False) -> int:
        try:
            target = cmake_file_api.resolve_executable(
                self.build_dir,
                self.command.target,
            )
        except cmake_file_api.FileApiError as exc:
            print(f"dev.py: {exc}", file=sys.stderr)
            return 1
        if not target.path.is_file():
            print(
                f"dev.py: built executable is missing: {target.path}\n"
                f"dev.py: run iree-cmake-build {self.command.target} first",
                file=sys.stderr,
            )
            return 1
        if self.command.print_path:
            print(target.path)
            return 0
        if not os.access(target.path, os.X_OK):
            print(
                f"dev.py: built output is not executable: {target.path}",
                file=sys.stderr,
            )
            return 1
        try:
            os.chdir(self.command.run_cwd)
            os.execvpe(
                str(target.path),
                [str(target.path), *self.command.program_args],
                self.env or os.environ,
            )
        except OSError as exc:
            print(
                "dev.py: failed to exec "
                + quote_command([str(target.path), *self.command.program_args])
                + f": {exc}",
                file=sys.stderr,
            )
            return 127
        raise AssertionError("os.execvpe returned unexpectedly")


def run_plan(
    tool_env: ToolEnvironment,
    *,
    configured_build_dir: Path | None,
    backend_args: list[str],
    run_cwd: Path | None = None,
) -> CommandPlan:
    command = parse_run_args(backend_args, run_cwd=run_cwd)
    return CommandPlan(
        [
            CMakeRunStep(
                command,
                build_dir(configured_build_dir),
                env=tool_env.path_env(),
            )
        ]
    )


def fuzz_plan(
    tool_env: ToolEnvironment,
    *,
    configured_build_dir: Path | None,
    backend_args: list[str],
) -> CommandPlan:
    return cmake_fuzz.fuzz_plan(
        tool_env,
        configured_build_dir=build_dir(configured_build_dir),
        backend_args=backend_args,
    )


@dataclass(frozen=True)
class CMakeCompileCommandsStep:
    command: CMakeCompileCommandsCommand
    build_dir: Path

    @property
    def source(self) -> Path:
        return self.build_dir / "compile_commands.json"

    @property
    def output_path(self) -> Path:
        return (self.command.output or self.source).resolve()

    def describe(self) -> str:
        lines = [f"# cmake compile-commands from {self.build_dir}"]
        if self.command.output is not None:
            lines.append(
                "cp " + quote_command([str(self.source), str(self.command.output)])
            )
        lines.append("print " + quote_command([str(self.output_path)]))
        return "\n".join(lines)

    def run(self, verbose: bool = False) -> int:
        if not self.source.is_file():
            print(
                f"dev.py: CMake compile_commands.json is missing: {self.source}",
                file=sys.stderr,
            )
            print("dev.py: run python dev.py cmake configure first", file=sys.stderr)
            return 1
        if self.command.output is not None:
            self.command.output.parent.mkdir(parents=True, exist_ok=True)
            if self.command.output.exists():
                self.command.output.chmod(self.command.output.stat().st_mode | 0o200)
            shutil.copyfile(self.source, self.command.output)
        print(self.output_path)
        return 0


def compile_commands_plan(
    tool_env: ToolEnvironment,
    *,
    configured_build_dir: Path | None,
    backend_args: list[str],
    run_cwd: Path | None = None,
) -> CommandPlan:
    del tool_env
    command = parse_compile_commands_args(backend_args, run_cwd=run_cwd)
    return CommandPlan(
        [
            CMakeCompileCommandsStep(
                command,
                build_dir(configured_build_dir),
            )
        ]
    )


def try_plan(
    tool_env: ToolEnvironment,
    *,
    configured_build_dir: Path | None,
    backend_args: list[str],
    run_cwd: Path | None = None,
) -> CommandPlan:
    return cmake_try.try_plan(
        tool_env,
        configured_build_dir=build_dir(configured_build_dir),
        backend_args=backend_args,
        run_cwd=run_cwd,
    )


def build_plan(
    tool_env: ToolEnvironment,
    *,
    configured_build_dir: Path | None,
    backend_args: list[str],
) -> CommandPlan:
    return CommandPlan(
        [
            CommandStep(
                [
                    tool_env.tool("cmake"),
                    "--build",
                    str(build_dir(configured_build_dir)),
                    *build_args(
                        backend_args,
                        configured_build_dir=configured_build_dir,
                    ),
                ],
                cwd=REPO_ROOT,
                env=tool_env.path_env(),
                label="cmake build",
            )
        ]
    )


def test_plan(
    tool_env: ToolEnvironment,
    *,
    configured_build_dir: Path | None,
    backend_args: list[str],
) -> CommandPlan:
    return CommandPlan(
        [
            CommandStep(
                [
                    tool_env.tool("ctest"),
                    "--test-dir",
                    str(build_dir(configured_build_dir)),
                    "--output-on-failure",
                    *backend_args,
                ],
                cwd=REPO_ROOT,
                env=tool_env.path_env(),
                label="cmake test",
            )
        ]
    )
