# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CMake libFuzzer command helpers."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from build_tools.devtools import cmake_file_api, fuzz
from build_tools.devtools.command_plan import CommandPlan, CommandStep, quote_command
from build_tools.devtools.environment import REPO_ROOT, ToolEnvironment


@dataclass(frozen=True)
class CMakeFuzzCommand:
    target: str
    build_args: list[str] = field(default_factory=list)
    fuzzer_args: list[str] = field(default_factory=list)


def split_program_args(arguments: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in arguments:
        return arguments, []
    separator_index = arguments.index("--")
    return arguments[:separator_index], arguments[separator_index + 1 :]


def parse_fuzz_args(arguments: list[str]) -> CMakeFuzzCommand:
    tool_args, fuzzer_args = split_program_args(arguments)
    target = ""
    build_args = []
    for arg in tool_args:
        if not target and not arg.startswith("-"):
            target = arg
        else:
            build_args.append(arg)
    if not target:
        raise ValueError("Target is required for cmake fuzz")
    return CMakeFuzzCommand(
        target=target,
        build_args=build_args,
        fuzzer_args=fuzzer_args,
    )


@dataclass(frozen=True)
class CMakeFuzzStep:
    cmake: str
    command: CMakeFuzzCommand
    build_dir: Path
    env: dict[str, str] | None = None

    def describe(self) -> str:
        resolved_target = cmake_file_api.resolve_target_name(
            self.build_dir,
            self.command.target,
        )
        return "\n".join(
            [
                f"# cmake fuzz {self.command.target}",
                quote_command(
                    [
                        self.cmake,
                        "--build",
                        str(self.build_dir),
                        "--target",
                        resolved_target,
                        *self.command.build_args,
                    ]
                ),
                "exec " + quote_command(["<built fuzzer>", "<corpus>"]),
            ]
        )

    def run(self, verbose: bool = False) -> int:
        if not cache_bool_enabled(self.build_dir, "IREE_ENABLE_FUZZING"):
            print(
                "dev.py: CMake build tree is not fuzz-enabled; run "
                "iree-cmake-configure -DIREE_ENABLE_FUZZING=ON first",
                file=sys.stderr,
            )
            return 1
        resolved_target = cmake_file_api.resolve_target_name(
            self.build_dir,
            self.command.target,
        )
        build_result = CommandStep(
            [
                self.cmake,
                "--build",
                str(self.build_dir),
                "--target",
                resolved_target,
                *self.command.build_args,
            ],
            cwd=REPO_ROOT,
            env=self.env,
            label="cmake fuzz build",
        ).run(verbose=verbose)
        if build_result != 0:
            return build_result

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
                f"dev.py: built fuzzer is missing: {target.path}",
                file=sys.stderr,
            )
            return 1
        if not os.access(target.path, os.X_OK):
            print(
                f"dev.py: built fuzzer is not executable: {target.path}",
                file=sys.stderr,
            )
            return 1

        target_dir = fuzz.cmake_fuzz_target_dir(self.command.target)
        corpus_dir = target_dir / "corpus"
        artifact_dir = target_dir / "artifacts"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            str(target.path),
            str(corpus_dir),
            f"-artifact_prefix={artifact_dir}/",
            *self.command.fuzzer_args,
        ]
        try:
            os.chdir(REPO_ROOT)
            os.execvpe(argv[0], argv, self.env or os.environ)
        except OSError as exc:
            print(
                f"dev.py: failed to exec {quote_command(argv)}: {exc}",
                file=sys.stderr,
            )
            return 127
        raise AssertionError("os.execvpe returned unexpectedly")


def cache_bool_enabled(build_dir: Path, name: str) -> bool:
    cache_path = build_dir / "CMakeCache.txt"
    try:
        with cache_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("//") or line.startswith("#"):
                    continue
                key_type, separator, value = line.rstrip("\n").partition("=")
                if not separator:
                    continue
                key = key_type.split(":", 1)[0]
                if key == name:
                    return value.upper() in ("1", "ON", "TRUE", "YES", "Y")
    except FileNotFoundError:
        raise cmake_file_api.FileApiError(
            f"CMake cache is missing in {build_dir}; run iree-cmake-configure first"
        )
    return False


def fuzz_plan(
    tool_env: ToolEnvironment,
    *,
    configured_build_dir: Path,
    backend_args: list[str],
) -> CommandPlan:
    command = parse_fuzz_args(backend_args)
    return CommandPlan(
        [
            CMakeFuzzStep(
                tool_env.tool("cmake"),
                command,
                configured_build_dir,
                env=tool_env.path_env(),
            )
        ]
    )
