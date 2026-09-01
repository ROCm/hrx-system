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
from build_tools.devtools import ctest as ctest_dev
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
CMAKE_BUILD_TREE_MARKER = Path(".iree/devtools-cmake-build-tree")
CMAKE_BUILD_TREE_MARKER_CONTENT = (
    f"iree-devtools-cmake-build-tree-v1\n{REPO_ROOT.resolve()}\n"
)


@dataclass(frozen=True)
class _CMakeGenerator:
    name: str
    platform: str = ""
    toolset: str = ""
    instance: str = ""

    def describe(self) -> str:
        details = []
        if self.platform:
            details.append(f"platform {self.platform}")
        if self.toolset:
            details.append(f"toolset {self.toolset}")
        if self.instance:
            details.append(f"instance {self.instance}")
        if not details:
            return self.name
        return f"{self.name} ({', '.join(details)})"


@dataclass(frozen=True)
class _PrepareFreshCMakeConfigureStep:
    build_dir: Path
    message: str
    reset_build_tree: bool

    def describe(self) -> str:
        return f"# {self.message}"

    def run(self, verbose: bool = False) -> int:
        if not self.reset_build_tree:
            if verbose:
                print(f"dev.py: {self.message}")
            return 0
        if not self.build_dir.exists():
            return 0
        if not _is_managed_cmake_build_tree(self.build_dir):
            print(
                f"dev.py: refusing to remove unrecognized CMake build tree "
                f"{self.build_dir}; the directory has neither an IREE devtool "
                "marker nor a CMake cache for this source tree",
                file=sys.stderr,
            )
            return 1
        print(f"dev.py: {self.message}")
        try:
            shutil.rmtree(self.build_dir)
        except OSError as exc:
            print(
                f"dev.py: failed to remove CMake build tree {self.build_dir}: {exc}",
                file=sys.stderr,
            )
            return 1
        return 0


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
    resolved_target_names = cmake_file_api.resolve_target_names(
        requested_build_dir,
        target_names,
    )
    for target_name in resolved_target_names:
        cmake_args.extend(["--target", target_name])
    cmake_args.extend(raw_args)
    return cmake_args


def _is_managed_cmake_build_tree(build_dir: Path) -> bool:
    marker_path = build_dir / CMAKE_BUILD_TREE_MARKER
    try:
        if marker_path.read_text(encoding="utf-8") == CMAKE_BUILD_TREE_MARKER_CONTENT:
            return True
    except (OSError, UnicodeError):
        pass

    try:
        entries = cmake_try.load_cmake_cache(build_dir)
    except cmake_file_api.FileApiError:
        return False
    values = {entry.name: entry.value for entry in entries}
    home_directory = values.get("CMAKE_HOME_DIRECTORY", "")
    if not home_directory:
        return False
    try:
        return Path(home_directory).resolve() == REPO_ROOT.resolve()
    except OSError:
        return False


def _cmake_short_option_value(arguments: list[str], option: str) -> str | None:
    value = None
    for index, argument in enumerate(arguments):
        if argument == option:
            if index + 1 < len(arguments):
                value = arguments[index + 1]
        elif argument.startswith(option) and len(argument) > len(option):
            value = argument[len(option) :]
    return value


def _cmake_definition_value(arguments: list[str], name: str) -> str | None:
    value = None
    for index, argument in enumerate(arguments):
        definition = None
        if argument == "-D":
            if index + 1 < len(arguments):
                definition = arguments[index + 1]
        elif argument.startswith("-D"):
            definition = argument[2:]
        if definition is None:
            continue
        name_and_type, separator, candidate_value = definition.partition("=")
        if separator and name_and_type.partition(":")[0] == name:
            value = candidate_value
    return value


def _configured_cmake_generator(build_dir: Path) -> _CMakeGenerator | None:
    try:
        entries = cmake_try.load_cmake_cache(build_dir)
    except cmake_file_api.FileApiError:
        return None
    values = {entry.name: entry.value for entry in entries}
    name = values.get("CMAKE_GENERATOR", "")
    if not name:
        return None
    return _CMakeGenerator(
        name=name,
        platform=values.get("CMAKE_GENERATOR_PLATFORM", ""),
        toolset=values.get("CMAKE_GENERATOR_TOOLSET", ""),
        instance=values.get("CMAKE_GENERATOR_INSTANCE", ""),
    )


def _select_fresh_cmake_generator(
    configured_generator: _CMakeGenerator,
    backend_args: list[str],
    env: Mapping[str, str],
) -> tuple[_CMakeGenerator, list[str]]:
    requested_name = _cmake_short_option_value(backend_args, "-G")
    environment_name = env.get("CMAKE_GENERATOR", "")
    selected_name = requested_name or environment_name or configured_generator.name
    inherited_generator = (
        configured_generator if selected_name == configured_generator.name else None
    )

    requested_platform = _cmake_short_option_value(backend_args, "-A")
    environment_platform = env.get("CMAKE_GENERATOR_PLATFORM", "")
    selected_platform = requested_platform or environment_platform
    if not selected_platform and inherited_generator is not None:
        selected_platform = inherited_generator.platform

    requested_toolset = _cmake_short_option_value(backend_args, "-T")
    environment_toolset = env.get("CMAKE_GENERATOR_TOOLSET", "")
    selected_toolset = requested_toolset or environment_toolset
    if not selected_toolset and inherited_generator is not None:
        selected_toolset = inherited_generator.toolset

    requested_instance = _cmake_definition_value(
        backend_args, "CMAKE_GENERATOR_INSTANCE"
    )
    environment_instance = env.get("CMAKE_GENERATOR_INSTANCE", "")
    selected_instance = requested_instance or environment_instance
    if not selected_instance and inherited_generator is not None:
        selected_instance = inherited_generator.instance

    selected_generator = _CMakeGenerator(
        name=selected_name,
        platform=selected_platform,
        toolset=selected_toolset,
        instance=selected_instance,
    )
    preservation_args = []
    if requested_name is None and not environment_name:
        preservation_args.extend(["-G", selected_generator.name])
    if (
        selected_generator.platform
        and requested_platform is None
        and not environment_platform
    ):
        preservation_args.extend(["-A", selected_generator.platform])
    if (
        selected_generator.toolset
        and requested_toolset is None
        and not environment_toolset
    ):
        preservation_args.extend(["-T", selected_generator.toolset])
    if (
        selected_generator.instance
        and requested_instance is None
        and not environment_instance
    ):
        preservation_args.append(
            f"-DCMAKE_GENERATOR_INSTANCE={selected_generator.instance}"
        )
    return selected_generator, preservation_args


def _prepare_fresh_configure(
    build_dir: Path,
    backend_args: list[str],
    env: Mapping[str, str],
) -> tuple[list[_PrepareFreshCMakeConfigureStep], list[str]]:
    if "--fresh" not in backend_args:
        return [], backend_args

    configured_generator = _configured_cmake_generator(build_dir)
    if configured_generator is None:
        if not build_dir.exists():
            return [], backend_args
        message = (
            f"remove incomplete CMake build tree {build_dir} before fresh configure"
        )
        return [_PrepareFreshCMakeConfigureStep(build_dir, message, True)], backend_args

    selected_generator, preservation_args = _select_fresh_cmake_generator(
        configured_generator, backend_args, env
    )
    if selected_generator == configured_generator:
        message = (
            f"preserve CMake generator {selected_generator.describe()} for fresh "
            "configure"
        )
        step = _PrepareFreshCMakeConfigureStep(build_dir, message, False)
    else:
        message = (
            f"switch CMake generator from {configured_generator.describe()} to "
            f"{selected_generator.describe()}; remove build tree {build_dir}"
        )
        step = _PrepareFreshCMakeConfigureStep(build_dir, message, True)
    return [step], [*preservation_args, *backend_args]


def configure_plan(
    tool_env: ToolEnvironment,
    *,
    configured_build_dir: Path | None,
    backend_args: list[str],
    env: dict[str, str] | None = None,
) -> CommandPlan:
    requested_build_dir = build_dir(configured_build_dir)
    codemodel_query_path = cmake_file_api.codemodel_query_path(requested_build_dir)
    command_env = tool_env.path_env() if env is None else env
    fresh_steps, configure_args = _prepare_fresh_configure(
        requested_build_dir, backend_args, command_env
    )
    return CommandPlan(
        [
            *fresh_steps,
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
                    *configure_args,
                ],
                cwd=REPO_ROOT,
                env=command_env,
                label="configure cmake",
            ),
            WriteFileStep(
                requested_build_dir / CMAKE_BUILD_TREE_MARKER,
                CMAKE_BUILD_TREE_MARKER_CONTENT,
                label="mark managed CMake build tree",
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
    env: dict[str, str] | None = None,
) -> CommandPlan:
    command = parse_run_args(backend_args, run_cwd=run_cwd)
    return CommandPlan(
        [
            CMakeRunStep(
                command,
                build_dir(configured_build_dir),
                env=tool_env.path_env() if env is None else env,
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
    env: dict[str, str] | None = None,
) -> CommandPlan:
    requested_build_dir = build_dir(configured_build_dir)
    cmake_args = build_args(
        backend_args,
        configured_build_dir=configured_build_dir,
    )
    target_names = [
        cmake_args[index + 1]
        for index, argument in enumerate(cmake_args[:-1])
        if argument == "--target"
    ]
    target_description = ", ".join(target_names) if target_names else "default"
    configured_generator = _configured_cmake_generator(requested_build_dir)
    generator_description = (
        configured_generator.describe()
        if configured_generator is not None
        else "unconfigured"
    )
    return CommandPlan(
        [
            CommandStep(
                [
                    tool_env.tool("cmake"),
                    "--build",
                    str(requested_build_dir),
                    *cmake_args,
                ],
                cwd=REPO_ROOT,
                env=tool_env.path_env() if env is None else env,
                label=(
                    f"cmake build: targets={target_description}; "
                    f"generator={generator_description}; "
                    f"tree={requested_build_dir}"
                ),
                announce=True,
            )
        ]
    )


def test_plan(
    tool_env: ToolEnvironment,
    *,
    configured_build_dir: Path | None,
    backend_args: list[str],
    env: dict[str, str] | None = None,
) -> CommandPlan:
    requested_build_dir = build_dir(configured_build_dir)
    command_env = tool_env.path_env() if env is None else env
    if ctest_dev.is_inspection_only(backend_args):
        return CommandPlan(
            [
                CommandStep(
                    ctest_dev.ctest_run_command(
                        tool_env.tool("ctest"),
                        requested_build_dir,
                        backend_args,
                    ),
                    cwd=REPO_ROOT,
                    env=command_env,
                    label="inspect cmake tests",
                )
            ]
        )
    return CommandPlan(
        [
            ctest_dev.CTestBuildAndRunStep(
                cmake=tool_env.tool("cmake"),
                ctest=tool_env.tool("ctest"),
                build_dir=requested_build_dir,
                arguments=backend_args,
                cwd=REPO_ROOT,
                env=command_env,
            )
        ]
    )
