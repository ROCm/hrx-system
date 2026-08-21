# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Host-side handoff for Bazel launch scripts that must preserve caller cwd."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

ARGUMENT_SEPARATOR_ENV = "IREE_BAZEL_LAUNCH_ARGUMENT_SEPARATOR"
CALLER_ARGUMENTS_ENV = "IREE_BAZEL_LAUNCH_CALLER_ARGUMENTS"
CALLER_CWD_ENV = "IREE_BAZEL_LAUNCH_CALLER_CWD"
MATERIALIZE_ENV = "IREE_BAZEL_LAUNCH_MATERIALIZE"
RUNFILES_ARGUMENTS_ENV = "IREE_BAZEL_LAUNCH_RUNFILES_ARGUMENTS"
RUNFILES_ENVIRONMENT_NAMES_ENV = "IREE_BAZEL_LAUNCH_RUNFILES_ENVIRONMENT_NAMES"
SCRIPT_PATH_ENV = "IREE_BAZEL_LAUNCH_SCRIPT_PATH"
CONTROL_ENVIRONMENT_NAMES = (
    ARGUMENT_SEPARATOR_ENV,
    CALLER_ARGUMENTS_ENV,
    CALLER_CWD_ENV,
    MATERIALIZE_ENV,
    RUNFILES_ARGUMENTS_ENV,
    RUNFILES_ENVIRONMENT_NAMES_ENV,
    SCRIPT_PATH_ENV,
)
RUNFILES_PATH_BEGIN = "__IREE_BAZEL_RUNFILE_PATH_BEGIN__"
RUNFILES_PATH_END = "__IREE_BAZEL_RUNFILE_PATH_END__"
TARGET_EXECUTABLE_OPTION = "--target-executable"


@dataclass(frozen=True)
class PreparedLaunch:
    argv: list[str]
    cwd: Path
    env: dict[str, str]
    script_path: Path
    materialize: bool


@dataclass(frozen=True)
class ProcessLaunch:
    argv: list[str]
    cwd: Path
    env: dict[str, str]


def environment_name_key(name: str) -> str:
    """Returns the host-platform comparison key for an environment name."""
    return name.upper() if os.name == "nt" else name


def environment_value(environment: Mapping[str, str], name: str) -> str | None:
    """Finds an environment value using native host name semantics."""
    name_key = environment_name_key(name)
    for candidate_name, value in environment.items():
        if environment_name_key(candidate_name) == name_key:
            return value
    return None


def configured_environment(
    base_env: Mapping[str, str] | None,
    *,
    caller_cwd: Path,
    argument_separator: str,
    caller_arguments: list[str],
    runfiles_arguments: list[str],
    marked_runfiles_arguments: list[str],
    runfiles_environment_names: list[str],
    script_path: Path,
    materialize: bool = False,
) -> dict[str, str]:
    """Adds the private handoff contract consumed by this launcher."""
    control_name_keys = {
        environment_name_key(name) for name in CONTROL_ENVIRONMENT_NAMES
    }
    environment = {
        name: value
        for name, value in (os.environ if base_env is None else base_env).items()
        if environment_name_key(name) not in control_name_keys
    }
    environment[ARGUMENT_SEPARATOR_ENV] = argument_separator
    environment[CALLER_ARGUMENTS_ENV] = json.dumps(
        caller_arguments,
        separators=(",", ":"),
    )
    environment[CALLER_CWD_ENV] = str(caller_cwd)
    environment[RUNFILES_ARGUMENTS_ENV] = json.dumps(
        {
            "arguments": runfiles_arguments,
            "marked_arguments": marked_runfiles_arguments,
        },
        separators=(",", ":"),
    )
    environment[RUNFILES_ENVIRONMENT_NAMES_ENV] = json.dumps(
        runfiles_environment_names,
        separators=(",", ":"),
    )
    environment[SCRIPT_PATH_ENV] = str(script_path)
    if materialize:
        environment[MATERIALIZE_ENV] = "1"
    return environment


def resolve_marked_runfiles_arguments(
    marked_arguments: list[str],
    *,
    bazel_cwd: Path,
) -> list[str]:
    """Absolutizes the graph-proven path spans in configured arguments."""
    rewritten_arguments = []
    inside_runfile_path = False
    for argument in marked_arguments:
        rewritten_argument = ""
        cursor = 0
        while cursor < len(argument):
            if inside_runfile_path:
                end = argument.find(RUNFILES_PATH_END, cursor)
                path_end = len(argument) if end == -1 else end
                path_value = argument[cursor:path_end]
                if not path_value:
                    raise ValueError(
                        "configured runfile argument contains an empty path"
                    )
                path = Path(path_value)
                if not path.is_absolute():
                    path = bazel_cwd / path
                if not path.exists():
                    raise FileNotFoundError(
                        f"configured runfile argument points to missing path {path}"
                    )
                rewritten_argument += str(path)
                if end == -1:
                    cursor = len(argument)
                else:
                    inside_runfile_path = False
                    cursor = end + len(RUNFILES_PATH_END)
            else:
                start = argument.find(RUNFILES_PATH_BEGIN, cursor)
                if start == -1:
                    rewritten_argument += argument[cursor:]
                    cursor = len(argument)
                else:
                    rewritten_argument += argument[cursor:start]
                    inside_runfile_path = True
                    cursor = start + len(RUNFILES_PATH_BEGIN)
        rewritten_arguments.append(rewritten_argument)
    if inside_runfile_path:
        raise ValueError("configured runfile argument has an unterminated path marker")
    return rewritten_arguments


def prepare_launch(
    argv: list[str],
    *,
    environ: Mapping[str, str] | None = None,
    initial_cwd: Path | None = None,
) -> PreparedLaunch:
    """Resolves graph-declared target and runfile paths before leaving Bazel's cwd."""
    if len(argv) < 3 or argv[0] != TARGET_EXECUTABLE_OPTION:
        raise ValueError("Bazel launcher did not provide the configured executable")
    executable_path = Path(argv[1])
    if not executable_path.is_absolute():
        raise ValueError(
            f"configured Bazel target executable is not absolute: {executable_path}"
        )
    bazel_argv = argv[2:]

    environment = dict(os.environ if environ is None else environ)
    try:
        argument_separator = environment.pop(ARGUMENT_SEPARATOR_ENV)
        encoded_caller_arguments = environment.pop(CALLER_ARGUMENTS_ENV)
        caller_cwd = Path(environment.pop(CALLER_CWD_ENV))
        encoded_arguments = environment.pop(RUNFILES_ARGUMENTS_ENV)
        encoded_names = environment.pop(RUNFILES_ENVIRONMENT_NAMES_ENV)
        script_path = Path(environment.pop(SCRIPT_PATH_ENV))
    except KeyError as exc:
        raise ValueError(
            f"missing launcher environment variable {exc.args[0]}"
        ) from exc
    materialize_value = environment.pop(MATERIALIZE_ENV, None)
    if materialize_value not in (None, "1"):
        raise ValueError(f"invalid {MATERIALIZE_ENV} value")
    if not argument_separator:
        raise ValueError("Bazel launcher argument separator is empty")

    try:
        caller_arguments = json.loads(encoded_caller_arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid caller argument list") from exc
    if not isinstance(caller_arguments, list) or any(
        not isinstance(argument, str) for argument in caller_arguments
    ):
        raise ValueError("caller arguments must contain strings")

    try:
        runfiles_arguments = json.loads(encoded_arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid runfiles argument metadata") from exc
    if not isinstance(runfiles_arguments, dict) or set(runfiles_arguments) != {
        "arguments",
        "marked_arguments",
    }:
        raise ValueError("invalid runfiles argument metadata fields")
    expected_arguments = runfiles_arguments["arguments"]
    marked_arguments = runfiles_arguments["marked_arguments"]
    for name, arguments in (
        ("arguments", expected_arguments),
        ("marked_arguments", marked_arguments),
    ):
        if not isinstance(arguments, list) or any(
            not isinstance(argument, str) for argument in arguments
        ):
            raise ValueError(f"runfiles {name} must contain strings")
    if len(expected_arguments) != len(marked_arguments):
        raise ValueError("runfiles argument metadata lengths do not match")
    unmarked_arguments = [
        argument.replace(RUNFILES_PATH_BEGIN, "").replace(RUNFILES_PATH_END, "")
        for argument in marked_arguments
    ]
    if unmarked_arguments != expected_arguments:
        raise ValueError("runfiles argument metadata does not match its path markers")

    separator_positions = [
        index
        for index, argument in enumerate(bazel_argv)
        if argument == argument_separator
    ]
    if len(separator_positions) != 1:
        raise ValueError("Bazel launcher argument separator is missing or duplicated")
    separator_position = separator_positions[0]
    default_arguments = bazel_argv[1:separator_position]

    try:
        runfiles_environment_names = json.loads(encoded_names)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid runfiles environment name list") from exc
    if not isinstance(runfiles_environment_names, list) or any(
        not isinstance(name, str) or not name for name in runfiles_environment_names
    ):
        raise ValueError("runfiles environment names must be non-empty strings")
    runfiles_environment_name_keys = [
        environment_name_key(name) for name in runfiles_environment_names
    ]
    if len(runfiles_environment_name_keys) != len(set(runfiles_environment_name_keys)):
        raise ValueError("runfiles environment names must be unique")

    bazel_cwd = Path.cwd() if initial_cwd is None else initial_cwd
    if expected_arguments:
        matching_positions = [
            position
            for position in range(len(default_arguments) - len(expected_arguments) + 1)
            if default_arguments[position : position + len(expected_arguments)]
            == expected_arguments
        ]
        if len(matching_positions) != 1:
            raise ValueError(
                "configured runfiles arguments do not uniquely match Bazel's launcher"
            )
        rewritten_arguments = resolve_marked_runfiles_arguments(
            marked_arguments,
            bazel_cwd=bazel_cwd,
        )
        position = matching_positions[0]
        default_arguments = [
            *default_arguments[:position],
            *rewritten_arguments,
            *default_arguments[position + len(expected_arguments) :],
        ]
    # Bazel formats --run_under commands through a host shell, making native
    # Windows paths in both the target token and caller arguments lossy. The
    # configured graph path and original caller arguments are authoritative.
    if not executable_path.is_file():
        raise FileNotFoundError(
            f"configured Bazel target executable does not exist: {executable_path}"
        )
    target_argv = [str(executable_path), *default_arguments, *caller_arguments]
    for name in runfiles_environment_names:
        value = environment_value(environment, name)
        if value is None:
            raise ValueError(f"Bazel launcher did not set environment variable {name}")
        path = Path(value)
        if not path.is_absolute():
            path = bazel_cwd / path
        if not path.is_file():
            raise FileNotFoundError(
                f"runfile environment variable {name} points to missing file {path}"
            )
        name_key = environment_name_key(name)
        environment = {
            candidate_name: candidate_value
            for candidate_name, candidate_value in environment.items()
            if environment_name_key(candidate_name) != name_key
        }
        environment[name] = str(path)

    if not caller_cwd.is_dir():
        raise NotADirectoryError(
            f"caller working directory does not exist: {caller_cwd}"
        )
    return PreparedLaunch(
        argv=target_argv,
        cwd=caller_cwd,
        env=environment,
        script_path=script_path,
        materialize=materialize_value == "1",
    )


def encode_process_launch(launch: PreparedLaunch) -> str:
    """Encodes a prepared target process for a parent launcher."""
    return json.dumps(
        {
            "argv": launch.argv,
            "cwd": str(launch.cwd),
            "environment": launch.env,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def decode_process_launch(encoded_launch: str) -> ProcessLaunch:
    """Validates and decodes a target process emitted by this launcher."""
    try:
        payload = json.loads(encoded_launch)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid materialized Bazel launch") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "argv",
        "cwd",
        "environment",
    }:
        raise ValueError("invalid materialized Bazel launch fields")
    argv = payload["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(arg, str) for arg in argv)
    ):
        raise ValueError("materialized Bazel launch argv must contain strings")
    cwd_value = payload["cwd"]
    if not isinstance(cwd_value, str) or not cwd_value:
        raise ValueError("materialized Bazel launch cwd must be a non-empty string")
    cwd = Path(cwd_value)
    if not cwd.is_dir():
        raise ValueError(f"materialized Bazel launch cwd does not exist: {cwd}")
    environment = payload["environment"]
    if not isinstance(environment, dict) or any(
        not isinstance(name, str) or not name or not isinstance(value, str)
        for name, value in environment.items()
    ):
        raise ValueError("materialized Bazel launch environment must contain strings")
    return ProcessLaunch(argv=argv, cwd=cwd, env=environment)


def remove_launch_script(script_path: Path) -> None:
    """Removes the generated script before overlaying the target process."""
    try:
        script_path.unlink()
    except FileNotFoundError:
        return
    except PermissionError as exc:
        # Keep target execution correct, but make the retained launch-script
        # path visible instead of silently leaking it.
        print(
            f"iree-bazel-launcher: unable to remove {script_path}: {exc}",
            file=sys.stderr,
        )


def handoff_process(argv: list[str], environment: Mapping[str, str]) -> int:
    """Overlays on POSIX or waits for a direct child on Windows."""
    if os.name == "nt":
        # Windows cannot replace the current process image. The CRT exec family
        # starts a child and terminates the caller, which signals the launcher's
        # process handle while the target is still running. Remain as the
        # target's direct parent so callers retain a valid lifetime and exit-code
        # contract without introducing a command interpreter.
        return subprocess.run(argv, env=environment).returncode
    os.execvpe(argv[0], argv, environment)
    raise AssertionError("os.execvpe returned unexpectedly")


def main(argv: list[str] | None = None) -> int:
    target_argv = list(sys.argv[1:] if argv is None else argv)
    script_path_value = environment_value(os.environ, SCRIPT_PATH_ENV)
    parent_owns_script = environment_value(os.environ, MATERIALIZE_ENV) is not None
    try:
        launch = prepare_launch(target_argv)
        if launch.materialize:
            sys.stdout.write(encode_process_launch(launch))
            sys.stdout.flush()
            return 0
        remove_launch_script(launch.script_path)
        os.chdir(launch.cwd)
        return handoff_process(launch.argv, launch.env)
    except (OSError, ValueError) as exc:
        if script_path_value is not None and not parent_owns_script:
            remove_launch_script(Path(script_path_value))
        print(f"iree-bazel-launcher: {exc}", file=sys.stderr)
        return 127


if __name__ == "__main__":
    sys.exit(main())
