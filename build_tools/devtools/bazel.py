# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bazel command helpers for dev.py."""

from __future__ import annotations

import json
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from build_tools.bazel import compile_commands_merge
from build_tools.devtools import bazel_launcher, fuzz
from build_tools.devtools.command_plan import quote_command
from build_tools.devtools.environment import (
    BAZEL_SH_ENV,
    LOCAL_TMP_ROOT,
    REPO_ROOT,
)

LOCAL_STATE_ROOT = REPO_ROOT / ".iree"
BAZEL_TRY_ROOT = LOCAL_STATE_ROOT / "bazel-try"
BAZEL_TRY_LABEL_ROOT = "//.iree/bazel-try"
BAZEL_TRY_COMMON_ARGS = ("--check_visibility=false",)
BAZEL_LAUNCH_ROOT = LOCAL_TMP_ROOT / "iree-bazel-launch"
BAZEL_COMPILE_COMMANDS_ROOT = LOCAL_TMP_ROOT / "iree-bazel-compile-commands"
DEFAULT_TRY_BINARY_NAME = "snippet"
DEFAULT_COMPILE_COMMANDS_OUTPUT = REPO_ROOT / "compile_commands.json"
DEFAULT_COMPILE_COMMANDS_TARGETS = ("//runtime/...", "//libhrx/...", "//loom/...")
COMPILE_COMMANDS_ASPECT = (
    "//build_tools/bazel:compile_commands.bzl%collect_compile_commands_aspect"
)
CLANG_TIDY_ASPECT = "//build_tools/clang_tidy:clang_tidy.bzl%collect_clang_tidy_aspect"
CLANG_TIDY_OUTPUT_GROUP = "iree_clang_tidy_reports"
CLANG_TIDY_REPO_ENV = "--repo_env=IREE_CLANG_TIDY_LLVM=auto"
RUNFILES_ENVIRONMENT_PROVIDER_SUFFIX = "%IreeRunfilesEnvironmentInfo"
RUNFILES_ARGUMENTS_PROVIDER_SUFFIX = "%IreeRunfilesArgumentsInfo"
BAZEL_LAUNCH_METADATA_QUERY_EXPRESSION = (
    'json.encode({"label": str(target.label), '
    '"argument_provider_count": len(['
    "key for key in providers(target).keys() "
    f'if key.endswith("{RUNFILES_ARGUMENTS_PROVIDER_SUFFIX}")]), '
    '"arguments": ['
    "argument for key, value in providers(target).items() "
    f'if key.endswith("{RUNFILES_ARGUMENTS_PROVIDER_SUFFIX}") '
    "for argument in value.arguments], "
    '"marked_arguments": ['
    "argument for key, value in providers(target).items() "
    f'if key.endswith("{RUNFILES_ARGUMENTS_PROVIDER_SUFFIX}") '
    "for argument in value.marked_arguments], "
    '"environment_names": sorted(['
    "name for key, value in providers(target).items() "
    f'if key.endswith("{RUNFILES_ENVIRONMENT_PROVIDER_SUFFIX}") '
    "for name in value.environment"
    ']), "inherited_environment_names": sorted('
    'providers(target)["RunEnvironmentInfo"].inherited_environment '
    'if "RunEnvironmentInfo" in providers(target) else []), '
    '"run_environment_names": sorted('
    'providers(target)["RunEnvironmentInfo"].environment.keys() '
    'if "RunEnvironmentInfo" in providers(target) else [])})'
)
HEADER_ROOTS: tuple[tuple[str, Path], ...] = (
    ("iree/", REPO_ROOT / "runtime/src"),
    ("loom/", REPO_ROOT / "loom/src"),
    ("loomc/", REPO_ROOT / "loom/binding/c/include"),
)
SPECIAL_HEADER_DEPS = {
    "iree/testing/gtest_harness.h": "//runtime/src/iree/testing:gtest_harness",
    "iree/testing/gbenchmark_harness.h": (
        "//runtime/src/iree/testing:gbenchmark_harness"
    ),
}


@dataclass(frozen=True)
class BazelTargetCommand:
    target: str
    bazel_args: list[str] = field(default_factory=list)
    program_args: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BazelRunCommand(BazelTargetCommand):
    print_path: bool = False
    run_cwd: Path = field(default_factory=Path.cwd)


@dataclass(frozen=True)
class BazelTryCommand:
    files: list[Path] = field(default_factory=list)
    inline_sources: list[str] = field(default_factory=list)
    language: str | None = None
    compile_only: bool = False
    output: Path | None = None
    explicit_deps: list[str] = field(default_factory=list)
    infer_deps: bool = True
    keep: bool = False
    bazel_args: list[str] = field(default_factory=list)
    program_args: list[str] = field(default_factory=list)
    run_cwd: Path = field(default_factory=Path.cwd)


@dataclass(frozen=True)
class BazelCompileCommandsCommand:
    targets: list[str] = field(default_factory=list)
    bazel_args: list[str] = field(default_factory=list)
    output: Path = DEFAULT_COMPILE_COMMANDS_OUTPUT
    keep: bool = False
    run_cwd: Path = field(default_factory=Path.cwd)


@dataclass(frozen=True)
class BazelLaunchMetadata:
    runfiles_arguments: list[str] = field(default_factory=list)
    marked_runfiles_arguments: list[str] = field(default_factory=list)
    runfiles_environment_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BazelLaunch:
    target: str
    script_path: Path
    run_cwd: Path
    argument_separator: str
    runfiles_arguments: list[str] = field(default_factory=list)
    marked_runfiles_arguments: list[str] = field(default_factory=list)
    runfiles_environment_names: list[str] = field(default_factory=list)

    def argv(self, env: dict[str, str] | None) -> list[str]:
        if os.name != "nt":
            return [str(self.script_path)]
        environment = os.environ if env is None else env
        command_interpreter = bazel_launcher.environment_value(environment, "COMSPEC")
        if not command_interpreter:
            raise ValueError("COMSPEC is required to execute Bazel launch scripts")
        return [command_interpreter, "/d", "/c", str(self.script_path)]

    def environment(
        self,
        base_env: dict[str, str] | None,
        *,
        materialize: bool = False,
    ) -> dict[str, str]:
        return bazel_launcher.configured_environment(
            base_env,
            caller_cwd=self.run_cwd,
            argument_separator=self.argument_separator,
            runfiles_arguments=self.runfiles_arguments,
            marked_runfiles_arguments=self.marked_runfiles_arguments,
            runfiles_environment_names=self.runfiles_environment_names,
            script_path=self.script_path,
            materialize=materialize,
        )


@dataclass(frozen=True)
class BazelProcess:
    target: str
    argv: list[str]
    cwd: Path
    env: dict[str, str]
    script_path: Path | None = None


def split_program_args(arguments: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in arguments:
        return arguments, []
    separator_index = arguments.index("--")
    return arguments[:separator_index], arguments[separator_index + 1 :]


def parse_bazel_run_args(
    arguments: list[str], *, run_cwd: Path | None = None
) -> BazelRunCommand:
    tool_args, program_args = split_program_args(arguments)
    target = ""
    bazel_args = []
    print_path = False
    for arg in tool_args:
        if arg in ("-p", "--print-path", "--print_path"):
            print_path = True
        elif arg.startswith("-"):
            bazel_args.append(arg)
        elif not target:
            target = arg
        else:
            bazel_args.append(arg)
    if not target:
        raise ValueError("Target is required for bazel run")
    return BazelRunCommand(
        target=target,
        bazel_args=bazel_args,
        program_args=program_args,
        print_path=print_path,
        run_cwd=run_cwd or Path.cwd(),
    )


def parse_bazel_fuzz_args(arguments: list[str]) -> BazelTargetCommand:
    tool_args, fuzzer_args = split_program_args(arguments)
    target = ""
    bazel_args = []
    for arg in tool_args:
        if arg.startswith("-"):
            bazel_args.append(arg)
        elif not target:
            target = arg
        else:
            bazel_args.append(arg)
    if not target:
        raise ValueError("Target is required for bazel fuzz")
    return BazelTargetCommand(
        target=target,
        bazel_args=bazel_args,
        program_args=fuzzer_args,
    )


def parse_bazel_try_args(
    arguments: list[str], *, run_cwd: Path | None = None
) -> BazelTryCommand:
    tool_args, program_args = split_program_args(arguments)
    files = []
    inline_sources = []
    explicit_deps = []
    bazel_args = []
    language = None
    compile_only = False
    output = None
    infer_deps = True
    keep = False

    index = 0
    while index < len(tool_args):
        arg = tool_args[index]
        if arg in ("-e", "--execute"):
            index += 1
            if index >= len(tool_args):
                raise ValueError(f"{arg} requires source text")
            inline_sources.append(tool_args[index])
        elif arg.startswith("--execute="):
            inline_sources.append(arg.split("=", 1)[1])
        elif arg == "-x":
            index += 1
            if index >= len(tool_args):
                raise ValueError("-x requires c or c++")
            language = normalize_language(tool_args[index])
        elif arg.startswith("-x="):
            language = normalize_language(arg.split("=", 1)[1])
        elif arg in ("-c", "--compile-only", "--compile_only"):
            compile_only = True
        elif arg in ("-o", "--output"):
            index += 1
            if index >= len(tool_args):
                raise ValueError(f"{arg} requires a path")
            output = Path(tool_args[index])
        elif arg.startswith("--output="):
            output = Path(arg.split("=", 1)[1])
        elif arg == "--dep":
            index += 1
            if index >= len(tool_args):
                raise ValueError("--dep requires a Bazel label")
            explicit_deps.append(tool_args[index])
        elif arg.startswith("--dep="):
            explicit_deps.append(arg.split("=", 1)[1])
        elif arg in ("--no-infer", "--no_infer"):
            infer_deps = False
        elif arg in ("-k", "--keep"):
            keep = True
        elif arg.startswith("-"):
            bazel_args.append(arg)
        else:
            files.append(Path(arg))
        index += 1

    return BazelTryCommand(
        files=files,
        inline_sources=inline_sources,
        language=language,
        compile_only=compile_only,
        output=output,
        explicit_deps=explicit_deps,
        infer_deps=infer_deps,
        keep=keep,
        bazel_args=bazel_args,
        program_args=program_args,
        run_cwd=run_cwd or Path.cwd(),
    )


def parse_bazel_compile_commands_args(
    arguments: list[str], *, run_cwd: Path | None = None
) -> BazelCompileCommandsCommand:
    tool_args = forwarded_tool_args(arguments)
    targets = []
    bazel_args = []
    output = DEFAULT_COMPILE_COMMANDS_OUTPUT
    keep = False
    command_cwd = run_cwd or Path.cwd()

    index = 0
    while index < len(tool_args):
        arg = tool_args[index]
        if arg in ("-o", "--output"):
            index += 1
            if index >= len(tool_args):
                raise ValueError(f"{arg} requires a path")
            output = Path(tool_args[index])
        elif arg.startswith("--output="):
            output = Path(arg.split("=", 1)[1])
        elif arg in ("-k", "--keep"):
            keep = True
        elif is_negative_bazel_target_pattern(arg):
            targets.append(arg)
        elif arg.startswith("-"):
            bazel_args.append(arg)
        else:
            targets.append(arg)
        index += 1

    if not output.is_absolute():
        output = command_cwd / output
    if not targets:
        targets = list(DEFAULT_COMPILE_COMMANDS_TARGETS)
    elif all(is_negative_bazel_target_pattern(target) for target in targets):
        targets = [*DEFAULT_COMPILE_COMMANDS_TARGETS, *targets]

    return BazelCompileCommandsCommand(
        targets=targets,
        bazel_args=bazel_args,
        output=output,
        keep=keep,
        run_cwd=command_cwd,
    )


def forwarded_tool_args(arguments: list[str]) -> list[str]:
    if arguments and arguments[0] == "--":
        return arguments[1:]
    return arguments


def clang_tidy_build_argv(
    bazel: str,
    targets: list[str],
    *,
    keep_going: bool = False,
) -> list[str]:
    argv = [bazel, "build"]
    if keep_going:
        argv.append("--keep_going")
    argv += [
        CLANG_TIDY_REPO_ENV,
        f"--aspects={CLANG_TIDY_ASPECT}",
        f"--output_groups={CLANG_TIDY_OUTPUT_GROUP}",
        "--",
        *targets,
    ]
    return argv


def is_negative_bazel_target_pattern(arg: str) -> bool:
    return arg.startswith(("-//", "-@"))


def normalize_language(value: str) -> str:
    if value in ("c", "C"):
        return "c"
    if value in ("c++", "cc", "cpp", "cxx", "C++"):
        return "c++"
    raise ValueError(f"-x expects c or c++, got {value!r}")


def run_quietly(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    verbose: bool,
) -> int:
    if verbose:
        print("dev.py:", quote_command(argv), flush=True)
        return subprocess.run(argv, cwd=cwd, env=env).returncode

    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0 and completed.stdout:
        print(completed.stdout.rstrip(), file=sys.stderr)
    return completed.returncode


def run_captured(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def validate_bazel_launch_args(bazel_args: list[str]) -> None:
    """Rejects Bazel launch options owned by the lock-free wrapper."""
    managed_options = {"run_under", "script_path"}
    for arg in bazel_args:
        if not arg.startswith("--"):
            continue
        option_name = arg[2:].split("=", 1)[0].replace("-", "_")
        if option_name in managed_options:
            raise ValueError(
                f"{arg.split('=', 1)[0]} is managed by the IREE Bazel launcher"
            )


def validate_bazel_launch_environment(env: dict[str, str] | None) -> None:
    """Rejects shell paths Bazel cannot quote in Windows launch scripts."""
    if os.name != "nt":
        return
    environment = os.environ if env is None else env
    bazel_shell = bazel_launcher.environment_value(environment, BAZEL_SH_ENV)
    if bazel_shell and any(character.isspace() for character in bazel_shell):
        raise ValueError(
            "BAZEL_SH contains whitespace that Bazel cannot quote in generated "
            "Windows launch scripts; configure a space-free shell path or enable "
            "Win32 short path names"
        )


def bazel_label_path_key(label: str) -> str:
    """Returns the repository-independent package and target portion of a label."""
    package_start = label.find("//")
    return label[package_start:] if package_start != -1 else label


def parse_bazel_launch_metadata(
    payload: object,
    *,
    target: str,
) -> BazelLaunchMetadata | None:
    """Validates one configured target's host-handoff metadata."""
    if not isinstance(payload, dict):
        print(f"dev.py: invalid Bazel launch metadata for {target}", file=sys.stderr)
        return None

    argument_fields = ("arguments", "marked_arguments")
    for field_name in argument_fields:
        value = payload.get(field_name)
        if not isinstance(value, list) or any(
            not isinstance(argument, str) for argument in value
        ):
            print(
                f"dev.py: invalid {field_name} in Bazel launch metadata for {target}",
                file=sys.stderr,
            )
            return None

    environment_fields = (
        "environment_names",
        "inherited_environment_names",
        "run_environment_names",
    )
    for field_name in environment_fields:
        value = payload.get(field_name)
        if not isinstance(value, list) or any(
            not isinstance(name, str) or not name for name in value
        ):
            print(
                f"dev.py: invalid {field_name} in Bazel launch metadata for {target}",
                file=sys.stderr,
            )
            return None

    argument_provider_count = payload.get("argument_provider_count")
    if argument_provider_count not in (0, 1):
        print(
            f"dev.py: invalid argument provider count in Bazel launch metadata "
            f"for {target}",
            file=sys.stderr,
        )
        return None
    arguments = payload["arguments"]
    marked_arguments = payload["marked_arguments"]
    if len(arguments) != len(marked_arguments) or (
        argument_provider_count == 0 and arguments
    ):
        print(
            f"dev.py: inconsistent runfiles arguments in Bazel launch metadata "
            f"for {target}",
            file=sys.stderr,
        )
        return None

    environment_names = payload["environment_names"]
    environment_name_keys = [
        bazel_launcher.environment_name_key(name) for name in environment_names
    ]
    if len(environment_name_keys) != len(set(environment_name_keys)):
        print(
            f"dev.py: duplicate runfiles environment metadata for {target}",
            file=sys.stderr,
        )
        return None
    target_environment_names = set(payload["run_environment_names"])
    target_environment_names.update(payload["inherited_environment_names"])
    control_name_keys = {
        bazel_launcher.environment_name_key(name)
        for name in bazel_launcher.CONTROL_ENVIRONMENT_NAMES
    }
    control_collisions = {
        name
        for name in target_environment_names
        if bazel_launcher.environment_name_key(name) in control_name_keys
    }
    if control_collisions:
        print(
            f"dev.py: {target} uses reserved Bazel launcher environment "
            f"{', '.join(sorted(control_collisions))}",
            file=sys.stderr,
        )
        return None
    return BazelLaunchMetadata(
        runfiles_arguments=arguments,
        marked_runfiles_arguments=marked_arguments,
        runfiles_environment_names=environment_names,
    )


def resolve_bazel_launch_metadata_for_targets(
    *,
    bazel: str,
    targets: list[str],
    bazel_args: list[str],
    cwd: Path,
    env: dict[str, str] | None,
) -> tuple[int, dict[str, BazelLaunchMetadata] | None]:
    """Queries host-handoff metadata for configured executable targets."""
    if not targets:
        return 0, {}
    target_expression = targets[0] if len(targets) == 1 else f"set({' '.join(targets)})"
    cquery = run_captured(
        [
            bazel,
            "cquery",
            "--output=starlark",
            f"--starlark:expr={BAZEL_LAUNCH_METADATA_QUERY_EXPRESSION}",
            *bazel_args,
            target_expression,
        ],
        cwd=cwd,
        env=env,
    )
    if cquery.returncode != 0:
        print_process_failure(cquery)
        return cquery.returncode, None
    records = [line for line in cquery.stdout.splitlines() if line.strip()]
    if len(records) != len(targets):
        if len(targets) > 1:
            metadata_by_target = {}
            for target in targets:
                result, metadata = resolve_bazel_launch_metadata(
                    bazel=bazel,
                    target=target,
                    bazel_args=bazel_args,
                    cwd=cwd,
                    env=env,
                )
                if result != 0 or metadata is None:
                    return result, None
                metadata_by_target[target] = metadata
            return 0, metadata_by_target
        print(
            f"dev.py: expected {len(targets)} configured launch targets, got "
            f"{len(records)}",
            file=sys.stderr,
        )
        return 1, None

    decoded_payloads = []
    for record in records:
        try:
            payload = json.loads(record)
        except json.JSONDecodeError as exc:
            print(f"dev.py: invalid Bazel launch metadata: {exc}", file=sys.stderr)
            return 1, None
        if not isinstance(payload, dict):
            print("dev.py: invalid Bazel launch metadata", file=sys.stderr)
            return 1, None
        label = payload.get("label")
        if not isinstance(label, str) or not label:
            print("dev.py: Bazel launch metadata has no target label", file=sys.stderr)
            return 1, None
        decoded_payloads.append(payload)

    if len(targets) == 1:
        metadata = parse_bazel_launch_metadata(decoded_payloads[0], target=targets[0])
        if metadata is None:
            return 1, None
        return 0, {targets[0]: metadata}

    payloads_by_key = {}
    for payload in decoded_payloads:
        label = payload["label"]
        label_key = bazel_label_path_key(label)
        if label_key in payloads_by_key:
            payloads_by_key = {}
            break
        payloads_by_key[label_key] = payload

    metadata_by_target = {}
    unmatched_targets = []
    for target in targets:
        target_key = bazel_label_path_key(target)
        payload = payloads_by_key.get(target_key)
        if payload is None:
            unmatched_targets.append(target)
            continue
        metadata = parse_bazel_launch_metadata(payload, target=target)
        if metadata is None:
            return 1, None
        metadata_by_target[target] = metadata
    for target in unmatched_targets:
        result, metadata = resolve_bazel_launch_metadata(
            bazel=bazel,
            target=target,
            bazel_args=bazel_args,
            cwd=cwd,
            env=env,
        )
        if result != 0 or metadata is None:
            return result, None
        metadata_by_target[target] = metadata
    return 0, metadata_by_target


def resolve_bazel_launch_metadata(
    *,
    bazel: str,
    target: str,
    bazel_args: list[str],
    cwd: Path,
    env: dict[str, str] | None,
) -> tuple[int, BazelLaunchMetadata | None]:
    """Queries host-handoff metadata carried by one configured target."""
    result, metadata_by_target = resolve_bazel_launch_metadata_for_targets(
        bazel=bazel,
        targets=[target],
        bazel_args=bazel_args,
        cwd=cwd,
        env=env,
    )
    if result != 0 or metadata_by_target is None:
        return result, None
    return 0, metadata_by_target[target]


def create_bazel_launch_script_path() -> Path:
    BAZEL_LAUNCH_ROOT.mkdir(parents=True, exist_ok=True)
    descriptor, path = tempfile.mkstemp(
        dir=BAZEL_LAUNCH_ROOT,
        prefix=f"run-{os.getpid()}-",
        suffix=".bat" if os.name == "nt" else ".sh",
    )
    os.close(descriptor)
    return Path(path)


def create_bazel_argument_separator() -> str:
    return f"__IREE_BAZEL_ARGUMENT_SEPARATOR_{secrets.token_hex(16)}__"


def bazel_run_under_command() -> str:
    launcher_path = Path(bazel_launcher.__file__).resolve()
    return shlex.join([Path(sys.executable).as_posix(), launcher_path.as_posix()])


def generate_bazel_launch(
    *,
    bazel: str,
    target: str,
    bazel_args: list[str],
    program_args: list[str],
    run_cwd: Path,
    env: dict[str, str] | None,
    verbose: bool,
    metadata: BazelLaunchMetadata | None = None,
) -> tuple[int, BazelLaunch | None]:
    """Builds a target and writes its canonical launcher without running it."""
    try:
        validate_bazel_launch_args(bazel_args)
        validate_bazel_launch_environment(env)
    except ValueError as exc:
        print(f"dev.py: {exc}", file=sys.stderr)
        return 2, None

    if metadata is None:
        query_result, metadata = resolve_bazel_launch_metadata(
            bazel=bazel,
            target=target,
            bazel_args=bazel_args,
            cwd=REPO_ROOT,
            env=env,
        )
        if query_result != 0 or metadata is None:
            return query_result, None

    script_path = create_bazel_launch_script_path()
    argument_separator = create_bazel_argument_separator()
    run_argv = [
        bazel,
        "run",
        *bazel_args,
        f"--script_path={script_path}",
        "--norun_in_cwd",
        f"--run_under={bazel_run_under_command()}",
        target,
        "--",
        argument_separator,
    ]
    if program_args:
        run_argv.extend(program_args)
    run_result = run_quietly(
        run_argv,
        cwd=REPO_ROOT,
        env=env,
        verbose=verbose,
    )
    if run_result != 0:
        bazel_launcher.remove_launch_script(script_path)
        return run_result, None
    if not script_path.is_file() or script_path.stat().st_size == 0:
        print(
            f"dev.py: Bazel did not write launch script {script_path}", file=sys.stderr
        )
        bazel_launcher.remove_launch_script(script_path)
        return 1, None
    return 0, BazelLaunch(
        target=target,
        script_path=script_path,
        run_cwd=run_cwd,
        argument_separator=argument_separator,
        runfiles_arguments=metadata.runfiles_arguments,
        marked_runfiles_arguments=metadata.marked_runfiles_arguments,
        runfiles_environment_names=metadata.runfiles_environment_names,
    )


def prepare_bazel_process(
    launch: BazelLaunch,
    *,
    env: dict[str, str] | None,
) -> tuple[int, BazelProcess | None]:
    """Prepares a lock-free process while preserving native host execution."""
    try:
        launch_argv = launch.argv(env)
    except ValueError as exc:
        print(f"dev.py: {exc}", file=sys.stderr)
        bazel_launcher.remove_launch_script(launch.script_path)
        return 1, None

    if os.name != "nt":
        return 0, BazelProcess(
            target=launch.target,
            argv=launch_argv,
            cwd=REPO_ROOT,
            env=launch.environment(env),
            script_path=launch.script_path,
        )

    # Bazel emits a batch file on Windows. Executing it as the final process
    # would leave cmd.exe between PID/signal-based tools and the target. Run
    # the batch file only long enough for the helper to emit Bazel's resolved
    # process contract, then execute that target directly below.
    try:
        completed = subprocess.run(
            launch_argv,
            cwd=REPO_ROOT,
            env=launch.environment(env, materialize=True),
            stdout=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        print(
            f"dev.py: failed to materialize Bazel launch for {launch.target}: {exc}",
            file=sys.stderr,
        )
        return 127, None
    finally:
        bazel_launcher.remove_launch_script(launch.script_path)
    if completed.returncode != 0:
        return process_exit_code(completed.returncode), None
    try:
        process_launch = bazel_launcher.decode_process_launch(completed.stdout)
    except ValueError as exc:
        print(
            f"dev.py: failed to materialize Bazel launch for {launch.target}: {exc}",
            file=sys.stderr,
        )
        return 1, None
    return 0, BazelProcess(
        target=launch.target,
        argv=process_launch.argv,
        cwd=process_launch.cwd,
        env=process_launch.env,
    )


def execute_bazel_launch(
    launch: BazelLaunch,
    *,
    env: dict[str, str] | None,
) -> int:
    """Executes one prepared Bazel target with direct-process semantics."""
    process_result, process_launch = prepare_bazel_process(launch, env=env)
    if process_result != 0 or process_launch is None:
        return process_result
    result = exec_path(
        process_launch.argv,
        cwd=process_launch.cwd,
        env=process_launch.env,
    )
    if process_launch.script_path is not None:
        bazel_launcher.remove_launch_script(process_launch.script_path)
    return result


def resolve_bazel_output_path(
    *,
    bazel: str,
    target: str,
    bazel_args: list[str],
    cwd: Path,
    env: dict[str, str] | None,
) -> Path | None:
    cquery = run_captured(
        [bazel, "cquery", "--output=files", *bazel_args, target],
        cwd=cwd,
        env=env,
    )
    if cquery.returncode != 0:
        print_process_failure(cquery)
        return None
    output_paths = [
        line.strip()
        for line in cquery.stdout.splitlines()
        if line.strip() and not line.startswith("INFO:")
    ]
    if not output_paths:
        return None

    execution_root = bazel_execution_root(bazel=bazel, cwd=cwd, env=env)
    candidates = []
    for output_path in output_paths:
        path = Path(output_path)
        if not path.is_absolute() and execution_root is not None:
            path = execution_root / path
        elif not path.is_absolute():
            path = cwd / path
        candidates.append(path)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return candidates[0]


def bazel_execution_root(
    *, bazel: str, cwd: Path, env: dict[str, str] | None
) -> Path | None:
    completed = run_captured([bazel, "info", "execution_root"], cwd=cwd, env=env)
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line:
            return Path(line)
    return None


def print_process_failure(completed: subprocess.CompletedProcess[str]) -> None:
    if completed.stdout:
        print(completed.stdout.rstrip(), file=sys.stderr)
    if completed.stderr:
        print(completed.stderr.rstrip(), file=sys.stderr)


def exec_path(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
) -> int:
    try:
        os.chdir(cwd)
        bazel_launcher.exec_process(argv, env or os.environ)
    except OSError as exc:
        print(f"dev.py: failed to exec {quote_command(argv)}: {exc}", file=sys.stderr)
        return 127
    raise AssertionError("os.execvpe returned unexpectedly")


def cleanup_try_scratch(scratch_dir: Path) -> None:
    shutil.rmtree(scratch_dir, ignore_errors=True)
    try:
        BAZEL_TRY_ROOT.rmdir()
    except OSError:
        pass
    try:
        LOCAL_STATE_ROOT.rmdir()
    except OSError:
        pass


def cleanup_compile_commands_scratch(scratch_dir: Path) -> None:
    shutil.rmtree(scratch_dir, ignore_errors=True)
    try:
        BAZEL_COMPILE_COMMANDS_ROOT.rmdir()
    except OSError:
        pass


def run_fuzzers(
    fuzzer_processes: list[BazelProcess],
) -> int:
    processes: list[tuple[BazelProcess, subprocess.Popen]] = []
    try:
        for process_launch in fuzzer_processes:
            processes.append(
                (
                    process_launch,
                    subprocess.Popen(
                        process_launch.argv,
                        cwd=process_launch.cwd,
                        env=process_launch.env,
                    ),
                )
            )
    except (OSError, ValueError) as exc:
        print(f"dev.py: failed to start fuzzer: {exc}", file=sys.stderr)
        stop_fuzzers(processes)
        for _, process in processes:
            process.wait()
        for process_launch in fuzzer_processes:
            if process_launch.script_path is not None:
                bazel_launcher.remove_launch_script(process_launch.script_path)
        return 127

    result = 0
    try:
        while processes:
            for process_launch, process in list(processes):
                process_result = process.poll()
                if process_result is None:
                    continue
                processes.remove((process_launch, process))
                if process_launch.script_path is not None:
                    bazel_launcher.remove_launch_script(process_launch.script_path)
                if process_result != 0 and result == 0:
                    print(
                        f"dev.py: fuzzer failed: {process_launch.target} exited "
                        f"{process_result}",
                        file=sys.stderr,
                    )
                    result = process_exit_code(process_result)
            if result != 0:
                stop_fuzzers(processes)
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_fuzzers(processes)
        result = 130
    for process_launch, process in processes:
        process.wait()
        if process_launch.script_path is not None:
            bazel_launcher.remove_launch_script(process_launch.script_path)
    return result


def process_exit_code(process_result: int) -> int:
    if process_result < 0:
        return 128 + abs(process_result)
    return process_result


def stop_fuzzers(processes: list[tuple[BazelProcess, subprocess.Popen]]) -> None:
    for _, process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)


@dataclass(frozen=True)
class BazelRunStep:
    bazel: str
    command: BazelRunCommand
    env: dict[str, str] | None = None

    def describe(self) -> str:
        lines = [f"# bazel run {self.command.target}"]
        if self.command.print_path:
            lines.extend(
                [
                    quote_command(
                        [
                            self.bazel,
                            "build",
                            *self.command.bazel_args,
                            self.command.target,
                        ]
                    ),
                    quote_command(
                        [
                            self.bazel,
                            "cquery",
                            "--output=files",
                            *self.command.bazel_args,
                            self.command.target,
                        ]
                    ),
                ]
            )
            lines.append("# print built executable path")
        else:
            lines.append(
                quote_command(
                    [
                        self.bazel,
                        "cquery",
                        "--output=starlark",
                        *self.command.bazel_args,
                        self.command.target,
                    ]
                )
            )
            run_argv = [
                self.bazel,
                "run",
                *self.command.bazel_args,
                "--script_path=<launch script>",
                "--norun_in_cwd",
                "--run_under=<host Python launcher>",
                self.command.target,
            ]
            if self.command.program_args:
                run_argv.extend(["--", *self.command.program_args])
            lines.append(quote_command(run_argv))
            lines.append("exec " + quote_command(["<Bazel launch script>"]))
        return "\n".join(lines)

    def run(self, verbose: bool = False) -> int:
        if not self.command.print_path:
            launch_result, launch = generate_bazel_launch(
                bazel=self.bazel,
                target=self.command.target,
                bazel_args=self.command.bazel_args,
                program_args=self.command.program_args,
                run_cwd=self.command.run_cwd,
                env=self.env,
                verbose=verbose,
            )
            if launch_result != 0 or launch is None:
                return launch_result
            return execute_bazel_launch(launch, env=self.env)

        build_argv = [
            self.bazel,
            "build",
            *self.command.bazel_args,
            self.command.target,
        ]
        build_result = run_quietly(
            build_argv,
            cwd=REPO_ROOT,
            env=self.env,
            verbose=verbose,
        )
        if build_result != 0:
            return build_result
        binary_path = resolve_bazel_output_path(
            bazel=self.bazel,
            target=self.command.target,
            bazel_args=self.command.bazel_args,
            cwd=REPO_ROOT,
            env=self.env,
        )
        if binary_path is None or not binary_path.is_file():
            print(
                f"dev.py: could not find built binary for {self.command.target}",
                file=sys.stderr,
            )
            return 1
        if self.command.print_path:
            print(binary_path)
            return 0
        raise AssertionError("non-print Bazel run returned to output-path execution")


@dataclass(frozen=True)
class BazelFuzzStep:
    bazel: str
    command: BazelTargetCommand
    env: dict[str, str] | None = None

    @property
    def bazel_args(self) -> list[str]:
        return ["--config=fuzzer", *self.command.bazel_args]

    def describe(self) -> str:
        lines = [f"# bazel fuzz {self.command.target}"]
        if "..." in self.command.target:
            lines.append(quote_command([self.bazel, "query", self.command.target]))
            lines.append(
                quote_command([self.bazel, "build", *self.bazel_args, "<fuzz targets>"])
            )
            lines.append("# generate one canonical Bazel launch script per fuzzer")
            lines.append("run launch scripts concurrently without the Bazel lock")
        else:
            lines.append("# build and generate canonical Bazel launch script")
            lines.append(
                quote_command(
                    [
                        self.bazel,
                        "run",
                        *self.bazel_args,
                        "--script_path=<launch script>",
                        "--norun_in_cwd",
                        "--run_under=<host Python launcher>",
                        self.command.target,
                        "--",
                        "<corpus>",
                        "-artifact_prefix=<artifacts>/",
                        *self.command.program_args,
                    ]
                )
            )
            lines.append("exec " + quote_command(["<Bazel launch script>"]))
        return "\n".join(lines)

    def run(self, verbose: bool = False) -> int:
        if "..." in self.command.target:
            return self.run_target_pattern(verbose=verbose)
        return self.run_target(self.command.target, verbose=verbose)

    def run_target_pattern(self, *, verbose: bool) -> int:
        discovered = run_captured(
            [self.bazel, "query", self.command.target],
            cwd=REPO_ROOT,
            env=self.env,
        )
        if discovered.returncode != 0:
            print_process_failure(discovered)
            return discovered.returncode
        targets = sorted(
            line.strip()
            for line in discovered.stdout.splitlines()
            if line.strip().endswith("_fuzz")
        )
        if not targets:
            print(
                f"dev.py: no fuzz targets found under {self.command.target}",
                file=sys.stderr,
            )
            return 1
        build_result = run_quietly(
            [self.bazel, "build", *self.bazel_args, *targets],
            cwd=REPO_ROOT,
            env=self.env,
            verbose=verbose,
        )
        if build_result != 0:
            return build_result
        metadata_result, metadata_by_target = resolve_bazel_launch_metadata_for_targets(
            bazel=self.bazel,
            targets=targets,
            bazel_args=self.bazel_args,
            cwd=REPO_ROOT,
            env=self.env,
        )
        if metadata_result != 0 or metadata_by_target is None:
            return metadata_result
        fuzzer_launches = []
        for target in targets:
            launch_result, launch = self.generate_fuzzer_launch(
                target,
                verbose=verbose,
                metadata=metadata_by_target[target],
            )
            if launch_result != 0 or launch is None:
                for pending_launch in fuzzer_launches:
                    bazel_launcher.remove_launch_script(pending_launch.script_path)
                return launch_result
            fuzzer_launches.append(launch)
        fuzzer_processes = []
        for launch in fuzzer_launches:
            process_result, process_launch = prepare_bazel_process(
                launch,
                env=self.env,
            )
            if process_result != 0 or process_launch is None:
                for pending_launch in fuzzer_launches:
                    bazel_launcher.remove_launch_script(pending_launch.script_path)
                return process_result
            fuzzer_processes.append(process_launch)
        return run_fuzzers(fuzzer_processes)

    def run_target(self, target: str, *, verbose: bool) -> int:
        launch_result, launch = self.generate_fuzzer_launch(
            target,
            verbose=verbose,
        )
        if launch_result != 0 or launch is None:
            return launch_result
        return execute_bazel_launch(launch, env=self.env)

    def generate_fuzzer_launch(
        self,
        target: str,
        *,
        verbose: bool,
        metadata: BazelLaunchMetadata | None = None,
    ) -> tuple[int, BazelLaunch | None]:
        return generate_bazel_launch(
            bazel=self.bazel,
            target=target,
            bazel_args=self.bazel_args,
            program_args=self.fuzzer_program_args(target),
            run_cwd=REPO_ROOT,
            env=self.env,
            verbose=verbose,
            metadata=metadata,
        )

    def fuzzer_program_args(self, target: str) -> list[str]:
        target_dir = fuzz.bazel_fuzz_target_dir(target)
        corpus_dir = target_dir / "corpus"
        artifact_dir = target_dir / "artifacts"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return [
            str(corpus_dir),
            f"-artifact_prefix={artifact_dir}/",
            *self.command.program_args,
        ]


@dataclass(frozen=True)
class BazelCompileCommandsStep:
    bazel: str
    command: BazelCompileCommandsCommand
    env: dict[str, str] | None = None

    def describe(self) -> str:
        scratch = BAZEL_COMPILE_COMMANDS_ROOT / "run-<pid>"
        lines = [
            "# bazel compile-commands",
            quote_command(self.aspect_build_argv(scratch / "build_events.json")),
            "merge compile command fragments from "
            + quote_command([str(scratch / "build_events.json")]),
            "write " + quote_command([str(self.command.output)]),
        ]
        return "\n".join(lines)

    def run(self, verbose: bool = False) -> int:
        scratch_dir = BAZEL_COMPILE_COMMANDS_ROOT / f"run-{os.getpid()}"
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir)
        scratch_dir.mkdir(parents=True)
        build_events_path = scratch_dir / "build_events.json"
        try:
            build_result = run_quietly(
                self.aspect_build_argv(build_events_path),
                cwd=REPO_ROOT,
                env=self.env,
                verbose=verbose,
            )
            if build_result != 0:
                return build_result
            command_directory = bazel_execution_root(
                bazel=self.bazel,
                cwd=REPO_ROOT,
                env=self.env,
            )
            if command_directory is None:
                print(
                    "dev.py: could not resolve Bazel execution root",
                    file=sys.stderr,
                )
                return 1
            fragment_paths = compile_commands_merge.fragment_paths_from_bep(
                build_events_path,
            )
            if not fragment_paths:
                print(
                    "dev.py: no C/C++ compile command fragments were produced",
                    file=sys.stderr,
                )
                return 1
            compile_commands_merge.write_merged_compile_commands(
                output_path=self.command.output,
                fragment_paths=fragment_paths,
                command_directory=command_directory,
            )
            print(self.command.output)
            return 0
        finally:
            if not self.command.keep:
                cleanup_compile_commands_scratch(scratch_dir)

    def aspect_build_argv(self, build_events_path: Path) -> list[str]:
        return [
            self.bazel,
            "build",
            *self.command.bazel_args,
            f"--aspects={COMPILE_COMMANDS_ASPECT}",
            "--output_groups=" + compile_commands_merge.COMPILE_COMMANDS_OUTPUT_GROUP,
            f"--build_event_json_file={build_events_path}",
            "--ui_event_filters=-info",
            "--",
            *self.command.targets,
        ]


@dataclass(frozen=True)
class BazelTryStep:
    bazel: str
    command: BazelTryCommand
    env: dict[str, str] | None = None

    def _bazel_args(self) -> list[str]:
        return [*BAZEL_TRY_COMMON_ARGS, *self.command.bazel_args]

    def _build_command(self, label: str) -> list[str]:
        return [self.bazel, "build", *self._bazel_args(), label]

    def describe(self) -> str:
        scratch = BAZEL_TRY_ROOT / "run-<pid>"
        label = f"{BAZEL_TRY_LABEL_ROOT}/run-<pid>:snippet"
        lines = [f"# write {scratch}/BUILD.bazel"]
        if self.command.compile_only:
            lines.append(quote_command(self._build_command(label)))
            lines.append("# compile only")
        else:
            lines.append("# build and generate canonical Bazel launch script")
            lines.append("exec " + quote_command(["<Bazel launch script>"]))
        return "\n".join(lines)

    def run(self, verbose: bool = False) -> int:
        scratch_dir = BAZEL_TRY_ROOT / f"run-{os.getpid()}"
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir)
        scratch_dir.mkdir(parents=True)
        launch = None
        try:
            try:
                source_names, source_texts = self.materialize_sources(scratch_dir)
            except (FileNotFoundError, ValueError) as exc:
                print(f"dev.py: {exc}", file=sys.stderr)
                return 2
            deps = list(dict.fromkeys(self.command.explicit_deps))
            if self.command.infer_deps:
                deps = list(dict.fromkeys([*deps, *self.infer_deps(source_texts)]))
            write_try_build_file(
                scratch_dir / "BUILD.bazel",
                source_names=source_names,
                deps=deps,
                testonly=True,
            )
            label = (
                f"{BAZEL_TRY_LABEL_ROOT}/{scratch_dir.name}:{DEFAULT_TRY_BINARY_NAME}"
            )
            if self.command.compile_only:
                build_result = run_quietly(
                    self._build_command(label),
                    cwd=REPO_ROOT,
                    env=self.env,
                    verbose=verbose,
                )
                if build_result != 0:
                    return build_result
            else:
                launch_result, launch = generate_bazel_launch(
                    bazel=self.bazel,
                    target=label,
                    bazel_args=self._bazel_args(),
                    program_args=self.command.program_args,
                    run_cwd=self.command.run_cwd,
                    env=self.env,
                    verbose=verbose,
                )
                if launch_result != 0 or launch is None:
                    return launch_result

            if self.command.output is not None:
                copy_result = self.copy_output(label, self.command.output)
                if copy_result != 0:
                    return copy_result
            if self.command.compile_only:
                return 0
            if not self.command.keep:
                cleanup_try_scratch(scratch_dir)
            return execute_bazel_launch(launch, env=self.env)
        finally:
            if launch is not None:
                bazel_launcher.remove_launch_script(launch.script_path)
            if not self.command.keep:
                cleanup_try_scratch(scratch_dir)

    def copy_output(self, label: str, output_path: Path) -> int:
        """Copies a built try executable to the caller-selected path."""
        binary_path = resolve_bazel_output_path(
            bazel=self.bazel,
            target=label,
            bazel_args=self._bazel_args(),
            cwd=REPO_ROOT,
            env=self.env,
        )
        if binary_path is None or not binary_path.is_file():
            print("dev.py: could not find built snippet binary", file=sys.stderr)
            return 1
        if not output_path.is_absolute():
            output_path = self.command.run_cwd / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(binary_path, output_path)
        output_path.chmod(output_path.stat().st_mode | 0o755)
        return 0

    def materialize_sources(self, scratch_dir: Path) -> tuple[list[str], list[str]]:
        source_names = []
        source_texts = []
        for index, source in enumerate(self.command.inline_sources):
            suffix = (
                "cc" if inline_source_is_cxx(self.command.language, source) else "c"
            )
            source_name = f"inline_{index}.{suffix}"
            (scratch_dir / source_name).write_text(source + "\n", encoding="utf-8")
            source_names.append(source_name)
            source_texts.append(source)
        for file_index, source_path in enumerate(self.command.files):
            resolved_path = source_path
            if not resolved_path.is_absolute():
                resolved_path = self.command.run_cwd / resolved_path
            if not resolved_path.is_file():
                raise FileNotFoundError(f"source file not found: {source_path}")
            source_name = scratch_input_name(
                resolved_path, self.command.run_cwd, file_index, source_names
            )
            scratch_source_path = scratch_dir / source_name
            scratch_source_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved_path, scratch_source_path)
            text = resolved_path.read_text(encoding="utf-8", errors="ignore")
            source_names.append(source_name)
            source_texts.append(text)
        if not source_names:
            if sys.stdin.isatty():
                raise ValueError("bazel try requires a file, -e source, or stdin")
            stdin_source = sys.stdin.read()
            if not stdin_source:
                raise ValueError("bazel try received empty stdin")
            suffix = (
                "cc"
                if inline_source_is_cxx(self.command.language, stdin_source)
                else "c"
            )
            source_name = f"stdin.{suffix}"
            (scratch_dir / source_name).write_text(stdin_source, encoding="utf-8")
            source_names.append(source_name)
            source_texts.append(stdin_source)
        return source_names, source_texts

    def infer_deps(self, source_texts: list[str]) -> list[str]:
        deps = []
        for header in sorted(extract_quoted_includes(source_texts)):
            dep = infer_dep_for_header(self.bazel, header, env=self.env)
            if dep is not None:
                deps.append(dep)
        return deps


def infer_language(source_names: list[str], source_texts: list[str]) -> str:
    if any(Path(name).suffix in (".cc", ".cpp", ".cxx", ".C") for name in source_names):
        return "c++"
    joined_source = "\n".join(source_texts)
    if (
        "iree/testing/gtest_harness.h" in joined_source
        or "iree/testing/gbenchmark_harness.h" in joined_source
        or re.search(r"\b(TEST|BENCHMARK)\s*\(", joined_source)
    ):
        return "c++"
    return "c"


def inline_source_is_cxx(language: str | None, source: str) -> bool:
    if language is not None:
        return language == "c++"
    return infer_language(["inline.c"], [source]) == "c++"


def extract_quoted_includes(source_texts: list[str]) -> set[str]:
    headers = set()
    include_pattern = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
    for source in source_texts:
        headers.update(include_pattern.findall(source))
    return headers


def scratch_input_name(
    resolved_path: Path, run_cwd: Path, file_index: int, existing_names: list[str]
) -> str:
    try:
        relative_path = resolved_path.resolve().relative_to(run_cwd.resolve())
    except ValueError:
        relative_path = Path("external") / str(file_index) / resolved_path.name

    if not relative_path.parts or any(part == ".." for part in relative_path.parts):
        relative_path = Path("external") / str(file_index) / resolved_path.name

    source_name = relative_path.as_posix()
    if source_name in existing_names:
        raise ValueError(f"multiple try inputs map to {source_name}")
    return source_name


def infer_dep_for_header(
    bazel: str, header: str, *, env: dict[str, str] | None
) -> str | None:
    special_dep = SPECIAL_HEADER_DEPS.get(header)
    if special_dep is not None:
        return special_dep

    header_path = header_path_for_include(header)
    if header_path is None:
        return None
    return infer_dep_for_header_path(bazel, header_path, env=env)


def header_path_for_include(header: str) -> Path | None:
    for prefix, root in HEADER_ROOTS:
        if header.startswith(prefix):
            return root / header
    return None


def infer_dep_for_header_path(
    bazel: str, header_path: Path, *, env: dict[str, str] | None
) -> str | None:
    package_dir = header_path.parent
    if not header_path.is_file():
        return fallback_dep_for_header_dir(package_dir)
    package_dir = nearest_package_dir(package_dir)
    if package_dir is None:
        return None
    package_label = "//" + package_dir.relative_to(REPO_ROOT).as_posix()
    header_label_path = re.escape(header_path.relative_to(package_dir).as_posix())
    labels = query_rules_with_header(
        bazel,
        header_label_path=header_label_path,
        target_pattern=f"{package_label}:*",
        env=env,
    )
    if not labels:
        labels = query_rules_with_header(
            bazel,
            header_label_path=header_label_path,
            target_pattern=f"{package_label}/...",
            env=env,
        )
    if labels:
        return labels[0]
    return fallback_dep_for_header_dir(package_dir)


def query_rules_with_header(
    bazel: str,
    *,
    header_label_path: str,
    target_pattern: str,
    env: dict[str, str] | None,
) -> list[str]:
    query_expression = (
        f'kind(".* rule", attr("hdrs", "{header_label_path}", {target_pattern}))'
    )
    completed = run_captured([bazel, "query", query_expression], cwd=REPO_ROOT, env=env)
    if completed.returncode == 0:
        return sorted(
            line.strip() for line in completed.stdout.splitlines() if line.strip()
        )
    return []


def nearest_package_dir(path: Path) -> Path | None:
    current = path
    while current != REPO_ROOT and current != current.parent:
        if (current / "BUILD.bazel").is_file():
            return current
        current = current.parent
    return None


def fallback_dep_for_header_dir(package_dir: Path) -> str | None:
    current = nearest_package_dir(package_dir)
    if current is not None:
        package = current.relative_to(REPO_ROOT).as_posix()
        target_name = current.name
        return f"//{package}:{target_name}"
    return None


def write_try_build_file(
    path: Path,
    *,
    source_names: list[str],
    deps: list[str],
    testonly: bool,
) -> None:
    lines = [
        'load("//build_tools/bazel:cc.bzl", "iree_cc_binary")',
        "",
        "iree_cc_binary(",
        f'    name = "{DEFAULT_TRY_BINARY_NAME}",',
        "    srcs = [",
    ]
    lines.extend(f'        "{source_name}",' for source_name in source_names)
    lines.append("    ],")
    if testonly:
        lines.append("    testonly = True,")
    if deps:
        lines.append("    deps = [")
        lines.extend(f'        "{dep}",' for dep in deps)
        lines.append("    ],")
    lines.append(")")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
