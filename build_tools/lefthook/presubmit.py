#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Root presubmit dispatcher.

The root owns repository-wide hygiene and maps changed files to project
presubmit entry points. Project-specific test policy stays in each project.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from build_tools.devtools.source_lock import (
    NonEmptyTrackedFileSnapshot,
    source_mutation_lock,
)

BUILDIFIER_EXTENSIONS = {".bzl", ".BUILD"}
BUILDIFIER_NAMES = {
    "BUILD",
    "BUILD.bazel",
    "MODULE.bazel",
    "WORKSPACE",
    "WORKSPACE.bazel",
    "WORKSPACE.bzlmod",
}
C_FORMAT_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".m",
    ".mm",
}
C_INCLUDE_FRAGMENT_EXTENSIONS = {
    ".def",
    ".inc",
    ".inl",
}
C_ANALYSIS_EXTENSIONS = C_FORMAT_EXTENSIONS | C_INCLUDE_FRAGMENT_EXTENSIONS
C_FORMAT_BATCH_SIZE = 64
C_FORMAT_DEFAULT_MAX_JOBS = 16
SEMGREP_CONFIG = "build_tools/static_analysis/semgrep/iree.yml"
SEMGREP_EXTENSIONS = C_ANALYSIS_EXTENSIONS
SEMGREP_PATH_PREFIXES = (
    "runtime/src/iree/",
    "loom/src/loom/",
    "libhrx/",
)
SEMGREP_DEFAULT_MAX_JOBS = 14
CLANG_TIDY_DEFAULT_CORE_DIVISOR = 2
CLANG_TIDY_DEFAULT_MAX_JOBS = 96
CLANG_TIDY_ASPECT = "//build_tools/clang_tidy:clang_tidy.bzl%collect_clang_tidy_aspect"
CLANG_TIDY_CMAKE_GENERATED_INPUTS_TARGET = "iree_generated_compile_inputs"
CLANG_TIDY_TRANSLATION_UNIT_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
}
CLANG_TIDY_HEADER_EXTENSIONS = {
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
} | C_INCLUDE_FRAGMENT_EXTENSIONS
CLANG_TIDY_EXTENSIONS = (
    CLANG_TIDY_TRANSLATION_UNIT_EXTENSIONS | CLANG_TIDY_HEADER_EXTENSIONS
)
CLANG_TIDY_INFRA_PREFIX = "build_tools/clang_tidy/"
CLANG_TIDY_FIXES_OUTPUT_GROUP = "iree_clang_tidy_fixes"
CLANG_TIDY_LOCAL_FIXES_OUTPUT_GROUP = "iree_clang_tidy_local_fixes"
CLANG_TIDY_LOCAL_OUTPUT_GROUP = "iree_clang_tidy_local_reports"
CLANG_TIDY_OUTPUT_GROUP = "iree_clang_tidy_reports"
CLANG_TIDY_PATH_PREFIXES = SEMGREP_PATH_PREFIXES
CLANG_TIDY_REPO_ENV = "--repo_env=IREE_CLANG_TIDY_LLVM=auto"
CLANG_TIDY_CHECKS = "-*,iree-*"
CLANG_TIDY_CMAKE_BUILD_DIR = REPO_ROOT / ".tmp" / "iree-clang-tidy-plugin"
CLANG_TIDY_FIXES_ROOT = "iree-clang-tidy-fixes"
CLANG_TIDY_SETUP_HINT = (
    "install LLVM clang-tidy/clang++/llvm-config or set "
    "IREE_CLANG_TIDY_LLVM_CONFIG/IREE_CLANG_TIDY_LLVM_ROOT"
)
CMAKE_BUILD_DIR_ENV = "IREE_CMAKE_BUILD_DIR"
DEVTOOLS_TMP_ENV = "IREE_DEVTOOLS_TMP"
CMAKE_STATE_SUBPATH = ("iree", "cmake_build_dir")
CMAKE_DEFAULT_BUILD_DIR = REPO_ROOT / "build" / "cmake"
CMAKE_CLANG_TIDY_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".m",
    ".mm",
}
PYTHON_EXTENSIONS = {".py"}
WATCHWORD_PATTERNS = (
    re.compile("DO " + "NOT SUBMIT"),
    re.compile("DO_" + "NOT_SUBMIT"),
    re.compile("DO" + "NOTSUBMIT"),
    re.compile("TODO before " + "submit", re.IGNORECASE),
)
FAILURE_OUTPUT_LINE_LIMIT = 240


def git_worktree_dir() -> Path:
    """Returns the worktree-specific Git directory for scratch files."""
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--absolute-git-dir"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    )


CONFLICT_MARKER_PATTERN = re.compile(r"^(<<<<<<< .+|>>>>>>> .+)")


@dataclass(frozen=True)
class Project:
    name: str
    root: str
    script: str


PROJECTS = (
    Project(
        name="runtime",
        root="runtime/",
        script="runtime/build_tools/presubmit.py",
    ),
    Project(
        name="libhrx",
        root="libhrx/",
        script="libhrx/build_tools/presubmit.py",
    ),
    Project(
        name="loom",
        root="loom/",
        script="loom/build_tools/presubmit.py",
    ),
)

GLOBAL_PROJECT_TRIGGERS = (
    "BUILD.bazel",
    "MODULE.bazel",
    ".bazelrc",
    ".bazel_to_cmake.cfg.py",
    "requirements",
)
ROOT_DEVTOOLS_TRIGGERS = (
    ".github/workflows/presubmit.yml",
    "CONTRIBUTING.md",
    "README.md",
    "dev.py",
    "lefthook.yml",
    "requirements-analysis",
    "requirements-dev",
    "build_tools/static_analysis/",
    "build_tools/devtools/",
    "build_tools/lefthook/",
)
CLANG_TIDY_FULL_SCOPE_EXACT_PATHS = {
    ".bazelrc",
    ".bazelversion",
    ".github/workflows/presubmit.yml",
    "MODULE.bazel",
    "MODULE.bazel.lock",
    "WORKSPACE",
    "WORKSPACE.bazel",
    "WORKSPACE.bzlmod",
    "dev.py",
}
CLANG_TIDY_FULL_SCOPE_PREFIXES = (
    "build_tools/bazel/",
    "build_tools/devtools/",
    "build_tools/lefthook/",
    CLANG_TIDY_INFRA_PREFIX,
)


@dataclass(frozen=True)
class PresubmitInputs:
    mode: str
    selected_paths: list[str]
    changed_paths: list[str]
    tracked_paths: list[str] | None = None
    base: str | None = None

    def all_paths(self) -> list[str]:
        tracked_paths = self.tracked_paths
        if tracked_paths is None:
            tracked_paths = git_list(["ls-files", "-z"])
        return unique_paths([*tracked_paths, *self.changed_paths])


@dataclass(frozen=True)
class StaticAnalysisProvider:
    name: str
    profiles: frozenset[str]
    runner: Callable[[PresubmitInputs, str, str, bool], bool]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repository presubmit checks.")
    parser.add_argument(
        "--lane",
        choices=("bazel", "cmake"),
        default="bazel",
        help="Build-system lane used for project tests. Defaults to bazel.",
    )
    parser.add_argument(
        "--profile",
        choices=("default", "paranoid", "ci"),
        default=os.environ.get("IREE_PRESUBMIT_PROFILE", "default"),
        help="Presubmit profile. Defaults to IREE_PRESUBMIT_PROFILE or default.",
    )

    mutation = parser.add_mutually_exclusive_group()
    mutation.add_argument("--fix", action="store_true", help="Apply local fixups.")
    mutation.add_argument(
        "--check",
        action="store_true",
        help="Check without writing files. This is the CI-safe mode.",
    )

    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--changed",
        action="store_true",
        help="Use local staged, unstaged, and untracked files.",
    )
    inputs.add_argument(
        "--staged", action="store_true", help="Use files staged for commit."
    )
    inputs.add_argument(
        "--commit",
        action="store_true",
        help=(
            "Use files staged for commit plus files changed by HEAD. This is "
            "the Git hook scope and covers amended commit contents."
        ),
    )
    inputs.add_argument("--all", action="store_true", help="Use all tracked files.")
    inputs.add_argument(
        "--base",
        metavar="GIT_REF",
        help=(
            "Use files changed from the merge base with GIT_REF through HEAD, "
            "plus local staged, unstaged, and untracked files."
        ),
    )
    inputs.add_argument(
        "--since",
        metavar="GIT_REF",
        help="Use tracked files changed since the given Git ref.",
    )
    inputs.add_argument(
        "--files-from",
        metavar="PATH",
        help="Use newline-separated repo-relative files from PATH.",
    )

    parser.add_argument(
        "--hygiene",
        action="store_true",
        help="Run formatting, generated-file, and cheap invariant checks.",
    )
    parser.add_argument(
        "--tests",
        action="store_true",
        help="Run affected project tests for the selected lane.",
    )
    parser.add_argument(
        "--no-project-tests",
        "--no_project_tests",
        dest="project_tests",
        action="store_false",
        default=True,
        help="Skip runtime/libhrx/loom project tests while still running root devtools tests.",
    )
    parser.add_argument(
        "--static-analysis",
        action="store_true",
        help="Run configured static-analysis providers.",
    )
    parser.add_argument(
        "--clang-tidy",
        action="store_true",
        help="Run only the clang-tidy static-analysis provider.",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the selected plan before running it.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream command output and print exact commands.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Explicit repo-relative paths. Used by the generated Git hook.",
    )

    args = parser.parse_args()
    args.verbose = args.verbose or env_flag("IREE_PRESUBMIT_VERBOSE")
    if args.paths and (
        args.changed
        or args.staged
        or args.commit
        or args.all
        or args.base
        or args.since
        or args.files_from
    ):
        parser.error("explicit paths cannot be combined with another input mode")
    apply_profile_defaults(args)
    return args


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def apply_profile_defaults(args: argparse.Namespace) -> None:
    if not args.fix and not args.check:
        args.check = args.profile == "ci"
        args.fix = not args.check

    if (
        not args.paths
        and not args.changed
        and not args.staged
        and not args.commit
        and not args.all
        and not args.base
        and not args.since
        and not args.files_from
    ):
        args.all = args.profile == "ci"
        args.changed = not args.all

    if (
        not args.hygiene
        and not args.tests
        and not args.static_analysis
        and not args.clang_tidy
    ):
        args.hygiene = True
        if args.profile in ("paranoid", "ci"):
            args.tests = True
            args.static_analysis = True


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes} m {remainder:.0f} s"


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def elide_failure_output(output: str) -> str:
    stripped_output = output.rstrip()
    if not stripped_output:
        return ""

    lines = stripped_output.splitlines()
    if len(lines) <= FAILURE_OUTPUT_LINE_LIMIT:
        return stripped_output

    head_line_count = FAILURE_OUTPUT_LINE_LIMIT // 3
    tail_line_count = FAILURE_OUTPUT_LINE_LIMIT - head_line_count
    omitted_line_count = len(lines) - head_line_count - tail_line_count
    return "\n".join(
        [
            *lines[:head_line_count],
            (
                f"[... omitted {omitted_line_count} lines; rerun with --verbose "
                "for full streaming output ...]"
            ),
            *lines[-tail_line_count:],
        ]
    )


def print_section(title: str) -> None:
    print(f"\n== {title} ==")


def skip_step(description: str, reason: str) -> bool:
    print(f"[skip] {description}: {reason}")
    return True


def print_step_failure(
    description: str,
    elapsed_seconds: float,
    *,
    command: list[str] | None = None,
    output: str | None = None,
    exit_code: int | None = None,
) -> None:
    if exit_code is None:
        print(f"[fail] {description} ({format_duration(elapsed_seconds)})")
    else:
        print(
            f"[fail] {description} ({format_duration(elapsed_seconds)}, "
            f"exit {exit_code})"
        )
    if command is not None:
        print("command:")
        print("  " + command_text(command))
    if output:
        print("output:")
        print(elide_failure_output(output))


def run_command(command: list[str], description: str, verbose: bool) -> bool:
    print(f"[run] {description}")
    sys.stdout.flush()
    start_time = time.monotonic()
    if verbose:
        print("  " + command_text(command))
        sys.stdout.flush()
        result = subprocess.run(command, cwd=REPO_ROOT)
        output = None
    else:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = result.stdout
    elapsed_seconds = time.monotonic() - start_time
    if result.returncode == 0:
        print(f"[ok] {description} ({format_duration(elapsed_seconds)})")
        return True
    print_step_failure(
        description,
        elapsed_seconds,
        command=None if verbose else command,
        output=output,
        exit_code=result.returncode,
    )
    return False


@dataclass(frozen=True)
class CommandBatchResult:
    command: list[str]
    returncode: int
    output: str


def run_command_batch(command: list[str]) -> CommandBatchResult:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return CommandBatchResult(
        command=command,
        returncode=result.returncode,
        output=result.stdout,
    )


def run_parallel_commands(
    commands: list[list[str]],
    description: str,
    verbose: bool,
    *,
    jobs: int,
) -> bool:
    if not commands:
        return True
    if len(commands) == 1:
        return run_command(commands[0], description, verbose)

    print(f"[run] {description}")
    sys.stdout.flush()
    start_time = time.monotonic()
    if verbose:
        print(f"  {len(commands)} batches, {jobs} jobs")
        for command in commands:
            print("  " + command_text(command))
        sys.stdout.flush()

    results: list[CommandBatchResult | None] = [None] * len(commands)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_index = {
            executor.submit(run_command_batch, command): index
            for index, command in enumerate(commands)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            results[future_to_index[future]] = future.result()

    elapsed_seconds = time.monotonic() - start_time
    completed_results = [result for result in results if result is not None]
    failed_results = [result for result in completed_results if result.returncode != 0]
    if not failed_results:
        print(f"[ok] {description} ({format_duration(elapsed_seconds)})")
        return True

    failure_output_parts = []
    for result in failed_results:
        failure_output_parts.append("command:")
        failure_output_parts.append("  " + command_text(result.command))
        if result.output:
            failure_output_parts.append("output:")
            failure_output_parts.append(result.output.rstrip())
    print_step_failure(
        description,
        elapsed_seconds,
        output="\n".join(failure_output_parts),
        exit_code=max(result.returncode for result in failed_results),
    )
    return False


def run_inline_check(description: str, action, verbose: bool) -> bool:
    print(f"[run] {description}")
    sys.stdout.flush()
    start_time = time.monotonic()
    if verbose:
        ok = action()
        elapsed_seconds = time.monotonic() - start_time
        if ok:
            print(f"[ok] {description} ({format_duration(elapsed_seconds)})")
        else:
            print(f"[fail] {description} ({format_duration(elapsed_seconds)})")
        return ok

    output_stream = io.StringIO()
    with (
        contextlib.redirect_stdout(output_stream),
        contextlib.redirect_stderr(output_stream),
    ):
        try:
            ok = action()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            ok = False
    elapsed_seconds = time.monotonic() - start_time
    output = output_stream.getvalue()
    if ok:
        print(f"[ok] {description} ({format_duration(elapsed_seconds)})")
        return True
    print_step_failure(description, elapsed_seconds, output=output)
    return False


def require_tool(tool: str, description: str) -> bool:
    if shutil.which(tool):
        return True
    print(f"[fail] {description}: required tool '{tool}' was not found on PATH.")
    return False


def require_static_tool(tool: str, description: str, profile: str) -> bool:
    if shutil.which(tool):
        return True
    if profile == "ci":
        print(f"[fail] {description}: required tool '{tool}' was not found on PATH.")
        return False
    return skip_step(
        description,
        f"tool '{tool}' was not found on PATH; run python dev.py bazel setup --venv",
    )


def git_list(args: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [path.decode("utf-8") for path in result.stdout.split(b"\0") if path]


def unique_paths(paths: list[str]) -> list[str]:
    unique = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def changed_files() -> list[str]:
    return unique_paths(
        [
            *git_list(["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"]),
            *git_list(["diff", "--name-only", "-z", "--diff-filter=ACMR"]),
            *git_list(["ls-files", "--others", "--exclude-standard", "-z"]),
        ]
    )


def head_commit_files() -> list[str]:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return []
    return git_list(
        ["diff", "--name-only", "-z", "--diff-filter=ACMR", "HEAD^", "HEAD", "--"]
    )


def commit_files() -> list[str]:
    return unique_paths(
        [
            *git_list(["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"]),
            *head_commit_files(),
        ]
    )


def tracked_files() -> list[str]:
    return git_list(["ls-files", "-z"])


def presubmit_inputs(args: argparse.Namespace) -> PresubmitInputs:
    if args.paths:
        return PresubmitInputs(
            mode="explicit",
            selected_paths=args.paths,
            changed_paths=args.paths,
        )
    if args.changed:
        paths = changed_files()
        return PresubmitInputs(
            mode="changed",
            selected_paths=paths,
            changed_paths=paths,
        )
    if args.staged:
        paths = git_list(
            ["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"]
        )
        return PresubmitInputs(
            mode="staged",
            selected_paths=paths,
            changed_paths=paths,
        )
    if args.commit:
        paths = commit_files()
        return PresubmitInputs(
            mode="commit",
            selected_paths=paths,
            changed_paths=paths,
        )
    if args.all:
        paths = tracked_files()
        return PresubmitInputs(
            mode="all",
            selected_paths=paths,
            changed_paths=paths,
            tracked_paths=paths,
        )
    if args.base:
        paths = unique_paths(
            [
                *git_list(
                    [
                        "diff",
                        "--name-only",
                        "-z",
                        "--diff-filter=ACMR",
                        f"{args.base}...HEAD",
                        "--",
                    ]
                ),
                *changed_files(),
            ]
        )
        return PresubmitInputs(
            mode="base",
            selected_paths=paths,
            changed_paths=paths,
            tracked_paths=tracked_files(),
            base=args.base,
        )
    if args.files_from:
        with open(args.files_from, encoding="utf-8") as file_list:
            paths = [line.strip() for line in file_list if line.strip()]
        return PresubmitInputs(
            mode="files-from",
            selected_paths=paths,
            changed_paths=paths,
        )
    paths = git_list(
        ["diff", "--name-only", "-z", "--diff-filter=ACMR", args.since, "--"]
    )
    return PresubmitInputs(
        mode="since",
        selected_paths=paths,
        changed_paths=paths,
    )


def selected_files(args: argparse.Namespace) -> list[str]:
    return presubmit_inputs(args).selected_paths


def existing_files(paths: list[str]) -> list[str]:
    return [path for path in paths if (REPO_ROOT / path).is_file()]


def has_unstaged_changes(paths: list[str]) -> bool:
    if not paths:
        return False
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", *paths],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        return False
    if result.returncode == 1:
        return True
    raise subprocess.CalledProcessError(result.returncode, result.args)


def stage_files(paths: list[str], verbose: bool) -> bool:
    if not paths:
        return True
    if not has_unstaged_changes(paths):
        return True
    return run_command(["git", "add", "--", *paths], "Stage local fixups", verbose)


def is_buildifier_file(path: str) -> bool:
    file_path = Path(path)
    return (
        file_path.name in BUILDIFIER_NAMES or file_path.suffix in BUILDIFIER_EXTENSIONS
    )


def is_c_format_file(path: str) -> bool:
    return Path(path).suffix in C_FORMAT_EXTENSIONS


def is_semgrep_candidate_file(path: str) -> bool:
    return path.startswith(SEMGREP_PATH_PREFIXES) and (
        Path(path).suffix in SEMGREP_EXTENSIONS
    )


def is_clang_tidy_candidate_file(path: str) -> bool:
    return path.startswith(CLANG_TIDY_PATH_PREFIXES) and (
        Path(path).suffix in CLANG_TIDY_EXTENSIONS
    )


def is_clang_tidy_infra_file(path: str) -> bool:
    return path.startswith(CLANG_TIDY_INFRA_PREFIX)


def is_bazel_graph_file(path: str) -> bool:
    file_path = Path(path)
    return (
        path in CLANG_TIDY_FULL_SCOPE_EXACT_PATHS
        or file_path.name in BUILDIFIER_NAMES
        or file_path.suffix in BUILDIFIER_EXTENSIONS
    )


def is_clang_tidy_full_scope_trigger(path: str) -> bool:
    if is_bazel_graph_file(path):
        return True
    if any(path.startswith(prefix) for prefix in CLANG_TIDY_FULL_SCOPE_PREFIXES):
        return True
    return path.startswith(CLANG_TIDY_PATH_PREFIXES) and (
        Path(path).suffix in CLANG_TIDY_HEADER_EXTENSIONS
    )


def clang_tidy_uses_full_scope(inputs: PresubmitInputs) -> bool:
    return inputs.mode == "base" and any(
        is_clang_tidy_full_scope_trigger(path) for path in inputs.changed_paths
    )


def clang_tidy_analysis_paths(inputs: PresubmitInputs) -> list[str]:
    if clang_tidy_uses_full_scope(inputs):
        return inputs.all_paths()
    return inputs.selected_paths


def is_executable_path(path: str) -> bool:
    return Path(path).is_file() and os.access(path, os.X_OK)


def executable_from_env(env_names: tuple[str, ...]) -> str | None:
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value and is_executable_path(value):
            return value
    return None


def executable_from_roots(
    root_env_names: tuple[str, ...], tool_names: tuple[str, ...]
) -> str | None:
    for root_env_name in root_env_names:
        root = os.environ.get(root_env_name)
        if not root:
            continue
        for tool_name in tool_names:
            candidate = Path(root) / "bin" / tool_name
            if is_executable_path(str(candidate)):
                return str(candidate)
    return None


def executable_from_path(tool_names: tuple[str, ...]) -> str | None:
    for tool_name in tool_names:
        path = shutil.which(tool_name)
        if path:
            return path
    return None


def find_llvm_tool(
    *,
    env_names: tuple[str, ...] = (),
    root_env_names: tuple[str, ...] = (),
    tool_names: tuple[str, ...],
) -> str | None:
    return (
        executable_from_env(env_names)
        or executable_from_roots(root_env_names, tool_names)
        or executable_from_path(tool_names)
    )


def clang_tidy_llvm_available() -> bool:
    return clang_tidy_llvm_tools() is not None


def clang_tidy_llvm_tools() -> tuple[str, str, str] | None:
    root_env_names = ("IREE_CLANG_TIDY_LLVM_ROOT", "IREE_LLVM_ROOT", "LLVM_ROOT")
    llvm_config = find_llvm_tool(
        env_names=("IREE_CLANG_TIDY_LLVM_CONFIG", "LLVM_CONFIG"),
        root_env_names=root_env_names,
        tool_names=("llvm-config", "llvm-config-22"),
    )
    clang_tidy = find_llvm_tool(
        env_names=("IREE_CLANG_TIDY_BINARY", "CLANG_TIDY"),
        root_env_names=root_env_names,
        tool_names=("clang-tidy", "clang-tidy-22"),
    )
    clangxx = find_llvm_tool(
        env_names=(
            "IREE_CLANG_TIDY_CLANGXX_BINARY",
            "IREE_CLANGXX_BINARY",
            "CLANGXX",
        ),
        root_env_names=root_env_names,
        tool_names=("clang++", "clang++-22"),
    )
    if not llvm_config or not clang_tidy or not clangxx:
        return None
    return llvm_config, clang_tidy, clangxx


def clang_tidy_apply_replacements_tool() -> str | None:
    return find_llvm_tool(
        env_names=(
            "IREE_CLANG_APPLY_REPLACEMENTS_BINARY",
            "CLANG_APPLY_REPLACEMENTS",
        ),
        root_env_names=("IREE_CLANG_TIDY_LLVM_ROOT", "IREE_LLVM_ROOT", "LLVM_ROOT"),
        tool_names=("clang-apply-replacements", "clang-apply-replacements-22"),
    )


def clang_tidy_run_tool() -> str | None:
    return find_llvm_tool(
        env_names=("IREE_RUN_CLANG_TIDY_BINARY", "RUN_CLANG_TIDY"),
        root_env_names=("IREE_CLANG_TIDY_LLVM_ROOT", "IREE_LLVM_ROOT", "LLVM_ROOT"),
        tool_names=("run-clang-tidy", "run-clang-tidy-22"),
    )


def clang_tidy_required(profile: str) -> bool:
    return profile == "ci" and env_flag("IREE_CLANG_TIDY_REQUIRED")


def clang_format_jobs() -> int:
    configured_jobs = os.environ.get("IREE_CLANG_FORMAT_JOBS")
    if configured_jobs:
        try:
            return max(1, int(configured_jobs))
        except ValueError:
            return 1
    return min(os.cpu_count() or 1, C_FORMAT_DEFAULT_MAX_JOBS)


def bazel_package_target_for_path(path: str) -> str | None:
    current = (REPO_ROOT / path).parent
    while current != current.parent:
        if (current / "BUILD.bazel").is_file():
            relative = current.relative_to(REPO_ROOT).as_posix()
            return f"//{relative}:all" if relative else "//:all"
        if current == REPO_ROOT:
            break
        current = current.parent
    return None


def semgrep_jobs() -> int:
    configured_jobs = os.environ.get("IREE_SEMGREP_JOBS")
    if configured_jobs:
        try:
            jobs = int(configured_jobs)
        except ValueError:
            return 1
        return max(1, jobs)
    cpu_count = os.cpu_count() or 1
    return min(SEMGREP_DEFAULT_MAX_JOBS, max(1, int(cpu_count * 0.85)))


def clang_tidy_jobs() -> int:
    configured_jobs = os.environ.get("IREE_CLANG_TIDY_JOBS")
    if configured_jobs:
        try:
            jobs = int(configured_jobs)
        except ValueError:
            return 1
        return max(1, jobs)
    cpu_count = os.cpu_count() or 1
    default_jobs = max(1, (cpu_count + CLANG_TIDY_DEFAULT_CORE_DIVISOR - 1) // 2)
    return min(cpu_count, CLANG_TIDY_DEFAULT_MAX_JOBS, default_jobs)


def semgrep_scan_command(files: list[str]) -> list[str]:
    return [
        "semgrep",
        "scan",
        "--metrics=off",
        "--disable-version-check",
        "--strict",
        "--error",
        "--severity",
        "ERROR",
        "--jobs",
        str(semgrep_jobs()),
        "--config",
        SEMGREP_CONFIG,
        "--",
        *files,
    ]


def semgrep_validate_command() -> list[str]:
    return [
        "semgrep",
        "scan",
        "--metrics=off",
        "--disable-version-check",
        "--strict",
        "--validate",
        "--config",
        SEMGREP_CONFIG,
    ]


def is_python_file(path: str) -> bool:
    return Path(path).suffix in PYTHON_EXTENSIONS


def run_buildifier(paths: list[str], fix: bool, verbose: bool) -> bool:
    files = existing_files([path for path in paths if is_buildifier_file(path)])
    if not files:
        return skip_step("Buildifier", "no Bazel files")
    if not require_tool("buildifier", "Buildifier"):
        return False
    command = ["buildifier", "-lint=off"]
    if not fix:
        command.append("-mode=check")
    command += files
    ok = run_command(command, "Buildifier", verbose)
    if fix and ok:
        ok = stage_files(files, verbose)
    return ok


def run_ruff(paths: list[str], fix: bool, verbose: bool) -> bool:
    files = existing_files([path for path in paths if is_python_file(path)])
    if not files:
        return skip_step("Ruff", "no Python files")
    if not require_tool("ruff", "Ruff"):
        return False
    ok = True
    lint_command = ["ruff", "check", "--cache-dir", ".ruff_cache"]
    if fix:
        lint_command.append("--fix")
    lint_command += files
    ok = run_command(lint_command, "Ruff lint", verbose) and ok

    format_command = ["ruff", "format", "--cache-dir", ".ruff_cache"]
    if not fix:
        format_command.append("--check")
    format_command += files
    ok = run_command(format_command, "Ruff format", verbose) and ok

    if fix:
        ok = stage_files(files, verbose) and ok
    return ok


def run_clang_format(paths: list[str], fix: bool, verbose: bool) -> bool:
    files = existing_files([path for path in paths if is_c_format_file(path)])
    if not files:
        return skip_step("clang-format", "no C/C++ files")
    if not require_tool("clang-format", "clang-format"):
        return False
    command_prefix = ["clang-format"]
    if fix:
        command_prefix.append("-i")
    else:
        command_prefix += ["--dry-run", "--Werror"]
    commands = [
        command_prefix + files[index : index + C_FORMAT_BATCH_SIZE]
        for index in range(0, len(files), C_FORMAT_BATCH_SIZE)
    ]
    ok = run_parallel_commands(
        commands,
        "clang-format",
        verbose,
        jobs=min(clang_format_jobs(), len(commands)),
    )
    if fix and ok:
        ok = stage_files(files, verbose)
    return ok


def text_for_path(path: str) -> str | None:
    try:
        data = (REPO_ROOT / path).read_bytes()
    except OSError:
        return None
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def fix_text_hygiene(path: str, text: str) -> bool:
    lines = text.splitlines(keepends=True)
    fixed_lines = []
    for line in lines:
        if line.endswith("\r\n"):
            fixed_lines.append(line[:-2].rstrip(" \t") + "\n")
        elif line.endswith("\n"):
            fixed_lines.append(line[:-1].rstrip(" \t") + "\n")
        else:
            fixed_lines.append(line.rstrip(" \t"))
    fixed_text = "".join(fixed_lines)
    if fixed_text and not fixed_text.endswith("\n"):
        fixed_text += "\n"
    if fixed_text == text:
        return False
    (REPO_ROOT / path).write_text(fixed_text, encoding="utf-8")
    return True


def run_text_hygiene(paths: list[str], fix: bool, verbose: bool) -> bool:
    ok = True
    changed = []
    for path in existing_files(paths):
        text = text_for_path(path)
        if text is None:
            continue
        for index, line in enumerate(text.splitlines(), start=1):
            if CONFLICT_MARKER_PATTERN.search(line):
                print(f"{path}:{index}: merge conflict marker")
                ok = False
        if fix:
            if fix_text_hygiene(path, text):
                changed.append(path)
        else:
            lines = text.splitlines(keepends=True)
            for index, line in enumerate(lines, start=1):
                content = line[:-1] if line.endswith("\n") else line
                if content.endswith((" ", "\t")):
                    print(f"{path}:{index}: trailing whitespace")
                    ok = False
            if text and not text.endswith("\n"):
                print(f"{path}: missing final newline")
                ok = False
    if fix and changed:
        ok = stage_files(changed, verbose) and ok
    return ok


def run_watchwords(paths: list[str]) -> bool:
    ok = True
    for path in existing_files(paths):
        text = text_for_path(path)
        if text is None:
            continue
        for index, line in enumerate(text.splitlines(), start=1):
            for pattern in WATCHWORD_PATTERNS:
                match = pattern.search(line)
                if match:
                    print(f'{path}:{index}: forbidden watchword "{match.group(0)}"')
                    ok = False
    return ok


def run_build_filename_check(paths: list[str]) -> bool:
    ok = True
    for path in paths:
        if Path(path).name == "BUILD":
            print(f"{path}: use BUILD.bazel instead of BUILD")
            ok = False
    return ok


def run_bazel_to_cmake(fix: bool, verbose: bool) -> bool:
    command = ["python", "build_tools/bazel_to_cmake/bazel_to_cmake.py"]
    command.append("--stage-updates" if fix else "--check")
    return run_command(command, "Bazel-to-CMake", verbose)


def run_amdgpu_target_map(paths: list[str], fix: bool, verbose: bool) -> bool:
    relevant_prefixes = (
        "build_tools/amdgpu/target_map.",
        "runtime/src/iree/hal/drivers/amdgpu/util/target_id_map.inl",
    )
    if not any(path.startswith(relevant_prefixes) for path in paths):
        return skip_step("AMDGPU target map", "no AMDGPU target-map inputs")
    command = ["python", "build_tools/amdgpu/target_map.py"]
    if not fix:
        command.append("--check")
    ok = run_command(command, "AMDGPU target map", verbose)
    if fix and ok:
        ok = stage_files(
            [
                "build_tools/amdgpu/target_map.bzl",
                "build_tools/amdgpu/target_map.cmake",
                "build_tools/amdgpu/target_map.h",
                "runtime/src/iree/hal/drivers/amdgpu/util/target_id_map.inl",
            ],
            verbose,
        )
    return ok


def run_hygiene(paths: list[str], fix: bool, verbose: bool) -> bool:
    print_section("Hygiene")
    ok = True
    ok = (
        run_inline_check(
            "BUILD filename policy", lambda: run_build_filename_check(paths), verbose
        )
        and ok
    )
    ok = (
        run_inline_check(
            "Text hygiene",
            lambda: run_text_hygiene(paths, fix=fix, verbose=verbose),
            verbose,
        )
        and ok
    )
    ok = run_inline_check("Watchwords", lambda: run_watchwords(paths), verbose) and ok
    ok = run_buildifier(paths, fix=fix, verbose=verbose) and ok
    ok = run_ruff(paths, fix=fix, verbose=verbose) and ok
    ok = run_clang_format(paths, fix=fix, verbose=verbose) and ok
    ok = run_bazel_to_cmake(fix=fix, verbose=verbose) and ok
    ok = run_amdgpu_target_map(paths, fix=fix, verbose=verbose) and ok
    return ok


def existing_project_scripts() -> list[Project]:
    return [project for project in PROJECTS if (REPO_ROOT / project.script).is_file()]


def projects_for_paths(paths: list[str]) -> list[Project]:
    projects = existing_project_scripts()
    if any(is_global_trigger(path) for path in paths):
        return projects
    selected = []
    for project in projects:
        if any(path.startswith(project.root) for path in paths):
            selected.append(project)
    return selected


def is_global_trigger(path: str) -> bool:
    if "build_tools" in Path(path).parts:
        return True
    if path.startswith("requirements") and path.endswith(".txt"):
        return True
    return any(
        path == trigger or path.startswith(trigger)
        for trigger in GLOBAL_PROJECT_TRIGGERS
    )


def run_project_tests(
    projects: list[Project],
    paths: list[str],
    fix: bool,
    verbose: bool,
    lane: str,
) -> bool:
    if not projects:
        return skip_step("Project tests", "no affected project entry points")
    ok = True
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=git_worktree_dir(),
        prefix="iree-x-presubmit-files-",
    ) as file_list:
        for path in paths:
            file_list.write(path + "\n")
        file_list_path = file_list.name
    try:
        for project in projects:
            command = [
                "python",
                project.script,
                "--lane",
                lane,
                "--files-from",
                file_list_path,
                "--tests",
            ]
            command.append("--fix" if fix else "--check")
            ok = run_command(command, f"{project.name} tests", verbose) and ok
    finally:
        Path(file_list_path).unlink(missing_ok=True)
    return ok


def run_root_devtools_tests(paths: list[str], verbose: bool) -> bool:
    if not any(is_root_devtools_trigger(path) for path in paths):
        return skip_step("Root devtools tests", "no root devtools inputs")
    ok = True
    ok = (
        run_command(
            [
                "bazel",
                "test",
                "--config=presubmit",
                "//build_tools/devtools:all",
                "//build_tools/lefthook:all",
            ],
            "Root devtools Bazel tests",
            verbose,
        )
        and ok
    )
    ok = (
        run_command(
            [
                "python",
                "build_tools/devtools/cli_smoke_test.py",
                "--from-working-tree",
                "--scenario",
                "dry-run",
            ],
            "Root devtools CLI smoke test",
            verbose,
        )
        and ok
    )
    ok = (
        run_command(
            [
                "python",
                "build_tools/devtools/bazel_smoke_test.py",
                "--from-working-tree",
                "--scenario",
                "dry-run",
            ],
            "Root devtools Bazel smoke test",
            verbose,
        )
        and ok
    )
    ok = (
        run_command(
            [
                "python",
                "build_tools/devtools/cmake_smoke_test.py",
                "--from-working-tree",
                "--scenario",
                "dry-run",
            ],
            "Root devtools CMake smoke test",
            verbose,
        )
        and ok
    )
    return ok


def run_root_devtools_tests_for_lane(
    paths: list[str], lane: str, verbose: bool
) -> bool:
    if lane == "bazel":
        return run_root_devtools_tests(paths, verbose)
    if lane != "cmake":
        raise ValueError(f"unknown lane: {lane}")
    if not any(is_root_devtools_trigger(path) for path in paths):
        return skip_step("Root devtools tests", "no root devtools inputs")
    return run_command(
        [
            "python",
            "-B",
            "-m",
            "unittest",
            "build_tools.devtools.cli_test",
            "build_tools.devtools.command_plan_test",
            "build_tools.devtools.setup_test",
        ],
        "Root devtools Python tests",
        verbose,
    )


def is_root_devtools_trigger(path: str) -> bool:
    if path.startswith("requirements-") and (
        path.endswith(".in") or path.endswith(".lock.txt")
    ):
        return True
    return any(
        path == trigger or path.startswith(trigger)
        for trigger in ROOT_DEVTOOLS_TRIGGERS
    )


def run_semgrep(inputs: PresubmitInputs, profile: str, verbose: bool) -> bool:
    paths = inputs.selected_paths
    files = existing_files([path for path in paths if is_semgrep_candidate_file(path)])
    validate_config = SEMGREP_CONFIG in paths
    if not files and not validate_config:
        return skip_step("Semgrep", "no C/C++ runtime inputs")
    if not require_static_tool("semgrep", "Semgrep", profile):
        return False

    ok = True
    if validate_config:
        ok = run_command(
            semgrep_validate_command(), "Semgrep config validation", verbose
        )
    if files:
        ok = (
            run_command(semgrep_scan_command(files), "Semgrep hard rules", verbose)
            and ok
        )
    return ok


def clang_tidy_bazel_command(
    targets: list[str],
    keep_going: bool = False,
    *,
    build_events_path: Path | None = None,
    emit_fixes: bool = False,
    local_outputs: bool = False,
) -> list[str]:
    command = [
        "bazel",
        "build",
    ]
    if keep_going:
        command.append("--keep_going")
    output_groups = [
        CLANG_TIDY_LOCAL_OUTPUT_GROUP if local_outputs else CLANG_TIDY_OUTPUT_GROUP
    ]
    if emit_fixes:
        output_groups.append(
            CLANG_TIDY_LOCAL_FIXES_OUTPUT_GROUP
            if local_outputs
            else CLANG_TIDY_FIXES_OUTPUT_GROUP
        )
    command += [
        CLANG_TIDY_REPO_ENV,
        f"--aspects={CLANG_TIDY_ASPECT}",
        f"--output_groups={','.join(output_groups)}",
    ]
    if emit_fixes:
        command.append("--aspects_parameters=emit_fixes=true")
    if build_events_path is not None:
        command += [
            f"--build_event_json_file={build_events_path}",
            "--ui_event_filters=-info",
        ]
    command += [
        "--",
        *targets,
    ]
    return command


def bazel_file_uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"expected file URI for Bazel output, got {uri}")
    if parsed.netloc:
        raise ValueError(f"file URI has unsupported host component: {uri}")
    return Path(unquote(parsed.path))


def bazel_output_paths_from_bep(
    build_events_path: Path,
    *,
    output_group: str,
    suffix: str,
) -> list[Path]:
    named_sets: dict[str, dict[str, object]] = {}
    root_set_ids: list[str] = []
    with build_events_path.open(encoding="utf-8") as build_events_file:
        for line in build_events_file:
            event = json.loads(line)
            named_set_id = event.get("id", {}).get("namedSet", {}).get("id")
            if named_set_id is not None:
                named_sets[named_set_id] = event.get("namedSetOfFiles", {})
            for group in event.get("completed", {}).get("outputGroup", []):
                if group.get("name") == output_group:
                    root_set_ids.extend(
                        file_set["id"] for file_set in group.get("fileSets", [])
                    )

    seen_set_ids: set[str] = set()
    seen_paths: set[Path] = set()
    output_paths: list[Path] = []

    def visit_set(set_id: str) -> None:
        if set_id in seen_set_ids:
            return
        seen_set_ids.add(set_id)
        named_set = named_sets.get(set_id)
        if named_set is None:
            raise ValueError(f"{build_events_path}: missing named set {set_id}")
        for file_set in named_set.get("fileSets", []):
            visit_set(file_set["id"])
        for file_entry in named_set.get("files", []):
            name = file_entry.get("name", "")
            if not name.endswith(suffix):
                continue
            output_path = bazel_file_uri_to_path(file_entry["uri"])
            if output_path in seen_paths:
                continue
            seen_paths.add(output_path)
            output_paths.append(output_path)

    for root_set_id in root_set_ids:
        visit_set(root_set_id)

    output_paths.sort()
    return output_paths


def normalize_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def devtools_tmp_root() -> Path:
    configured_tmp_root = os.environ.get(DEVTOOLS_TMP_ENV)
    if not configured_tmp_root:
        return REPO_ROOT / ".tmp"
    return normalize_repo_path(Path(configured_tmp_root).expanduser())


def cmake_build_dir_state_file() -> Path:
    return devtools_tmp_root().joinpath(*CMAKE_STATE_SUBPATH)


def cmake_build_dir_from_env() -> Path:
    configured_build_dir = os.environ.get(CMAKE_BUILD_DIR_ENV)
    if configured_build_dir:
        return normalize_repo_path(Path(configured_build_dir).expanduser())
    try:
        recorded_build_dir = (
            cmake_build_dir_state_file().read_text(encoding="utf-8").strip()
        )
    except FileNotFoundError:
        recorded_build_dir = ""
    if recorded_build_dir:
        return normalize_repo_path(Path(recorded_build_dir).expanduser())
    return CMAKE_DEFAULT_BUILD_DIR


def cmake_clang_tidy_candidate_files(paths: list[str]) -> list[str]:
    return existing_files(
        [
            path
            for path in paths
            if path.startswith(CLANG_TIDY_PATH_PREFIXES)
            and Path(path).suffix in CMAKE_CLANG_TIDY_EXTENSIONS
        ]
    )


def llvm_cmake_dir(llvm_config: str) -> str | None:
    result = subprocess.run(
        [llvm_config, "--cmakedir"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print(f"[fail] clang-tidy CMake plugin: {llvm_config} --cmakedir failed")
        if result.stdout:
            print(result.stdout.rstrip())
        return None
    return result.stdout.strip()


def cmake_clang_tidy_plugin_path(build_dir: Path) -> Path | None:
    names = (
        "IREEClangTidyPlugin.dll",
        "IREEClangTidyPlugin.so",
        "libIREEClangTidyPlugin.dylib",
        "libIREEClangTidyPlugin.so",
    )
    for name in names:
        candidate = build_dir / name
        if candidate.is_file():
            return candidate
    for pattern in (
        "*IREEClangTidyPlugin*.dll",
        "*IREEClangTidyPlugin*.dylib",
        "*IREEClangTidyPlugin*.so",
    ):
        matches = sorted(build_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def cmake_clang_tidy_command(
    *,
    run_clang_tidy: str,
    clang_tidy: str,
    plugin: Path,
    compile_commands_dir: Path,
    files: list[str],
) -> list[str]:
    return [
        run_clang_tidy,
        "-clang-tidy-binary",
        clang_tidy,
        "-p",
        str(compile_commands_dir),
        f"-checks={CLANG_TIDY_CHECKS}",
        f"-load={plugin}",
        "-j",
        str(clang_tidy_jobs()),
        "-warnings-as-errors=*",
        *files,
    ]


def cmake_generated_compile_inputs_command(*, compile_commands_dir: Path) -> list[str]:
    return [
        "cmake",
        "--build",
        str(compile_commands_dir),
        "--target",
        CLANG_TIDY_CMAKE_GENERATED_INPUTS_TARGET,
        "--parallel",
        str(clang_tidy_jobs()),
    ]


def cmake_run_clang_tidy_fix_command(
    *,
    run_clang_tidy: str,
    clang_tidy: str,
    clang_apply_replacements: str,
    plugin: Path,
    compile_commands_dir: Path,
    files: list[str],
) -> list[str]:
    return [
        run_clang_tidy,
        "-clang-tidy-binary",
        clang_tidy,
        "-clang-apply-replacements-binary",
        clang_apply_replacements,
        "-p",
        str(compile_commands_dir),
        f"-checks={CLANG_TIDY_CHECKS}",
        f"-load={plugin}",
        "-j",
        str(clang_tidy_jobs()),
        "-fix",
        "-format",
        "-style=file",
        *files,
    ]


def prepare_clang_tidy_replacements_dir(
    scratch_dir: Path, fix_paths: list[Path]
) -> Path:
    replacements_dir = scratch_dir / "replacements"
    replacements_dir.mkdir(parents=True, exist_ok=True)
    for fix_path in fix_paths:
        shutil.copy2(fix_path, replacements_dir / fix_path.name)
    return replacements_dir


def clang_tidy_sanitize_artifact_path(path: str) -> str:
    result = path
    for old, new in (
        ("/", "_"),
        ("\\", "_"),
        (":", "_"),
        ("+", "_"),
        ("-", "_"),
        (".", "_"),
    ):
        result = result.replace(old, new)
    return result


def clang_tidy_fix_paths_for_files(
    fix_paths: list[Path], candidate_files: list[str]
) -> list[Path]:
    selected_suffixes = tuple(
        f".{clang_tidy_sanitize_artifact_path(path)}.clang_tidy_fixes.yaml"
        for path in candidate_files
    )
    return [
        fix_path for fix_path in fix_paths if fix_path.name.endswith(selected_suffixes)
    ]


def clang_apply_replacements_command(
    clang_apply_replacements: str, replacements_dir: Path
) -> list[str]:
    return [
        clang_apply_replacements,
        "--format",
        "--style=file",
        "--remove-change-desc-files",
        str(replacements_dir),
    ]


def run_clang_tidy_cmake(
    inputs: PresubmitInputs, profile: str, verbose: bool, fix: bool = False
) -> bool:
    paths = clang_tidy_analysis_paths(inputs)
    candidate_files = cmake_clang_tidy_candidate_files(paths)
    infra_files = existing_files(
        [path for path in inputs.changed_paths if is_clang_tidy_infra_file(path)]
    )
    if not candidate_files and not infra_files:
        return skip_step("clang-tidy", "no C/C++ runtime inputs")

    tools = clang_tidy_llvm_tools()
    if not tools:
        if clang_tidy_required(profile):
            print(f"[fail] clang-tidy: {CLANG_TIDY_SETUP_HINT}.")
            return False
        return skip_step("clang-tidy", CLANG_TIDY_SETUP_HINT)
    llvm_config, clang_tidy, _ = tools
    run_clang_tidy = None
    if candidate_files:
        run_clang_tidy = clang_tidy_run_tool()
        if not run_clang_tidy:
            if clang_tidy_required(profile):
                print(
                    f"[fail] clang-tidy: {CLANG_TIDY_SETUP_HINT} with run-clang-tidy."
                )
                return False
            return skip_step(
                "clang-tidy", f"{CLANG_TIDY_SETUP_HINT} with run-clang-tidy"
            )
    clang_apply_replacements = None
    if fix:
        clang_apply_replacements = clang_tidy_apply_replacements_tool()
        if not clang_apply_replacements:
            print(
                f"[fail] clang-tidy: {CLANG_TIDY_SETUP_HINT} with "
                "clang-apply-replacements for --fix."
            )
            return False
    if not require_tool("cmake", "clang-tidy CMake plugin"):
        return False

    cmake_dir = llvm_cmake_dir(llvm_config)
    if not cmake_dir:
        return False

    ok = True
    configure_command = [
        "cmake",
        "-S",
        "build_tools/clang_tidy",
        "-B",
        str(CLANG_TIDY_CMAKE_BUILD_DIR),
        f"-DLLVM_DIR={cmake_dir}",
    ]
    ok = (
        run_command(
            configure_command,
            "clang-tidy CMake plugin configure",
            verbose,
        )
        and ok
    )
    ok = (
        run_command(
            ["cmake", "--build", str(CLANG_TIDY_CMAKE_BUILD_DIR)],
            "clang-tidy CMake plugin build",
            verbose,
        )
        and ok
    )
    if infra_files:
        ok = (
            run_command(
                [
                    "ctest",
                    "--test-dir",
                    str(CLANG_TIDY_CMAKE_BUILD_DIR),
                    "--output-on-failure",
                ],
                "clang-tidy CMake plugin tests",
                verbose,
            )
            and ok
        )
    if not candidate_files:
        return ok

    plugin = cmake_clang_tidy_plugin_path(CLANG_TIDY_CMAKE_BUILD_DIR)
    if not plugin:
        print(
            f"[fail] clang-tidy CMake plugin: built plugin was not found under "
            f"{CLANG_TIDY_CMAKE_BUILD_DIR}"
        )
        return False

    compile_commands_dir = cmake_build_dir_from_env()
    compile_commands = compile_commands_dir / "compile_commands.json"
    if not compile_commands.is_file():
        print(
            f"[fail] clang-tidy: CMake compile_commands.json is missing: "
            f"{compile_commands}"
        )
        print("hint: run python dev.py cmake configure")
        return False

    ok = (
        run_command(
            cmake_generated_compile_inputs_command(
                compile_commands_dir=compile_commands_dir,
            ),
            "clang-tidy CMake generated compile inputs",
            verbose,
        )
        and ok
    )
    if not ok:
        return False

    if fix:
        ok = (
            run_command(
                cmake_run_clang_tidy_fix_command(
                    run_clang_tidy=run_clang_tidy,
                    clang_tidy=clang_tidy,
                    clang_apply_replacements=clang_apply_replacements,
                    plugin=plugin,
                    compile_commands_dir=compile_commands_dir,
                    files=candidate_files,
                ),
                "clang-tidy CMake fix",
                verbose,
            )
            and ok
        )
        if ok:
            ok = (
                run_command(
                    cmake_clang_tidy_command(
                        run_clang_tidy=run_clang_tidy,
                        clang_tidy=clang_tidy,
                        plugin=plugin,
                        compile_commands_dir=compile_commands_dir,
                        files=candidate_files,
                    ),
                    "clang-tidy CMake compile database after fixes",
                    verbose,
                )
                and ok
            )
    else:
        ok = (
            run_command(
                cmake_clang_tidy_command(
                    run_clang_tidy=run_clang_tidy,
                    clang_tidy=clang_tidy,
                    plugin=plugin,
                    compile_commands_dir=compile_commands_dir,
                    files=candidate_files,
                ),
                "clang-tidy CMake compile database",
                verbose,
            )
            and ok
        )
    return ok


def run_clang_tidy_bazel_fix(
    *,
    candidate_files: list[str],
    package_targets: list[str],
    profile: str,
    verbose: bool,
    local_outputs: bool,
) -> bool:
    clang_apply_replacements = clang_tidy_apply_replacements_tool()
    if not clang_apply_replacements:
        print(
            f"[fail] clang-tidy: {CLANG_TIDY_SETUP_HINT} with "
            "clang-apply-replacements for --fix."
        )
        return False

    scratch_dir = devtools_tmp_root() / CLANG_TIDY_FIXES_ROOT / f"run-{os.getpid()}"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True)
    build_events_path = scratch_dir / "build_events.json"
    try:
        ok = run_command(
            clang_tidy_bazel_command(
                package_targets,
                keep_going=True,
                build_events_path=build_events_path,
                emit_fixes=True,
                local_outputs=local_outputs,
            ),
            "clang-tidy Bazel fix export",
            verbose,
        )
        if not ok:
            return False

        fix_paths = bazel_output_paths_from_bep(
            build_events_path,
            output_group=CLANG_TIDY_LOCAL_FIXES_OUTPUT_GROUP
            if local_outputs
            else CLANG_TIDY_FIXES_OUTPUT_GROUP,
            suffix=".clang_tidy_fixes.yaml",
        )
        if not fix_paths:
            print("[fail] clang-tidy: Bazel produced no replacement artifacts.")
            return False
        fix_paths = clang_tidy_fix_paths_for_files(fix_paths, candidate_files)
        if not fix_paths:
            ok = (
                skip_step(
                    "clang-tidy apply fixes",
                    "no replacement artifacts for selected translation units",
                )
                and ok
            )
            return (
                run_command(
                    clang_tidy_bazel_command(
                        package_targets,
                        keep_going=profile == "ci",
                        local_outputs=local_outputs,
                    ),
                    "clang-tidy Bazel actions after fixes",
                    verbose,
                )
                and ok
            )

        replacements_dir = prepare_clang_tidy_replacements_dir(scratch_dir, fix_paths)
        ok = run_command(
            clang_apply_replacements_command(
                clang_apply_replacements,
                replacements_dir,
            ),
            "clang-tidy apply fixes",
            verbose,
        )
        if not ok:
            return False

        return run_command(
            clang_tidy_bazel_command(
                package_targets,
                keep_going=profile == "ci",
                local_outputs=local_outputs,
            ),
            "clang-tidy Bazel actions after fixes",
            verbose,
        )
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def run_clang_tidy(
    inputs: PresubmitInputs,
    profile: str,
    lane: str,
    verbose: bool,
    fix: bool = False,
) -> bool:
    if lane == "cmake":
        return run_clang_tidy_cmake(inputs, profile, verbose, fix=fix)
    if lane != "bazel":
        return skip_step("clang-tidy", "unknown build-system lane")

    paths = clang_tidy_analysis_paths(inputs)
    candidate_files = existing_files(
        [path for path in paths if is_clang_tidy_candidate_file(path)]
    )
    infra_files = existing_files(
        [path for path in inputs.changed_paths if is_clang_tidy_infra_file(path)]
    )
    if not candidate_files and not infra_files:
        return skip_step("clang-tidy", "no C/C++ runtime inputs")

    if not clang_tidy_llvm_available():
        if clang_tidy_required(profile):
            print(f"[fail] clang-tidy: {CLANG_TIDY_SETUP_HINT}.")
            return False
        return skip_step("clang-tidy", CLANG_TIDY_SETUP_HINT)

    ok = True
    if infra_files:
        ok = (
            run_command(
                [
                    "bazel",
                    "test",
                    "--config=presubmit",
                    CLANG_TIDY_REPO_ENV,
                    "//build_tools/clang_tidy:plugin_smoke_test",
                    "//build_tools/clang_tidy:refcount_checks_test",
                    "//build_tools/clang_tidy:status_checks_test",
                    "//build_tools/clang_tidy:style_checks_test",
                    "//build_tools/clang_tidy:trace_checks_test",
                ],
                "clang-tidy plugin tests",
                verbose,
            )
            and ok
        )
        ok = (
            run_command(
                [
                    "bazel",
                    "build",
                    CLANG_TIDY_REPO_ENV,
                    "//build_tools/clang_tidy:action_smoke",
                ],
                "clang-tidy action smoke",
                verbose,
            )
            and ok
        )

    package_targets = unique_paths(
        [
            target
            for target in (
                bazel_package_target_for_path(path) for path in candidate_files
            )
            if target is not None
        ]
    )
    if package_targets:
        local_outputs = True
        if fix:
            ok = (
                run_clang_tidy_bazel_fix(
                    candidate_files=candidate_files,
                    package_targets=package_targets,
                    profile=profile,
                    verbose=verbose,
                    local_outputs=local_outputs,
                )
                and ok
            )
        else:
            ok = (
                run_command(
                    clang_tidy_bazel_command(
                        package_targets,
                        keep_going=profile == "ci",
                        local_outputs=local_outputs,
                    ),
                    "clang-tidy Bazel actions",
                    verbose,
                )
                and ok
            )
    elif candidate_files:
        ok = skip_step("clang-tidy", "no owning Bazel packages for inputs") and ok
    return ok


STATIC_ANALYSIS_PROVIDERS = (
    StaticAnalysisProvider(
        name="Semgrep",
        profiles=frozenset(("paranoid", "ci")),
        runner=lambda inputs, profile, lane, verbose: run_semgrep(
            inputs, profile, verbose
        ),
    ),
    StaticAnalysisProvider(
        name="clang-tidy",
        profiles=frozenset(("paranoid", "ci")),
        runner=run_clang_tidy,
    ),
)


def run_static_analysis(
    inputs: PresubmitInputs, profile: str, lane: str, verbose: bool
) -> bool:
    if not inputs.selected_paths:
        return True
    ok = True
    selected_providers = [
        provider
        for provider in STATIC_ANALYSIS_PROVIDERS
        if profile in provider.profiles
    ]
    if not selected_providers:
        return skip_step("Static analysis", f"no providers for {profile} profile")
    for provider in selected_providers:
        ok = provider.runner(inputs, profile, lane, verbose) and ok
    return ok


def print_plan(
    args: argparse.Namespace, inputs: PresubmitInputs, projects: list[Project]
) -> None:
    paths = inputs.selected_paths
    mutation = "fix" if args.fix else "check"
    if args.paths:
        input_mode = "explicit paths"
    elif args.changed:
        input_mode = "changed"
    elif args.commit:
        input_mode = "commit"
    elif args.all:
        input_mode = "all"
    elif args.base:
        input_mode = f"base {args.base}"
    elif args.since:
        input_mode = f"since {args.since}"
    elif args.files_from:
        input_mode = f"files from {args.files_from}"
    else:
        input_mode = "staged"
    scopes = []
    if args.hygiene:
        scopes.append("hygiene")
    if args.tests:
        scopes.append("tests")
    if args.static_analysis:
        scopes.append("static-analysis")
    if args.clang_tidy:
        scopes.append("clang-tidy")
    print("presubmit plan:")
    print(f"  lane: {args.lane}")
    print(f"  profile: {args.profile}")
    print(f"  mode: {mutation}")
    print(f"  input: {input_mode} ({len(paths)} file(s))")
    print("  scopes: " + ", ".join(scopes))
    if projects:
        print("  projects: " + ", ".join(project.name for project in projects))
    else:
        print("  projects: none")
    print(f"  project tests: {'enabled' if args.project_tests else 'disabled'}")
    sys.stdout.flush()


def dev_py_rerun_command(args: argparse.Namespace, verbose: bool) -> list[str]:
    if args.all:
        command = [
            "python",
            "dev.py",
            args.lane,
            "presubmit",
            "--profile",
            args.profile,
        ]
        if not args.project_tests:
            command.append("--no-project-tests")
    else:
        command = [
            "python",
            "dev.py",
            args.lane,
            "precommit",
            "--profile",
            args.profile,
        ]
        if args.base:
            command += ["--base", args.base]
        elif args.commit:
            command.append("--commit")
        elif args.staged or args.paths:
            command.append("--staged")
    if verbose:
        command.append("--verbose")
    return command


def precommit_invocation_autofixes(args: argparse.Namespace) -> bool:
    return args.tests and (args.commit or args.staged or bool(args.paths))


def print_suggested_actions(args: argparse.Namespace, ok: bool) -> None:
    if ok:
        return

    print_section("Suggested Actions")
    if args.fix:
        print(
            "[next] Inspect the failing fixer/test output above, adjust files, then run:"
        )
        print("  " + command_text(dev_py_rerun_command(args, verbose=False)))
    elif precommit_invocation_autofixes(args):
        print(
            "[next] Apply mechanical fixes and rerun the selected test-bearing check:"
        )
        print("  " + command_text(dev_py_rerun_command(args, verbose=False)))
    else:
        if args.all:
            print("[next] Repair the diagnostics above. For staged mechanical fixups:")
        else:
            print("[next] Stage the intended files and apply mechanical fixes:")
        print(f"  python dev.py {args.lane} fix")
        print("[verify] Re-run the selected check:")
        print("  " + command_text(dev_py_rerun_command(args, verbose=False)))
    print("[debug] Re-run with full tool output:")
    print("  " + command_text(dev_py_rerun_command(args, verbose=True)))


def run_presubmit(
    args: argparse.Namespace, inputs: PresubmitInputs, projects: list[Project]
) -> int:
    paths = inputs.selected_paths
    if args.print_plan:
        print_plan(args, inputs, projects)

    ok = True
    if args.hygiene:
        ok = run_hygiene(paths, fix=args.fix, verbose=args.verbose) and ok
    if args.tests:
        print_section("Tests")
        ok = (
            run_root_devtools_tests_for_lane(
                paths, lane=args.lane, verbose=args.verbose
            )
            and ok
        )
        if args.project_tests:
            ok = (
                run_project_tests(
                    projects,
                    paths,
                    fix=args.fix,
                    verbose=args.verbose,
                    lane=args.lane,
                )
                and ok
            )
        else:
            ok = skip_step("Project tests", "disabled by --no-project-tests") and ok
    if args.static_analysis:
        print_section("Static Analysis")
        ok = (
            run_static_analysis(
                inputs, profile=args.profile, lane=args.lane, verbose=args.verbose
            )
            and ok
        )
    if args.clang_tidy:
        print_section("clang-tidy")
        ok = (
            run_clang_tidy(
                inputs,
                profile=args.profile,
                lane=args.lane,
                verbose=args.verbose,
                fix=args.fix,
            )
            and ok
        )
    print_suggested_actions(args, ok)
    return 0 if ok else 1


def run_presubmit_with_source_guard(
    args: argparse.Namespace, inputs: PresubmitInputs, projects: list[Project]
) -> int:
    snapshot = NonEmptyTrackedFileSnapshot.capture_tracked_package_initializers(
        REPO_ROOT
    )
    result = run_presubmit(args, inputs, projects)
    if not snapshot.verify(REPO_ROOT):
        result = 1
    return result


def main() -> int:
    args = parse_arguments()
    inputs = presubmit_inputs(args)
    projects = projects_for_paths(inputs.selected_paths)
    if args.fix:
        with source_mutation_lock(REPO_ROOT, "root-presubmit-fix"):
            return run_presubmit_with_source_guard(args, inputs, projects)
    return run_presubmit_with_source_guard(args, inputs, projects)


if __name__ == "__main__":
    sys.exit(main())
