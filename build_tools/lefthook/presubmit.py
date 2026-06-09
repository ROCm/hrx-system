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
import contextlib
import io
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

REPO_ROOT = Path(__file__).resolve().parents[2]

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
SEMGREP_CONFIG = "build_tools/static_analysis/semgrep/iree.yml"
SEMGREP_EXTENSIONS = C_FORMAT_EXTENSIONS
SEMGREP_PATH_PREFIXES = (
    "runtime/src/iree/",
    "loom/src/loom/",
    "libhrx/",
)
SEMGREP_DEFAULT_MAX_JOBS = 14
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


@dataclass(frozen=True)
class StaticAnalysisProvider:
    name: str
    profiles: frozenset[str]
    runner: Callable[[list[str], str, bool], bool]


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

    if not args.hygiene and not args.tests and not args.static_analysis:
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


def selected_files(args: argparse.Namespace) -> list[str]:
    if args.paths:
        return args.paths
    if args.changed:
        return changed_files()
    if args.staged:
        return git_list(["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"])
    if args.commit:
        return commit_files()
    if args.all:
        return git_list(["ls-files", "-z"])
    if args.base:
        return unique_paths(
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
    if args.files_from:
        with open(args.files_from, encoding="utf-8") as file_list:
            return [line.strip() for line in file_list if line.strip()]
    return git_list(
        ["diff", "--name-only", "-z", "--diff-filter=ACMR", args.since, "--"]
    )


def existing_files(paths: list[str]) -> list[str]:
    return [path for path in paths if (REPO_ROOT / path).is_file()]


def stage_files(paths: list[str], verbose: bool) -> bool:
    if not paths:
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
    command = ["clang-format"]
    if fix:
        command.append("-i")
    else:
        command += ["--dry-run", "--Werror"]
    command += files
    ok = run_command(command, "clang-format", verbose)
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
        mode="w", encoding="utf-8", delete=False, dir=git_worktree_dir()
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


def run_semgrep(paths: list[str], profile: str, verbose: bool) -> bool:
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


STATIC_ANALYSIS_PROVIDERS = (
    StaticAnalysisProvider(
        name="Semgrep",
        profiles=frozenset(("paranoid", "ci")),
        runner=run_semgrep,
    ),
)


def run_static_analysis(paths: list[str], profile: str, verbose: bool) -> bool:
    if not paths:
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
        ok = provider.runner(paths, profile, verbose) and ok
    return ok


def print_plan(
    args: argparse.Namespace, paths: list[str], projects: list[Project]
) -> None:
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


def main() -> int:
    args = parse_arguments()
    paths = selected_files(args)
    projects = projects_for_paths(paths)
    if args.print_plan:
        print_plan(args, paths, projects)

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
            run_static_analysis(paths, profile=args.profile, verbose=args.verbose)
            and ok
        )
    print_suggested_actions(args, ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
