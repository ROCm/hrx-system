#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom project presubmit entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "loom/py"))

from loom.gen import checked_in_artifacts

from build_tools.devtools import project_presubmit
from build_tools.devtools.source_lock import NonEmptyTrackedFileSnapshot

PROJECT_NAME = "loom"
PROJECT_ROOT = "loom/"
CMAKE_TEST_REGEX = "^loom/"
GLOBAL_TEST_TRIGGERS = (
    "BUILD.bazel",
    "MODULE.bazel",
    ".bazelrc",
    ".bazel_to_cmake.cfg.py",
    "requirements",
)
RESOURCE_TEST_TAG_FILTERS = (
    "-iree-run-requirement=runtime.resource.amd_gpu",
    "-iree-run-requirement=runtime.resource.vulkan_device",
    "-iree-run-requirement=runtime.resource.webgpu_device",
)
CTEST_RESOURCE_LABEL_EXCLUDE_REGEX = "runtime-resource="
CMAKE_SOURCE_FORMAT_TARGET_DEFINES = (
    ("amdgpu", "LOOM_TARGET_AMDGPU"),
    ("llvmir", "LOOM_TARGET_LLVMIR"),
    ("spirv", "LOOM_TARGET_SPIRV"),
    ("wasm", "LOOM_TARGET_WASM"),
    ("xdna", "LOOM_TARGET_XDNA"),
    ("x86", "LOOM_TARGET_X86"),
)
CI_LOOM_TARGETS = ",".join(
    target for target, _define in CMAKE_SOURCE_FORMAT_TARGET_DEFINES
)
BAZEL_SOURCE_TOOL_ARGS = (
    "--config=locked",
    f"--//loom/config/target:enable={CI_LOOM_TARGETS}",
)
LOOM_FORMAT_BAZEL_TARGET = "//loom/src/loom/tools/loom-format:loom-format"
LOOM_FORMAT_CMAKE_TARGET = "loom::tools::loom-format"
LOOM_LINT_BAZEL_TARGET = "//loom/py/loom/tools:loom-lint"
LOOM_LINT_CMAKE_TARGET = "loom::py::loom::tools::loom-lint"
LOOM_LINT_PYTHON_SOURCE = "loom/py/loom/tools/source_lint.py"
LOOM_LINT_SUFFIXES = frozenset({".loom", ".loom-test"})
LOOM_FORMAT_SUFFIXES = frozenset({".loom", ".loom-test"})
# Syntax-corpus modules retain their exact parser/printer fixture contract rather
# than the verified canonical-source contract enforced by loom-format.
LOOM_FORMAT_EXCLUDED_PREFIXES = ("loom/src/loom/test/corpus/text/",)
# These exact modules intentionally fail semantic verification to test public
# diagnostics. New invalid-looking filenames are not excluded automatically.
LOOM_FORMAT_EXCLUDED_PATHS = frozenset(
    {
        "loom/src/loom/tooling/target/amdgpu/test/amdgpu_bad_return.loom",
        "loom/src/loom/tools/iree-benchmark-loom/testdata/duplicate_symbol.loom",
    }
)

# CreateProcess limits its command line to 32,767 UTF-16 code units including
# the terminator. Keep one portable bound below that ceiling so repository-wide
# file checks have the same batching behavior on every host.
MAX_PORTABLE_COMMAND_LINE_UTF16_UNITS = 30_000


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Loom project presubmit.")
    project_presubmit.add_common_arguments(parser, project_name=PROJECT_NAME)
    return parser.parse_args()


def run_command(command: list[str], description: str) -> bool:
    return project_presubmit.run_command(
        PROJECT_NAME, command, description, cwd=REPO_ROOT
    )


def command_line_utf16_units(command: list[str]) -> int:
    rendered_command = subprocess.list2cmdline(command)
    return len(rendered_command.encode("utf-16-le")) // 2 + 1


def batch_path_commands(
    command_prefix: list[str],
    paths: list[str],
    *,
    max_command_line_utf16_units: int = MAX_PORTABLE_COMMAND_LINE_UTF16_UNITS,
) -> list[list[str]]:
    if not command_prefix:
        raise ValueError("command prefix must not be empty")
    if max_command_line_utf16_units <= 0:
        raise ValueError("command-line limit must be positive")
    if command_line_utf16_units(command_prefix) > max_command_line_utf16_units:
        raise ValueError("command prefix exceeds the portable command-line limit")

    commands: list[list[str]] = []
    command = list(command_prefix)
    prefix_length = len(command_prefix)
    for path in paths:
        candidate = [*command, path]
        if command_line_utf16_units(candidate) <= max_command_line_utf16_units:
            command.append(path)
            continue
        if len(command) > prefix_length:
            commands.append(command)
        command = [*command_prefix, path]
        if command_line_utf16_units(command) > max_command_line_utf16_units:
            raise ValueError(f"path exceeds the portable command-line limit: {path}")
    if len(command) > prefix_length:
        commands.append(command)
    return commands


def run_batched_path_command(
    command_prefix: list[str], paths: list[str], description: str
) -> bool:
    try:
        commands = batch_path_commands(command_prefix, paths)
    except ValueError as exc:
        print(f"loom presubmit: {exc}", file=sys.stderr)
        return False

    ok = True
    for index, command in enumerate(commands, start=1):
        batch_description = (
            description
            if len(commands) == 1
            else f"{description} ({index}/{len(commands)})"
        )
        ok = run_command(command, batch_description) and ok
    return ok


def is_global_trigger(path: str) -> bool:
    if "build_tools" in Path(path).parts:
        return True
    if path.startswith("requirements") and path.endswith(".txt"):
        return True
    return any(
        path == trigger or path.startswith(trigger) for trigger in GLOBAL_TEST_TRIGGERS
    )


def selected_files(files_from: str | None) -> list[str]:
    if not files_from:
        return []
    with open(files_from, encoding="utf-8") as file_list:
        return [line.strip() for line in file_list if line.strip()]


def should_run_presubmit(files_from: str | None) -> bool:
    paths = selected_files(files_from)
    if not paths:
        return files_from is None
    return any(
        path.startswith(PROJECT_ROOT) or is_global_trigger(path) for path in paths
    )


def run_generated_artifact_maintenance(
    fix: bool, files_from: str | None = None
) -> bool:
    print("loom presubmit: Checked-in generated artifacts")
    if fix and files_from is not None:
        result = checked_in_artifacts.maintain_checked_in_artifacts(
            "update", selected_paths=selected_files(files_from)
        )
    else:
        result = checked_in_artifacts.maintain_checked_in_artifacts(
            "update" if fix else "check"
        )
    if not result.ok:
        return False
    if not fix:
        return True
    return project_presubmit.stage_changed_paths(
        PROJECT_NAME,
        REPO_ROOT,
        result.changed_paths,
    )


def is_format_source_path(path: str) -> bool:
    source_path = PurePosixPath(path)
    if "\\" in path or source_path.as_posix() != path or ".." in source_path.parts:
        return False
    if (
        not path.startswith(PROJECT_ROOT)
        or source_path.suffix not in LOOM_FORMAT_SUFFIXES
    ):
        return False
    if path in LOOM_FORMAT_EXCLUDED_PATHS:
        return False
    return not any(path.startswith(prefix) for prefix in LOOM_FORMAT_EXCLUDED_PREFIXES)


def existing_format_source_paths(paths: list[str]) -> list[str]:
    return sorted(
        {
            path
            for path in paths
            if is_format_source_path(path) and (REPO_ROOT / path).is_file()
        }
    )


def _tracked_project_paths() -> list[str] | None:
    result = subprocess.run(
        ["git", "--literal-pathspecs", "ls-files", "-z", "--", PROJECT_ROOT],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        print(
            "loom presubmit: failed to query tracked Loom project files",
            file=sys.stderr,
        )
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        return None
    return result.stdout.split("\0")


def tracked_format_source_paths() -> list[str] | None:
    paths = _tracked_project_paths()
    return None if paths is None else existing_format_source_paths(paths)


def is_lint_source_path(path: str) -> bool:
    source_path = PurePosixPath(path)
    return (
        "\\" not in path
        and source_path.as_posix() == path
        and ".." not in source_path.parts
        and path.startswith(PROJECT_ROOT)
        and source_path.suffix in LOOM_LINT_SUFFIXES
    )


def existing_lint_source_paths(paths: list[str]) -> list[str]:
    return sorted(
        {
            path
            for path in paths
            if is_lint_source_path(path) and (REPO_ROOT / path).is_file()
        }
    )


def tracked_lint_source_paths() -> list[str] | None:
    paths = _tracked_project_paths()
    return None if paths is None else existing_lint_source_paths(paths)


def validate_cmake_source_format_configuration() -> bool:
    build_dir = project_presubmit.cmake_build_dir(REPO_ROOT)
    if not project_presubmit.validate_cmake_build_tree(PROJECT_NAME, build_dir):
        return False

    cmake_true_values = frozenset({"1", "ON", "TRUE", "YES", "Y"})
    missing_targets = [
        (target, define)
        for target, define in CMAKE_SOURCE_FORMAT_TARGET_DEFINES
        if (project_presubmit.cmake_cache_value(build_dir, define) or "").upper()
        not in cmake_true_values
    ]
    if not missing_targets:
        return True

    missing_names = ", ".join(target for target, _define in missing_targets)
    missing_options = " ".join(f"-D{define}=ON" for _target, define in missing_targets)
    print(
        "loom presubmit: canonical source formatting requires the CMake "
        f"build tree to enable {CI_LOOM_TARGETS}; missing: {missing_names}"
    )
    print(
        "loom presubmit: reconfigure that tree with "
        f"`{missing_options}` or select a full-target tree with "
        f"{project_presubmit.CMAKE_BUILD_DIR_ENV}"
    )
    return False


def run_source_format_maintenance(
    *, lane: str, files_from: str | None, fix: bool
) -> bool:
    tracked_paths = tracked_format_source_paths()
    if tracked_paths is None:
        return False
    if files_from is None:
        selected_paths = tracked_paths
    else:
        selected_paths = existing_format_source_paths(selected_files(files_from))
    check_paths = sorted(set(tracked_paths).union(selected_paths))
    if not check_paths:
        return True
    if lane == "cmake" and not validate_cmake_source_format_configuration():
        return False

    formatter_path = project_presubmit.build_and_resolve_executable(
        PROJECT_NAME,
        REPO_ROOT,
        lane=lane,
        bazel_target=LOOM_FORMAT_BAZEL_TARGET,
        cmake_target=LOOM_FORMAT_CMAKE_TARGET,
        bazel_args=BAZEL_SOURCE_TOOL_ARGS,
    )
    if formatter_path is None:
        return False

    if fix and selected_paths:
        if not run_batched_path_command(
            [str(formatter_path), "--in-place"],
            selected_paths,
            "Canonicalize selected Loom source",
        ):
            return False
        if not project_presubmit.stage_changed_paths(
            PROJECT_NAME,
            REPO_ROOT,
            selected_paths,
        ):
            return False

    return run_batched_path_command(
        [str(formatter_path), "--check"],
        check_paths,
        "Canonical Loom source",
    )


def run_source_lint(*, lane: str, files_from: str | None) -> bool:
    tracked_paths = tracked_lint_source_paths()
    if tracked_paths is None:
        return False
    selected_paths = (
        []
        if files_from is None
        else existing_lint_source_paths(selected_files(files_from))
    )
    check_paths = sorted(set(tracked_paths).union(selected_paths))

    public_lint_ok = True
    if check_paths:
        if lane == "bazel":
            linter_path = project_presubmit.build_and_resolve_executable(
                PROJECT_NAME,
                REPO_ROOT,
                lane=lane,
                bazel_target=LOOM_LINT_BAZEL_TARGET,
                cmake_target=LOOM_LINT_CMAKE_TARGET,
                bazel_args=BAZEL_SOURCE_TOOL_ARGS,
            )
            linter_command = [] if linter_path is None else [str(linter_path)]
        elif lane == "cmake":
            # CMake models Python entrypoints as source-bearing custom targets,
            # not native executable artifacts. The public linter is deliberately
            # standalone, so the source lane can invoke that same entrypoint.
            linter_command = [sys.executable, LOOM_LINT_PYTHON_SOURCE]
        else:
            raise ValueError(f"unknown lane: {lane}")
        public_lint_ok = bool(linter_command) and run_batched_path_command(
            linter_command, check_paths, "Loom authoring policy"
        )

    repository_lint_ok = run_command(
        [
            sys.executable,
            "loom/build_tools/linters/loom_source_lint.py",
        ],
        "Loom repository invariants",
    )
    return public_lint_ok and repository_lint_ok


def bazel_test_command() -> list[str]:
    return [
        "bazel",
        "test",
        "--config=presubmit",
        f"--//loom/config/target:enable={CI_LOOM_TARGETS}",
        "--test_tag_filters=" + ",".join(RESOURCE_TEST_TAG_FILTERS),
        "//loom/...",
    ]


def run_bazel_tests() -> bool:
    return run_command(
        bazel_test_command(),
        "Bazel tests",
    )


def run_cmake_tests() -> bool:
    build_dir = project_presubmit.cmake_build_dir(REPO_ROOT)
    if not project_presubmit.validate_cmake_build_tree(PROJECT_NAME, build_dir):
        return False
    if not run_command(
        ["cmake", "--build", str(build_dir), "--parallel"],
        "CMake build",
    ):
        return False
    return run_command(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "--output-on-failure",
            "-R",
            CMAKE_TEST_REGEX,
            "-LE",
            CTEST_RESOURCE_LABEL_EXCLUDE_REGEX,
        ],
        "CTest tests",
    )


def run_presubmit(args: argparse.Namespace) -> int:
    if not args.hygiene and not args.tests:
        return 0
    if not should_run_presubmit(args.files_from):
        print("loom presubmit: no Loom-affecting files")
        return 0
    ok = True
    if args.hygiene:
        ok = run_generated_artifact_maintenance(args.fix, args.files_from) and ok
        ok = (
            run_source_format_maintenance(
                lane=args.lane,
                files_from=args.files_from,
                fix=args.fix,
            )
            and ok
        )
        ok = run_source_lint(lane=args.lane, files_from=args.files_from) and ok
    if args.tests:
        if args.lane == "bazel":
            ok = run_bazel_tests() and ok
        elif args.lane == "cmake":
            ok = run_cmake_tests() and ok
        else:
            raise ValueError(f"unknown lane: {args.lane}")
    return 0 if ok else 1


def main() -> int:
    args = parse_arguments()
    snapshot = NonEmptyTrackedFileSnapshot.capture_tracked_package_initializers(
        REPO_ROOT
    )
    result = run_presubmit(args)
    if not snapshot.verify(REPO_ROOT):
        result = 1
    return result


if __name__ == "__main__":
    sys.exit(main())
