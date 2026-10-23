# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Builds the concrete CMake closure selected by CTest."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from build_tools.devtools import cmake_cache, cmake_file_api, cmake_make
from build_tools.devtools.command_plan import CommandStep, quote_command

BUILD_TARGETS_CATALOG_FILENAME = "iree_ctest_build_targets.json"
REFRESH_TARGET = "iree-ctest-refresh"
MAX_BUILD_COMMAND_LENGTH = 8192
SKIP_GLOBAL_TMPDIR_CLEANUP_ENV = "IREE_CTEST_SKIP_GLOBAL_TMPDIR_CLEANUP"


class CTestMetadataError(ValueError):
    """Raised when CTest selection or build-root metadata is invalid."""


@dataclass(frozen=True)
class CTestTemporaryDirectory:
    test_name: str
    path: Path


@dataclass(frozen=True)
class CTestSelection:
    test_names: tuple[str, ...]
    build_targets: tuple[str, ...]
    temporary_directory_root: Path
    temporary_directories: tuple[CTestTemporaryDirectory, ...]


def _parse_test_temporary_directory(
    test: dict, test_index: int, test_name: str
) -> CTestTemporaryDirectory | None:
    properties = test.get("properties")
    if not isinstance(properties, list):
        raise CTestMetadataError(
            f"CTest JSON test {test_name} at index {test_index} has no properties list"
        )

    temporary_directory = None
    for property_index, property_value in enumerate(properties):
        if not isinstance(property_value, dict):
            raise CTestMetadataError(
                f"CTest JSON property at index {property_index} for test "
                f"{test_name} is not an object"
            )
        if property_value.get("name") != "ENVIRONMENT":
            continue
        environment = property_value.get("value")
        if not isinstance(environment, list) or not all(
            isinstance(value, str) for value in environment
        ):
            raise CTestMetadataError(
                f"CTest JSON ENVIRONMENT for test {test_name} is not a string list"
            )
        for value in environment:
            name, separator, path = value.partition("=")
            if name != "TEST_TMPDIR":
                continue
            if not separator or not path:
                raise CTestMetadataError(
                    f"CTest test {test_name} has an empty TEST_TMPDIR"
                )
            if temporary_directory is not None:
                raise CTestMetadataError(
                    f"CTest test {test_name} has more than one TEST_TMPDIR"
                )
            temporary_directory = CTestTemporaryDirectory(test_name, Path(path))
    return temporary_directory


def parse_ctest_selection(
    ctest_payload: str,
    build_target_catalog_payload: str,
) -> CTestSelection:
    try:
        model = json.loads(ctest_payload)
    except json.JSONDecodeError as exc:
        raise CTestMetadataError(f"CTest did not emit valid JSON: {exc}") from exc

    if not isinstance(model, dict) or model.get("kind") != "ctestInfo":
        raise CTestMetadataError("CTest JSON is not a ctestInfo model")
    version = model.get("version")
    if not isinstance(version, dict) or version.get("major") != 1:
        raise CTestMetadataError(f"unsupported CTest JSON model version: {version!r}")
    tests = model.get("tests")
    if not isinstance(tests, list):
        raise CTestMetadataError("CTest JSON model has no tests list")

    try:
        build_target_catalog = json.loads(build_target_catalog_payload)
    except json.JSONDecodeError as exc:
        raise CTestMetadataError(
            f"CMake build-target catalog is not valid JSON: {exc}"
        ) from exc
    if (
        not isinstance(build_target_catalog, dict)
        or build_target_catalog.get("kind") != "ireeCtestBuildTargets"
        or build_target_catalog.get("version") != 2
    ):
        raise CTestMetadataError("CMake build-target catalog has an invalid schema")
    temporary_directory_root_value = build_target_catalog.get("temporaryDirectoryRoot")
    if (
        not isinstance(temporary_directory_root_value, str)
        or not temporary_directory_root_value
    ):
        raise CTestMetadataError(
            "CMake build-target catalog has no temporary-directory root"
        )
    temporary_directory_root = Path(temporary_directory_root_value)
    catalog_tests = build_target_catalog.get("tests")
    if not isinstance(catalog_tests, dict):
        raise CTestMetadataError("CMake build-target catalog has no tests object")

    test_names = []
    build_targets = []
    temporary_directories = []
    seen_build_targets = set()
    for test_index, test in enumerate(tests):
        if not isinstance(test, dict):
            raise CTestMetadataError(
                f"CTest JSON test at index {test_index} is not an object"
            )
        test_name = test.get("name")
        if not isinstance(test_name, str) or not test_name:
            raise CTestMetadataError(
                f"CTest JSON test at index {test_index} has no name"
            )
        test_names.append(test_name)
        temporary_directory = _parse_test_temporary_directory(
            test, test_index, test_name
        )
        if temporary_directory is not None:
            temporary_directories.append(temporary_directory)

        if test_name not in catalog_tests:
            raise CTestMetadataError(
                f"selected CTest test {test_name} is missing from the "
                "CMake build-target catalog"
            )
        test_build_targets = catalog_tests[test_name]
        if not isinstance(test_build_targets, list):
            raise CTestMetadataError(
                f"selected CTest test {test_name} has a non-list "
                f"build-target catalog entry: {test_build_targets!r}"
            )

        for build_target in test_build_targets:
            if not isinstance(build_target, str) or not build_target:
                raise CTestMetadataError(
                    f"selected CTest test {test_name} has an invalid "
                    f"build-target catalog entry: {build_target!r}"
                )
            if build_target in seen_build_targets:
                continue
            seen_build_targets.add(build_target)
            build_targets.append(build_target)

    return CTestSelection(
        tuple(test_names),
        tuple(build_targets),
        temporary_directory_root,
        tuple(temporary_directories),
    )


def prepare_selected_test_temporary_directories(
    selection: CTestSelection,
) -> None:
    """Creates empty TEST_TMPDIRs for exactly the selected CTest tests."""
    if not selection.temporary_directory_root.is_absolute():
        raise CTestMetadataError(
            "CMake build-target catalog has a relative temporary-directory "
            f"root: {selection.temporary_directory_root}"
        )
    temporary_root = selection.temporary_directory_root.resolve()
    validated_directories = []
    owners = {}
    for temporary_directory in selection.temporary_directories:
        path = temporary_directory.path
        if not path.is_absolute():
            raise CTestMetadataError(
                f"CTest test {temporary_directory.test_name} has a relative "
                f"TEST_TMPDIR: {path}"
            )
        resolved_path = path.resolve()
        if resolved_path == temporary_root or not resolved_path.is_relative_to(
            temporary_root
        ):
            raise CTestMetadataError(
                f"CTest test {temporary_directory.test_name} has TEST_TMPDIR "
                f"outside {temporary_root}: {path}"
            )
        previous_owner = owners.get(resolved_path)
        if previous_owner is not None:
            raise CTestMetadataError(
                f"CTest tests {previous_owner} and {temporary_directory.test_name} "
                f"share TEST_TMPDIR {path}"
            )
        owners[resolved_path] = temporary_directory.test_name
        validated_directories.append((temporary_directory.test_name, path))

    for test_name, path in validated_directories:
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True)
        except OSError as exc:
            raise CTestMetadataError(
                f"failed to prepare TEST_TMPDIR for selected CTest test "
                f"{test_name} at {path}: {exc}"
            ) from exc


def ctest_build_config(arguments: list[str]) -> str | None:
    build_config = None
    for index, argument in enumerate(arguments):
        if argument in ("-C", "--build-config"):
            if index + 1 < len(arguments):
                build_config = arguments[index + 1]
        elif argument.startswith("--build-config="):
            build_config = argument.partition("=")[2]
        elif argument.startswith("-C") and len(argument) > 2:
            build_config = argument[2:]
    return build_config


def is_inspection_only(arguments: list[str]) -> bool:
    for argument in arguments:
        if argument == "-N" or argument.startswith("--show-only"):
            return True
        if argument in ("--list-presets", "--print-labels"):
            return True
    return False


def ctest_selection_command(
    ctest: str,
    build_dir: Path,
    arguments: list[str],
) -> list[str]:
    return [
        ctest,
        "--test-dir",
        str(build_dir),
        *arguments,
        "--show-only=json-v1",
    ]


def ctest_run_command(
    ctest: str,
    build_dir: Path,
    arguments: list[str],
) -> list[str]:
    return [
        ctest,
        "--test-dir",
        str(build_dir),
        "--output-on-failure",
        *arguments,
    ]


def cmake_refresh_command(
    cmake: str,
    build_dir: Path,
    *,
    build_config: str | None = None,
) -> list[str]:
    command = [cmake, "--build", str(build_dir)]
    if build_config:
        command.extend(["--config", build_config])
    command.extend(["--target", REFRESH_TARGET])
    return command


@contextmanager
def cmake_build_commands(
    cmake: str,
    build_dir: Path,
    build_targets: tuple[str, ...],
    *,
    build_config: str | None = None,
    max_command_length: int = MAX_BUILD_COMMAND_LENGTH,
) -> Iterator[list[list[str]]]:
    if not build_targets:
        yield []
        return

    command_prefix = [cmake, "--build", str(build_dir)]
    if build_config:
        command_prefix.extend(["--config", build_config])

    if len(build_targets) > 1:
        cache = {
            entry.name: entry.value for entry in cmake_cache.load_cmake_cache(build_dir)
        }
        if cache["CMAKE_GENERATOR"] == "Unix Makefiles":
            with cmake_make.selected_build_arguments(
                build_dir, build_targets
            ) as arguments:
                yield [[*command_prefix, *arguments]]
            return

    command_prefix.append("--target")

    commands = []
    batch = []
    prefix_length = sum(len(argument) + 1 for argument in command_prefix)
    batch_length = prefix_length
    for build_target in build_targets:
        target_length = len(build_target) + 1
        if prefix_length + target_length > max_command_length:
            raise CTestMetadataError(
                f"CMake build target exceeds the command-length limit: {build_target}"
            )
        if batch and batch_length + target_length > max_command_length:
            commands.append([*command_prefix, *batch])
            batch = []
            batch_length = prefix_length
        batch.append(build_target)
        batch_length += target_length
    if batch:
        commands.append([*command_prefix, *batch])
    yield commands


@dataclass(frozen=True)
class CTestBuildAndRunStep:
    cmake: str
    ctest: str
    build_dir: Path
    arguments: list[str]
    cwd: Path
    env: dict[str, str] | None = None

    def describe(self) -> str:
        command_env = self.env if self.env is not None else os.environ
        build_config = ctest_build_config(self.arguments)
        if build_config is None:
            build_config = command_env.get("CTEST_CONFIGURATION_TYPE")
        return "\n".join(
            [
                CommandStep(
                    cmake_refresh_command(
                        self.cmake,
                        self.build_dir,
                        build_config=build_config,
                    ),
                    cwd=self.cwd,
                    env=self.env,
                    label="refresh the CMake test graph",
                ).describe(),
                CommandStep(
                    ctest_selection_command(
                        self.ctest,
                        self.build_dir,
                        self.arguments,
                    ),
                    cwd=self.cwd,
                    env=self.env,
                    label="enumerate the exact CTest selection",
                ).describe(),
                CommandStep(
                    [
                        self.cmake,
                        "--build",
                        str(self.build_dir),
                        "--target",
                        "<selected-ctest-build-roots...>",
                    ],
                    cwd=self.cwd,
                    env=self.env,
                    label="build the selected CTest roots",
                ).describe(),
                CommandStep(
                    ctest_run_command(
                        self.ctest,
                        self.build_dir,
                        self.arguments,
                    ),
                    cwd=self.cwd,
                    env=self.env,
                    label="run the same CTest selection",
                ).describe(),
            ]
        )

    def run(self, verbose: bool = False) -> int:
        command_env = self.env if self.env is not None else os.environ
        ctest_env = dict(command_env)
        ctest_env[SKIP_GLOBAL_TMPDIR_CLEANUP_ENV] = "1"
        build_config = ctest_build_config(self.arguments)
        if build_config is None:
            build_config = command_env.get("CTEST_CONFIGURATION_TYPE")
        refresh_command = cmake_refresh_command(
            self.cmake,
            self.build_dir,
            build_config=build_config,
        )
        if verbose:
            print("dev.py: refresh CMake test graph")
            print("  " + quote_command(refresh_command))
            sys.stdout.flush()
        try:
            refresh_result = subprocess.run(
                refresh_command,
                cwd=self.cwd,
                env=self.env,
            )
        except OSError as exc:
            print(
                f"dev.py: failed to run {quote_command(refresh_command)}: {exc}",
                file=sys.stderr,
            )
            return 127
        if refresh_result.returncode != 0:
            return refresh_result.returncode

        select_command = ctest_selection_command(
            self.ctest,
            self.build_dir,
            self.arguments,
        )
        if verbose:
            print("dev.py: enumerate selected CTests")
            print("  " + quote_command(select_command))
            sys.stdout.flush()
        try:
            selection_result = subprocess.run(
                select_command,
                cwd=self.cwd,
                env=ctest_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            print(
                f"dev.py: failed to run {quote_command(select_command)}: {exc}",
                file=sys.stderr,
            )
            return 127

        if selection_result.stderr:
            print(selection_result.stderr.rstrip(), file=sys.stderr)
        if selection_result.returncode != 0:
            if selection_result.stdout:
                print(selection_result.stdout.rstrip())
            return selection_result.returncode

        build_target_catalog_path = self.build_dir / BUILD_TARGETS_CATALOG_FILENAME
        try:
            build_target_catalog_payload = build_target_catalog_path.read_text()
        except OSError as exc:
            print(
                "dev.py: failed to read CTest build-target catalog "
                f"{build_target_catalog_path}: {exc}",
                file=sys.stderr,
            )
            return 1

        try:
            selection = parse_ctest_selection(
                selection_result.stdout,
                build_target_catalog_payload,
            )
            with cmake_build_commands(
                self.cmake,
                self.build_dir,
                selection.build_targets,
                build_config=build_config,
            ) as build_commands:
                if build_commands:
                    print(
                        f"dev.py: building {len(selection.build_targets)} target(s) "
                        f"for {len(selection.test_names)} selected CTest test(s)"
                    )
                    sys.stdout.flush()
                elif verbose:
                    print(
                        f"dev.py: {len(selection.test_names)} selected CTest test(s) "
                        "require no build"
                    )

                for build_command in build_commands:
                    if verbose:
                        print("dev.py: build selected CTest roots")
                        print("  " + quote_command(build_command))
                        sys.stdout.flush()
                    try:
                        build_result = subprocess.run(
                            build_command,
                            cwd=self.cwd,
                            env=self.env,
                        )
                    except OSError as exc:
                        print(
                            f"dev.py: failed to run {quote_command(build_command)}: {exc}",
                            file=sys.stderr,
                        )
                        return 127
                    if build_result.returncode != 0:
                        return build_result.returncode
        except (
            CTestMetadataError,
            cmake_make.MakeBuildError,
            cmake_file_api.FileApiError,
            OSError,
        ) as exc:
            print(
                f"dev.py: failed to prepare selected CTest build closure: {exc}",
                file=sys.stderr,
            )
            return 1

        try:
            prepare_selected_test_temporary_directories(selection)
        except (CTestMetadataError, OSError) as exc:
            print(
                f"dev.py: failed to prepare selected CTest temporary "
                f"directories: {exc}",
                file=sys.stderr,
            )
            return 1

        run_command = ctest_run_command(
            self.ctest,
            self.build_dir,
            self.arguments,
        )
        if verbose:
            print("dev.py: run selected CTests")
            print("  " + quote_command(run_command))
            sys.stdout.flush()
        try:
            return subprocess.run(
                run_command,
                cwd=self.cwd,
                env=ctest_env,
            ).returncode
        except OSError as exc:
            print(
                f"dev.py: failed to run {quote_command(run_command)}: {exc}",
                file=sys.stderr,
            )
            return 127
