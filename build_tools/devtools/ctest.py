# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Builds the concrete CMake closure selected by CTest."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from build_tools.devtools.command_plan import CommandStep, quote_command

BUILD_TARGETS_CATALOG_FILENAME = "iree_ctest_build_targets.json"
REFRESH_TARGET = "iree-ctest-refresh"
MAX_BUILD_COMMAND_LENGTH = 8192


class CTestMetadataError(ValueError):
    """Raised when CTest selection or build-root metadata is invalid."""


@dataclass(frozen=True)
class CTestSelection:
    test_names: tuple[str, ...]
    build_targets: tuple[str, ...]


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
        or build_target_catalog.get("version") != 1
    ):
        raise CTestMetadataError("CMake build-target catalog has an invalid schema")
    catalog_tests = build_target_catalog.get("tests")
    if not isinstance(catalog_tests, dict):
        raise CTestMetadataError("CMake build-target catalog has no tests object")

    test_names = []
    build_targets = []
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

    return CTestSelection(tuple(test_names), tuple(build_targets))


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


def _command_length(arguments: list[str]) -> int:
    return sum(len(argument) + 1 for argument in arguments)


def cmake_build_commands(
    cmake: str,
    build_dir: Path,
    build_targets: tuple[str, ...],
    *,
    build_config: str | None = None,
    max_command_length: int = MAX_BUILD_COMMAND_LENGTH,
) -> list[list[str]]:
    if not build_targets:
        return []

    command_prefix = [cmake, "--build", str(build_dir)]
    if build_config:
        command_prefix.extend(["--config", build_config])
    command_prefix.append("--target")

    commands = []
    batch = []
    for build_target in build_targets:
        candidate = [*command_prefix, *batch, build_target]
        if batch and _command_length(candidate) > max_command_length:
            commands.append([*command_prefix, *batch])
            batch = []
            candidate = [*command_prefix, build_target]
        if _command_length(candidate) > max_command_length:
            raise CTestMetadataError(
                f"CMake build target exceeds the command-length limit: {build_target}"
            )
        batch.append(build_target)
    if batch:
        commands.append([*command_prefix, *batch])
    return commands


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
                env=self.env,
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
            build_commands = cmake_build_commands(
                self.cmake,
                self.build_dir,
                selection.build_targets,
                build_config=build_config,
            )
        except CTestMetadataError as exc:
            print(
                f"dev.py: invalid selected CTest build closure: {exc}", file=sys.stderr
            )
            return 1

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
                env=self.env,
            ).returncode
        except OSError as exc:
            print(
                f"dev.py: failed to run {quote_command(run_command)}: {exc}",
                file=sys.stderr,
            )
            return 127
