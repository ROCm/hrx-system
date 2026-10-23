# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CMAKE_COMMAND = os.environ["IREE_TEST_CMAKE_COMMAND"]
CTEST_COMMAND = os.environ["IREE_TEST_CTEST_COMMAND"]
CONFIGURATION = os.environ.get("IREE_TEST_CMAKE_BUILD_TYPE") or "Release"
REPO_ROOT = Path(__file__).resolve().parents[2]


def configured_cmake_arguments() -> list[str]:
    """Returns the outer build's generator and toolchain arguments."""
    arguments = ["-G", os.environ["IREE_TEST_CMAKE_GENERATOR"]]

    generator_platform = os.environ.get("IREE_TEST_CMAKE_GENERATOR_PLATFORM")
    if generator_platform:
        arguments.extend(["-A", generator_platform])
    generator_toolset = os.environ.get("IREE_TEST_CMAKE_GENERATOR_TOOLSET")
    if generator_toolset:
        arguments.extend(["-T", generator_toolset])

    cmake_variables = (
        "CMAKE_GENERATOR_INSTANCE",
        "CMAKE_MAKE_PROGRAM",
        "CMAKE_BUILD_TYPE",
        "CMAKE_C_COMPILER",
        "CMAKE_CXX_COMPILER",
        "CMAKE_AR",
        "CMAKE_LINKER",
        "CMAKE_RC_COMPILER",
        "CMAKE_MT",
    )
    for cmake_variable in cmake_variables:
        value = os.environ.get(f"IREE_TEST_{cmake_variable}")
        if value:
            arguments.append(f"-D{cmake_variable}={value}")

    return arguments


def run_command(*arguments, expect_failure=False):
    result = subprocess.run(
        arguments, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if (result.returncode != 0) != expect_failure:
        raise AssertionError(
            f"Command {arguments} returned {result.returncode}:\n{result.stdout}"
        )
    return result.stdout


def configure_project(source, build, *arguments, expect_failure=False):
    return run_command(
        CMAKE_COMMAND,
        "-S",
        str(source),
        "-B",
        str(build),
        *configured_cmake_arguments(),
        f"-DIREE_REPO_ROOT={REPO_ROOT.as_posix()}",
        f"-DPython3_EXECUTABLE={sys.executable}",
        *arguments,
        expect_failure=expect_failure,
    )


def build_project(build, *targets):
    return run_command(
        CMAKE_COMMAND,
        "--build",
        str(build),
        "--config",
        CONFIGURATION,
        "--parallel",
        "4",
        *(["--target", *targets] if targets else []),
    )


def test_project(build, *arguments):
    return run_command(
        CTEST_COMMAND,
        "--test-dir",
        str(build),
        "--build-config",
        CONFIGURATION,
        "--output-on-failure",
        "--no-tests=error",
        *arguments,
    )


def install_project(build, prefix):
    run_command(
        CMAKE_COMMAND,
        "--install",
        str(build),
        "--config",
        CONFIGURATION,
        "--prefix",
        str(prefix),
        "--component",
        "FixtureTests",
    )
