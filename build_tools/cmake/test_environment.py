# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os


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
