# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import bazel_to_cmake_config

# This runner consumes Loom's generated artifacts and compiler build rules while
# retaining the ordinary experimental target namespace.
_LOOM = bazel_to_cmake_config.include_project(
    __file__, "../../loom/.bazel_to_cmake.cfg.py"
)

PROJECT_CONFIG = bazel_to_cmake_config.ProjectConfig(
    name="loom_serve",
    package_prefixes=["experimental/loom_serve"],
    build_file_functions=_LOOM.build_file_functions,
)
