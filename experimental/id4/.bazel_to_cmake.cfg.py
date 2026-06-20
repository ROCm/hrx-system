# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import bazel_to_cmake_config

_RUNTIME_PROJECT_CONFIG = bazel_to_cmake_config.include_project(
    __file__,
    "../../runtime/.bazel_to_cmake.cfg.py",
)

PROJECT_CONFIG = bazel_to_cmake_config.ProjectConfig(
    name="experimental_id4",
    package_prefixes=["experimental/id4"],
    build_file_functions=_RUNTIME_PROJECT_CONFIG.build_file_functions,
    target_mappings=_RUNTIME_PROJECT_CONFIG.target_mappings,
    convert_unmatched_target=_RUNTIME_PROJECT_CONFIG.convert_unmatched_target,
)
