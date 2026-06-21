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
_LOOM_PROJECT_CONFIG = bazel_to_cmake_config.include_project(
    __file__,
    "../../loom/.bazel_to_cmake.cfg.py",
)


class Id4BuildFileFunctions(
    _LOOM_PROJECT_CONFIG.build_file_functions,
    _RUNTIME_PROJECT_CONFIG.build_file_functions,
):
    def _custom_initialize(self):
        _LOOM_PROJECT_CONFIG.build_file_functions._custom_initialize(self)
        _RUNTIME_PROJECT_CONFIG.build_file_functions._custom_initialize(self)


def convert_unmatched_target(converter, target):
    return _RUNTIME_PROJECT_CONFIG.convert_unmatched_target(converter, target)


_TARGET_MAPPINGS = dict(_RUNTIME_PROJECT_CONFIG.target_mappings)
_TARGET_MAPPINGS.update(_LOOM_PROJECT_CONFIG.target_mappings)

PROJECT_CONFIG = bazel_to_cmake_config.ProjectConfig(
    name="experimental_id4",
    package_prefixes=["experimental/id4"],
    build_file_functions=Id4BuildFileFunctions,
    target_mappings=_TARGET_MAPPINGS,
    convert_unmatched_target=convert_unmatched_target,
)
