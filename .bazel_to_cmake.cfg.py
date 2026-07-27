# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import bazel_to_cmake_config

BAZEL_TO_CMAKE_REPO_ROOT = True

DEFAULT_ROOT_DIRS = ["runtime/src/iree", "libhrx", "loom"]

REPO_MAP = {
    "@iree": "",
}

PROJECTS = bazel_to_cmake_config.include_projects(
    __file__,
    [
        "runtime/.bazel_to_cmake.cfg.py",
        "libhrx/.bazel_to_cmake.cfg.py",
        "loom/.bazel_to_cmake.cfg.py",
    ],
)


def convert_unmatched_target(converter, target):
    return ["iree::" + converter._convert_to_cmake_path(target)]
