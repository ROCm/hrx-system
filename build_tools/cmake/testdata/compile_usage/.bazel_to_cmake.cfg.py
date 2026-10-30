# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from bazel_to_cmake_targets import TargetConverter

BAZEL_TO_CMAKE_REPO_ROOT = True
DEFAULT_ROOT_DIRS = ["package"]


class CustomTargetConverter(TargetConverter):
    def _convert_unmatched_target(self, target):
        return ["iree::fixture::" + target.rsplit(":", 1)[-1]]
