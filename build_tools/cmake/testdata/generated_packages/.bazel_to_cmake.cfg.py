# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from bazel_to_cmake_converter import BuildFileFunctions
from bazel_to_cmake_targets import TargetConverter

BAZEL_TO_CMAKE_REPO_ROOT = True
DEFAULT_ROOT_DIRS = ["package"]


class CustomBuildFileFunctions(BuildFileFunctions):
    def _should_emit_python_target(self):
        return True


class CustomTargetConverter(TargetConverter):
    def _convert_unmatched_target(self, target):
        return ["fixture::" + self._convert_to_cmake_path(target)]

    def _initialize(self):
        self._update_target_mappings(
            {
                "//external:usage": ["authored_usage"],
                "//external:suppressions": ["native_suppression_provider"],
                "//package/producer:generator": [
                    "fixture::package::producer::generator"
                ],
            }
        )
