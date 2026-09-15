# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

import bazel_to_cmake_config
import bazel_to_cmake_converter
import bazel_to_cmake_requirements


class XdnaBuildFileFunctions(bazel_to_cmake_converter.BuildFileFunctions):
    def _custom_initialize(self):
        self._xdna_policy = bazel_to_cmake_requirements.load_project_policy(
            self._repo_root, "experimental/xdna"
        )

    def _apply_xdna_policy(self, kwargs, include_run_requirements=False):
        package_name = os.path.relpath(self._build_dir, self._repo_root).replace(
            "\\", "/"
        )
        policy = self._xdna_policy.collect(package_name)
        kwargs = dict(kwargs)
        kwargs["target_compatible_with"] = (
            bazel_to_cmake_requirements.append_cmake_conditions(
                kwargs.get("target_compatible_with"),
                policy.cmake_conditions(),
            )
        )
        kwargs["tags"] = list(kwargs.get("tags") or []) + policy.tags(
            include_run_requirements=include_run_requirements
        )
        if include_run_requirements and policy.resource_group:
            kwargs.setdefault("resource_group", policy.resource_group)
        return kwargs

    def xdna_cc_library(self, deps=[], **kwargs):
        self.cc_library(
            deps=deps + ["//runtime/src:defines"], **self._apply_xdna_policy(kwargs)
        )

    def xdna_cc_binary(self, deps=[], **kwargs):
        self.cc_binary(
            deps=deps + ["//runtime/src:defines"], **self._apply_xdna_policy(kwargs)
        )

    def xdna_cc_test(self, deps=[], **kwargs):
        self.cc_test(
            deps=deps + ["//runtime/src:defines"],
            **self._apply_xdna_policy(kwargs, include_run_requirements=True),
        )

    def xdna_execution_test_suite(self, **kwargs):
        self.iree_execution_test_suite(**self._apply_xdna_policy(kwargs))

    def xdna_cc_benchmark(self, deps=[], **kwargs):
        self.cc_binary_benchmark(
            deps=deps + ["//runtime/src:defines", "//third_party:google_benchmark"],
            **self._apply_xdna_policy(kwargs, include_run_requirements=True),
        )


PROJECT_CONFIG = bazel_to_cmake_config.ProjectConfig(
    name="experimental",
    package_prefixes=["experimental"],
    build_file_functions=XdnaBuildFileFunctions,
)
