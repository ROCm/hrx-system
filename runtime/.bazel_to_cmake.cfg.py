# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

import bazel_to_cmake_config
import bazel_to_cmake_converter
import bazel_to_cmake_requirements


class RuntimeBuildFileFunctions(bazel_to_cmake_converter.BuildFileFunctions):
    def _custom_initialize(self):
        self._runtime_requirement_policy = (
            bazel_to_cmake_requirements.load_project_policy(
                self._repo_root,
                "runtime",
            )
        )

    def _package_name(self):
        return os.path.relpath(self._build_dir, self._repo_root).replace("\\", "/")

    def _runtime_package_policy(self):
        return self._runtime_requirement_policy.collect(self._package_name())

    def _apply_runtime_cmake_policy(self, kwargs, include_run_requirements=False):
        policy = self._runtime_package_policy()
        kwargs = dict(kwargs)
        kwargs["target_compatible_with"] = (
            bazel_to_cmake_requirements.append_cmake_conditions(
                kwargs.get("target_compatible_with"),
                policy.cmake_conditions(),
            )
        )
        policy_tags = policy.tags(include_run_requirements=include_run_requirements)
        if policy_tags or kwargs.get("tags"):
            tags = list(kwargs.get("tags") or [])
            tags.extend(policy_tags)
            kwargs["tags"] = tags
        if (
            include_run_requirements
            and policy.resource_group
            and not kwargs.get("resource_group")
        ):
            kwargs["resource_group"] = policy.resource_group
        return kwargs

    def iree_runtime_cc_library(self, deps=[], **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        self.cc_library(deps=deps + ["//runtime/src:defines"], **kwargs)

    def iree_runtime_cc_binary(self, deps=[], **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        self.cc_binary(deps=deps + ["//runtime/src:defines"], **kwargs)

    def iree_runtime_cc_test(self, deps=[], resource_group=None, **kwargs):
        if resource_group:
            kwargs["resource_group"] = resource_group
        kwargs = self._apply_runtime_cmake_policy(
            kwargs,
            include_run_requirements=True,
        )
        resource_group = kwargs.pop("resource_group", None)
        tags = kwargs.get("tags", [])
        if not resource_group and tags:
            for tag in tags:
                if tag.startswith("resource_group:"):
                    resource_group = tag[len("resource_group:") :]
                    break
        self.cc_test(
            deps=deps + ["//runtime/src:defines"],
            resource_group=resource_group,
            **kwargs,
        )

    def iree_runtime_cc_benchmark(self, deps=[], **kwargs):
        kwargs = self._apply_runtime_cmake_policy(
            kwargs,
            include_run_requirements=True,
        )
        self.cc_binary_benchmark(
            deps=deps
            + [
                "//runtime/src:defines",
                "//third_party:google_benchmark",
            ],
            **kwargs,
        )

    def iree_runtime_cc_fuzz(self, deps=[], **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        self.iree_cc_fuzz(deps=deps + ["//runtime/src:defines"], **kwargs)

    def iree_executable_test(self, src, **kwargs):
        self.native_test(src=src, **kwargs)

    def iree_runtime_flatbuffer_c_library(
        self, flatcc_includes=None, deps=None, **kwargs
    ):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        if deps:
            kwargs["deps"] = deps
        self.iree_flatbuffer_c_library(includes=flatcc_includes, **kwargs)

    def iree_checked_glob(self, files, **kwargs):
        return files

    def iree_generated_files(self, name, srcs, outs, args, output_args, tool, **kwargs):
        if tool != ":generate_vm_isa":
            raise NotImplementedError(f"iree_generated_files tool: {tool}")
        cmd_parts = ["python3 $(rootpath generate_vm_isa.py)"]
        cmd_parts.extend(arg.replace("$(location ", "$(rootpath ") for arg in args)
        for out in outs:
            cmd_parts.extend([output_args[out], f"$(execpath {out})"])
        self.iree_genrule(
            name=name,
            srcs=["generate_vm_isa.py"] + srcs,
            outs=outs,
            cmd=" ".join(cmd_parts),
        )

    def iree_runtime_vmasm_module(self, deps=[], **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        self.iree_vmasm_module(deps=deps + ["//runtime/src:defines"], **kwargs)

    def _emit_iree_hal_cts_test_suite(self, kwargs):
        if "backends" in kwargs and "backends_lib" not in kwargs:
            kwargs["backends_lib"] = kwargs.pop("backends")
        self._iree_hal_cts_test_suite(**kwargs)

    def iree_runtime_hal_cts_test_suite(self, **kwargs):
        kwargs = self._apply_runtime_cmake_policy(
            kwargs,
            include_run_requirements=True,
        )
        self._emit_iree_hal_cts_test_suite(kwargs)

    def iree_hal_cts_test_suite(self, **kwargs):
        kwargs = self._apply_runtime_cmake_policy(
            kwargs,
            include_run_requirements=True,
        )
        self._emit_iree_hal_cts_test_suite(kwargs)

    def iree_hal_cts_testdata(self, **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        self._iree_hal_cts_testdata(**kwargs)

    def iree_amdgpu_hal_cts_testdata(self, **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        self._iree_amdgpu_hal_cts_testdata(**kwargs)

    def iree_amdgpu_binary(self, **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        self._iree_amdgpu_binary(**kwargs)

    def iree_amdgpu_binary_variants(self, **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        self._iree_amdgpu_binary_variants(**kwargs)

    def iree_amdgpu_binary_variants_embed_data(self, **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        self._iree_amdgpu_binary_variants_embed_data(**kwargs)

    def iree_amdgpu_target_selectors_flag(self, *args, **kwargs):
        pass

    def iree_hal_amdgpu_source_device_binaries(self, name):
        self._iree_hal_amdgpu_source_device_binaries()


def convert_unmatched_target(converter, target):
    return ["iree::" + converter._convert_to_cmake_path(target)]


PROJECT_CONFIG = bazel_to_cmake_config.ProjectConfig(
    name="runtime",
    package_prefixes=["runtime"],
    build_file_functions=RuntimeBuildFileFunctions,
    target_mappings={
        "//runtime/src:defines": [],
    },
    convert_unmatched_target=convert_unmatched_target,
)
