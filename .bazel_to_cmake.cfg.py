# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import bazel_to_cmake_converter
import bazel_to_cmake_targets


DEFAULT_ROOT_DIRS = ["runtime/src/iree", "libhrx"]

REPO_MAP = {
    "@iree_core": "",
}


class CustomBuildFileFunctions(bazel_to_cmake_converter.BuildFileFunctions):
    def _drop_selects(self, values):
        if isinstance(values, bazel_to_cmake_converter.MixedDeps):
            return values.unconditional
        return values

    def iree_select(self, selector):
        return self.select(selector)

    def hrx_cc_library(self, deps=[], **kwargs):
        self.cc_library(deps=deps + ["//runtime/src:defines", "//libhrx:defines"], **kwargs)

    def hrx_cc_binary(self, deps=[], **kwargs):
        self.cc_binary(deps=deps + ["//runtime/src:defines", "//libhrx:defines"], **kwargs)

    def hrx_cc_test(self, deps=[], resource_group=None, **kwargs):
        self.cc_test(
            deps=deps + ["//runtime/src:defines", "//libhrx:defines"],
            resource_group=resource_group,
            **kwargs,
        )

    def hrx_cc_benchmark(self, deps=[], **kwargs):
        self.cc_binary_benchmark(
            deps=deps + ["//runtime/src:defines", "//libhrx:defines"], **kwargs
        )

    def hrx_cc_shared_library(self, deps=[], **kwargs):
        kwargs["copts"] = self._drop_selects(kwargs.get("copts"))
        kwargs["hdrs"] = [
            hdr
            for hdr in (kwargs.get("hdrs") or [])
            if hdr != "//libhrx/src:iree_hal_compat.h"
        ]
        self.cc_library(
            deps=deps + ["//runtime/src:defines", "//libhrx:defines"],
            shared=True,
            **kwargs,
        )

    def hrx_cts_test(self, name, deps=[], **kwargs):
        srcs = kwargs.get("srcs")
        copts = kwargs.get("copts")
        defines = kwargs.get("defines")
        includes = kwargs.get("includes")
        tags = kwargs.get("tags")
        full_deps = [
            ":core",
            "@catch2//:catch2",
        ] + deps + ["//runtime/src:defines", "//libhrx:defines"]
        name_block = self._convert_string_arg_block("NAME", "hrx_cts_" + name, quote=False)
        srcs_block = self._convert_srcs_block(srcs)
        copts_block = self._convert_string_list_block("COPTS", copts, sort=False)
        defines_block = self._convert_string_list_block("DEFINES", defines)
        deps_block, platform_deps_block = self._convert_platform_select_deps("hrx_cts_" + name, full_deps)
        includes_block = self._convert_includes_block(includes)
        args_block = self._convert_string_list_block(
            "ARGS",
            [
                "--hrx-library",
                "$<TARGET_FILE:libhrx::src::libhrx::hrx>",
            ],
        )
        if platform_deps_block:
            self._converter.body += platform_deps_block
        labels_block = self._convert_string_list_block("LABELS", tags)
        self._converter.body += (
            f"hrx_cc_test(\n"
            f"{name_block}"
            f"{srcs_block}"
            f"{copts_block}"
            f"{defines_block}"
            f"{deps_block}"
            f"{args_block}"
            f"{includes_block}"
            f"{labels_block}"
            f")\n\n"
        )

    def iree_runtime_cc_library(self, deps=[], **kwargs):
        self.cc_library(deps=deps + ["//runtime/src:defines"], **kwargs)

    def iree_runtime_cc_binary(self, deps=[], **kwargs):
        self.cc_binary(deps=deps + ["//runtime/src:defines"], **kwargs)

    def iree_runtime_cc_test(self, deps=[], resource_group=None, **kwargs):
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
        self.cc_binary_benchmark(deps=deps + ["//runtime/src:defines"], **kwargs)

    def iree_runtime_cc_fuzz(self, deps=[], **kwargs):
        self.iree_cc_fuzz(deps=deps + ["//runtime/src:defines"], **kwargs)

    def iree_executable_test(self, src, **kwargs):
        self.native_test(src=src, **kwargs)

    def iree_runtime_flatbuffer_c_library(
        self, flatcc_includes=None, deps=None, **kwargs
    ):
        if deps:
            kwargs["deps"] = deps
        self.iree_flatbuffer_c_library(includes=flatcc_includes, **kwargs)

    def iree_checked_glob(self, files, **kwargs):
        return files

    def iree_generated_files(self, name, srcs, outs, args, output_args, tool, **kwargs):
        if tool != ":generate_vm_isa":
            self._convert_unimplemented_function("iree_generated_files", tool)
            return
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
        self.iree_vmasm_module(deps=deps + ["//runtime/src:defines"], **kwargs)

    def iree_runtime_hal_cts_test_suite(self, **kwargs):
        if "backends" in kwargs and "backends_lib" not in kwargs:
            kwargs["backends_lib"] = kwargs.pop("backends")
        self.iree_hal_cts_test_suite(**kwargs)

    def iree_execution_test_suite(self, **kwargs):
        # The reduced HRX runtime CMake bring-up builds the tools needed by
        # runtime targets, but does not port the YAML execution-test harness yet.
        pass


class CustomTargetConverter(bazel_to_cmake_targets.TargetConverter):
    def _initialize(self):
        self._update_target_mappings(
            {
                "//runtime/src:defines": [],
                "//libhrx:defines": ["libhrx_defs"],
                "//libhrx/src/libhrx:hrx_static": [
                    "libhrx::src::libhrx::hrx"
                ],
                # Temporary AQL profile SDK header shape. See the comments in
                # build_tools/cmake/hrx_dependencies.cmake and
                # build_tools/third_party/BUILD.bazel: this should become a
                # real ROCm-provided package/target once upstream publishes
                # package metadata for include/aqlprofile-sdk.
                "//build_tools/third_party:aqlprofile_sdk_headers": [
                    "aqlprofile-sdk::headers"
                ],
                "@flatcc//:compiler": ["iree-flatcc-cli"],
                "@flatcc//:flatcc": ["flatcc"],
                "@flatcc//:parsing": ["flatcc::parsing"],
                "@flatcc//:runtime": ["flatcc::runtime"],
                "//build_tools/third_party/flatcc": ["flatcc"],
                "//build_tools/third_party/flatcc:flatcc": ["flatcc"],
                "//build_tools/third_party/flatcc:parsing": ["flatcc::parsing"],
                "//build_tools/third_party/flatcc:runtime": ["flatcc::runtime"],
                "@catch2//:catch2": ["Catch2::Catch2"],
            }
        )

    def _convert_unmatched_target(self, target: str) -> str:
        if target.startswith("//libhrx"):
            cmake_path = self._convert_to_cmake_path(target)
            if cmake_path == "libhrx":
                return ["libhrx"]
            if cmake_path.startswith("libhrx::"):
                cmake_path = cmake_path[len("libhrx::") :]
            return ["libhrx::" + cmake_path]
        return ["iree::" + self._convert_to_cmake_path(target)]
