# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import bazel_to_cmake_config
import bazel_to_cmake_converter


class HrxBuildFileFunctions(bazel_to_cmake_converter.BuildFileFunctions):
    def select(self, selector):
        if all(
            condition == "//conditions:default"
            or condition.startswith(":amdgpu_executable_target_")
            for condition in selector
        ):
            return selector.get("//conditions:default", [])
        return super().select(selector)

    def _drop_selects(self, values):
        if isinstance(values, bazel_to_cmake_converter.MixedDeps):
            return values.unconditional
        return values

    def _normalize_label(self, source):
        path = super()._normalize_label(source)
        project_prefix = "${PROJECT_SOURCE_DIR}/"
        if not path.startswith(project_prefix):
            return path

        # libhrx is a nested CMake project, so PROJECT_SOURCE_DIR names the
        # libhrx subtree rather than the repository root used by Bazel labels.
        repo_path = path[len(project_prefix) :]
        if repo_path.startswith("libhrx/"):
            return "${LIBHRX_SOURCE_DIR}/" + repo_path[len("libhrx/") :]
        return "${IREE_ROOT_DIR}/" + repo_path

    def hrx_cc_library(self, deps=[], **kwargs):
        self.cc_library(
            deps=deps + ["//runtime/src:defines", "//libhrx:defines"],
            **kwargs,
        )

    def hrx_cc_binary(self, deps=[], **kwargs):
        self.cc_binary(
            deps=deps + ["//runtime/src:defines", "//libhrx:defines"],
            **kwargs,
        )

    def hrx_cc_test(self, deps=[], resource_group=None, **kwargs):
        self.cc_test(
            deps=deps + ["//runtime/src:defines", "//libhrx:defines"],
            resource_group=resource_group,
            **kwargs,
        )

    def hrx_cc_benchmark(self, deps=[], **kwargs):
        self.cc_binary_benchmark(
            deps=deps
            + [
                "//runtime/src:defines",
                "//libhrx:defines",
                "//third_party:google_benchmark",
            ],
            **kwargs,
        )

    def hrx_cc_shared_library(self, deps=[], **kwargs):
        kwargs["copts"] = self._drop_selects(kwargs.get("copts"))
        self.cc_library(
            deps=deps + ["//runtime/src:defines", "//libhrx:defines"],
            shared=True,
            **kwargs,
        )

    def iree_amdgpu_binary(self, **kwargs):
        self._iree_amdgpu_binary(**kwargs)

    def iree_amdgpu_binary_variants(self, **kwargs):
        self._iree_amdgpu_binary_variants(**kwargs)

    def iree_amdgpu_binary_variants_embed_data(self, **kwargs):
        self._iree_amdgpu_binary_variants_embed_data(**kwargs)

    def iree_amdgpu_target_label_fragment(self, target):
        return target.replace("-", "_").replace(".", "_")

    def iree_amdgpu_target_selector_config_settings(
        self, name, flag, code_object_targets
    ):
        del name
        del flag
        requested = {
            target: ":amdgpu_executable_target_{}_requested".format(
                self.iree_amdgpu_target_label_fragment(target)
            )
            for target in code_object_targets
        }
        return type("AmdgpuTargetSelection", (), {"requested": requested})()

    def iree_amdgpu_target_selectors_flag(self, *args, **kwargs):
        pass

    def hrx_cts_test(self, name, deps=[], **kwargs):
        srcs = kwargs.get("srcs")
        copts = kwargs.get("copts")
        data = kwargs.get("data")
        defines = kwargs.get("defines")
        env = kwargs.get("env")
        includes = kwargs.get("includes")
        tags = kwargs.get("tags")
        args = kwargs.get("args")
        target_compatible_with = kwargs.get("target_compatible_with")
        full_deps = (
            [
                ":core",
                "//third_party:catch2",
            ]
            + deps
            + ["//runtime/src:defines", "//libhrx:defines"]
        )
        name_block = self._convert_string_arg_block(
            "NAME",
            "hrx_cts_" + name,
            quote=False,
        )
        srcs_block = self._convert_srcs_block(srcs)
        copts_block = self._convert_string_list_block("COPTS", copts, sort=False)
        defines_block = self._convert_string_list_block("DEFINES", defines)
        data_block = self._convert_target_list_block(
            "DATA",
            ["//libhrx/src/libhrx:hrx"] + (data or []),
            omit_empty=True,
        )
        deps_block, platform_deps_block = self._convert_platform_select_deps(
            "hrx_cts_" + name,
            full_deps,
        )
        includes_block = self._convert_includes_block(includes)
        args_block = self._convert_string_list_block(
            "ARGS",
            self._convert_location_args(
                [
                    "--hrx-library",
                    "$<TARGET_FILE:libhrx::src::libhrx::hrx>",
                ]
                + (args or [])
            ),
            sort=False,
        )
        env_block = self._convert_string_list_block(
            "ENV", self._convert_native_test_env(env), sort=False
        )
        if platform_deps_block:
            self._converter.body += platform_deps_block
        labels_block = self._convert_string_list_block("LABELS", tags)
        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"hrx_cc_test(\n"
            f"{name_block}"
            f"{srcs_block}"
            f"{copts_block}"
            f"{defines_block}"
            f"{data_block}"
            f"{deps_block}"
            f"{args_block}"
            f"{env_block}"
            f"{includes_block}"
            f"{labels_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)


def convert_unmatched_target(converter, target):
    cmake_path = converter._convert_to_cmake_path(target)
    if cmake_path == "libhrx":
        return ["libhrx"]
    if cmake_path.startswith("libhrx::"):
        cmake_path = cmake_path[len("libhrx::") :]
    return ["libhrx::" + cmake_path]


PROJECT_CONFIG = bazel_to_cmake_config.ProjectConfig(
    name="libhrx",
    package_prefixes=["libhrx"],
    build_file_functions=HrxBuildFileFunctions,
    target_mappings={
        "//build_tools/amdgpu:target_map_h": [],
        "//libhrx:defines": ["libhrx_defs"],
    },
    convert_unmatched_target=convert_unmatched_target,
)
