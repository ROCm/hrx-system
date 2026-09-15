# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

import bazel_to_cmake_config
import bazel_to_cmake_converter
import bazel_to_cmake_requirements

_AMDF_CONFIG_CMAKE_OPTIONS = {
    "//libamdf/config:enabled_setting": "AMDF_BUILD",
    "//libamdf/config/family:cdna": "AMDF_FAMILY_CDNA",
    "//libamdf/config/family:gpu": "AMDF_FAMILY_RDNA OR AMDF_FAMILY_CDNA",
    "//libamdf/config/family:rdna": "AMDF_FAMILY_RDNA",
    "//libamdf/config/family:xdna": "AMDF_FAMILY_XDNA",
    "//libamdf/config/provider:gpu": '(AMDF_FAMILY_RDNA OR AMDF_FAMILY_CDNA) AND (CMAKE_SYSTEM_NAME STREQUAL "Windows" OR (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND IREE_ARCH STREQUAL "x86_64"))',
    "//libamdf/config/provider:wddm": '(AMDF_FAMILY_RDNA OR AMDF_FAMILY_CDNA) AND CMAKE_SYSTEM_NAME STREQUAL "Windows"',
}


class AmdfBuildFileFunctions(bazel_to_cmake_converter.BuildFileFunctions):
    def _custom_initialize(self):
        self._amdf_requirement_policy = bazel_to_cmake_requirements.load_project_policy(
            self._repo_root,
            "libamdf",
        )

    def _package_name(self):
        return os.path.relpath(self._build_dir, self._repo_root).replace("\\", "/")

    def _apply_amdf_cmake_policy(self, kwargs, include_run_requirements=False):
        policy = self._amdf_requirement_policy.collect(self._package_name())
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

    def _convert_select_condition(self, label):
        if label in _AMDF_CONFIG_CMAKE_OPTIONS:
            return _AMDF_CONFIG_CMAKE_OPTIONS[label]
        return super()._convert_select_condition(label)

    def amdf_cc_library(self, deps=None, **kwargs):
        kwargs = self._apply_amdf_cmake_policy(kwargs)
        self.cc_library(deps=(deps or []) + ["//libamdf:headers"], **kwargs)

    def amdf_cc_binary(self, deps=None, **kwargs):
        kwargs = self._apply_amdf_cmake_policy(kwargs)
        self.cc_binary(deps=(deps or []) + ["//libamdf:headers"], **kwargs)

    def amdf_cc_test(self, deps=None, **kwargs):
        kwargs = self._apply_amdf_cmake_policy(
            kwargs,
            include_run_requirements=True,
        )
        self.cc_test(deps=(deps or []) + ["//libamdf:headers"], **kwargs)

    def amdf_cc_benchmark(self, deps=None, **kwargs):
        kwargs = self._apply_amdf_cmake_policy(kwargs, include_run_requirements=True)
        self.cc_binary_benchmark(
            deps=(deps or []) + ["//libamdf:headers", "//third_party:google_benchmark"],
            **kwargs,
        )

    def amdf_windows_sidecar_library(self, **kwargs):
        kwargs = dict(kwargs)
        kwargs["target_compatible_with"] = (
            kwargs.get("target_compatible_with") or []
        ) + [
            "@platforms//cpu:x86_64",
            "@platforms//os:windows",
        ]
        body_start = len(self._converter.body)
        self.amdf_cc_binary(**kwargs)
        emitted_body = self._converter.body[body_start:]
        self._converter.body = self._converter.body[:body_start]
        self._converter.body += emitted_body.replace(
            "iree_cc_binary(",
            "amdf_windows_sidecar_library(",
            1,
        )

    def iree_dynamic_library_bundle(self, **kwargs):
        # CMake colocates private companions with the public library, matching
        # the installed layout. Bazel's runfiles tree requires the explicit
        # environment bindings carried by this rule.
        del kwargs

    def amdf_library(
        self,
        name,
        components,
        hdrs,
        win_def_file,
        runtime_data=None,
        **kwargs,
    ):
        kwargs = self._apply_amdf_cmake_policy(kwargs)
        target_compatible_with = kwargs.pop("target_compatible_with", None)
        tags = kwargs.pop("tags", None)
        if self._should_skip_target(tags=tags, **kwargs):
            return

        self._target_file_paths[
            self._current_target_label(name + "_shared_artifact")
        ] = f"$<TARGET_FILE:{name}>"
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        components_block = self._convert_target_list_block("COMPONENTS", components)
        runtime_data_block, runtime_data_select_block = (
            self._convert_platform_select_deps(
                name,
                runtime_data,
                block_name="RUNTIME_DATA",
            )
        )
        del hdrs
        win_def_file_block = self._convert_srcs_block(
            [win_def_file],
            block_name="WINDOWS_DEF_FILE",
        )

        self._converter.header += f"amdf_declare_headers(\n{name_block})\n\n"
        condition = self._target_compatible_condition(target_compatible_with)
        if condition:
            self._converter.header += f"if({condition})\n"
        self._converter.header += f"amdf_declare_library(\n{name_block})\n"
        if condition:
            self._converter.header += "endif()\n"
        self._converter.header += "\n"

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += runtime_data_select_block
        self._converter.body += (
            f"amdf_library(\n{name_block}{components_block}{runtime_data_block}"
            f"{win_def_file_block})\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def amdf_cts_test_suite(
        self,
        name,
        srcs,
        deps,
        tags=None,
        resource_group=None,
        visibility=None,
        **kwargs,
    ):
        del visibility
        policy = self._apply_amdf_cmake_policy(
            dict(kwargs, tags=tags, resource_group=resource_group),
            include_run_requirements=True,
        )
        corpus_name = name + "_cases"
        self.amdf_cc_library(
            name=corpus_name,
            testonly=True,
            srcs=srcs,
            deps=deps,
            alwayslink=True,
            **kwargs,
        )
        common_deps = [":" + corpus_name, "//libamdf/cts/util:test_main"]
        for mode in ("static", "shared", "dynamic"):
            binary_name = name + "_" + mode + "_bin"
            data = None if mode == "static" else ["//libamdf:amdf_shared_artifact"]
            body_start = len(self._converter.body)
            self.amdf_cc_binary(
                name=binary_name,
                testonly=True,
                data=data,
                deps=common_deps + ["//libamdf/cts/util:" + mode + "_provider"],
                **kwargs,
            )
            emitted_body = self._converter.body[body_start:]
            self._converter.body = self._converter.body[:body_start]
            self._converter.body += emitted_body.replace(
                "iree_cc_binary(",
                "amdf_cts_binary(",
                1,
            )
            provider_args = (
                ["--amdf_library=$(rootpath //libamdf:amdf_shared_artifact)"]
                if mode == "dynamic"
                else []
            )
            for lifetime, suffix in (("process", ""), ("instance", "_instance")):
                self.iree_executable_test(
                    name=name + "_" + mode + suffix,
                    src=":" + binary_name,
                    args=["--amdf_native_lifetime=" + lifetime] + provider_args,
                    data=data,
                    **policy,
                )


def convert_unmatched_target(converter, target):
    cmake_path = converter._convert_to_cmake_path(target)
    if cmake_path.startswith("libamdf::"):
        return [cmake_path]
    return ["libamdf::" + cmake_path]


PROJECT_CONFIG = bazel_to_cmake_config.ProjectConfig(
    name="libamdf",
    package_prefixes=["libamdf"],
    build_file_functions=AmdfBuildFileFunctions,
    target_mappings={
        "//libamdf:amdf": ["amdf::amdf"],
        "//libamdf:amdf_shared_artifact": ["amdf::amdf"],
        "//libamdf:amdf_static": ["amdf::amdf_static"],
        "//libamdf:headers": ["amdf::headers"],
        "//libamdf/src/gpu/umd/wddm/wkmi:runtime": [
            "libamdf::src::gpu::umd::wddm::wkmi::amdf_wkmi_bridge",
        ],
    },
    convert_unmatched_target=convert_unmatched_target,
)
