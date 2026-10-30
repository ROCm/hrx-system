# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

import bazel_to_cmake_config
import bazel_to_cmake_converter
import bazel_to_cmake_requirements
from conditions import Equal, Matches, Not, Option, all_of, any_of
from loom_binary import LoomBinaryBuildFileFunctions


class RuntimeBuildFileFunctions(
    LoomBinaryBuildFileFunctions, bazel_to_cmake_converter.BuildFileFunctions
):
    def _declarative_load_bindings(self):
        return {**super()._declarative_load_bindings(), "select": self.select}

    def _custom_initialize(self):
        self._runtime_requirement_policy = (
            bazel_to_cmake_requirements.load_project_policy(
                self._repo_root,
                "runtime",
            )
        )

    def _package_name(self):
        return os.path.relpath(self._build_dir, self._repo_root).replace("\\", "/")

    def _should_emit_python_target(self):
        package_name = self._package_name()
        specification_package = "runtime/src/iree/vm/bytecode/spec"
        return package_name == specification_package or package_name.startswith(
            specification_package + "/"
        )

    def _runtime_package_policy(self):
        return self._runtime_requirement_policy.collect(self._package_name())

    def _apply_runtime_cmake_policy(self, kwargs, include_run_requirements=False):
        policy = self._runtime_package_policy()
        kwargs = dict(kwargs)
        kwargs["target_compatible_with"] = (
            bazel_to_cmake_requirements.append_conditions(
                kwargs.get("target_compatible_with"),
                policy.conditions(),
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
        if (
            self._package_name()
            == "runtime/src/iree/hal/drivers/task/executable/loaders/registration"
        ):
            deps = deps + ["${IREE_HAL_EXECUTABLE_LOADER_EXTRA_DEPS}"]
        self.cc_library(deps=deps + ["//runtime/src:defines"], **kwargs)

    def iree_cc_library(self, **kwargs):
        if self._package_name() == "runtime/src":
            kwargs["cmake_package"] = "iree"
        self.cc_library(**kwargs)

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

    def iree_runtime_tracing_provider(self, name, header, **kwargs):
        self.iree_runtime_cc_library(
            name=name,
            defines=[
                f'IREE_TRACING_PROVIDER_H=\\"{header}\\"',
                "IREE_TRACING_MODE=${IREE_TRACING_MODE}",
            ],
            **kwargs,
        )

    def iree_hal_driver_registry(self, name, modules, **kwargs):
        self._check_no_unhandled_kwargs("iree_hal_driver_registry", kwargs)
        if modules:
            raise NotImplementedError(
                "Native external drivers use iree_register_external_hal_driver"
            )
        self._converter.body += (
            "iree_hal_driver_registry(\n"
            + self._convert_string_arg_block("NAME", name)
            + ")\n\n"
        )

    def iree_runtime_benchmark_library(
        self, name, srcs, deps, disabled_srcs, disabled_deps, **kwargs
    ):
        if self._configuration is not None:
            enabled = all_of(
                "IREE_TARGET_HAS_THREADS",
                any_of("IREE_BUILD_TESTS", "IREE_BUILD_BENCHMARKS"),
            ).evaluate(self._configuration)
            self.iree_runtime_cc_library(
                name=name,
                srcs=srcs if enabled else disabled_srcs,
                deps=deps if enabled else disabled_deps,
                **kwargs,
            )
            return
        self._converter.body += "if(IREE_TARGET_HAS_THREADS AND (IREE_BUILD_TESTS OR IREE_BUILD_BENCHMARKS))\n"
        self.iree_runtime_cc_library(name=name, srcs=srcs, deps=deps, **kwargs)
        self._converter.body += "else()\n"
        self.iree_runtime_cc_library(
            name=name, srcs=disabled_srcs, deps=disabled_deps, **kwargs
        )
        self._converter.body += "endif()\n\n"

    def iree_runtime_cc_fuzz(self, deps=[], **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        self.iree_cc_fuzz(deps=deps + ["//runtime/src:defines"], **kwargs)

    def iree_executable_test(self, src, **kwargs):
        self.native_test(src=src, **kwargs)

    def iree_checked_glob(self, files, **kwargs):
        return files

    def iree_runtime_hal_cts_test_suite(self, **kwargs):
        kwargs = self._apply_runtime_cmake_policy(
            kwargs,
            include_run_requirements=True,
        )
        self._iree_runtime_hal_cts_test_suite(**kwargs)

    def iree_runtime_hal_cts_testdata(
        self,
        name,
        srcs,
        backend_names,
        target_name,
        target_family,
        target_key,
        **kwargs,
    ):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        compatibility = kwargs.pop("target_compatible_with", None)
        if self._should_skip_target(**kwargs):
            return
        self._emit_platform_guard_begin(compatibility)
        self._converter.body += (
            "iree_runtime_hal_cts_testdata(\n"
            + self._convert_string_arg_block("NAME", name)
            + self._convert_data_srcs_block(srcs, sort=False)
            + self._convert_string_list_block("BACKEND_NAMES", backend_names)
            + self._convert_string_arg_block("TARGET_NAME", target_name)
            + self._convert_string_arg_block("TARGET_FAMILY", target_family)
            + self._convert_string_arg_block("TARGET_KEY", target_key)
            + ")\n\n"
        )
        self._emit_platform_guard_end(compatibility)

    def iree_hal_amd_gpu_cts_testdata(self, name, srcs, backend_names, **kwargs):
        if not getattr(self, "_amd_gpu_cts_profiles_emitted", False):
            self._amd_gpu_cts_profiles_emitted = True
            capabilities = self._loaded_modules.symbol(
                "//loom/build_tools/amdgpu:target_config.bzl",
                "LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITY_BY_TARGET",
                self._build_dir,
            )
            self._converter.body += (
                'include("${IREE_SOURCE_DIR}/runtime/build_tools/cmake/iree_runtime_amd_gpu_cts.cmake")\n\n'
                "set(_IREE_HAL_AMD_GPU_CTS_TARGETS\n"
                + "".join(
                    f'  "{target}=LOOM_ARCH_AMDGPU_{capability.upper()}"\n'
                    for target, capability in capabilities.items()
                )
                + ")\n\n"
            )
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        compatibility = (kwargs.pop("target_compatible_with", None) or []) + [
            "//loom/config/emit:amdgpu_elf"
        ]
        self._emit_platform_guard_begin(compatibility)
        self._converter.body += (
            "iree_hal_amd_gpu_cts_testdata(\n"
            + self._convert_string_arg_block("NAME", name)
            + self._convert_data_srcs_block(srcs, sort=False)
            + self._convert_string_list_block("BACKEND_NAMES", backend_names)
            + "  TARGETS ${_IREE_HAL_AMD_GPU_CTS_TARGETS}\n"
            + ")\n\n"
        )
        self._emit_platform_guard_end(compatibility)

    def iree_amdgpu_hal_cts_testdata(self, **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        self._iree_amdgpu_hal_cts_testdata(**kwargs)

    def iree_amdgpu_binary(self, **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        self._iree_amdgpu_binary(**kwargs)

    def iree_amdgpu_library(self, **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        self._iree_amdgpu_library(**kwargs)

    def iree_amdgpu_library_variants(self, **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        self._iree_amdgpu_library_variants(**kwargs)

    def iree_amdgpu_binary_variants(self, **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        self._iree_amdgpu_binary_variants(**kwargs)

    def iree_amdgpu_binary_variants_embed_data(self, **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        self._iree_amdgpu_binary_variants_embed_data(**kwargs)

    def iree_amdgpu_target_selectors_flag(self, *args, **kwargs):
        pass

    def iree_hal_amdgpu_source_device_binaries(self, name):
        module = (
            "//runtime/src/iree/hal/drivers/amdgpu/device/binaries/source:targets.bzl"
        )
        inputs = list(
            self._loaded_modules.symbol(module, "_GENERATOR_INPUTS", self._build_dir)
        )
        inputs += self._loaded_modules.symbol(
            module, "_GENERATOR_SCRIPT_DEPS", self._build_dir
        )
        self._converter.body += (
            "iree_hal_amdgpu_source_device_binaries(\n"
            + self._convert_string_arg_block("NAME", name)
            + self._convert_srcs_block(inputs, block_name="INPUTS")
            + ")\n\n"
        )

    def iree_hal_amdgpu_device_binaries(
        self,
        name,
        target_selectors_flag="//runtime/src/iree/hal/drivers/amdgpu:targets",
        target="amdgcn-amd-amdhsa",
    ):
        self._converter.body += (
            "iree_hal_amdgpu_device_binaries(\n"
            + self._convert_string_arg_block("NAME", name)
            + self._convert_string_arg_block("TARGET", target)
            + self._convert_amdgpu_target_selectors_block(target_selectors_flag)
            + ")\n\n"
        )


def convert_unmatched_target(converter, target):
    return ["iree::" + converter._convert_to_cmake_path(target)]


PROJECT_CONFIG = bazel_to_cmake_config.ProjectConfig(
    name="runtime",
    package_prefixes=["runtime"],
    build_file_functions=RuntimeBuildFileFunctions,
    target_mappings={
        "//runtime/src:defines": ["iree::defines"],
        "//runtime/config/hal:external_driver_registry": [
            "iree::hal::drivers::configured_external_driver_registry"
        ],
        "@iree_macos_toolchain//:Foundation": ['"-framework Foundation"'],
    },
    condition_mappings={
        "//runtime/src/iree/math:gnu_x86": all_of(
            all_of(Matches("CMAKE_C_COMPILER_ID", "Clang|GNU"), Not(Option("MSVC"))),
            any_of(Equal("IREE_ARCH", "x86_32"), Equal("IREE_ARCH", "x86_64")),
        ),
        "//runtime/src/iree/math:clang_cl_x86": all_of(
            all_of(Matches("CMAKE_C_COMPILER_ID", "Clang"), "MSVC"),
            any_of(Equal("IREE_ARCH", "x86_32"), Equal("IREE_ARCH", "x86_64")),
        ),
        "//runtime/src/iree/math:msvc_x86_32": all_of(
            Equal("CMAKE_C_COMPILER_ID", "MSVC"), Equal("IREE_ARCH", "x86_32")
        ),
        "//runtime/config/base:allocator_system_custom": Equal(
            "IREE_ALLOCATOR_SYSTEM", "custom"
        ),
        "//runtime/config/base:allocator_system_mimalloc": Equal(
            "IREE_ALLOCATOR_SYSTEM", "mimalloc"
        ),
        ":windows_x86_64": all_of("WIN32", Equal("IREE_ARCH", "x86_64")),
    },
    convert_unmatched_target=convert_unmatched_target,
)
