# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re

import bazel_to_cmake_config
import bazel_to_cmake_converter
import bazel_to_cmake_requirements


def _load_generated_amdgpu_target_config():
    config_path = os.path.join(
        os.path.dirname(__file__),
        "build_tools",
        "amdgpu",
        "target_config.bzl",
    )
    env = {}
    with open(config_path, encoding="utf-8") as config_file:
        exec(compile(config_file.read(), config_path, "exec"), env)
    return env


_LOOM_AMDGPU_TARGET_CONFIG = _load_generated_amdgpu_target_config()


def _loom_amdgpu_config_cmake_options():
    config = _LOOM_AMDGPU_TARGET_CONFIG
    options = {}
    for processor in config["LOOM_AMDGPU_SUPPORTED_EXACT_PROCESSORS"]:
        processor_upper = processor.upper()
        options[f"//loom/config/target/amdgpu:processor_{processor}"] = (
            f"LOOM_TARGET_AMDGPU_PROCESSOR_{processor_upper}"
        )
        options[f"//loom/config/target/amdgpu:iree_hal_processor_{processor}"] = (
            f"LOOM_TARGET_AMDGPU_IREE_HAL_PROCESSOR_{processor_upper}"
        )
    for capability in config["LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITIES"]:
        capability_upper = capability.upper()
        options[f"//loom/config/target/amdgpu:{capability}"] = (
            f"LOOM_TARGET_AMDGPU_{capability_upper}"
        )
        options[f"//loom/config/target/amdgpu:iree_hal_{capability}"] = (
            f"LOOM_TARGET_AMDGPU_IREE_HAL_{capability_upper}"
        )
    return options


_LOOM_CONFIG_CMAKE_OPTIONS = {
    "//loom/config/emit:amdgpu": "LOOM_EMIT_AMDGPU",
    "//loom/config/emit:spirv": "LOOM_EMIT_SPIRV",
    "//loom/config/emit:wasm": "LOOM_EMIT_WASM",
    "//loom/config/emit:xdna": "LOOM_EMIT_XDNA",
    "//loom/config/execute:amdgpu_hal": "LOOM_TARGET_ARCH_AMDGPU AND LOOM_EMIT_AMDGPU AND LOOM_EXECUTE_IREE_HAL AND IREE_HAL_DRIVER_AMDGPU",
    "//loom/config/execute:iree_hal": "LOOM_EXECUTE_IREE_HAL",
    "//loom/config/execute:iree_hal_amdgpu": "LOOM_EXECUTE_IREE_HAL AND IREE_HAL_DRIVER_AMDGPU",
    "//loom/config/execute:iree_hal_vulkan": "LOOM_EXECUTE_IREE_HAL AND IREE_HAL_DRIVER_VULKAN",
    "//loom/config/execute:spirv_vulkan_hal": "LOOM_TARGET_ARCH_SPIRV AND LOOM_EMIT_SPIRV AND LOOM_EXECUTE_IREE_HAL AND IREE_HAL_DRIVER_VULKAN",
    "//loom/config/import:mlir": "LOOM_IMPORT_MLIR",
    "//loom/config/import:tilelang": "LOOM_IMPORT_TILELANG",
    "//loom/config/target:amdgpu": "LOOM_TARGET_AMDGPU",
    "//loom/config/target:spirv": "LOOM_TARGET_SPIRV",
    "//loom/config/target:vm": "LOOM_TARGET_VM",
    "//loom/config/target:wasm": "LOOM_TARGET_WASM",
    "//loom/config/target:xdna": "LOOM_TARGET_XDNA",
    "//loom/config/target:x86": "LOOM_TARGET_X86",
    "//loom/config/target/arch:amdgpu": "LOOM_TARGET_ARCH_AMDGPU",
    "//loom/config/target/arch:spirv": "LOOM_TARGET_ARCH_SPIRV",
    "//loom/config/target/arch:vm": "LOOM_TARGET_ARCH_VM",
    "//loom/config/target/arch:wasm": "LOOM_TARGET_ARCH_WASM",
    "//loom/config/target/arch:xdna": "LOOM_TARGET_ARCH_XDNA",
    "//loom/config/target/arch:x86": "LOOM_TARGET_ARCH_X86",
}
_LOOM_CONFIG_CMAKE_OPTIONS.update(_loom_amdgpu_config_cmake_options())

_GENERATED_ROOTPATH_PATTERN = re.compile(r"\$\(rootpath ([^)]+)\)")
_GENERATED_LOCATION_PATTERN = re.compile(r"\$\(location ([^)]+)\)")


class LoomBuildFileFunctions(bazel_to_cmake_converter.BuildFileFunctions):
    def _declarative_load_bindings(self):
        return {
            **super()._declarative_load_bindings(),
            "loom_execution_profile": self.loom_execution_profile,
        }

    def _custom_initialize(self):
        self._loom_module_targets = set()
        self._loom_generated_file_families = {}
        self._loom_low_descriptor_archive_source_vars = {}
        self._loom_low_descriptor_archive_targets = {}
        self._loom_generated_external_files = {}
        self.LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITY_BY_KEY = _LOOM_AMDGPU_TARGET_CONFIG[
            "LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITY_BY_KEY"
        ]
        self._loom_amdgpu_descriptor_set_capabilities = _LOOM_AMDGPU_TARGET_CONFIG[
            "LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITIES"
        ]
        self._loom_amdgpu_descriptor_set_defines = _LOOM_AMDGPU_TARGET_CONFIG[
            "LOOM_AMDGPU_DESCRIPTOR_SET_DEFINES"
        ]
        self._loom_amdgpu_descriptor_set_generator_targets = _LOOM_AMDGPU_TARGET_CONFIG[
            "LOOM_AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS"
        ]
        self._loom_amdgpu_descriptor_set_capabilities_by_storage_generator_target = (
            _LOOM_AMDGPU_TARGET_CONFIG[
                "LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITIES_BY_STORAGE_GENERATOR_TARGET"
            ]
        )
        self._loom_requirement_policy = bazel_to_cmake_requirements.load_project_policy(
            self._repo_root,
            "loom",
        )

    def _package_name(self):
        return os.path.relpath(self._build_dir, self._repo_root).replace("\\", "/")

    def _loom_package_policy(self):
        return self._loom_requirement_policy.collect(self._package_name())

    def _apply_loom_cmake_policy(self, kwargs, include_run_requirements=False):
        policy = self._loom_package_policy()
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

    def _apply_loom_target_compatible_with(self, target_compatible_with):
        policy = self._loom_package_policy()
        return bazel_to_cmake_requirements.append_cmake_conditions(
            target_compatible_with,
            policy.cmake_conditions(),
        )

    def _cmake_guard_condition(self, target_compatible_with):
        if not target_compatible_with:
            return None
        if isinstance(target_compatible_with, bazel_to_cmake_converter.ConditionSelect):
            compatible_conditions = []
            for label, value in target_compatible_with.conditions.items():
                if label == "//conditions:default":
                    continue
                if value == []:
                    condition = self._convert_select_condition(label)
                    if condition:
                        compatible_conditions.append(condition)
            if compatible_conditions:
                return " OR ".join(compatible_conditions)
            return None
        conditions = [
            self._convert_select_condition(label) for label in target_compatible_with
        ]
        if not all(conditions):
            raise NotImplementedError(
                f"target_compatible_with: {target_compatible_with}"
            )
        return " AND ".join(conditions)

    def _guard_cmake_text(self, text, target_compatible_with):
        condition = self._cmake_guard_condition(target_compatible_with)
        if not condition:
            return text
        return f"\nif({condition})\n{text}endif()\n\n"

    def _convert_select_condition(self, label):
        if label in _LOOM_CONFIG_CMAKE_OPTIONS:
            return _LOOM_CONFIG_CMAKE_OPTIONS[label]
        return super()._convert_select_condition(label)

    def loom_config_compatible_with(self, config_labels):
        return list(config_labels)

    def loom_target_profile(
        self, name, family, selector, target_compatible_with=None, **kwargs
    ):
        condition = self._cmake_guard_condition(target_compatible_with)
        requires = f"  REQUIRES\n    {condition}\n" if condition else ""
        self._converter.body += (
            "loom_target_profile(\n"
            + self._convert_string_arg_block("NAME", name)
            + self._convert_string_arg_block("FAMILY", family)
            + self._convert_string_arg_block("SELECTOR", selector)
            + requires
            + ")\n\n"
        )

    def loom_amdgpu_target_profile(
        self, name, target, target_compatible_with=None, **kwargs
    ):
        capability = _LOOM_AMDGPU_TARGET_CONFIG[
            "LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITY_BY_TARGET"
        ][target]
        self.loom_target_profile(
            name=name,
            family="amdgpu",
            selector=target,
            target_compatible_with=list(target_compatible_with or [])
            + [
                "//loom/config/target/arch:amdgpu",
                "//loom/config/target/amdgpu:" + capability,
            ],
            **kwargs,
        )

    def loom_kernel_binary(self, name, tags=None, **kwargs):
        if not self._should_skip_target(tags=tags):
            raise NotImplementedError(
                f"loom_kernel_binary requires a CMake projection: {name}"
            )

    def loom_module(
        self,
        name,
        srcs,
        libraries=None,
        roots=None,
        configs=None,
        mode="merge",
        output=None,
        output_format="text",
        include_input_exports=False,
        include_input_tests=False,
        strip_check=False,
        require_resolved_config=False,
        strict_deps=False,
        tags=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )
        output = output or (name + (".loombc" if output_format == "bc" else ".loom"))
        self._loom_module_targets.add(self._current_target_label(name))
        self._target_file_paths[self._current_target_label(name)] = (
            f"${{CMAKE_CURRENT_BINARY_DIR}}/{output}"
        )
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_loom_module_inputs("SRCS", srcs)
        libraries_block = self._convert_loom_module_inputs("LIBRARIES", libraries)
        roots_block = self._convert_string_list_block("ROOTS", roots, sort=False)
        configs_block = self._convert_string_list_block("CONFIGS", configs, sort=False)
        mode_block = self._convert_string_arg_block("MODE", mode)
        output_block = self._convert_string_arg_block("OUTPUT", output)
        output_format_block = self._convert_string_arg_block(
            "OUTPUT_FORMAT", output_format
        )
        include_input_exports_block = self._convert_option_block(
            "INCLUDE_INPUT_EXPORTS", include_input_exports
        )
        include_input_tests_block = self._convert_option_block(
            "INCLUDE_INPUT_TESTS", include_input_tests
        )
        strip_check_block = self._convert_option_block("STRIP_CHECK", strip_check)
        require_resolved_config_block = self._convert_option_block(
            "REQUIRE_RESOLVED_CONFIG", require_resolved_config
        )
        strict_deps_block = self._convert_option_block("STRICT_DEPS", strict_deps)
        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"loom_module(\n"
            f"{name_block}"
            f"{srcs_block}"
            f"{libraries_block}"
            f"{roots_block}"
            f"{configs_block}"
            f"{mode_block}"
            f"{output_block}"
            f"{output_format_block}"
            f"{include_input_exports_block}"
            f"{include_input_tests_block}"
            f"{strip_check_block}"
            f"{require_resolved_config_block}"
            f"{strict_deps_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def loom_library(
        self,
        name,
        srcs=None,
        deps=None,
        data=None,
        input_format="",
        inputopts=None,
        tags=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        self._loom_module_targets.add(self._current_target_label(name))
        blocks = [
            self._convert_string_arg_block("NAME", name, quote=False),
            self._convert_loom_module_inputs("SRCS", srcs),
            self._convert_loom_module_inputs("LIBRARIES", deps),
            self._convert_data_list_block(data),
            self._convert_string_arg_block("INPUT_FORMAT", input_format or None),
            self._convert_string_list_block(
                "INPUTOPTS", self._convert_location_args(inputopts), sort=False
            ),
            "  MODE merge\n  OUTPUT_FORMAT bc\n  STRICT_DEPS\n",
        ]
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )
        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += "loom_module(\n" + "".join(blocks) + ")\n\n"
        self._emit_platform_guard_end(target_compatible_with)

    @staticmethod
    def _reject_workload_args(name, args):
        for arg in args or []:
            if arg.split("=", 1)[0] in ("--config", "--case"):
                raise ValueError(
                    f"{name}: use loom_test configs and case for workload selection, not {arg}"
                )

    def loom_execution_profile(
        self,
        name,
        target_family,
        target_class,
        executor,
        runner_args=None,
        build_requirements=None,
        run_requirements=None,
        resource_group=None,
        tags=None,
    ):
        self._reject_workload_args(name, runner_args)
        return {
            "kind": "loom_execution_profile",
            "name": name,
            "target_family": target_family,
            "target_class": target_class,
            "executor": executor,
            "runner_args": runner_args,
            "build_requirements": build_requirements or [],
            "run_requirements": run_requirements or [],
            "resource_group": resource_group,
            "tags": tags or [],
        }

    @staticmethod
    def _loom_test_name_suffix(value):
        if ":" in value:
            value = value.split(":")[-1]
        elif "/" in value:
            value = value.split("/")[-1]
        return re.sub(r"[-.+]", "_", value)

    def _loom_test_variants(self, name, configs, case, variants):
        if variants is None:
            return {name: (configs, case)}
        if not variants:
            raise ValueError(
                f"{name} variants must contain at least one named workload"
            )
        workloads = {}
        for variant_name, variant in variants.items():
            if not isinstance(variant_name, str) or not variant_name:
                raise ValueError(f"{name} variant names must be non-empty strings")
            for key in variant:
                if key not in ("configs", "case"):
                    raise ValueError(
                        f"{name} variant {variant_name} has unknown field {key}"
                    )
            workload_name = name + "_" + self._loom_test_name_suffix(variant_name)
            if workload_name in workloads:
                raise ValueError(f"{name} has colliding variant names: {variant_name}")
            workload_configs = dict(configs)
            workload_configs.update(variant.get("configs", {}))
            workloads[workload_name] = (workload_configs, variant.get("case", case))
        return workloads

    def loom_test_module(
        self,
        name,
        srcs,
        deps=None,
        data=None,
        input_format="",
        inputopts=None,
        tags=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        if not srcs:
            raise ValueError(f"{name} requires at least one authored test source")
        self._loom_module_targets.add(self._current_target_label(name))
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )
        blocks = [
            self._convert_string_arg_block("NAME", name, quote=False),
            self._convert_loom_module_inputs("SRCS", srcs),
            self._convert_loom_module_inputs("LIBRARIES", deps),
            self._convert_data_list_block(data),
            self._convert_string_arg_block("INPUT_FORMAT", input_format or None),
            self._convert_string_list_block(
                "INPUTOPTS", self._convert_location_args(inputopts), sort=False
            ),
        ]
        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += "loom_test_module(\n" + "".join(blocks) + ")\n\n"
        self._emit_platform_guard_end(target_compatible_with)

    def loom_test(
        self,
        name,
        srcs=None,
        deps=None,
        data=None,
        input_format="",
        inputopts=None,
        module=None,
        args=None,
        configs=None,
        case="",
        variants=None,
        execution_profiles=None,
        compile_targets=None,
        tags=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        if module and (srcs or deps or data or input_format or inputopts):
            raise ValueError(
                f"{name} module is exclusive with source and import options"
            )
        if not module and not srcs:
            raise ValueError(f"{name} requires srcs or a test module")
        if not execution_profiles and not compile_targets:
            raise ValueError(f"{name} requires execution_profiles or compile_targets")
        self._reject_workload_args(name, args)
        workloads = self._loom_test_variants(name, configs or {}, case, variants)
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )
        if not module:
            module = ":" + name + "_module"
            self.loom_test_module(
                name=name + "_module",
                srcs=srcs,
                deps=deps,
                data=data,
                input_format=input_format,
                inputopts=inputopts,
                target_compatible_with=target_compatible_with,
            )
        self._emit_platform_guard_begin(target_compatible_with)
        for workload_name, (workload_configs, workload_case) in workloads.items():
            for key, value in workload_configs.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ValueError(
                        f"{workload_name} configs must map symbol names to string values"
                    )
            if not isinstance(workload_case, str):
                raise ValueError(f"{workload_name} case must be a string")
            config_args = [
                f"--config={key}={workload_configs[key]}"
                for key in sorted(workload_configs)
            ]
            workload_args = config_args + (
                ["--case=" + workload_case] if workload_case else []
            )
            execution_names = set()
            for profile in execution_profiles or []:
                if profile.get("kind") != "loom_execution_profile":
                    raise ValueError(
                        f"{name} execution profile was not created by loom_execution_profile"
                    )
                suffix = self._loom_test_name_suffix(profile["name"])
                execution_name = f"{workload_name}_execute_{suffix}_test"
                if execution_name in execution_names:
                    raise ValueError(
                        f"{name} has colliding execution profiles: {profile['name']}"
                    )
                execution_names.add(execution_name)
                self._loom_execution_test(
                    execution_name,
                    self._convert_single_target(module),
                    profile,
                    args,
                    tags,
                    workload_args,
                )
            self._loom_check_compile_tests(
                name=workload_name,
                src=module,
                targets=compile_targets,
                data=None,
                env=None,
                tags=tags,
                target_compatible_with=None,
                args=config_args or None,
            )
        self._emit_platform_guard_end(target_compatible_with)

    def _loom_execution_test(self, name, module, profile, args, tags, workload_args):
        policy = bazel_to_cmake_requirements.CollectedPackagePolicy(
            build_requirements=profile["build_requirements"],
            run_requirements=profile["run_requirements"],
            resource_group=profile["resource_group"],
        )
        labels = list(tags or []) + profile["tags"]
        labels.extend(policy.tags(include_run_requirements=True))
        labels.extend(
            [
                "loom-execution-profile=" + profile["name"],
                "loom-target-family=" + profile["target_family"],
                "loom-target-class=" + profile["target_class"],
                "loom-executor=" + profile["executor"],
            ]
        )
        blocks = [
            self._convert_string_arg_block("NAME", name, quote=False),
            self._convert_string_arg_block("MODULE", module),
            self._convert_string_list_block(
                "ARGS", self._convert_test_location_args(args), sort=False
            ),
            self._convert_string_list_block(
                "RUNNER_ARGS",
                self._convert_test_location_args(
                    (profile["runner_args"] or []) + workload_args or None
                ),
                sort=False,
            ),
            self._convert_string_list_block("LABELS", labels, sort=False),
            self._convert_string_arg_block("RESOURCE_GROUP", policy.resource_group),
        ]
        self._emit_platform_guard_begin(policy.cmake_conditions())
        self._converter.body += "loom_execution_test(\n" + "".join(blocks) + ")\n\n"
        self._emit_platform_guard_end(policy.cmake_conditions())

    def _convert_loom_module_inputs(self, block_name, inputs):
        if inputs is None:
            return ""
        converted_inputs = []
        for input_value in inputs:
            if input_value.startswith(":") or input_value.startswith("//"):
                label = self._canonical_location_label(input_value)
                if self._is_source_data_label(input_value) or (
                    label in self._target_file_paths
                    and label not in self._loom_module_targets
                ):
                    converted_inputs.extend(self._cmake_location_paths(input_value))
                else:
                    converted_inputs.append(self._convert_single_target(input_value))
            else:
                converted_inputs.append(input_value)
        return self._convert_string_list_block(block_name, converted_inputs, sort=False)

    def loom_amdgpu_target_selectors_flag(self, **kwargs):
        return None

    def loom_amdgpu_target_config_settings(self, **kwargs):
        return None

    def _loom_amdgpu_descriptor_set_config_label(self, capability):
        if capability not in self._loom_amdgpu_descriptor_set_capabilities:
            raise ValueError(
                f"Unknown Loom AMDGPU descriptor-set capability: {capability}"
            )
        return "//loom/config/target/amdgpu:" + capability

    def loom_amdgpu_descriptor_set_compatible_with(self, capability):
        return self.loom_config_compatible_with(
            [self._loom_amdgpu_descriptor_set_config_label(capability)]
        )

    def loom_amdgpu_descriptor_table_compatible_with(self, storage_generator_target):
        capabilities = self._loom_amdgpu_descriptor_set_capabilities_by_storage_generator_target.get(
            storage_generator_target
        )
        if not capabilities:
            raise ValueError(
                f"Unknown AMDGPU descriptor storage target: {storage_generator_target}"
            )
        compatibility = {
            self._loom_amdgpu_descriptor_set_config_label(capability): []
            for capability in capabilities
        }
        compatibility["//conditions:default"] = ["@platforms//:incompatible"]
        return self.select(compatibility)

    def loom_amdgpu_selected_descriptor_set_defines(self):
        defines = []
        for capability in self._loom_amdgpu_descriptor_set_capabilities:
            defines = defines + self.select(
                {
                    self._loom_amdgpu_descriptor_set_config_label(capability): [
                        self._loom_amdgpu_descriptor_set_defines[capability],
                    ],
                    "//conditions:default": [],
                }
            )
        return defines

    def loom_amdgpu_selected_descriptor_set_values(self, values):
        result = []
        for capability in self._loom_amdgpu_descriptor_set_capabilities:
            selected_values = values.get(capability, [])
            if isinstance(selected_values, str):
                selected_values = [selected_values]
            result = result + self.select(
                {
                    self._loom_amdgpu_descriptor_set_config_label(
                        capability
                    ): selected_values,
                    "//conditions:default": [],
                }
            )
        return result

    def loom_amdgpu_selected_descriptor_set_generator_args(self, args):
        return self.loom_amdgpu_selected_descriptor_set_values(args)

    def loom_amdgpu_selected_descriptor_set_key_args(self):
        return self.loom_amdgpu_selected_descriptor_set_values(
            {
                capability: "--descriptor-set=" + descriptor_set_key
                for descriptor_set_key, capability in self.LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITY_BY_KEY.items()
            }
        )

    def loom_amdgpu_selected_descriptor_set_deps(self, targets):
        return self.loom_amdgpu_selected_descriptor_set_values(targets)

    def loom_amdgpu_low_descriptor_deps(self):
        return self.loom_amdgpu_selected_descriptor_set_deps(
            {
                capability: ":"
                + self._loom_amdgpu_descriptor_set_generator_targets[capability]
                for capability in self._loom_amdgpu_descriptor_set_capabilities
            }
        )

    def loom_amdgpu_encoding_table_deps(self):
        return self.loom_amdgpu_selected_descriptor_set_deps(
            {
                capability: ":"
                + self._loom_amdgpu_descriptor_set_generator_targets[capability]
                + "_encoding_tables"
                for capability in self._loom_amdgpu_descriptor_set_capabilities
            }
        )

    def _should_emit_python_target(self):
        return self._current_package().startswith(("loom/py/loom", "loom/src/loom"))

    def _python_package_dirs(self):
        return [
            "${PROJECT_SOURCE_DIR}/loom/py",
            "${PROJECT_BINARY_DIR}/loom/py",
            "${PROJECT_SOURCE_DIR}",
        ]

    def _emit_rewritten_cmake_rule(self, old_rule_name, new_rule_name, emit):
        body_start = len(self._converter.body)
        emit()
        emitted_body = self._converter.body[body_start:]
        self._converter.body = self._converter.body[:body_start] + emitted_body.replace(
            f"{old_rule_name}(",
            f"{new_rule_name}(",
            1,
        )

    def loom_cc_library(self, deps=[], **kwargs):
        kwargs = self._apply_loom_cmake_policy(kwargs)
        self._emit_rewritten_cmake_rule(
            "iree_cc_library",
            "loom_cc_library",
            lambda: self.cc_library(
                deps=deps + ["//runtime/src:defines", "//loom/src:defines"],
                **kwargs,
            ),
        )

    def loom_cc_binary(self, deps=[], **kwargs):
        kwargs = self._apply_loom_cmake_policy(kwargs)
        self._emit_rewritten_cmake_rule(
            "iree_cc_binary",
            "loom_cc_binary",
            lambda: self.cc_binary(
                deps=deps + ["//runtime/src:defines", "//loom/src:defines"],
                **kwargs,
            ),
        )

    def loom_cc_test(self, deps=[], resource_group=None, **kwargs):
        if resource_group:
            kwargs["resource_group"] = resource_group
        kwargs = self._apply_loom_cmake_policy(
            kwargs,
            include_run_requirements=True,
        )
        resource_group = kwargs.pop("resource_group", None)
        self._emit_rewritten_cmake_rule(
            "iree_cc_test",
            "loom_cc_test",
            lambda: self.cc_test(
                deps=deps + ["//runtime/src:defines", "//loom/src:defines"],
                resource_group=resource_group,
                **kwargs,
            ),
        )

    def loom_cc_benchmark(self, deps=[], **kwargs):
        kwargs = self._apply_loom_cmake_policy(
            kwargs,
            include_run_requirements=True,
        )
        self._emit_rewritten_cmake_rule(
            "iree_cc_binary_benchmark",
            "loom_cc_benchmark",
            lambda: self.cc_binary_benchmark(
                deps=deps
                + [
                    "//runtime/src:defines",
                    "//loom/src:defines",
                ],
                **kwargs,
            ),
        )

    def loom_cc_fuzz(self, deps=[], **kwargs):
        kwargs = self._apply_loom_cmake_policy(kwargs)
        self._emit_rewritten_cmake_rule(
            "iree_cc_fuzz",
            "loom_cc_fuzz",
            lambda: self.iree_cc_fuzz(
                deps=deps + ["//runtime/src:defines", "//loom/src:defines"],
                **kwargs,
            ),
        )

    def _cmake_identifier(self, name):
        return re.sub(r"[^A-Za-z0-9_]", "_", name)

    def _parse_external_file_label(self, label):
        match = re.fullmatch(r"@([^/]+)//(?:(.*):)?([^:]+)", label)
        if not match:
            raise ValueError(f"Unsupported external file label: {label}")
        repo_name = match.group(1)
        package = match.group(2) or ""
        filename = match.group(3)
        if package:
            return repo_name, f"{package}/{filename}"
        return repo_name, filename

    def _convert_generated_file_label(self, label):
        generated_external_file = self._loom_generated_external_files.get(label)
        if generated_external_file is not None:
            return generated_external_file
        if label.startswith("@"):
            repo_name, filename = self._parse_external_file_label(label)
            try:
                source_var = self._loom_low_descriptor_archive_source_vars[repo_name]
            except KeyError as exc:
                raise ValueError(
                    f"Generated input {label} has no CMake source mapping for "
                    f"external repo '{repo_name}'"
                ) from exc
            return f"${{{source_var}}}/{filename}"
        return self._normalize_label(label)

    def _cmake_location_paths(self, label):
        generated_external_file = self._loom_generated_external_files.get(label)
        if generated_external_file is not None:
            return [generated_external_file]
        outputs = self._loom_generated_file_families.get(
            self._canonical_location_label(label)
        )
        if outputs is not None:
            return outputs
        return super()._cmake_location_paths(label)

    def loom_spirv_registry_sources(
        self,
        name,
        tags=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        _ = name
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )
        self._loom_generated_external_files[
            "@spirv_headers//:spirv_core_grammar_unified1"
        ] = "${_LOOM_SPIRV_CORE_GRAMMAR}"
        self._loom_generated_external_files["@vulkan_headers//:vulkan_xml_registry"] = (
            "${_LOOM_VULKAN_XML_REGISTRY}"
        )
        self._converter.header += self._guard_cmake_text(
            "\n"
            "iree_get_spirv_registry_sources(\n"
            "  _LOOM_SPIRV_CORE_GRAMMAR\n"
            "  _LOOM_VULKAN_XML_REGISTRY\n"
            ")\n\n",
            target_compatible_with,
        )

    def _convert_generated_input_label(self, label):
        inputs = [self._convert_generated_file_label(label)]
        if label.startswith("@") and label not in self._loom_generated_external_files:
            repo_name, _ = self._parse_external_file_label(label)
            inputs.append(self._loom_low_descriptor_archive_targets[repo_name])
        return inputs

    def _convert_generated_arg(self, arg):
        def replace_rootpath(match):
            return self._convert_generated_file_label(match.group(1))

        def replace_external_location(match):
            label = match.group(1)
            if label.startswith("@"):
                return self._convert_generated_file_label(label)
            return match.group(0)

        arg = _GENERATED_ROOTPATH_PATTERN.sub(replace_rootpath, arg)
        return _GENERATED_LOCATION_PATTERN.sub(replace_external_location, arg)

    def _convert_generated_args(self, args):
        def convert(args):
            converted_args = []
            for arg in args or []:
                arg = self._convert_generated_arg(arg)
                converted_args.extend(self._convert_location_arg(arg))
            return converted_args

        return self._convert_generated_values(args, convert)

    def _convert_generated_inputs(self, inputs):
        def convert(inputs):
            converted_inputs = []
            for label in inputs or []:
                for converted_input in self._convert_generated_input_label(label):
                    if converted_input not in converted_inputs:
                        converted_inputs.append(converted_input)
            return converted_inputs

        return self._convert_generated_values(inputs, convert)

    def _convert_generated_values(self, values, convert):
        if values is None:
            return None
        if isinstance(values, bazel_to_cmake_converter.ConditionSelect):
            return bazel_to_cmake_converter.ConditionSelect(
                {
                    label: convert(selected_values)
                    for label, selected_values in values.conditions.items()
                }
            )
        if isinstance(values, bazel_to_cmake_converter.MixedDeps):
            return bazel_to_cmake_converter.MixedDeps(
                unconditional=convert(values.unconditional),
                selects=[
                    bazel_to_cmake_converter.ConditionSelect(
                        {
                            label: convert(selected_values)
                            for label, selected_values in select.conditions.items()
                        }
                    )
                    for select in values.selects
                ],
            )
        return convert(values)

    def loom_generated_file(
        self,
        name,
        generator,
        output,
        output_flag,
        args=None,
        inputs=None,
        comment=None,
        tags=None,
        testonly=None,
        target_compatible_with=None,
        visibility=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )

        self._target_file_paths[self._current_target_label(name)] = (
            f"${{CMAKE_CURRENT_BINARY_DIR}}/{output}"
        )
        self._target_file_paths[self._current_target_label(output)] = (
            f"${{CMAKE_CURRENT_BINARY_DIR}}/{output}"
        )
        testonly_block = self._convert_option_block("TESTONLY", testonly)
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        generator_block = self._convert_single_target_block("GENERATOR", generator)
        output_block = self._convert_string_arg_block("OUTPUT", output)
        output_flag_block = self._convert_string_arg_block("OUTPUT_FLAG", output_flag)
        args_block, platform_args_block = self._convert_platform_select_strings(
            name,
            "ARGS",
            self._convert_generated_args(args),
            sort=False,
        )
        inputs_block, platform_inputs_block = self._convert_platform_select_strings(
            name,
            "INPUTS",
            self._convert_generated_inputs(inputs),
            sort=False,
        )
        comment_block = self._convert_string_arg_block("COMMENT", comment)

        self._emit_platform_guard_begin(target_compatible_with)
        if platform_args_block:
            self._converter.body += platform_args_block
        if platform_inputs_block:
            self._converter.body += platform_inputs_block
        self._converter.body += (
            f"loom_generated_file(\n"
            f"{name_block}"
            f"{generator_block}"
            f"{output_block}"
            f"{output_flag_block}"
            f"{args_block}"
            f"{inputs_block}"
            f"{comment_block}"
            f"{testonly_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def loom_generated_file_family(
        self,
        name,
        generator,
        outputs,
        output_flags,
        output_directories=None,
        args=None,
        inputs=None,
        comment=None,
        tags=None,
        testonly=None,
        target_compatible_with=None,
        visibility=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )

        for directory in output_directories or []:
            if directory not in outputs:
                raise ValueError(
                    f"generated output directory is not declared: {directory}"
                )
        for output in outputs:
            self._target_file_paths[self._current_target_label(output)] = (
                f"${{CMAKE_CURRENT_BINARY_DIR}}/{output}"
            )
        self._loom_generated_file_families[self._current_target_label(name)] = [
            f"${{CMAKE_CURRENT_BINARY_DIR}}/{output}" for output in outputs
        ]
        testonly_block = self._convert_option_block("TESTONLY", testonly)
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        generator_block = self._convert_single_target_block("GENERATOR", generator)
        outputs_block = self._convert_string_list_block("OUTPUTS", outputs, sort=False)
        output_flags_block = self._convert_string_list_block(
            "OUTPUT_FLAGS", output_flags, sort=False
        )
        args_block, platform_args_block = self._convert_platform_select_strings(
            name,
            "ARGS",
            self._convert_generated_args(args),
            sort=False,
        )
        inputs_block, platform_inputs_block = self._convert_platform_select_strings(
            name,
            "INPUTS",
            self._convert_generated_inputs(inputs),
            sort=False,
        )
        comment_block = self._convert_string_arg_block("COMMENT", comment)

        self._emit_platform_guard_begin(target_compatible_with)
        if platform_args_block:
            self._converter.body += platform_args_block
        if platform_inputs_block:
            self._converter.body += platform_inputs_block
        self._converter.body += (
            f"loom_generated_file_family(\n"
            f"{name_block}"
            f"{generator_block}"
            f"{outputs_block}"
            f"{output_flags_block}"
            f"{args_block}"
            f"{inputs_block}"
            f"{comment_block}"
            f"{testonly_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def loom_generated_cc_library(
        self,
        name,
        generator,
        source=None,
        srcs=None,
        textual_hdrs=None,
        generated_src_flags=None,
        generated_srcs=None,
        hdrs=None,
        generated_hdr_flags=None,
        generated_hdrs=None,
        args=None,
        inputs=None,
        extra_output_flags=None,
        extra_outputs=None,
        deps=None,
        tags=None,
        testonly=None,
        target_compatible_with=None,
        visibility=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )

        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        generator_block = self._convert_single_target_block("GENERATOR", generator)
        source_block = self._convert_string_arg_block("SOURCE", source)
        srcs_block = self._convert_srcs_block(srcs, block_name="SRCS")
        textual_hdrs_block = self._convert_srcs_block(
            textual_hdrs, block_name="TEXTUAL_HDRS"
        )
        generated_src_flags_block = self._convert_string_list_block(
            "GENERATED_SRC_FLAGS", generated_src_flags or None, sort=False
        )
        generated_srcs_block = self._convert_string_list_block(
            "GENERATED_SRCS", generated_srcs or None, sort=False
        )
        hdrs_block = self._convert_srcs_block(hdrs, block_name="HDRS")
        generated_hdr_flags_block = self._convert_string_list_block(
            "GENERATED_HDR_FLAGS", generated_hdr_flags or None, sort=False
        )
        generated_hdrs_block = self._convert_string_list_block(
            "GENERATED_HDRS", generated_hdrs or None, sort=False
        )
        args_block, platform_args_block = self._convert_platform_select_strings(
            name,
            "ARGS",
            self._convert_generated_args(args),
            sort=False,
        )
        inputs_block, platform_inputs_block = self._convert_platform_select_strings(
            name,
            "INPUTS",
            self._convert_generated_inputs(inputs),
            sort=False,
        )
        extra_output_flags_block = self._convert_string_list_block(
            "EXTRA_OUTPUT_FLAGS", extra_output_flags or None, sort=False
        )
        extra_outputs_block = self._convert_string_list_block(
            "EXTRA_OUTPUTS", extra_outputs or None, sort=False
        )
        deps_block = self._convert_target_list_block("DEPS", deps)
        testonly_block = self._convert_option_block("TESTONLY", testonly)

        self._emit_platform_guard_begin(target_compatible_with)
        if platform_args_block:
            self._converter.body += platform_args_block
        if platform_inputs_block:
            self._converter.body += platform_inputs_block
        self._converter.body += (
            f"loom_generated_cc_library(\n"
            f"{name_block}"
            f"{generator_block}"
            f"{source_block}"
            f"{srcs_block}"
            f"{textual_hdrs_block}"
            f"{generated_src_flags_block}"
            f"{generated_srcs_block}"
            f"{hdrs_block}"
            f"{generated_hdr_flags_block}"
            f"{generated_hdrs_block}"
            f"{args_block}"
            f"{inputs_block}"
            f"{extra_output_flags_block}"
            f"{extra_outputs_block}"
            f"{deps_block}"
            f"{testonly_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def loom_low_descriptor_data_archive(
        self,
        name,
        repo_name,
        urls,
        sha256,
        tags=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )
        source_var = "_LOOM_%s_SOURCE_DIR" % self._cmake_identifier(name).upper()
        archive_target = "loom_%s" % self._cmake_identifier(name)
        self._loom_low_descriptor_archive_source_vars[repo_name] = source_var
        self._loom_low_descriptor_archive_targets[repo_name] = archive_target

        source_dir = "${PROJECT_BINARY_DIR}/_deps/%s-src" % repo_name
        source_dir_block = self._convert_string_arg_block(
            "SOURCE_DIR", "${%s}" % source_var
        )
        name_block = self._convert_string_arg_block("NAME", archive_target, quote=False)
        urls_block = self._convert_string_list_block("URLS", urls, sort=False)
        sha256_block = self._convert_string_arg_block("SHA256", sha256)
        self._converter.header += self._guard_cmake_text(
            "\n"
            f"set({source_var}\n"
            f'  "{source_dir}"\n'
            f")\n"
            f"loom_low_descriptor_data_archive(\n"
            f"{name_block}"
            f"{source_dir_block}"
            f"{urls_block}"
            f"{sha256_block}"
            f")\n\n",
            target_compatible_with,
        )

    def _emit_loom_target_table_cc_library(
        self,
        name,
        generator=None,
        source=None,
        header=None,
        generated_hdr_flags=None,
        generated_hdrs=None,
        header_only=False,
        cmake_rule_name="loom_target_table_cc_library",
        args=None,
        inputs=None,
        deps=None,
        ids_deps=None,
        tags=None,
        testonly=None,
        visibility=None,
        **kwargs,
    ):
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        generator_block = (
            self._convert_single_target_block("GENERATOR", generator)
            if generator is not None
            else ""
        )
        source_block = self._convert_string_arg_block("SOURCE", source)
        header_block = self._convert_string_arg_block("HEADER", header)
        generated_hdr_flags_block = self._convert_string_list_block(
            "GENERATED_HDR_FLAGS", generated_hdr_flags, sort=False
        )
        generated_hdrs_block = self._convert_string_list_block(
            "GENERATED_HDRS", generated_hdrs, sort=False
        )
        args_block, platform_args_block = self._convert_platform_select_strings(
            name,
            "ARGS",
            self._convert_generated_args(args),
            sort=False,
        )
        inputs_block, platform_inputs_block = self._convert_platform_select_strings(
            name,
            "INPUTS",
            self._convert_generated_inputs(inputs),
            sort=False,
        )
        deps_block = self._convert_target_list_block("DEPS", deps)
        ids_deps_block = self._convert_target_list_block("IDS_DEPS", ids_deps)
        header_only_block = self._convert_option_block("HEADER_ONLY", header_only)
        testonly_block = self._convert_option_block("TESTONLY", testonly)

        if platform_args_block:
            self._converter.body += platform_args_block
        if platform_inputs_block:
            self._converter.body += platform_inputs_block
        self._converter.body += (
            f"{cmake_rule_name}(\n"
            f"{name_block}"
            f"{generator_block}"
            f"{source_block}"
            f"{header_block}"
            f"{generated_hdr_flags_block}"
            f"{generated_hdrs_block}"
            f"{args_block}"
            f"{inputs_block}"
            f"{deps_block}"
            f"{ids_deps_block}"
            f"{header_only_block}"
            f"{testonly_block}"
            f")\n\n"
        )

    def loom_target_table_cc_library(
        self,
        name,
        generator=None,
        source=None,
        header=None,
        generated_hdr_flags=None,
        generated_hdrs=None,
        header_only=False,
        args=None,
        inputs=None,
        deps=None,
        ids_deps=None,
        tags=None,
        testonly=None,
        target_compatible_with=None,
        visibility=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )
        self._emit_platform_guard_begin(target_compatible_with)
        self._emit_loom_target_table_cc_library(
            name=name,
            generator=generator,
            source=source,
            header=header,
            generated_hdr_flags=generated_hdr_flags,
            generated_hdrs=generated_hdrs,
            header_only=header_only,
            args=args,
            inputs=inputs,
            deps=deps,
            ids_deps=ids_deps,
            testonly=testonly,
            cmake_rule_name="loom_target_table_cc_library",
        )
        self._emit_platform_guard_end(target_compatible_with)

    def loom_target_contract_cc_libraries(
        self,
        name,
        contract_deps=None,
        lower_rule_deps=None,
        tags=None,
        testonly=None,
        target_compatible_with=None,
        visibility=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )

        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        contract_deps_block = self._convert_target_list_block(
            "CONTRACT_DEPS", contract_deps
        )
        lower_rule_deps_block = self._convert_target_list_block(
            "LOWER_RULE_DEPS", lower_rule_deps
        )
        testonly_block = self._convert_option_block("TESTONLY", testonly)

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"loom_target_contract_cc_libraries(\n"
            f"{name_block}"
            f"{contract_deps_block}"
            f"{lower_rule_deps_block}"
            f"{testonly_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def loom_target_contract_table_cc_libraries(
        self,
        name,
        generator,
        args=None,
        index_output=None,
        inputs=None,
        contract_deps=None,
        lower_rule_deps=None,
        tags=None,
        testonly=None,
        target_compatible_with=None,
        visibility=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )

        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        generator_block = self._convert_single_target_block("GENERATOR", generator)
        index_block = self._convert_string_arg_block("INDEX_OUTPUT", index_output)
        args_block, platform_args_block = self._convert_platform_select_strings(
            name,
            "ARGS",
            self._convert_generated_args(args),
            sort=False,
        )
        inputs_block, platform_inputs_block = self._convert_platform_select_strings(
            name,
            "INPUTS",
            self._convert_generated_inputs(inputs),
            sort=False,
        )
        contract_deps_block = self._convert_target_list_block(
            "CONTRACT_DEPS", contract_deps
        )
        lower_rule_deps_block = self._convert_target_list_block(
            "LOWER_RULE_DEPS", lower_rule_deps
        )
        testonly_block = self._convert_option_block("TESTONLY", testonly)

        self._emit_platform_guard_begin(target_compatible_with)
        if platform_args_block:
            self._converter.body += platform_args_block
        if platform_inputs_block:
            self._converter.body += platform_inputs_block
        self._converter.body += (
            f"loom_target_contract_table_cc_libraries(\n"
            f"{name_block}"
            f"{generator_block}"
            f"{index_block}"
            f"{args_block}"
            f"{inputs_block}"
            f"{contract_deps_block}"
            f"{lower_rule_deps_block}"
            f"{testonly_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def loom_target_contract_file_family(
        self,
        name,
        generator,
        fragments,
        args=None,
        index_output=None,
        inputs=None,
        comment=None,
        tags=None,
        target_compatible_with=None,
        visibility=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )

        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        generator_block = self._convert_single_target_block("GENERATOR", generator)
        index_block = self._convert_string_arg_block("INDEX_OUTPUT", index_output)
        fragments_block = self._convert_string_list_block(
            "FRAGMENTS",
            [
                f"{stem}={fragment_key}"
                for stem, fragment_key in sorted(fragments.items())
            ],
            sort=False,
        )
        args_block, platform_args_block = self._convert_platform_select_strings(
            name,
            "ARGS",
            self._convert_generated_args(args),
            sort=False,
        )
        inputs_block, platform_inputs_block = self._convert_platform_select_strings(
            name,
            "INPUTS",
            self._convert_generated_inputs(inputs),
            sort=False,
        )
        comment_block = self._convert_string_arg_block("COMMENT", comment)

        self._emit_platform_guard_begin(target_compatible_with)
        if platform_args_block:
            self._converter.body += platform_args_block
        if platform_inputs_block:
            self._converter.body += platform_inputs_block
        self._converter.body += (
            f"loom_target_contract_file_family(\n"
            f"{name_block}"
            f"{generator_block}"
            f"{index_block}"
            f"{fragments_block}"
            f"{args_block}"
            f"{inputs_block}"
            f"{comment_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def loom_low_descriptor_cc_library(
        self,
        name,
        generator=None,
        source=None,
        header=None,
        generated_hdr_flags=None,
        generated_hdrs=None,
        header_only=False,
        args=None,
        inputs=None,
        deps=None,
        ids_deps=None,
        tags=None,
        testonly=None,
        target_compatible_with=None,
        visibility=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )
        self._emit_platform_guard_begin(target_compatible_with)
        self._emit_loom_target_table_cc_library(
            name=name,
            generator=generator,
            source=source,
            header=header,
            generated_hdr_flags=generated_hdr_flags,
            generated_hdrs=generated_hdrs,
            header_only=header_only,
            args=args,
            inputs=inputs,
            deps=deps,
            ids_deps=ids_deps,
            testonly=testonly,
            cmake_rule_name="loom_low_descriptor_cc_library",
        )
        self._emit_platform_guard_end(target_compatible_with)

    def _loom_check_test_base_name(self, src):
        if src.startswith("${_GLOB_") and src.endswith("}"):
            return src
        base, separator, extension = src.rpartition(".")
        if not separator or not extension.endswith("-test") or extension == "-test":
            raise ValueError(
                f"loom_check_test source must use a .<format>-test extension: {src}"
            )
        return base

    def loom_check_runner_binary(
        self,
        name,
        src,
        deps=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(**kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )
        runner_deps = (deps or []) + ["//loom/src/loom/tools/loom-check:main"]
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_srcs_block([src])
        deps_block, platform_deps_block = self._convert_platform_select_deps(
            name, runner_deps
        )
        if platform_deps_block:
            self._converter.body += platform_deps_block
        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"iree_cc_binary(\n{name_block}{srcs_block}{deps_block})\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def loom_check_test(
        self,
        name,
        src,
        data=None,
        env=None,
        tags=None,
        runner="//loom/src/loom/tools/loom-check:loom-check-test",
        target_compatible_with=None,
        compile_targets=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )
        self._loom_check_test_base_name(src)
        combined_tags = (tags or []) + ["loom-check"]
        name_block = self._convert_string_arg_block("NAME", name)
        test_binary_block = self._convert_single_target_block("SRC", runner)
        args_block = self._convert_string_list_block(
            "ARGS",
            ["{{${CMAKE_CURRENT_SOURCE_DIR}/%s}}" % src],
            sort=False,
        )
        data_block = self._convert_data_list_block(data)
        env_block = self._convert_string_list_block(
            "ENV", self._convert_test_env(env), sort=False
        )
        labels_block = self._convert_string_list_block("LABELS", combined_tags)
        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"iree_native_test(\n"
            f"{name_block}"
            f'  WORKING_DIRECTORY "${{IREE_ROOT_DIR}}"\n'
            f"{args_block}"
            f"{test_binary_block}"
            f"{data_block}"
            f"{env_block}"
            f"{labels_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)
        self._loom_check_compile_tests(
            name, src, compile_targets, data, env, tags, target_compatible_with
        )

    def loom_check_compile_tests(
        self,
        name,
        src,
        targets,
        data=None,
        env=None,
        tags=None,
        args=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        self._loom_check_compile_tests(
            name,
            src,
            targets,
            data,
            env,
            tags,
            self._apply_loom_target_compatible_with(target_compatible_with),
            args=args,
        )

    def _loom_check_compile_tests(
        self,
        name,
        src,
        targets,
        data,
        env,
        tags,
        target_compatible_with,
        registered_srcs=None,
        args=None,
    ):
        if not targets:
            return
        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            "loom_check_compile_tests(\n"
            + self._convert_string_arg_block("NAME", name)
            + self._convert_loom_module_inputs("SRC", [src])
            + self._convert_string_list_block(
                "REGISTERED_SRCS", registered_srcs, sort=False
            )
            + self._convert_target_list_block("TARGETS", targets)
            + self._convert_string_list_block(
                "ARGS", self._convert_test_location_args(args), sort=False
            )
            + self._convert_data_list_block(data)
            + self._convert_string_list_block(
                "ENV", self._convert_test_env(env), sort=False
            )
            + self._convert_string_list_block("LABELS", tags)
            + ")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def loom_check_test_suite(
        self,
        name,
        srcs,
        size="small",
        data=None,
        env=None,
        tags=None,
        runner="//loom/src/loom/tools/loom-check:loom-check-test",
        test_name_prefix_to_strip="",
        resource_group=None,
        timeout=None,
        target_compatible_with=None,
        compile_targets=None,
        **kwargs,
    ):
        del size
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )
        for src in srcs:
            self._loom_check_test_base_name(src)
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_string_list_block(
            "SRCS", [self._normalize_label(src) for src in srcs], sort=False
        )
        default_runner = "//loom/src/loom/tools/loom-check:loom-check-test"
        runner_block = (
            ""
            if runner == default_runner
            else self._convert_single_target_block("RUNNER", runner)
        )
        data_block = self._convert_data_list_block(data)
        env_block = self._convert_string_list_block(
            "ENV", self._convert_test_env(env), sort=False
        )
        labels_block = self._convert_string_list_block("LABELS", tags)
        test_name_prefix_to_strip_block = self._convert_string_arg_block(
            "TEST_NAME_PREFIX_TO_STRIP", test_name_prefix_to_strip or None
        )
        resource_group_block = self._convert_string_arg_block(
            "RESOURCE_GROUP", resource_group, quote=False
        )
        timeout_block = self._convert_timeout_arg_block("TIMEOUT", timeout)
        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"loom_check_test_suite(\n"
            f"{name_block}"
            f"{srcs_block}"
            f"{runner_block}"
            f"{data_block}"
            f"{env_block}"
            f"{labels_block}"
            f"{test_name_prefix_to_strip_block}"
            f"{resource_group_block}"
            f"{timeout_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)
        for src, targets in (compile_targets or {}).items():
            if src not in srcs and not any(
                item.startswith("${_GLOB_") for item in srcs
            ):
                raise ValueError(
                    f"compiler qualification source is not registered in {name}: {src}"
                )
            test_name = self._loom_check_test_base_name(src)
            if test_name_prefix_to_strip and test_name.startswith(
                test_name_prefix_to_strip
            ):
                test_name = test_name[len(test_name_prefix_to_strip) :]
            test_name = test_name.replace("/", "_")
            self._loom_check_compile_tests(
                test_name, src, targets, data, env, tags, target_compatible_with, srcs
            )


def convert_unmatched_target(converter, target):
    cmake_path = converter._convert_to_cmake_path(target)
    if cmake_path == "loom":
        return ["loom"]
    if cmake_path == "loom::src::loom":
        return ["loom"]
    if cmake_path.startswith("loom::src::loom::"):
        cmake_path = cmake_path[len("loom::src::loom::") :]
        return ["loom::" + cmake_path]
    if cmake_path == "loom::src":
        return ["loom"]
    if cmake_path.startswith("loom::"):
        cmake_path = cmake_path[len("loom::") :]
    return ["loom::" + cmake_path]


PROJECT_CONFIG = bazel_to_cmake_config.ProjectConfig(
    name="loom",
    package_prefixes=["loom"],
    build_file_functions=LoomBuildFileFunctions,
    target_mappings={
        "//loom/src:defines": [],
    },
    convert_unmatched_target=convert_unmatched_target,
)
