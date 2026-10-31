# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import dataclasses
import importlib.util
import os
import re
import sys
from types import SimpleNamespace

import bazel_to_cmake_config
import bazel_to_cmake_converter
import bazel_to_cmake_requirements
from conditions import Option, all_of, any_of
from loom_binary import LoomBinaryBuildFileFunctions


def _load_loom_corpus_build_file_functions():
    module_path = os.path.join(
        os.path.dirname(__file__),
        "build_tools",
        "bazel_to_cmake",
        "loom_corpus.py",
    )
    spec = importlib.util.spec_from_file_location(
        "loom_bazel_to_cmake_corpus",
        module_path,
    )
    if not spec or not spec.loader:
        raise RuntimeError(f"could not load Loom corpus converter from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.LoomCorpusBuildFileFunctions


LoomCorpusBuildFileFunctions = _load_loom_corpus_build_file_functions()


@dataclasses.dataclass(frozen=True)
class _LoomExecutionProfile:
    kind: str
    name: str
    target_family: str
    target_class: str
    executor: str
    runner: str | None
    runner_args: list[str] | None
    build_requirements: list
    run_requirements: list
    resource_group: str | None
    tags: list[str]


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
        options[f"//loom/config/arch/amdgpu:processor_{processor}"] = (
            f"LOOM_ARCH_AMDGPU_PROCESSOR_{processor_upper}"
        )
        options[f"//loom/config/arch/amdgpu:iree_hal_processor_{processor}"] = (
            f"LOOM_ARCH_AMDGPU_IREE_HAL_PROCESSOR_{processor_upper}"
        )
    for capability in config["LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITIES"]:
        capability_upper = capability.upper()
        options[f"//loom/config/arch/amdgpu:{capability}"] = (
            f"LOOM_ARCH_AMDGPU_{capability_upper}"
        )
        options[f"//loom/config/arch/amdgpu:iree_hal_{capability}"] = (
            f"LOOM_ARCH_AMDGPU_IREE_HAL_{capability_upper}"
        )
    return options


_LOOM_NATIVE_ELF_CMAKE_CONDITION = any_of(
    "LOOM_ARTIFACT_AIE2P_ELF", "LOOM_ARTIFACT_AMDGPU_ELF", "LOOM_ARTIFACT_X86_ELF"
)
_LOOM_NATIVE_OBJECT_ELF_CMAKE_CONDITION = any_of(
    "LOOM_ARTIFACT_AIE2P_ELF", "LOOM_ARTIFACT_X86_ELF"
)

_LOOM_CONFIG_CMAKE_OPTIONS = {
    "//loom/config:shared_tools_enabled": "LOOM_LINK_SHARED_TOOLS",
    "//loom/config/arch:aie2p": "LOOM_ARCH_AIE2P",
    "//loom/config/arch:amdgpu": "LOOM_ARCH_AMDGPU",
    "//loom/config/arch:spirv": "LOOM_ARCH_SPIRV",
    "//loom/config/arch:vm": "LOOM_ARCH_VM",
    "//loom/config/arch:wasm": "LOOM_ARCH_WASM",
    "//loom/config/arch:x86": "LOOM_ARCH_X86",
    "//loom/config/emit:elf": "LOOM_EMIT_ELF",
    "//loom/config/emit:spirv": "LOOM_EMIT_SPIRV",
    "//loom/config/emit:vm": "LOOM_EMIT_VM",
    "//loom/config/emit:wasm": "LOOM_EMIT_WASM",
    "//loom/config/emit:aie2p_elf": "LOOM_ARTIFACT_AIE2P_ELF",
    "//loom/config/emit:amdgpu_elf": "LOOM_ARTIFACT_AMDGPU_ELF",
    "//loom/config/emit:spirv_binary": "LOOM_ARTIFACT_SPIRV_BINARY",
    "//loom/config/emit:vm_binary": "LOOM_ARTIFACT_VM_BINARY",
    "//loom/config/emit:wasm_binary": "LOOM_ARTIFACT_WASM_BINARY",
    "//loom/config/emit:x86_elf": "LOOM_ARTIFACT_X86_ELF",
    "//loom/config/emit:native": _LOOM_NATIVE_ELF_CMAKE_CONDITION,
    "//loom/config/emit:native_elf": _LOOM_NATIVE_ELF_CMAKE_CONDITION,
    "//loom/config/emit:native_object": _LOOM_NATIVE_OBJECT_ELF_CMAKE_CONDITION,
    "//loom/config/emit:native_object_elf": _LOOM_NATIVE_OBJECT_ELF_CMAKE_CONDITION,
    "//loom/config/execute:iree_hal": "LOOM_EXECUTE_IREE_HAL",
    "//loom/config/execute:iree_vm": "LOOM_EXECUTE_IREE_VM",
    "//loom/config/execute:wasm_host": "LOOM_EXECUTE_WASM_HOST",
    "//loom/config/execute:aie2p_xdna_hal": "LOOM_ROUTE_AIE2P_XDNA_HAL",
    "//loom/config/execute:amdgpu_hal": "LOOM_ROUTE_AMDGPU_HAL",
    "//loom/config/execute:spirv_vulkan_hal": "LOOM_ROUTE_SPIRV_VULKAN_HAL",
    "//loom/config/execute:vm_iree_vm": "LOOM_ROUTE_VM_IREE_VM",
    "//loom/config/execute:wasm_hosted": "LOOM_ROUTE_WASM_HOSTED",
    "//loom/config/execute:x86_task_hal": "LOOM_ROUTE_X86_TASK_HAL",
    "//loom/config/import/cxx:embed_includes_enabled": "LOOM_IMPORT_CXX_EMBED_INCLUDES",
    "//loom/config/import:cxx": "LOOM_IMPORT_CXX",
    "//loom/config/import:mlir": "LOOM_IMPORT_MLIR",
    "//loom/config/import:tilelang": "LOOM_IMPORT_TILELANG",
    "//loom/config/serve:audio_decoding": "LOOM_SERVE_AUDIO_DECODING",
    "//loom/config/serve:flac": "LOOM_SERVE_AUDIO_CODEC_FLAC",
    "//loom/config/serve:gif": "LOOM_SERVE_IMAGE_CODEC_GIF",
    "//loom/config/serve:image_input_enabled": all_of(
        "LOOM_BUILD", "LOOM_SERVE_IMAGE_INPUT"
    ),
    "//loom/config/serve:jpeg": "LOOM_SERVE_IMAGE_CODEC_JPEG",
    "//loom/config/serve:mp3": "LOOM_SERVE_AUDIO_CODEC_MP3",
    "//loom/config/serve:png": "LOOM_SERVE_IMAGE_CODEC_PNG",
    "//loom/config/serve:wav": "LOOM_SERVE_AUDIO_CODEC_WAV",
    "//loom/config/serve:webp": "LOOM_SERVE_IMAGE_CODEC_WEBP",
}
_LOOM_CONFIG_CMAKE_OPTIONS.update(_loom_amdgpu_config_cmake_options())

_GENERATED_ROOTPATH_PATTERN = re.compile(r"\$\(rootpath ([^)]+)\)")
_GENERATED_LOCATION_PATTERN = re.compile(r"\$\(location ([^)]+)\)")


class LoomBuildFileFunctions(
    LoomCorpusBuildFileFunctions,
    LoomBinaryBuildFileFunctions,
    bazel_to_cmake_converter.BuildFileFunctions,
):
    def _declarative_load_bindings(self):
        return {
            **super()._declarative_load_bindings(),
            "loom_corpus_catalog": self.loom_corpus_catalog,
            "loom_corpus_manifest": self.loom_corpus_manifest,
            "loom_execution_profile": self.loom_execution_profile,
            "native": SimpleNamespace(package_name=self._current_package),
            "select": self.select,
            "iree_select": self.iree_select,
            "iree_cmake_extra_content": self.iree_cmake_extra_content,
            "apply_loom_target_policy": self._apply_loom_cmake_policy,
            **{
                name: getattr(self, name)
                for name in (
                    "loom_cc_library",
                    "loom_cc_library_group",
                    "loom_cc_instance_library",
                    "loom_cc_binary",
                    "loom_cc_test",
                    "loom_cc_benchmark",
                    "loom_cc_fuzz",
                )
            },
        }

    def _custom_initialize(self):
        self._loom_module_targets = set()
        self._loom_generated_external_repositories = {}
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
        self._loom_amdgpu_target_capabilities_by_representation_capability = (
            _LOOM_AMDGPU_TARGET_CONFIG[
                "LOOM_AMDGPU_TARGET_CAPABILITIES_BY_REPRESENTATION_CAPABILITY"
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

    def _apply_loom_target_compatible_with(self, target_compatible_with):
        policy = self._loom_package_policy()
        return bazel_to_cmake_requirements.append_conditions(
            target_compatible_with,
            policy.conditions(),
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
                "//loom/config/arch:amdgpu",
                "//loom/config/arch/amdgpu:" + capability,
            ],
            **kwargs,
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
        self._register_target_files(name, [f"${{CMAKE_CURRENT_BINARY_DIR}}/{output}"])
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

    def loom_model(self, tags=None, **kwargs):
        # Model publication actions use Bazel's declared runfiles layout. Native
        # serving coverage remains available through ordinary CMake fixtures.
        if not self._should_skip_target(tags=tags):
            raise NotImplementedError("loom_model requires the skip-bazel_to_cmake tag")

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
        runner=None,
        build_requirements=None,
        run_requirements=None,
        resource_group=None,
        tags=None,
    ):
        self._reject_workload_args(name, runner_args)
        return _LoomExecutionProfile(
            kind="loom_execution_profile",
            name=name,
            target_family=target_family,
            target_class=target_class,
            executor=executor,
            runner=runner,
            runner_args=runner_args,
            build_requirements=build_requirements or [],
            run_requirements=run_requirements or [],
            resource_group=resource_group,
            tags=tags or [],
        )

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
        benchmark_smoke=True,
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
                if profile.kind != "loom_execution_profile":
                    raise ValueError(
                        f"{name} execution profile was not created by loom_execution_profile"
                    )
                suffix = self._loom_test_name_suffix(profile.name)
                execution_name = f"{workload_name}_execute_{suffix}_test"
                if execution_name in execution_names:
                    raise ValueError(
                        f"{name} has colliding execution profiles: {profile.name}"
                    )
                execution_names.add(execution_name)
                self._loom_execution_test(
                    execution_name,
                    self._convert_single_target(module),
                    profile,
                    args,
                    benchmark_smoke,
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

    def _loom_execution_test(
        self, name, module, profile, args, benchmark_smoke, tags, workload_args
    ):
        if profile.runner:
            self._converter.body += (
                f"# {name} uses the Bazel execution runner {profile.runner}; "
                "no CMake execution target is available.\n\n"
            )
            return
        policy = bazel_to_cmake_requirements.CollectedPackagePolicy(
            build_requirements=profile.build_requirements,
            run_requirements=profile.run_requirements,
            resource_group=profile.resource_group,
        )
        labels = list(tags or []) + profile.tags
        labels.extend(policy.tags(include_run_requirements=True))
        labels.extend(
            [
                "loom-execution-profile=" + profile.name,
                "loom-target-family=" + profile.target_family,
                "loom-target-class=" + profile.target_class,
                "loom-executor=" + profile.executor,
            ]
        )
        blocks = [
            self._convert_string_arg_block("NAME", name, quote=False),
            "  CORRECTNESS_ONLY\n" if not benchmark_smoke else "",
            self._convert_string_arg_block("MODULE", module),
            self._convert_string_list_block(
                "ARGS", self._convert_test_location_args(args), sort=False
            ),
            self._convert_string_list_block(
                "RUNNER_ARGS",
                self._convert_test_location_args(
                    (profile.runner_args or []) + workload_args or None
                ),
                sort=False,
            ),
            self._convert_string_list_block("LABELS", labels, sort=False),
            self._convert_string_arg_block("RESOURCE_GROUP", policy.resource_group),
        ]
        self._emit_platform_guard_begin(policy.conditions())
        self._converter.body += "loom_execution_test(\n" + "".join(blocks) + ")\n\n"
        self._emit_platform_guard_end(policy.conditions())

    def loom_amdgpu_target_selectors_flag(self, **kwargs):
        return None

    def loom_amdgpu_target_config_settings(self, **kwargs):
        return None

    def _loom_amdgpu_descriptor_set_config_label(self, capability, prefix=""):
        if capability not in self._loom_amdgpu_descriptor_set_capabilities:
            raise ValueError(
                f"Unknown Loom AMDGPU descriptor-set capability: {capability}"
            )
        if prefix:
            capability = prefix + "_" + capability
        return "//loom/config/arch/amdgpu:" + capability

    def _loom_amdgpu_descriptor_set_capabilities_compatible_with(
        self, capabilities, prefix=""
    ):
        compatibility = {
            self._loom_amdgpu_descriptor_set_config_label(capability, prefix): []
            for capability in capabilities
        }
        compatibility["//conditions:default"] = ["@platforms//:incompatible"]
        return self.select(compatibility)

    def loom_amdgpu_descriptor_set_compatible_with(self, capability):
        return self.loom_config_compatible_with(
            [self._loom_amdgpu_descriptor_set_config_label(capability)]
        )

    def loom_amdgpu_iree_hal_representation_compatible_with(self, capability):
        target_capabilities = (
            self._loom_amdgpu_target_capabilities_by_representation_capability.get(
                capability
            )
        )
        if not target_capabilities:
            raise ValueError(
                f"Unknown Loom AMDGPU representation capability: {capability}"
            )
        return self._loom_amdgpu_descriptor_set_capabilities_compatible_with(
            target_capabilities, prefix="iree_hal"
        )

    def loom_amdgpu_descriptor_table_compatible_with(self, storage_generator_target):
        capabilities = self._loom_amdgpu_descriptor_set_capabilities_by_storage_generator_target.get(
            storage_generator_target
        )
        if not capabilities:
            raise ValueError(
                f"Unknown AMDGPU descriptor storage target: {storage_generator_target}"
            )
        return self._loom_amdgpu_descriptor_set_capabilities_compatible_with(
            capabilities
        )

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

    def loom_py_library(self, **kwargs):
        self.iree_py_library(**self._apply_loom_cmake_policy(kwargs))

    def loom_py_binary(self, **kwargs):
        self.iree_py_binary(**self._apply_loom_cmake_policy(kwargs))

    def loom_py_test(self, **kwargs):
        self.iree_py_test(
            **self._apply_loom_cmake_policy(kwargs, include_run_requirements=True)
        )

    def loom_cc_library(self, deps=[], **kwargs):
        kwargs = self._apply_loom_cmake_policy(kwargs)
        self.cc_library(
            deps=deps + ["//runtime/src:defines", "//loom/src:defines"],
            **kwargs,
        )

    def loom_cc_library_group(self, **kwargs):
        # The existing dependency-only CMake library is an INTERFACE target.
        self.cc_library(**self._apply_loom_cmake_policy(kwargs))

    def loom_cc_instance_library(self, tags=None, **kwargs):
        self.loom_cc_library(tags=(tags or []) + ["LINKABLE_MORE_THAN_ONCE"], **kwargs)

    def loom_cc_binary(self, deps=[], **kwargs):
        kwargs = self._apply_loom_cmake_policy(kwargs)
        self.cc_binary(
            deps=deps + ["//runtime/src:defines", "//loom/src:defines"],
            **kwargs,
        )

    def loomc_cc_binary(self, runtime_only=False, **kwargs):
        self.loom_cc_binary(
            _cmake_rule="loomc_cc_binary",
            _cmake_options=("RUNTIME_ONLY",) if runtime_only else (),
            _native_linking=Option("LOOM_LINK_SHARED_TOOLS").evaluate(
                self._configuration or {}
            ),
            **kwargs,
        )

    def loomc_interop_headers(self, name, deps):
        self._converter.body += (
            "loomc_interop_headers(\n"
            + self._convert_string_arg_block("NAME", name, quote=False)
            + self._convert_target_list_block("DEPS", deps)
            + ")\n\n"
        )

    def loomc_shared_libraries(
        self, name, components, runtime_deps, tool_components=[]
    ):
        components_block, variables_block = self._convert_platform_select_deps(
            name, components, "COMPONENTS"
        )
        self._converter.body += (
            variables_block
            + "loomc_shared_libraries(\n"
            + self._convert_string_arg_block("NAME", name, quote=False)
            + components_block
            + self._convert_target_list_block("TOOL_COMPONENTS", tool_components)
            + self._convert_target_list_block("RUNTIME_DEPS", runtime_deps)
            + ")\n\n"
        )

    def loom_cc_test(self, deps=[], resource_group=None, **kwargs):
        if resource_group:
            kwargs["resource_group"] = resource_group
        kwargs = self._apply_loom_cmake_policy(
            kwargs,
            include_run_requirements=True,
        )
        resource_group = kwargs.pop("resource_group", None)
        self.cc_test(
            deps=deps + ["//runtime/src:defines", "//loom/src:defines"],
            resource_group=resource_group,
            **kwargs,
        )

    def loom_cc_benchmark(self, deps=[], **kwargs):
        kwargs = self._apply_loom_cmake_policy(
            kwargs,
            include_run_requirements=True,
        )
        self.cc_binary_benchmark(
            deps=deps + ["//runtime/src:defines", "//loom/src:defines"],
            **kwargs,
        )

    def loom_cc_fuzz(self, deps=[], **kwargs):
        kwargs = self._apply_loom_cmake_policy(kwargs)
        self.iree_cc_fuzz(
            deps=deps + ["//runtime/src:defines", "//loom/src:defines"],
            **kwargs,
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
                source_dir = self._loom_generated_external_repositories[repo_name]
            except KeyError as exc:
                raise ValueError(
                    f"Generated input {label} has no CMake source mapping for "
                    f"external repo '{repo_name}'"
                ) from exc
            return f"{source_dir}/{filename}"
        return self._normalize_label(label)

    def _cmake_location_paths(self, label):
        generated_external_file = self._loom_generated_external_files.get(label)
        if generated_external_file is not None:
            return [generated_external_file]
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

        return self._map_configurable_list(args, convert)

    def _convert_generated_inputs(self, inputs):
        return self._map_configurable_list(
            inputs,
            lambda labels: list(
                dict.fromkeys(
                    self._convert_generated_file_label(label) for label in labels or []
                )
            ),
        )

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
        if not name or not repo_name or not urls or not sha256:
            raise ValueError(
                "descriptor archives require a name, repository, URLs, and hash"
            )
        source_dir = f"${{IREE_BINARY_DIR}}/_deps/{repo_name}-src"
        archive_name = "loom_" + self._cmake_identifier(name)
        self._loom_generated_external_repositories[repo_name] = source_dir
        urls_block = self._convert_string_list_block("URL", urls)
        self._converter.header += self._guard_cmake_text(
            "include(FetchContent)\n"
            f"FetchContent_Declare({archive_name}\n"
            f"{urls_block}"
            f'  URL_HASH "SHA256={sha256}"\n'
            f'  SOURCE_DIR "{source_dir}"\n'
            "  DOWNLOAD_NO_PROGRESS TRUE\n"
            "  DOWNLOAD_EXTRACT_TIMESTAMP FALSE\n"
            ")\n"
            f"FetchContent_MakeAvailable({archive_name})\n",
            target_compatible_with,
        )

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
        self.cc_binary(
            name=name,
            srcs=[src],
            deps=(deps or []) + ["//loom/src/loom/tools/loom-check:main"],
            target_compatible_with=target_compatible_with,
        )

    def loom_check_test(
        self,
        name,
        src,
        args=None,
        data=None,
        env=None,
        tags=None,
        runner="//loom/src/loom/tools/loom-check:loom-check-test",
        target_compatible_with=None,
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
            self._convert_test_location_args(args or [])
            + ["{{${_IREE_PACKAGE_SOURCE_DIR}/%s}}" % src],
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
        if isinstance(src, str) and src.endswith("-test"):
            raise ValueError(
                "compiler checks require an authored or linked module, "
                f"not a textual test fixture: {src}"
            )
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
        args=None,
    ):
        if not targets:
            return
        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            "loom_check_compile_tests(\n"
            + self._convert_string_arg_block("NAME", name)
            + self._convert_loom_module_inputs("SRC", [src])
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
        args=None,
        data=None,
        env=None,
        tags=None,
        runner="//loom/src/loom/tools/loom-check:loom-check-test",
        test_name_prefix_to_strip="",
        resource_group=None,
        timeout=None,
        target_compatible_with=None,
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
        args_block = self._convert_string_list_block(
            "ARGS", self._convert_test_location_args(args), sort=False
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
            f"{args_block}"
            f"{data_block}"
            f"{env_block}"
            f"{labels_block}"
            f"{test_name_prefix_to_strip_block}"
            f"{resource_group_block}"
            f"{timeout_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)


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
        "//loom/src:defines": ["loom_defs"],
    },
    condition_mappings=_LOOM_CONFIG_CMAKE_OPTIONS,
    convert_unmatched_target=convert_unmatched_target,
)
