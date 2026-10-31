# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom deployment-product projections shared by consuming projects."""

from dataclasses import dataclass, field

from bazel_to_cmake_converter import ConditionSelect, MixedDeps
from conditions import Condition, Not, all_of, any_of


@dataclass(frozen=True)
class Profile:
    # Compiler selector owned by the source profile declaration.
    compiler: str
    # Availability over bootstrap options; native target properties are opaque.
    available: Condition


@dataclass
class Profiles:
    # Source profiles and ordered sets, indexed by canonical BUILD label.
    profiles: dict[str, Profile | list[tuple[str, Condition]]] = field(
        default_factory=dict
    )

    def expand(self, label, active=()):
        if label in active:
            raise ValueError("Loom profile cycle: " + " -> ".join((*active, label)))
        if label not in self.profiles:
            raise ValueError(f"Unknown Loom profile or set: {label}")
        value = self.profiles[label]
        if isinstance(value, Profile):
            return {label: all_of()}
        result = {}
        for member, selected in value:
            for profile, inherited in self.expand(member, (*active, label)).items():
                predicate = all_of(selected, inherited)
                result[profile] = (
                    any_of(result[profile], predicate)
                    if profile in result
                    else predicate
                )
        return result


class LoomBinaryBuildFileFunctions:
    @property
    def _loom_profiles(self):
        if "loom.profiles" not in self._declarations:
            self._declarations["loom.profiles"] = Profiles()
        return self._declarations["loom.profiles"]

    def _loom_profile_condition(self, requirements):
        if isinstance(requirements, ConditionSelect):
            selected = [
                (self._select_condition(label), values)
                for label, values in requirements.conditions.items()
                if label != "//conditions:default"
            ]
            predicates = [predicate for predicate, values in selected if values == []]
            if requirements.conditions.get("//conditions:default") == []:
                predicates.append(
                    Not(any_of(*(predicate for predicate, _ in selected)))
                )
            return any_of(*predicates)
        if isinstance(requirements, MixedDeps):
            return all_of(
                self._loom_profile_condition(requirements.unconditional),
                *(
                    self._loom_profile_condition(value)
                    for value in requirements.selects
                ),
            )
        predicates = [self._select_condition(label) for label in requirements or []]
        if any(predicate is None for predicate in predicates):
            raise NotImplementedError(f"Loom profile requirements: {requirements}")
        return all_of(*predicates)

    def _loom_profile_members(self, targets):
        if isinstance(targets, MixedDeps):
            return [
                *self._loom_profile_members(targets.unconditional),
                *(
                    member
                    for value in targets.selects
                    for member in self._loom_profile_members(value)
                ),
            ]
        if isinstance(targets, ConditionSelect):
            predicates = [
                self._select_condition(label)
                for label in targets.conditions
                if label != "//conditions:default"
            ]
            result = []
            for label, members in targets.conditions.items():
                predicate = (
                    Not(any_of(*predicates))
                    if label == "//conditions:default"
                    else self._select_condition(label)
                )
                result.extend(
                    (self._canonical_location_label(member), predicate)
                    for member in members
                )
            return result
        return [
            (self._canonical_location_label(member), all_of()) for member in targets
        ]

    def _loom_target_identity_label(self, target):
        if target.startswith("@"):
            return target
        return self._canonical_location_label(target)

    def loom_target_set(
        self, name, targets, allow_empty=False, visibility=None, **kwargs
    ):
        self._check_no_unhandled_kwargs("loom_target_set", kwargs)
        label = self._current_target_label(name)
        if label in self._loom_profiles.profiles:
            raise ValueError(f"Duplicate Loom profile set: {label}")
        members = self._loom_profile_members(targets)
        if not members and not allow_empty:
            raise ValueError(f"Loom profile set {label} must not be empty")
        self._loom_profiles.profiles[label] = members
        targets_block, variable_block = self._convert_platform_select_deps(
            name,
            targets,
            "TARGETS",
            target_transform=self._loom_target_identity_label,
        )
        self._converter.body += (
            variable_block
            + "loom_target_set(\n"
            + self._convert_string_arg_block("NAME", name)
            + ("  ALLOW_EMPTY\n" if allow_empty else "")
            + targets_block
            + ")\n\n"
        )

    def _convert_select_condition(self, label):
        if isinstance(label, str) and label.startswith(
            "//loom/config/arch/amdgpu:descriptor_set_"
        ):
            return "LOOM_ARCH_AMDGPU_" + label.split(":", 1)[1].upper()
        return super()._convert_select_condition(label)

    def loom_config_compatible_with(self, config_labels):
        return list(config_labels)

    def _register_loom_module_target(self, name):
        if not hasattr(self, "_loom_module_targets"):
            self._loom_module_targets = set()
        self._loom_module_targets.add(self._current_target_label(name))

    def _convert_loom_module_inputs(self, block_name, inputs):
        if inputs is None:
            return ""
        if not hasattr(self, "_loom_module_targets"):
            self._loom_module_targets = set()
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
        visibility=None,
        **kwargs,
    ):
        del visibility
        if self._should_skip_target(tags=tags, **kwargs):
            return
        self._register_loom_module_target(name)
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
        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += "loom_module(\n" + "".join(blocks) + ")\n\n"
        self._emit_platform_guard_end(target_compatible_with)

    def loom_target_profile(
        self, name, family, selector, target_compatible_with=None, tags=None, **kwargs
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        label = self._current_target_label(name)
        if label in self._loom_profiles.profiles:
            raise ValueError(f"Duplicate Loom profile: {label}")
        self._loom_profiles.profiles[label] = Profile(
            family + ":" + selector,
            self._loom_profile_condition(target_compatible_with),
        )
        condition = self._target_compatible_condition(target_compatible_with)
        requires = f"  REQUIRES\n    {condition}\n" if condition else ""
        self._converter.body += (
            "loom_target_profile(\n"
            + self._convert_string_arg_block("NAME", name)
            + self._convert_string_arg_block("FAMILY", family)
            + self._convert_string_arg_block("SELECTOR", selector)
            + requires
            + ")\n\n"
        )

    def loom_kernel_binary(
        self,
        name,
        target,
        srcs=None,
        deps=None,
        roots=None,
        configs=None,
        out=None,
        testonly=False,
        tags=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        output = out or name
        self._register_target_files(name, [f"${{CMAKE_CURRENT_BINARY_DIR}}/{output}"])
        self._register_target_files(output, [f"${{CMAKE_CURRENT_BINARY_DIR}}/{output}"])
        blocks = [
            self._convert_string_arg_block("NAME", name),
            self._convert_string_arg_block(
                "COMPONENT", self._current_target_label(name)
            ),
            self._convert_string_arg_block(
                "TARGET", self._convert_single_target(target)
            ),
            self._convert_string_arg_block("OUTPUT", output),
            self._convert_data_srcs_block(srcs, sort=False),
            self._convert_string_list_block(
                "LIBRARIES",
                [self._convert_single_target(dependency) for dependency in deps]
                if deps
                else None,
                sort=False,
                quote=False,
            ),
            self._convert_string_list_block("ROOTS", roots, sort=False),
            self._convert_string_list_block(
                "CONFIGS",
                [f"{key}={value}" for key, value in sorted(configs.items())]
                if configs
                else None,
                sort=False,
            ),
            self._convert_option_block("TESTONLY", testonly),
        ]
        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += "loom_kernel_binary(\n" + "".join(blocks) + ")\n\n"
        self._emit_platform_guard_end(target_compatible_with)
