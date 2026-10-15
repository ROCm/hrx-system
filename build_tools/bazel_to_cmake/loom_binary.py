# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom deployment-product projections shared by consuming projects."""


class LoomBinaryBuildFileFunctions:
    def _convert_select_condition(self, label):
        if isinstance(label, str) and label.startswith(
            "//loom/config/target/amdgpu:descriptor_set_"
        ):
            return "LOOM_TARGET_AMDGPU_" + label.split(":", 1)[1].upper()
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
        self._target_file_paths[self._current_target_label(name)] = (
            f"${{CMAKE_CURRENT_BINARY_DIR}}/{output}"
        )
        self._target_file_paths[self._current_target_label(output)] = (
            f"${{CMAKE_CURRENT_BINARY_DIR}}/{output}"
        )
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
