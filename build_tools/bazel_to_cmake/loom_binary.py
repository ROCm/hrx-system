# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom deployment-product projections shared by consuming projects."""


class LoomBinaryBuildFileFunctions:
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
        data=None,
        input_format="",
        inputopts=None,
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
            self._convert_data_srcs_block(data, block_name="DATA", sort=False),
            self._convert_string_arg_block("INPUT_FORMAT", input_format or None),
            self._convert_string_list_block(
                "INPUTOPTS", self._convert_location_args(inputopts), sort=False
            ),
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
