# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Projects generated-file declarations into native generator actions."""


class GeneratedFileBuildFileFunctions:
    def iree_generated_files(
        self,
        name,
        tool,
        outs,
        output_args,
        srcs=None,
        data=None,
        args=None,
        tags=None,
        output_directories=None,
        use_param_file=False,
        testonly=False,
        target_compatible_with=None,
        progress_message=None,
        mnemonic=None,
        visibility=None,
    ):
        if self._should_skip_target(tags=tags):
            return
        outputs = list(outs) + list(output_directories or [])
        paths = {output: f"{self._package_binary_dir()}/{output}" for output in outputs}
        output_paths = list(paths.values())
        for output, path in paths.items():
            self._register_target_files(output, [path])
        self._register_target_files(name, output_paths)
        if self._libraries is not None:
            label = self._current_target_label(name)
            producer = self._convert_single_target(label).replace("::", "_")
            self._libraries.producers.update(
                (path, producer) for path in self._target_file_paths[label]
            )

        # Match the action's exact-path or unique-basename output selection.
        basenames = {}
        for output in outputs:
            basenames.setdefault(output.rsplit("/", 1)[-1], []).append(output)
        selected = set()
        arguments = []
        for key, argument in output_args.items():
            matches = [key] if key in paths else basenames.get(key, [])
            if len(matches) != 1 or matches[0] in selected:
                raise ValueError(f"{name}: output_args must select one output: {key}")
            output = matches[0]
            selected.add(output)
            if "{path}" in argument:
                arguments.append(argument.replace("{path}", paths[output]))
            else:
                arguments.extend([argument, paths[output]])
        if not outputs or selected != set(outputs):
            raise ValueError(f"{name}: output_args must cover all declared outputs")

        self._emit_platform_guard_begin(target_compatible_with)
        if testonly:
            self._converter.body += "if(IREE_BUILD_TESTS)\n"

        def render():
            args_block, args_variables = self._convert_platform_select_strings(
                name, "ARGS", (self._convert_generated_args(args) or []) + arguments
            )
            inputs_block, inputs_variables = self._convert_platform_select_strings(
                name,
                "INPUTS",
                self._convert_generated_inputs((srcs or []) + (data or [])),
            )
            return (
                args_variables
                + inputs_variables
                + "iree_generated_files(\n"
                + self._convert_string_arg_block("NAME", name, quote=False)
                + self._convert_single_target_block("GENERATOR", tool)
                + self._convert_string_list_block("OUTPUTS", output_paths)
                + args_block
                + inputs_block
                + self._convert_option_block("USE_PARAM_FILE", use_param_file)
                + self._convert_string_arg_block(
                    "COMMENT",
                    (progress_message or "Generating %{label}").replace(
                        "%{label}", self._current_target_label(name)
                    ),
                )
                + ")\n\n"
            )

        self._converter.defer(render)
        if testonly:
            self._converter.body += "endif()\n"
        self._emit_platform_guard_end(target_compatible_with)

    iree_generated_sources = iree_generated_files
