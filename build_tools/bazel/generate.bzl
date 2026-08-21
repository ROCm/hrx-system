# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared generated-file actions for IREE Bazel rules."""

IreeGeneratedFilesInfo = provider(
    doc = "Metadata for files produced by an IREE generator action.",
    fields = {
        "args": "Command-line arguments passed to the generator.",
        "outputs": "depset of generated output files.",
        "srcs": "depset of declared source input files.",
        "tool": "Generator executable label.",
    },
)

def _expand_output_arg(output_arg, output):
    if "{path}" in output_arg:
        return [output_arg.replace("{path}", output.path)]
    return [output_arg, output.path]

def _resolve_output_arguments(ctx, outputs):
    """Resolves output_args keys to declared outputs."""
    basename_counts = {}
    for output in outputs:
        basename_counts[output.basename] = basename_counts.get(output.basename, 0) + 1

    package_prefix = ctx.label.package + "/" if ctx.label.package else ""
    outputs_by_path = {}
    outputs_by_basename = {}
    for output in outputs:
        output_path = output.short_path
        if package_prefix and output_path.startswith(package_prefix):
            output_path = output_path[len(package_prefix):]
        outputs_by_path[output_path] = output
        if basename_counts[output.basename] == 1:
            outputs_by_basename[output.basename] = output

    resolved_outputs = {}
    selected_output_paths = {}
    unknown_outputs = []
    for output_key in ctx.attr.output_args.keys():
        output = outputs_by_path.get(output_key)
        if output == None:
            output = outputs_by_basename.get(output_key)
        if output == None:
            unknown_outputs.append(output_key)
            continue
        if output.path in selected_output_paths:
            fail("output_args selects generated output %r more than once" % output.path)
        resolved_outputs[output_key] = output
        selected_output_paths[output.path] = True
    if unknown_outputs:
        fail("output_args contains undeclared outputs: %s" % ", ".join(unknown_outputs))

    missing_outputs = []
    for output in outputs:
        if output.path not in selected_output_paths:
            missing_outputs.append(output.short_path)
    if missing_outputs:
        fail("output_args missing declared outputs: %s" % ", ".join(missing_outputs))
    return resolved_outputs

def _iree_generated_files_impl(ctx):
    outputs = ctx.outputs.outs
    outputs_by_argument = _resolve_output_arguments(ctx, outputs)

    location_targets = ctx.attr.srcs + ctx.attr.data
    command_args = []
    for arg in ctx.attr.args:
        command_args.append(ctx.expand_location(arg, targets = location_targets))

    for output_name, output_arg in ctx.attr.output_args.items():
        command_args.extend(_expand_output_arg(output_arg, outputs_by_argument[output_name]))

    args = ctx.actions.args()
    args.add_all(command_args)

    ctx.actions.run(
        arguments = [args],
        executable = ctx.attr.tool[DefaultInfo].files_to_run,
        inputs = depset(ctx.files.srcs + ctx.files.data),
        mnemonic = ctx.attr.mnemonic,
        outputs = outputs,
        progress_message = ctx.attr.progress_message,
    )

    return [
        DefaultInfo(files = depset(outputs)),
        IreeGeneratedFilesInfo(
            args = command_args,
            outputs = depset(outputs),
            srcs = depset(ctx.files.srcs),
            tool = ctx.attr.tool.label,
        ),
    ]

iree_generated_files = rule(
    implementation = _iree_generated_files_impl,
    attrs = {
        "args": attr.string_list(
            doc = "Generator arguments before output arguments. $(location) references may name srcs or data labels.",
        ),
        "data": attr.label_list(
            allow_files = True,
            doc = "Additional files available to $(location) expansion and declared as action inputs.",
        ),
        "mnemonic": attr.string(
            default = "IreeGenerate",
            doc = "Bazel action mnemonic.",
        ),
        "output_args": attr.string_dict(
            mandatory = True,
            doc = "Map of package-relative output path to generator argument. A unique basename may be used as shorthand. Use '{path}' to embed the output path in a single argument; otherwise the path is passed as the following argument.",
        ),
        "outs": attr.output_list(
            mandatory = True,
            doc = "Generated output files.",
        ),
        "progress_message": attr.string(
            doc = "Progress message displayed while the generator runs.",
        ),
        "srcs": attr.label_list(
            allow_files = True,
            doc = "Source inputs consumed by the generator.",
        ),
        "tool": attr.label(
            cfg = "exec",
            executable = True,
            mandatory = True,
            doc = "Executable generator tool.",
        ),
    },
    doc = "Runs an executable generator with declared inputs and outputs.",
)
