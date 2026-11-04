# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared generated-file and generated-source actions for IREE Bazel rules."""

IreeGeneratedFilesInfo = provider(
    doc = "Metadata for files produced by an IREE generator action.",
    fields = {
        "args": "Command-line arguments passed to the generator.",
        "outputs": "depset of generated output files.",
        "outputs_by_name": "Package-relative output name to generated File mapping.",
        "srcs": "depset of declared source input files.",
        "tool": "Generator executable label.",
    },
)

def _expand_output_arg(output_arg, output):
    if "{path}" in output_arg:
        return [output_arg.replace("{path}", output.path)]
    return [output_arg, output.path]

def _resolve_output_arguments(ctx, outputs_by_name):
    """Resolves output_args keys to declared outputs."""
    basename_counts = {}
    for output in outputs_by_name.values():
        basename_counts[output.basename] = basename_counts.get(output.basename, 0) + 1

    outputs_by_basename = {}
    for output in outputs_by_name.values():
        if basename_counts[output.basename] == 1:
            outputs_by_basename[output.basename] = output

    resolved_outputs = {}
    selected_output_paths = {}
    unknown_outputs = []
    for output_key in ctx.attr.output_args.keys():
        output = outputs_by_name.get(output_key)
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
    for name, output in outputs_by_name.items():
        if output.path not in selected_output_paths:
            missing_outputs.append(name)
    if missing_outputs:
        fail("output_args missing declared outputs: %s" % ", ".join(missing_outputs))
    return resolved_outputs

def _expand_action_location(ctx, argument, location_targets):
    """Expands a location into an execroot-relative action input path."""

    # Generator actions consume artifacts from the execroot, never runfiles.
    # Normalize all accepted location spellings before Bazel expands them so a
    # configurable argument list cannot preserve a runfiles-relative rootpath.
    argument = argument.replace("$(rootpath ", "$(location ")
    argument = argument.replace("$(execpath ", "$(location ")
    return ctx.expand_location(argument, targets = location_targets)

def _register_generated_action(ctx, outputs_by_name):
    outputs = outputs_by_name.values()
    outputs_by_argument = _resolve_output_arguments(ctx, outputs_by_name)

    location_targets = ctx.attr.srcs + ctx.attr.data
    command_args = []
    for arg in ctx.attr.args:
        command_args.append(_expand_action_location(ctx, arg, location_targets))

    for output_name, output_arg in ctx.attr.output_args.items():
        command_args.extend(_expand_output_arg(output_arg, outputs_by_argument[output_name]))

    args = ctx.actions.args()
    args.add_all(command_args)
    if ctx.attr.use_param_file:
        args.use_param_file("@%s", use_always = True)
        args.set_param_file_format("multiline")

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
            outputs_by_name = outputs_by_name,
            srcs = depset(ctx.files.srcs),
            tool = ctx.attr.tool.label,
        ),
    ]

def _iree_generated_files_impl(ctx):
    # Output labels and directory declarations already provide package-relative
    # names. Keep these identities independent of Bazel's artifact path layout.
    outputs_by_name = {
        label.name: output
        for label, output in zip(ctx.attr.outs, ctx.outputs.outs)
    }
    for name in ctx.attr.output_directories:
        outputs_by_name[name] = ctx.actions.declare_directory(name)
    return _register_generated_action(ctx, outputs_by_name)

_GENERATOR_ATTRS = {
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
    "output_directories": attr.string_list(
        doc = "Generated directory trees whose members are owned by the generator.",
    ),
    "progress_message": attr.string(
        default = "Generating files for %{label}",
        doc = "Progress message displayed while the generator runs. Defaults to the exact generating target label.",
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
    "use_param_file": attr.bool(
        doc = "Passes generator arguments through an @response file. The generator must support one argument per line.",
    ),
}

_GENERATED_FILES_ATTRS = dict(_GENERATOR_ATTRS)
_GENERATED_FILES_ATTRS["outs"] = attr.output_list(
    mandatory = True,
    doc = "Generated output files.",
)

iree_generated_files = rule(
    implementation = _iree_generated_files_impl,
    attrs = _GENERATED_FILES_ATTRS,
    doc = "Runs an executable generator with declared inputs and outputs.",
)

def _iree_generated_sources_action_impl(ctx):
    output_root = "_iree_generated/%s" % ctx.label.name
    outputs_by_name = {
        name: ctx.actions.declare_file("%s/%s" % (output_root, name))
        for name in ctx.attr.outs
    }
    return _register_generated_action(ctx, outputs_by_name)

_GENERATED_SOURCES_ATTRS = {
    name: value
    for name, value in _GENERATOR_ATTRS.items()
    if name != "output_directories"
}
_GENERATED_SOURCES_ATTRS["outs"] = attr.string_list(
    mandatory = True,
    doc = "Package-relative names of generated source files.",
)

_iree_generated_sources_action = rule(
    implementation = _iree_generated_sources_action_impl,
    attrs = _GENERATED_SOURCES_ATTRS,
)

def _iree_generated_source_outputs_impl(ctx):
    generated_outputs = ctx.attr.generator[IreeGeneratedFilesInfo].outputs_by_name
    for label, output in zip(ctx.attr.outs, ctx.outputs.outs):
        source = generated_outputs.get(label.name)
        if source == None:
            fail("unknown generated output: %s" % label.name)
        ctx.actions.symlink(
            output = output,
            target_file = source,
            progress_message = "Exposing generated source %s" % output.short_path,
        )
    return [DefaultInfo(files = depset(ctx.outputs.outs))]

_iree_generated_source_outputs = rule(
    implementation = _iree_generated_source_outputs_impl,
    attrs = {
        "generator": attr.label(
            cfg = "exec",
            mandatory = True,
            providers = [IreeGeneratedFilesInfo],
            doc = "Private source generator evaluated in the execution configuration.",
        ),
        "outs": attr.output_list(
            mandatory = True,
            doc = "Generated source files exposed in the consuming configuration.",
        ),
    },
)

def iree_generated_sources(
        name,
        tool,
        outs,
        output_args,
        args = [],
        data = [],
        mnemonic = "IreeGenerate",
        progress_message = None,
        srcs = [],
        tags = [],
        testonly = False,
        target_compatible_with = None,
        use_param_file = False,
        visibility = None):
    """Generates sources whose bytes do not vary across target/exec transition.

    The generator runs in Bazel's execution configuration. Target- and
    execution-configured consumers that resolve to the same execution
    configuration therefore share one generator action. Cheap built-in symlink
    actions preserve the public output labels and package-relative paths in
    each consuming configuration.

    Explicit project build settings may remain semantic generator inputs. This
    rule is only for source or data files whose content does not depend on the
    target/exec platform transition itself. Target artifacts such as object
    files, device binaries, and linked modules need a target-configured rule.

    Args:
      name: Public aggregate target name.
      tool: Executable generator tool label.
      outs: Package-relative generated file names.
      output_args: Map of output names to generator arguments.
      args: Generator arguments before output arguments.
      data: Additional generator inputs available to location expansion.
      mnemonic: Generator action mnemonic.
      progress_message: Generator action progress message.
      srcs: Source inputs consumed by the generator.
      tags: Tags applied to the public targets and private generator.
      testonly: Whether the generated sources are restricted to tests.
      target_compatible_with: Compatibility constraints for public targets.
      use_param_file: Whether to pass generator arguments through an @response
        file containing one argument per line.
      visibility: Visibility of public targets.
    """
    if not outs:
        fail("generated source actions require at least one output")
    if progress_message == None:
        progress_message = "Generating files for //%s:%s" % (native.package_name(), name)

    generator_name = name + "__source_generator"
    _iree_generated_sources_action(
        name = generator_name,
        tool = tool,
        outs = outs,
        output_args = output_args,
        args = args,
        data = data,
        mnemonic = mnemonic,
        progress_message = progress_message,
        srcs = srcs,
        tags = tags + ["manual"],
        testonly = testonly,
        use_param_file = use_param_file,
        visibility = ["//visibility:private"],
    )

    public_kwargs = {
        "tags": tags,
        "testonly": testonly,
    }
    if target_compatible_with != None:
        public_kwargs["target_compatible_with"] = target_compatible_with
    if visibility != None:
        public_kwargs["visibility"] = visibility

    _iree_generated_source_outputs(
        name = name,
        generator = ":" + generator_name,
        outs = outs,
        **public_kwargs
    )
