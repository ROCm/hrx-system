# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Repository-owned C/C++ compile command extraction."""

load("@rules_cc//cc:find_cc_toolchain.bzl", "CC_TOOLCHAIN_ATTRS", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load(
    "//build_tools/bazel:cc_introspection.bzl",
    "iree_cc_compile_command",
    "iree_cc_feature_configuration",
    "iree_cc_sanitize_label",
    "iree_cc_source_files",
)

IreeCompileCommandsInfo = provider(
    doc = "C/C++ compile command metadata collected from configured targets.",
    fields = {
        "entries": "depset of JSON-encoded compile command entries.",
        "fragments": "depset of JSON fragment files containing compile command entries.",
    },
)

def _compile_command_entry(ctx, target, cc_toolchain, feature_configuration, source):
    compile_command = iree_cc_compile_command(
        ctx,
        target,
        cc_toolchain,
        feature_configuration,
        source,
    )
    return json.encode({
        "arguments": compile_command.arguments,
        "directory": ".",
        "file": source.path,
    })

def _collect_compile_commands_aspect_impl(target, ctx):
    transitive_entries = []
    transitive_fragments = []
    for attr_name in [
        "actual",
        "deps",
        "implementation_deps",
    ]:
        if not hasattr(ctx.rule.attr, attr_name):
            continue
        attr_value = getattr(ctx.rule.attr, attr_name)
        if type(attr_value) == type([]):
            deps = attr_value
        elif attr_value:
            deps = [attr_value]
        else:
            deps = []
        for dep in deps:
            if IreeCompileCommandsInfo in dep:
                transitive_entries.append(dep[IreeCompileCommandsInfo].entries)
                transitive_fragments.append(dep[IreeCompileCommandsInfo].fragments)

    local_entries = []
    if CcInfo in target:
        cc_toolchain = find_cc_toolchain(ctx)
        feature_configuration = iree_cc_feature_configuration(ctx, cc_toolchain)
        for source in iree_cc_source_files(ctx):
            local_entries.append(_compile_command_entry(
                ctx,
                target,
                cc_toolchain,
                feature_configuration,
                source,
            ))

    local_fragments = []
    if local_entries:
        fragment = ctx.actions.declare_file(
            "%s.compile_commands.json" % iree_cc_sanitize_label(target.label),
        )
        ctx.actions.write(
            output = fragment,
            content = "[\n%s\n]\n" % ",\n".join(local_entries),
        )
        local_fragments.append(fragment)

    compile_commands = IreeCompileCommandsInfo(
        entries = depset(local_entries, transitive = transitive_entries),
        fragments = depset(local_fragments, transitive = transitive_fragments),
    )
    output_groups = OutputGroupInfo(
        iree_compile_commands_fragments = depset(
            local_fragments,
            transitive = transitive_fragments,
        ),
    )
    return [compile_commands, output_groups]

collect_compile_commands_aspect = aspect(
    implementation = _collect_compile_commands_aspect_impl,
    attr_aspects = [
        "actual",
        "deps",
        "implementation_deps",
    ],
    attrs = CC_TOOLCHAIN_ATTRS,
    fragments = ["cpp"],
    required_providers = [CcInfo],
    toolchains = use_cc_toolchain(),
)

def _iree_compile_commands_impl(ctx):
    transitive_entries = []
    transitive_fragments = []
    for target in ctx.attr.targets:
        if IreeCompileCommandsInfo not in target:
            continue
        compile_commands = target[IreeCompileCommandsInfo]
        transitive_entries.append(compile_commands.entries)
        transitive_fragments.append(compile_commands.fragments)

    fragments = depset(transitive = transitive_fragments)
    args = ctx.actions.args()
    args.add(ctx.outputs.out)
    args.add_all(fragments)
    ctx.actions.run(
        executable = ctx.executable._merge_tool,
        arguments = [args],
        inputs = fragments,
        outputs = [ctx.outputs.out],
        mnemonic = "IreeCompileCommands",
        progress_message = "Merging C/C++ compile commands %{output}",
    )

    return [
        DefaultInfo(files = depset([ctx.outputs.out])),
        IreeCompileCommandsInfo(
            entries = depset(transitive = transitive_entries),
            fragments = fragments,
        ),
    ]

iree_compile_commands = rule(
    implementation = _iree_compile_commands_impl,
    attrs = {
        "out": attr.output(
            mandatory = True,
            doc = "Output compile_commands.json file.",
        ),
        "targets": attr.label_list(
            aspects = [collect_compile_commands_aspect],
            doc = "C/C++ targets whose configured compile commands should be emitted.",
            mandatory = True,
            providers = [CcInfo],
        ),
        "_merge_tool": attr.label(
            cfg = "exec",
            default = Label("//build_tools/bazel:compile_commands_merge"),
            executable = True,
        ),
    },
    doc = "Emits a compile_commands.json file for configured C/C++ targets.",
)
