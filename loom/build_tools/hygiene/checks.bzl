# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Read-only validation actions with source and tool identity in their keys."""

load("//loom/config/target:targets.bzl", "LOOM_TARGET_ARCHITECTURES")

# Repository formatting owns every checked-in Loom source, independent of the
# product targets selected by its caller. Canonicalize the formatter's target
# configuration so subset builds recognize all repository IR and share one
# execution-tool action graph.
def _repository_format_transition_impl(_settings, _attr):
    return {
        "//loom/config/emit:enable": [],
        "//loom/config/target/arch:enable": [],
        "//loom/config/target:enable": LOOM_TARGET_ARCHITECTURES,
    }

_repository_format_transition = transition(
    implementation = _repository_format_transition_impl,
    inputs = [],
    outputs = [
        "//loom/config/emit:enable",
        "//loom/config/target:enable",
        "//loom/config/target/arch:enable",
    ],
)

def _source_check_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name + ".passed")
    arguments = ctx.actions.args()
    arguments.add("--tool", ctx.executable.tool)
    arguments.add("--check", ctx.attr.check)
    arguments.add("--sources", ctx.file.manifest)
    arguments.add("--output", output)
    ctx.actions.run(
        executable = ctx.executable._runner,
        arguments = [arguments],
        inputs = depset(ctx.files.srcs + [ctx.file.manifest]),
        tools = [ctx.attr.tool[DefaultInfo].files_to_run],
        outputs = [output],
        mnemonic = ctx.attr.mnemonic,
        progress_message = "Checking Loom " + ctx.attr.check,
    )
    return [DefaultInfo(files = depset([output]))]

_SOURCE_CHECK_ATTRS = {
    "check": attr.string(mandatory = True),
    "manifest": attr.label(allow_single_file = True, mandatory = True),
    "mnemonic": attr.string(mandatory = True),
    "srcs": attr.label_list(allow_files = True),
    "tool": attr.label(executable = True, cfg = "exec", mandatory = True),
    "_runner": attr.label(default = Label(":check"), executable = True, cfg = "exec"),
}

_source_check = rule(
    implementation = _source_check_impl,
    attrs = _SOURCE_CHECK_ATTRS,
)

_repository_format_source_check = rule(
    implementation = _source_check_impl,
    attrs = dict(
        _SOURCE_CHECK_ATTRS,
        _allowlist_function_transition = attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    ),
    cfg = _repository_format_transition,
)

_CHECKS = {
    "authoring": ("sources", "//loom/py/loom/tools:loom-lint", "LoomAuthoringPolicy"),
    "format": ("format_sources", "//loom/src/loom/tools/loom-format:loom-format", "LoomSourceFormat"),
    "repository": ("policy_sources", "//loom/build_tools:loom_source_lint", "LoomRepositoryPolicy"),
}

def loom_hygiene_check(name, check, **kwargs):
    """Declares one complete read-only source check and its executable closure."""
    source_group, tool, mnemonic = _CHECKS[check]
    source_check_rule = _repository_format_source_check if check == "format" else _source_check
    source_check_rule(
        name = name,
        check = check,
        tool = tool,
        mnemonic = mnemonic,
        srcs = ["@loom_hygiene_sources//:" + source_group],
        manifest = "@loom_hygiene_sources//:" + source_group + ".json",
        **kwargs
    )
