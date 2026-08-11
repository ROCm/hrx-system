# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom authoring toolchain declarations."""

_TOOLCHAIN_TYPES = {
    "benchmark": Label("//loom/build_tools/bazel:benchmark_toolchain_type"),
    "compile": Label("//loom/build_tools/bazel:compile_toolchain_type"),
    "format": Label("//loom/build_tools/bazel:format_toolchain_type"),
    "link": Label("//loom/build_tools/bazel:link_toolchain_type"),
    "test": Label("//loom/build_tools/bazel:test_toolchain_type"),
}

def _tool_info(target):
    default_info = target[DefaultInfo]
    return struct(
        executable = default_info.files_to_run.executable,
        files_to_run = default_info.files_to_run,
        runfiles = default_info.default_runfiles,
    )

def _loom_toolchain_impl(ctx):
    return [
        platform_common.ToolchainInfo(
            tool = _tool_info(ctx.attr.tool),
        ),
    ]

_loom_toolchain = rule(
    implementation = _loom_toolchain_impl,
    attrs = {
        "tool": attr.label(
            cfg = "exec",
            executable = True,
            mandatory = True,
            doc = "Executable implementing one Loom authoring tool role.",
        ),
    },
    doc = "Binds one Loom authoring tool role to an executable.",
)

def loom_tools_toolchains(
        name,
        benchmark_tool,
        compile_tool,
        format_tool,
        link_tool,
        test_tool,
        exec_compatible_with = [],
        target_compatible_with = [],
        target_settings = [],
        tags = [],
        visibility = None):
    """Declares independently resolved source-built or released Loom tools.

    The macro emits ``<name>_<role>`` implementations and
    ``<name>_<role>_toolchain`` registrations for the benchmark, compile,
    format, link, and test roles. Splitting the roles prevents a rule using one
    executable from configuring the dependency graphs of the other tools.
    """
    tools = {
        "benchmark": benchmark_tool,
        "compile": compile_tool,
        "format": format_tool,
        "link": link_tool,
        "test": test_tool,
    }
    for role, tool in tools.items():
        implementation_name = "%s_%s" % (name, role)
        _loom_toolchain(
            name = implementation_name,
            tags = tags,
            tool = tool,
            visibility = ["//visibility:private"],
        )
        toolchain_kwargs = {
            "name": implementation_name + "_toolchain",
            "exec_compatible_with": exec_compatible_with,
            "tags": tags,
            "target_compatible_with": target_compatible_with,
            "target_settings": target_settings,
            "toolchain": ":" + implementation_name,
            "toolchain_type": _TOOLCHAIN_TYPES[role],
        }
        if visibility != None:
            toolchain_kwargs["visibility"] = visibility
        native.toolchain(**toolchain_kwargs)
