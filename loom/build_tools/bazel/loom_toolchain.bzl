# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom authoring toolchain declarations."""

def _tool_info(target):
    default_info = target[DefaultInfo]
    return struct(
        executable = default_info.files_to_run.executable,
        files_to_run = default_info.files_to_run,
        runfiles = default_info.default_runfiles,
    )

def _loom_tools_toolchain_impl(ctx):
    return [
        platform_common.ToolchainInfo(
            benchmark = _tool_info(ctx.attr.benchmark_tool),
            compile = _tool_info(ctx.attr.compile_tool),
            format = _tool_info(ctx.attr.format_tool),
            link = _tool_info(ctx.attr.link_tool),
            test = _tool_info(ctx.attr.test_tool),
        ),
    ]

loom_tools_toolchain = rule(
    implementation = _loom_tools_toolchain_impl,
    attrs = {
        "benchmark_tool": attr.label(
            cfg = "exec",
            executable = True,
            mandatory = True,
            doc = "iree-benchmark-loom compatible benchmark and planning tool.",
        ),
        "compile_tool": attr.label(
            cfg = "exec",
            executable = True,
            mandatory = True,
            doc = "loom-compile compatible target compiler.",
        ),
        "format_tool": attr.label(
            cfg = "exec",
            executable = True,
            mandatory = True,
            doc = "loom-format compatible canonical formatting tool.",
        ),
        "link_tool": attr.label(
            cfg = "exec",
            executable = True,
            mandatory = True,
            doc = "loom-link compatible module linker.",
        ),
        "test_tool": attr.label(
            cfg = "exec",
            executable = True,
            mandatory = True,
            doc = "iree-test-loom compatible correctness runner.",
        ),
    },
    doc = "Binds Loom authoring rules to source-built or released tools.",
)
