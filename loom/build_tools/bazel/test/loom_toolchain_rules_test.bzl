# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Configuration ownership tests for the source Loom toolchain roles."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo")

_BUILD_ROLES = ["compile", "link"]
_DESTINATION_ROLES = ["test", "benchmark", "format", "lint"]
_TOOLCHAIN_TYPES = {
    role: Label("//loom/build_tools/bazel:" + role + "_toolchain_type")
    for role in _BUILD_ROLES + _DESTINATION_ROLES
}

ToolchainUsageInfo = provider(
    doc = "Toolchain executables resolved by a destination test consumer.",
    fields = {
        "executables": "Resolved source executables indexed by toolchain role.",
    },
)

def _toolchain_consumer_impl(ctx):
    return [ToolchainUsageInfo(
        executables = {
            role: ctx.toolchains[toolchain_type].tool.executable
            for role, toolchain_type in _TOOLCHAIN_TYPES.items()
        },
    )]

_toolchain_consumer = rule(
    implementation = _toolchain_consumer_impl,
    toolchains = _TOOLCHAIN_TYPES.values(),
)

def _test_role_configurations(env, target):
    executables = target[ToolchainUsageInfo].executables
    destination = target[TestingAspectInfo].bin_path
    for role in _BUILD_ROLES:
        if executables[role].root.path == destination:
            env.fail("%s must use the build execution configuration" % role)
    for role in _DESTINATION_ROLES:
        if executables[role].root.path != destination:
            env.fail("%s lost the destination test configuration: %s" % (
                role,
                executables[role].root.path,
            ))

def loom_toolchain_rules_test_suite(name):
    _toolchain_consumer(
        name = name + "_subject",
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {"size": "small"},
        config_settings = {
            "//command_line_option:compilation_mode": "dbg",
            "//command_line_option:extra_toolchains": [
                "//loom/build_tools/bazel:source_tools_" + role + "_toolchain"
                for role in _TOOLCHAIN_TYPES
            ],
        },
        impl = _test_role_configurations,
        target = name + "_subject",
    )
