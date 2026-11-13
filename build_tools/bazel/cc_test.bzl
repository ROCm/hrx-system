# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared C/C++ Bazel test macros for IREE repositories."""

# rules_cc currently exposes only its macro publicly. Rule inheritance needs
# the underlying rule object so ctx.super() can preserve its full behavior.
# buildifier: disable=bzl-visibility
load(
    "@rules_cc//cc/private/rules_impl:cc_test.bzl",
    rules_cc_test = "cc_test",
)
load(
    "//build_tools/sanitizer:build_defs.bzl",
    "declare_sanitizer_suppressions",
)
load(":cc_attrs.bzl", "cc_attrs")
load(
    ":cc_execution.bzl",
    "cc_execution_attrs",
    "cc_execution_impl",
    "cc_execution_initializer",
)
load(":cc_opts.bzl", "cc_opts")

# This is a rules_cc test rule extension, not a wrapper target. ctx.super()
# preserves the parent rule's compilation, coverage, launcher, platform, and
# test providers before the shared implementation conditionally adds runfiles.
cc_test = rule(
    implementation = cc_execution_impl,
    attrs = cc_execution_attrs,
    # rules_cc replaces Bazel's implicit test group with its optional C++
    # runner. Keep that runner and restore the standard target-platform
    # qualification so cross-compiled tests cannot run on an incompatible host.
    exec_groups = {
        "test": exec_group(toolchains = [
            "@bazel_tools//tools/test:default_test_toolchain_type",
            config_common.toolchain_type(
                "@bazel_tools//tools/cpp:test_runner_toolchain_type",
                mandatory = False,
            ),
        ]),
    },
    initializer = cc_execution_initializer,
    parent = rules_cc_test,
)

def _keep_debug_transition_impl(_settings, _attr):
    return {"//command_line_option:strip": "never"}

_keep_debug_transition = transition(
    implementation = _keep_debug_transition_impl,
    inputs = [],
    outputs = ["//command_line_option:strip"],
)

def _cc_unstripped_test_impl(ctx):
    return ctx.super()

# Diagnostic tests consume their own DWARF records as input. Retain those
# records through the normal C++ test rule, including its runner and coverage.
cc_unstripped_test = rule(
    implementation = _cc_unstripped_test_impl,
    parent = cc_test,
    cfg = _keep_debug_transition,
)

def _iree_cc_test_impl(
        name,
        visibility,
        srcs,
        deps,
        data,
        copts,
        conlyopts,
        cxxopts,
        defines,
        local_defines,
        includes,
        system_includes,
        linkopts,
        linkstatic,
        keep_debug,
        args,
        env,
        sanitizer_suppressions,
        size,
        tags,
        resource_group,
        **kwargs):
    data = declare_sanitizer_suppressions(name, data, sanitizer_suppressions, testonly = True)
    target_attrs = cc_attrs.collect(
        srcs = srcs,
        hdrs = None,
        textual_hdrs = None,
        deps = deps,
        data = data,
        copts = copts,
        conlyopts = conlyopts,
        cxxopts = cxxopts,
        defines = defines,
        local_defines = local_defines,
        includes = includes,
        system_includes = system_includes,
        linkopts = cc_opts.iree_code_link_options(linkopts),
        linkstatic = linkstatic,
    )
    target_attrs = cc_attrs.merge_dicts(kwargs, target_attrs)
    target_attrs["linkstatic"] = True
    cc_attrs.add_if_not_none(target_attrs, "args", args)
    cc_attrs.add_if_not_none(target_attrs, "env", env)
    target_attrs["size"] = size
    target_attrs["tags"] = cc_attrs.with_resource_group_tags(tags, resource_group)
    test_rule = cc_unstripped_test if keep_debug else cc_test
    test_rule(
        name = name,
        visibility = visibility,
        **target_attrs
    )

iree_cc_test = macro(
    doc = """Defines a shared IREE C/C++ test target.

    Tests link statically by default so each executable does not split
    process-local test state across Bazel-generated shared objects.

    `keep_debug` preserves debug sections consumed by introspection tests.
    Tests requiring DWARF must also request its emission in `copts`.

    `resource_group` serializes tests that compete for a named local resource.
    Bazel receives the conservative `exclusive-if-local` tag plus a structured
    `resource_group:<name>` tag that CI and other generators can inspect.
    """,
    implementation = _iree_cc_test_impl,
    inherit_attrs = cc_test,
    attrs = cc_attrs.merge_dicts(
        cc_attrs.compilation,
        cc_attrs.dependency,
        cc_attrs.binary_source,
        cc_attrs.link,
        {
            "args": attr.string_list(
                doc = "Command-line arguments passed to the test binary.",
            ),
            "env": attr.string_dict(
                configurable = False,
                doc = "Environment variables passed to the test binary.",
            ),
            "execution_data": None,
            "execution_deps": None,
            "keep_debug": attr.bool(
                configurable = False,
                default = False,
                doc = "Retain debug sections required by the test; copts controls their emission.",
            ),
            "resource_group": attr.string(
                configurable = False,
                doc = "Local resource name used to serialize tests competing for the same host resource.",
            ),
            "sanitizer_suppressions": attr.string_dict(
                configurable = False,
                doc = "Sanitizer suppression files keyed by sanitizer name.",
            ),
            "size": attr.string(
                configurable = False,
                default = "small",
                doc = "Bazel test size.",
            ),
        },
    ),
)
