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
    "//build_tools/sanitizer:suppressions.bzl",
    "iree_sanitizer_suppression_data",
    "iree_sanitizer_suppression_env",
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
    initializer = cc_execution_initializer,
    parent = rules_cc_test,
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
        args,
        env,
        sanitizer_suppressions,
        size,
        tags,
        resource_group,
        **kwargs):
    data = iree_sanitizer_suppression_data(data, sanitizer_suppressions)
    env = iree_sanitizer_suppression_env(env, sanitizer_suppressions)
    target_attrs = cc_attrs.collect(
        srcs = srcs,
        hdrs = None,
        textual_hdrs = None,
        deps = cc_opts.iree_code_link_dependencies(deps),
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
    cc_test(
        name = name,
        visibility = visibility,
        **target_attrs
    )

iree_cc_test = macro(
    doc = """Defines a shared IREE C/C++ test target.

    Tests link statically by default so each executable does not split
    process-local test state across Bazel-generated shared objects.

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
            "dynamic_library_data": None,
            "dynamic_library_deps": None,
            "env": attr.string_dict(
                configurable = False,
                doc = "Environment variables passed to the test binary.",
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
