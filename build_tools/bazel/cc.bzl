# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared C/C++ Bazel macros for IREE repositories.

This package owns repo-wide C/C++ mechanics that are deliberately independent of
any top-level project directory. Project-specific wrappers should load these
macros and add only the dependencies, flags, or validation that belong to that
project.
"""

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

# rules_cc currently exposes only its macro publicly. Rule inheritance needs
# the underlying rule object so ctx.super() can preserve its full behavior.
# buildifier: disable=bzl-visibility
load(
    "@rules_cc//cc/private/rules_impl:cc_binary.bzl",
    rules_cc_binary = "cc_binary",
)
load("//build_tools/sanitizer:build_defs.bzl", "declare_sanitizer_suppressions")
load(":cc_attrs.bzl", "cc_attrs")
load(
    ":cc_execution.bzl",
    "cc_execution_attrs",
    "cc_execution_impl",
    "cc_execution_initializer",
)
load(":cc_opts.bzl", "cc_opts")

# This isolated diagnostic branch inspects private types in exec-config tools.
# Bazel's exec transition otherwise forces strip=always after command-line flags.
def _diagnostic_debug_transition_impl(_settings, _attr):
    return {"//command_line_option:strip": "never"}

_diagnostic_debug_transition = transition(
    implementation = _diagnostic_debug_transition_impl,
    inputs = [],
    outputs = ["//command_line_option:strip"],
)

# Extending the rules_cc implementation keeps the public target as the actual
# C/C++ rule: compilation actions, implicit outputs, aspects, and every parent
# provider remain attached to the same label. The surrounding symbolic macro
# continues to own IREE's user-facing defaults.
cc_binary = rule(
    implementation = cc_execution_impl,
    attrs = cc_execution_attrs,
    initializer = cc_execution_initializer,
    parent = rules_cc_binary,
    cfg = _diagnostic_debug_transition,
)

def _iree_cc_library_group_impl(ctx):
    return [
        cc_common.merge_cc_infos(direct_cc_infos = [dep[CcInfo] for dep in ctx.attr.deps]),
        DefaultInfo(
            files = depset(transitive = [dep[DefaultInfo].files for dep in ctx.attr.deps]),
            default_runfiles = ctx.runfiles().merge_all([dep[DefaultInfo].default_runfiles for dep in ctx.attr.deps]),
            data_runfiles = ctx.runfiles().merge_all([dep[DefaultInfo].data_runfiles for dep in ctx.attr.deps]),
        ),
        coverage_common.instrumented_files_info(ctx, dependency_attributes = ["deps"]),
    ]

iree_cc_library_group = rule(
    doc = """Aggregates existing C/C++ libraries without introducing an implementation.

    Direct dependency headers remain direct headers of the group, including under
    Clang layering checks. Linker inputs retain their original owners. The group
    adds no compilation, archive, or module-map actions; narrower consumers can
    depend on its components directly.
    """,
    implementation = _iree_cc_library_group_impl,
    attrs = {
        "deps": attr.label_list(
            providers = [CcInfo],
            doc = "Libraries whose public headers and implementations are re-exported.",
        ),
    },
    provides = [CcInfo],
)

def _iree_cc_library_impl(
        name,
        visibility,
        srcs,
        hdrs,
        textual_hdrs,
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
        linkstamp,
        alwayslink,
        sanitizer_suppressions,
        **kwargs):
    data = declare_sanitizer_suppressions(name, data, sanitizer_suppressions, kwargs.get("testonly", False))
    target_attrs = cc_attrs.collect(
        srcs = srcs,
        hdrs = hdrs,
        textual_hdrs = textual_hdrs,
        deps = deps,
        data = data,
        copts = copts,
        conlyopts = conlyopts,
        cxxopts = cxxopts,
        defines = defines,
        local_defines = local_defines,
        includes = includes,
        system_includes = system_includes,
        linkopts = linkopts,
        linkstatic = linkstatic,
    )
    target_attrs = cc_attrs.merge_dicts(kwargs, target_attrs)
    cc_attrs.add_if_not_none(target_attrs, "alwayslink", alwayslink)
    cc_attrs.add_if_not_none(target_attrs, "linkstamp", linkstamp)
    cc_library(
        name = name,
        visibility = visibility,
        **target_attrs
    )

iree_cc_library = macro(
    doc = """Defines a shared IREE C/C++ library target.

    The shared wrapper intentionally adds no project policy. It exists so that
    project wrappers can share Bazel-specific mechanics, and so that attributes
    needed by non-Bazel generators can remain explicit at call sites.
    """,
    implementation = _iree_cc_library_impl,
    inherit_attrs = "common",
    attrs = cc_attrs.merge_dicts(
        cc_attrs.compilation,
        cc_attrs.dependency,
        cc_attrs.library_source,
        cc_attrs.link,
        {
            "alwayslink": attr.bool(
                doc = "Whether to force this library into binaries that depend on it.",
            ),
            "linkstamp": attr.label(
                allow_single_file = True,
                doc = "Source compiled at each final binary or shared-library link.",
            ),
            "sanitizer_suppressions": attr.string_dict(
                configurable = False,
                doc = "Sanitizer suppression files owned by this runtime dependency.",
            ),
        },
    ),
)

def _iree_cc_binary_impl(
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
        linkshared,
        linkstatic,
        args,
        **kwargs):
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
    cc_attrs.add_if_not_none(target_attrs, "linkshared", linkshared)
    cc_binary(
        name = name,
        visibility = visibility,
        **target_attrs
    )

iree_cc_binary = macro(
    doc = """Defines a shared IREE C/C++ binary target.

    Binaries link statically by default so test and tool executables do not
    split runtime or test state across Bazel-generated shared objects. Dynamic
    linking should be a deliberate target-specific rule, not an accidental
    default inherited from rules_cc.
    """,
    implementation = _iree_cc_binary_impl,
    inherit_attrs = cc_binary,
    attrs = cc_attrs.merge_dicts(
        cc_attrs.compilation,
        cc_attrs.dependency,
        cc_attrs.binary_source,
        cc_attrs.link,
        {
            "additional_linker_inputs": attr.label_list(
                allow_files = True,
                doc = "Extra files made available to the link action, such as linker version scripts referenced from linkopts.",
            ),
            "args": attr.string_list(
                doc = "Command-line arguments used when this binary is run by Bazel.",
            ),
            "execution_data": None,
            "execution_deps": None,
            "linkshared": attr.bool(
                doc = "Whether to create a shared library.",
            ),
        },
    ),
)
