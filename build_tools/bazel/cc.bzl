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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(":cc_attrs.bzl", "cc_attrs")

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
        alwayslink,
        **kwargs):
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
        linkopts = linkopts,
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
    inherit_attrs = "common",
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
            "linkshared": attr.bool(
                doc = "Whether to create a shared library.",
            ),
        },
    ),
)
