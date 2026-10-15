# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared C/C++ attribute helpers for IREE Bazel macros.

This file owns implementation mechanics only. User-facing policy belongs in the
macro files that define concrete target kinds, such as `cc.bzl`,
`cc_test.bzl`, `cc_benchmark.bzl`, and `cc_fuzz.bzl`.
"""

load(
    "//build_tools/bazel:test_resources.bzl",
    _with_resource_group_tags = "with_resource_group_tags",
)

def _merge_string_lists(first, second):
    if first == None:
        first = []
    if second == None:
        second = []
    return first + second

_COMPILATION_ATTRS = {
    "conlyopts": attr.string_list(
        doc = "Additional compiler options used only for C compile actions in this target.",
    ),
    "copts": attr.string_list(
        doc = "Additional compiler options for this target.",
    ),
    "cxxopts": attr.string_list(
        doc = "Additional compiler options used only for C++ compile actions in this target.",
    ),
    "defines": attr.string_list(
        doc = "Preprocessor definitions propagated to dependents.",
    ),
    "includes": attr.string_list(
        doc = "Include directories propagated to dependents.",
    ),
    "local_defines": attr.string_list(
        doc = "Preprocessor definitions used only while compiling this target.",
    ),
    "system_includes": attr.string_list(
        doc = """Include directories that should be system includes outside Bazel.

        Bazel's C/C++ rules do not expose a separate public system include
        attribute, so these paths are forwarded through `includes`. Other
        generators can preserve the caller's distinction.
        """,
    ),
}

_LINK_ATTRS = {
    "linkopts": attr.string_list(
        doc = "Additional linker options for this target.",
    ),
    "linkstatic": attr.bool(
        doc = "Whether to force static linking for this target.",
    ),
}

_DEPENDENCY_ATTRS = {
    "data": attr.label_list(
        allow_files = True,
        doc = "Runtime data dependencies for this target.",
    ),
    "deps": attr.label_list(
        doc = "C/C++ library dependencies for this target.",
    ),
}

_LIBRARY_SOURCE_ATTRS = {
    "hdrs": attr.label_list(
        allow_files = True,
        doc = "Public headers provided by this target.",
    ),
    "srcs": attr.label_list(
        allow_files = True,
        doc = "Source files compiled by this target.",
    ),
    "textual_hdrs": attr.label_list(
        allow_files = True,
        doc = "Headers that are textually included and not compiled independently.",
    ),
}

_BINARY_SOURCE_ATTRS = {
    "srcs": attr.label_list(
        allow_files = True,
        doc = "Source files compiled by this target.",
    ),
}

def _add_if_not_none(target, name, value):
    if value != None:
        target[name] = value

def _merge_dicts(*dicts):
    result = {}
    for source in dicts:
        result.update(source)
    return result

def _collect(
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
        linkstatic):
    result = {
        "includes": _merge_string_lists(includes, system_includes),
    }
    _add_if_not_none(result, "srcs", srcs)
    _add_if_not_none(result, "hdrs", hdrs)
    _add_if_not_none(result, "textual_hdrs", textual_hdrs)
    _add_if_not_none(result, "deps", deps)
    _add_if_not_none(result, "data", data)
    _add_if_not_none(result, "copts", copts)
    _add_if_not_none(result, "conlyopts", conlyopts)
    _add_if_not_none(result, "cxxopts", cxxopts)
    _add_if_not_none(result, "defines", defines)
    _add_if_not_none(result, "local_defines", local_defines)
    _add_if_not_none(result, "linkopts", linkopts)
    if linkstatic:
        result["linkstatic"] = linkstatic
    return result

cc_attrs = struct(
    add_if_not_none = _add_if_not_none,
    binary_source = _BINARY_SOURCE_ATTRS,
    collect = _collect,
    compilation = _COMPILATION_ATTRS,
    dependency = _DEPENDENCY_ATTRS,
    library_source = _LIBRARY_SOURCE_ATTRS,
    link = _LINK_ATTRS,
    merge_dicts = _merge_dicts,
    with_resource_group_tags = _with_resource_group_tags,
)
