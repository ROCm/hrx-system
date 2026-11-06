# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared C/C++ Bazel fuzz target macros for IREE repositories."""

load(":cc.bzl", "iree_cc_binary")
load(":cc_attrs.bzl", "cc_attrs")

_FUZZ_DEFINE = "FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION"
_FUZZ_LINKOPT = "-fsanitize=fuzzer"
_FUZZ_TAGS = [
    "fuzz",
    "manual",
]

def _fuzzing_engine_linkopts():
    return select({
        "//build_tools/bazel:external_fuzzing_engine_enabled": [],
        "//conditions:default": [_FUZZ_LINKOPT],
    })

def _append(values, appended_values):
    if values == None:
        values = []
    return values + appended_values

def _iree_cc_fuzz_impl(
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
        tags,
        testonly,
        **kwargs):
    iree_cc_binary(
        name = name,
        visibility = visibility,
        srcs = srcs,
        deps = deps,
        data = data,
        copts = copts,
        conlyopts = conlyopts,
        cxxopts = cxxopts,
        defines = _append(defines, [_FUZZ_DEFINE]),
        local_defines = local_defines,
        includes = includes,
        system_includes = system_includes,
        linkopts = _append(linkopts, _fuzzing_engine_linkopts()),
        linkstatic = linkstatic,
        args = args,
        tags = _append(tags, _FUZZ_TAGS),
        testonly = testonly,
        **kwargs
    )

iree_cc_fuzz = macro(
    doc = """Defines a libFuzzer C/C++ binary target.

    Fuzz targets are ordinary binaries whose `main` comes from libFuzzer via
    `-fsanitize=fuzzer` by default. An enclosing fuzz service may select
    `//build_tools/bazel:external_fuzzing_engine` and supply its engine through
    the build link options. Targets are tagged `manual` so wildcard builds do
    not require a fuzz-capable C/C++ toolchain; build them explicitly with an
    appropriate compiler configuration.
    """,
    implementation = _iree_cc_fuzz_impl,
    inherit_attrs = "common",
    attrs = cc_attrs.merge_dicts(
        cc_attrs.compilation,
        cc_attrs.dependency,
        cc_attrs.binary_source,
        cc_attrs.link,
        {
            "args": attr.string_list(
                doc = "Command-line arguments used when this fuzzer is run by Bazel.",
            ),
            "testonly": attr.bool(
                configurable = False,
                default = True,
                doc = "Whether the fuzz binary is only available to test targets.",
            ),
        },
    ),
)
