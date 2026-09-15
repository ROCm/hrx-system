# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C and C++ attribute policy for libamdf-owned code."""

load("//build_tools/bazel:cc_opts.bzl", "cc_opts")

_AMDF_DEPS = [Label("//libamdf:headers")]

def _with_amdf_deps(deps):
    if deps == None:
        deps = []
    return deps + _AMDF_DEPS

def _with_amdf_compiler_options(copts, conlyopts, cxxopts):
    # Public declarations opt in to ELF visibility through AMDF_API.
    copts = (copts or []) + select({
        "@platforms//os:windows": [],
        "//conditions:default": ["-fvisibility=hidden"],
    })
    return cc_opts.iree_code_compiler_options(
        copts = copts,
        conlyopts = conlyopts,
        cxxopts = cxxopts,
    )

amdf_cc_attrs = struct(
    with_amdf_compiler_options = _with_amdf_compiler_options,
    with_amdf_deps = _with_amdf_deps,
)
