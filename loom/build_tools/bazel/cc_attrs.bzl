# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom-specific C/C++ attribute helpers."""

load("//build_tools/bazel:cc_opts.bzl", "cc_opts")

_LOOM_DEPS = [
    Label("//runtime/src:defines"),
    Label("//loom/src:defines"),
]

_CXX_ATTRIBUTES = {
    "cxx_features": attr.string_list(
        configurable = False,
        doc = "Optional C++ runtime features: exceptions and rtti.",
    ),
    "cxx_standard": attr.string(
        configurable = False,
        default = "c++20",
        values = ["c++17", "c++20", "c++23"],
        doc = "C++ language standard for this target.",
    ),
}

def _with_loom_deps(deps):
    if deps == None:
        deps = []
    return deps + _LOOM_DEPS

def _with_loom_compiler_options(
        copts,
        conlyopts,
        cxxopts,
        cxx_standard = "c++20",
        cxx_features = None,
        features = None):
    return cc_opts.iree_code_compiler_options(
        copts = copts,
        conlyopts = conlyopts,
        cxxopts = cxxopts,
        cxx_standard = cxx_standard,
        cxx_features = cxx_features,
        features = features,
    )

loom_cc_attrs = struct(
    cxx_attributes = _CXX_ATTRIBUTES,
    with_loom_compiler_options = _with_loom_compiler_options,
    with_loom_deps = _with_loom_deps,
)
