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

def _with_loom_deps(deps):
    if deps == None:
        deps = []
    return deps + _LOOM_DEPS

def _with_loom_compiler_options(copts, conlyopts, cxxopts):
    return cc_opts.iree_code_compiler_options(
        copts = copts,
        conlyopts = conlyopts,
        cxxopts = cxxopts,
    )

loom_cc_attrs = struct(
    with_loom_compiler_options = _with_loom_compiler_options,
    with_loom_deps = _with_loom_deps,
)
