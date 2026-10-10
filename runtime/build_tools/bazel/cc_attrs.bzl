# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime-specific C/C++ attribute helpers."""

load("//build_tools/bazel:cc_opts.bzl", "cc_opts")

_RUNTIME_DEFINES_DEP = Label("//runtime/src:defines")

def _with_runtime_deps(deps):
    if deps == None:
        deps = []
    return deps + [_RUNTIME_DEFINES_DEP]

def _with_runtime_compiler_options(copts, conlyopts, cxxopts, features = None):
    return cc_opts.iree_code_compiler_options(
        copts = copts,
        conlyopts = conlyopts,
        cxxopts = cxxopts,
        features = features,
    )

runtime_cc_attrs = struct(
    with_runtime_compiler_options = _with_runtime_compiler_options,
    with_runtime_deps = _with_runtime_deps,
)
