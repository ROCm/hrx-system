# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""HRX-specific C/C++ attribute helpers."""

load("//build_tools/bazel:cc_opts.bzl", "cc_opts")

_HRX_DEPS = [
    Label("//runtime/src:defines"),
    Label("//libhrx:defines"),
]

def _with_hrx_deps(deps):
    if deps == None:
        deps = []
    return deps + _HRX_DEPS

def _with_hrx_compiler_options(copts, conlyopts, cxxopts, features = None):
    return cc_opts.iree_code_compiler_options(
        copts = copts,
        conlyopts = conlyopts,
        cxxopts = cxxopts,
        features = features,
    )

hrx_cc_attrs = struct(
    with_hrx_compiler_options = _with_hrx_compiler_options,
    with_hrx_deps = _with_hrx_deps,
)
