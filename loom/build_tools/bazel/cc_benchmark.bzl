# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom-specific C/C++ Bazel benchmark macros."""

load("//build_tools/bazel:cc_benchmark.bzl", "iree_cc_benchmark")
load(
    "//loom/requirements:package_policy.bzl",
    "apply_loom_test_policy",
)
load(":cc_attrs.bzl", "loom_cc_attrs")

_GOOGLE_BENCHMARK_DEP = Label("//third_party:google_benchmark")

def _loom_cc_benchmark_impl(
        name,
        visibility,
        copts,
        conlyopts,
        cxxopts,
        deps,
        **kwargs):
    policy_kwargs = dict(kwargs)
    policy_kwargs["deps"] = [] if deps == None else deps
    kwargs = apply_loom_test_policy(policy_kwargs, name = name)
    kwargs.pop("deps")
    if deps == None:
        deps = []
    compiler_options = loom_cc_attrs.with_loom_compiler_options(
        copts = copts,
        conlyopts = conlyopts,
        cxxopts = cxxopts,
    )
    iree_cc_benchmark(
        name = name,
        visibility = visibility,
        copts = compiler_options.copts,
        conlyopts = compiler_options.conlyopts,
        cxxopts = compiler_options.cxxopts,
        deps = loom_cc_attrs.with_loom_deps(deps + [_GOOGLE_BENCHMARK_DEP]),
        **kwargs
    )

loom_cc_benchmark = macro(
    doc = """Defines a Loom C/C++ benchmark binary and smoke test.""",
    implementation = _loom_cc_benchmark_impl,
    inherit_attrs = iree_cc_benchmark,
    attrs = {},
)
