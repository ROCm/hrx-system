# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime-specific C/C++ Bazel fuzz macros."""

load("//build_tools/bazel:cc_fuzz.bzl", "iree_cc_fuzz")
load(
    "//runtime/requirements:package_policy.bzl",
    "apply_runtime_target_policy",
)
load(":cc_attrs.bzl", "runtime_cc_attrs")

def _iree_runtime_cc_fuzz_impl(
        name,
        visibility,
        copts,
        conlyopts,
        cxxopts,
        deps,
        **kwargs):
    kwargs = apply_runtime_target_policy(kwargs)
    compiler_options = runtime_cc_attrs.with_runtime_compiler_options(
        copts = copts,
        conlyopts = conlyopts,
        cxxopts = cxxopts,
        features = kwargs.pop("features", None),
    )
    iree_cc_fuzz(
        name = name,
        visibility = visibility,
        copts = compiler_options.copts,
        conlyopts = compiler_options.conlyopts,
        cxxopts = compiler_options.cxxopts,
        features = compiler_options.features,
        deps = runtime_cc_attrs.with_runtime_deps(deps),
        **kwargs
    )

iree_runtime_cc_fuzz = macro(
    doc = """Defines a runtime libFuzzer C/C++ binary target.""",
    implementation = _iree_runtime_cc_fuzz_impl,
    inherit_attrs = iree_cc_fuzz,
    attrs = {},
)
