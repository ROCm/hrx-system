# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime-specific production C/C++ Bazel macros.

These macros encode policy for targets under `//runtime`. Shared mechanics live
in `//build_tools/bazel:cc.bzl`; this file adds only runtime-specific
dependencies for production library and binary targets.
"""

load(
    "//build_tools/bazel:cc.bzl",
    "iree_cc_binary",
    "iree_cc_library",
)
load(
    "//runtime/requirements:package_policy.bzl",
    "apply_runtime_target_policy",
)
load(":cc_attrs.bzl", "runtime_cc_attrs")

def _iree_runtime_cc_library_impl(
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
    iree_cc_library(
        name = name,
        visibility = visibility,
        copts = compiler_options.copts,
        conlyopts = compiler_options.conlyopts,
        cxxopts = compiler_options.cxxopts,
        features = compiler_options.features,
        deps = runtime_cc_attrs.with_runtime_deps(deps),
        **kwargs
    )

iree_runtime_cc_library = macro(
    doc = """Defines a runtime C/C++ library target.

    Runtime libraries receive the public runtime include root through
    `//runtime/src:defines`. Callers should depend on this wrapper rather than
    the shared `iree_cc_library` macro unless they are defining the runtime
    root itself.
    """,
    implementation = _iree_runtime_cc_library_impl,
    inherit_attrs = iree_cc_library,
    attrs = {},
)

def _iree_runtime_cc_binary_impl(
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
    iree_cc_binary(
        name = name,
        visibility = visibility,
        copts = compiler_options.copts,
        conlyopts = compiler_options.conlyopts,
        cxxopts = compiler_options.cxxopts,
        features = compiler_options.features,
        deps = runtime_cc_attrs.with_runtime_deps(deps),
        **kwargs
    )

iree_runtime_cc_binary = macro(
    doc = """Defines a runtime C/C++ binary target.""",
    implementation = _iree_runtime_cc_binary_impl,
    inherit_attrs = iree_cc_binary,
    attrs = {},
)
