# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Production C/C++ Bazel macros for libamdf."""

load(
    "//build_tools/bazel:cc.bzl",
    "iree_cc_binary",
    "iree_cc_library",
)
load(
    "//libamdf/requirements:package_policy.bzl",
    "apply_amdf_target_policy",
)
load(":cc_attrs.bzl", "amdf_cc_attrs")

def _amdf_cc_library_impl(
        name,
        visibility,
        copts,
        conlyopts,
        cxxopts,
        deps,
        **kwargs):
    policy_kwargs = dict(kwargs)
    policy_kwargs["deps"] = [] if deps == None else deps
    kwargs = apply_amdf_target_policy(policy_kwargs, name = name)
    kwargs.pop("deps")
    compiler_options = amdf_cc_attrs.with_amdf_compiler_options(
        copts = copts,
        conlyopts = conlyopts,
        cxxopts = cxxopts,
    )
    iree_cc_library(
        name = name,
        visibility = visibility,
        copts = compiler_options.copts,
        conlyopts = compiler_options.conlyopts,
        cxxopts = compiler_options.cxxopts,
        deps = amdf_cc_attrs.with_amdf_deps(deps),
        **kwargs
    )

amdf_cc_library = macro(
    doc = "Defines a libamdf C/C++ library target.",
    implementation = _amdf_cc_library_impl,
    inherit_attrs = iree_cc_library,
    attrs = {},
)

def _amdf_cc_binary_impl(
        name,
        visibility,
        copts,
        conlyopts,
        cxxopts,
        deps,
        **kwargs):
    policy_kwargs = dict(kwargs)
    policy_kwargs["deps"] = [] if deps == None else deps
    kwargs = apply_amdf_target_policy(policy_kwargs, name = name)
    kwargs.pop("deps")
    compiler_options = amdf_cc_attrs.with_amdf_compiler_options(
        copts = copts,
        conlyopts = conlyopts,
        cxxopts = cxxopts,
    )
    iree_cc_binary(
        name = name,
        visibility = visibility,
        copts = compiler_options.copts,
        conlyopts = compiler_options.conlyopts,
        cxxopts = compiler_options.cxxopts,
        deps = amdf_cc_attrs.with_amdf_deps(deps),
        **kwargs
    )

amdf_cc_binary = macro(
    doc = "Defines a libamdf C/C++ executable target.",
    implementation = _amdf_cc_binary_impl,
    inherit_attrs = iree_cc_binary,
    attrs = {},
)
