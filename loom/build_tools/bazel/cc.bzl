# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom-specific production C/C++ Bazel macros."""

load(
    "//build_tools/bazel:cc.bzl",
    "iree_cc_binary",
    "iree_cc_library",
    "iree_cc_library_group",
)
load(
    "//loom/requirements:package_policy.bzl",
    "apply_loom_target_policy",
)
load(":cc_attrs.bzl", "loom_cc_attrs")

def _loom_cc_library_group_impl(name, visibility, **kwargs):
    iree_cc_library_group(
        name = name,
        visibility = visibility,
        **apply_loom_target_policy(kwargs, name = name)
    )

loom_cc_library_group = macro(
    doc = "Aggregates Loom library capabilities without adding build actions.",
    implementation = _loom_cc_library_group_impl,
    inherit_attrs = iree_cc_library_group,
)

def _loom_cc_library_impl(
        name,
        visibility,
        copts,
        conlyopts,
        cxxopts,
        cxx_standard,
        cxx_features,
        deps,
        **kwargs):
    policy_kwargs = dict(kwargs)
    policy_kwargs["deps"] = [] if deps == None else deps
    kwargs = apply_loom_target_policy(policy_kwargs, name = name)
    kwargs.pop("deps")
    compiler_options = loom_cc_attrs.with_loom_compiler_options(
        copts = copts,
        conlyopts = conlyopts,
        cxxopts = cxxopts,
        cxx_standard = cxx_standard,
        cxx_features = cxx_features,
        features = kwargs.pop("features", None),
    )
    iree_cc_library(
        name = name,
        visibility = visibility,
        copts = compiler_options.copts,
        conlyopts = compiler_options.conlyopts,
        cxxopts = compiler_options.cxxopts,
        features = compiler_options.features,
        deps = loom_cc_attrs.with_loom_deps(deps),
        **kwargs
    )

loom_cc_library = macro(
    doc = """Defines a Loom C/C++ library target.

    Loom libraries receive both the public runtime include root through
    `//runtime/src:defines` and the public Loom C include root through
    `//loom/src:defines`.
    """,
    implementation = _loom_cc_library_impl,
    inherit_attrs = iree_cc_library,
    attrs = loom_cc_attrs.cxx_attributes,
)

def _loom_cc_instance_library_impl(name, visibility, tags, **kwargs):
    loom_cc_library(
        name = name,
        visibility = visibility,
        tags = (tags or []) + ["LINKABLE_MORE_THAN_ONCE"],
        **kwargs
    )

loom_cc_instance_library = macro(
    doc = """Defines support code whose state and identity come from explicit owners.

    Persistent state belongs to caller-supplied instances. Identity-bearing
    metadata is resolved through those owners, not image-local definitions.
    Local immutable tables describe values; their addresses are not identities.
    Function addresses are not used as implementation identities either.

    These libraries may be linked privately into both an executable and its
    exact-build shared compiler. This declares only the named library's contract:
    dependencies retain their own ownership checks, and no symbols are exported.
    Compilation and target ownership are otherwise identical to loom_cc_library.
    """,
    implementation = _loom_cc_instance_library_impl,
    inherit_attrs = loom_cc_library,
)

def _loom_cc_binary_impl(
        name,
        visibility,
        copts,
        conlyopts,
        cxxopts,
        deps,
        **kwargs):
    policy_kwargs = dict(kwargs)
    policy_kwargs["deps"] = [] if deps == None else deps
    kwargs = apply_loom_target_policy(policy_kwargs, name = name)
    kwargs.pop("deps")
    compiler_options = loom_cc_attrs.with_loom_compiler_options(
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
        deps = loom_cc_attrs.with_loom_deps(deps),
        **kwargs
    )

loom_cc_binary = macro(
    doc = """Defines a Loom C/C++ binary target.""",
    implementation = _loom_cc_binary_impl,
    inherit_attrs = iree_cc_binary,
    attrs = {},
)
