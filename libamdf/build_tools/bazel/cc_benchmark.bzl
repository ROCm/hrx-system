# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""libamdf benchmarks using the shared binary and hardware smoke-test rules."""

load("//build_tools/bazel:cc_benchmark.bzl", "iree_cc_benchmark")
load("//libamdf/requirements:package_policy.bzl", "apply_amdf_test_policy")
load(":cc_attrs.bzl", "amdf_cc_attrs")

def _amdf_cc_benchmark_impl(name, visibility, copts, conlyopts, cxxopts, deps, **kwargs):
    policy_kwargs = dict(kwargs)
    policy_kwargs["deps"] = deps or []
    kwargs = apply_amdf_test_policy(policy_kwargs, name = name)
    kwargs.pop("deps")
    compiler_options = amdf_cc_attrs.with_amdf_compiler_options(
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
        deps = amdf_cc_attrs.with_amdf_deps(deps) + ["//third_party:google_benchmark"],
        **kwargs
    )

amdf_cc_benchmark = macro(
    doc = "Defines a public-API benchmark and resource-qualified smoke test.",
    implementation = _amdf_cc_benchmark_impl,
    inherit_attrs = iree_cc_benchmark,
    attrs = {},
)
