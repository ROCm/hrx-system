# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for shared C/C++ benchmark Bazel macros."""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo", "util")
load("//build_tools/bazel:cc_benchmark.bzl", "iree_cc_benchmark")

def _all_compilation_paths(compilation_context):
    return [
        str(path)
        for path in (
            compilation_context.includes.to_list() +
            compilation_context.quote_includes.to_list() +
            compilation_context.system_includes.to_list()
        )
    ]

def _expect_path_suffix(env, paths, suffix):
    for path in paths:
        if path.endswith(suffix):
            return
    env.fail("expected one of %s to end with %r" % (paths, suffix))

def _test_cc_benchmark_binary_preserves_system_include_inputs(name, **kwargs):
    util.helper_target(
        iree_cc_benchmark,
        name = name + "_subject",
        srcs = [name + "_subject.cc"],
        includes = ["benchmark_include"],
        system_includes = ["benchmark_system_include"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_benchmark_binary_preserves_system_include_inputs_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_benchmark_binary_preserves_system_include_inputs_impl(env, target):
    paths = _all_compilation_paths(target[CcInfo].compilation_context)
    _expect_path_suffix(env, paths, "benchmark_include")
    _expect_path_suffix(env, paths, "benchmark_system_include")

def _test_cc_benchmark_smoke_test_args_and_tags(name, **kwargs):
    util.helper_target(
        iree_cc_benchmark,
        name = name + "_subject",
        args = ["--benchmark_filter=BM_Thing"],
        resource_group = "gpu",
        srcs = [name + "_subject.cc"],
        tags = [
            "requires-gpu-vulkan",
            "manual",
        ],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_benchmark_smoke_test_args_and_tags_impl,
        target = name + "_subject_test",
        **kwargs
    )

def _test_cc_benchmark_smoke_test_args_and_tags_impl(env, target):
    attrs = target[TestingAspectInfo].attrs
    expected_args = [
        "--benchmark_min_time=0s",
        "--benchmark_filter=BM_Thing",
    ]
    if attrs.args != expected_args:
        env.fail("expected smoke test args %r, got %r" % (expected_args, attrs.args))
    for expected_tag in [
        "requires-gpu-vulkan",
        "exclusive-if-local",
        "resource_group:gpu",
    ]:
        if expected_tag not in attrs.tags:
            env.fail("expected %r in smoke test tags %r" % (expected_tag, attrs.tags))

def cc_benchmark_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_cc_benchmark_binary_preserves_system_include_inputs,
            _test_cc_benchmark_smoke_test_args_and_tags,
        ],
    )
