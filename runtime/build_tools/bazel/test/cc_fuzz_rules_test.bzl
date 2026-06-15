# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for runtime C/C++ fuzz Bazel macros."""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo", "util")
load("//runtime/build_tools/bazel:cc_fuzz.bzl", "iree_runtime_cc_fuzz")

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

def _test_runtime_fuzz_adds_runtime_include_root(name, **kwargs):
    util.helper_target(
        iree_runtime_cc_fuzz,
        name = name + "_subject",
        srcs = [name + "_subject.cc"],
        tags = ["requires-gpu-amd"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_runtime_fuzz_adds_runtime_include_root_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_runtime_fuzz_adds_runtime_include_root_impl(env, target):
    paths = _all_compilation_paths(target[CcInfo].compilation_context)
    _expect_path_suffix(env, paths, "runtime/src")
    tags = target[TestingAspectInfo].attrs.tags
    for expected_tag in [
        "requires-gpu-amd",
        "fuzz",
        "manual",
    ]:
        if expected_tag not in tags:
            env.fail("expected %r in fuzz tags %r" % (expected_tag, tags))

def cc_fuzz_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_runtime_fuzz_adds_runtime_include_root,
        ],
    )
