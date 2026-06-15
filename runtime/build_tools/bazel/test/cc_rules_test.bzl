# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for runtime C/C++ Bazel macros."""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo", "util")
load("//runtime/build_tools/bazel:cc.bzl", "iree_runtime_cc_binary", "iree_runtime_cc_library")
load("//runtime/build_tools/bazel:cc_test.bzl", "iree_runtime_cc_test")

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

def _expect_value(env, values, expected_value):
    if expected_value not in values:
        env.fail("expected %r in %r" % (expected_value, values))

def _expect_no_value(env, values, unexpected_value):
    if unexpected_value in values:
        env.fail("did not expect %r in %r" % (unexpected_value, values))

def _test_runtime_library_adds_runtime_include_root(name, **kwargs):
    util.helper_target(
        iree_runtime_cc_library,
        name = name + "_subject",
        hdrs = [name + "_subject.h"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_runtime_library_adds_runtime_include_root_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_runtime_library_adds_runtime_include_root_impl(env, target):
    paths = _all_compilation_paths(target[CcInfo].compilation_context)
    _expect_path_suffix(env, paths, "runtime/src")

def _test_runtime_binary_adds_runtime_include_root(name, **kwargs):
    util.helper_target(
        iree_runtime_cc_binary,
        name = name + "_subject",
        srcs = [name + "_subject.cc"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_runtime_binary_adds_runtime_include_root_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_runtime_binary_adds_runtime_include_root_impl(env, target):
    paths = _all_compilation_paths(target[CcInfo].compilation_context)
    _expect_path_suffix(env, paths, "runtime/src")

def _test_runtime_c_library_applies_c_options(name, **kwargs):
    util.helper_target(
        iree_runtime_cc_library,
        name = name + "_subject",
        copts = ["-DUSER_COPT"] + select({
            "//conditions:default": ["-DUSER_SELECTED_COPT"],
        }),
        srcs = [name + "_subject.c"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_runtime_c_library_applies_c_options_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_runtime_c_library_applies_c_options_impl(env, target):
    copts = target[TestingAspectInfo].attrs.copts
    cxxopts = target[TestingAspectInfo].attrs.cxxopts
    _expect_value(env, copts, "-Wall")
    _expect_value(env, copts, "-Werror")
    _expect_value(env, copts, "-Wno-unused-function")
    _expect_value(env, copts, "-DUSER_COPT")
    _expect_value(env, copts, "-DUSER_SELECTED_COPT")
    _expect_no_value(env, copts, "-Wno-invalid-offsetof")
    _expect_value(env, cxxopts, "-Wno-invalid-offsetof")
    _expect_value(env, cxxopts, "-std=c++17")

def _test_runtime_cxx_binary_applies_cxx_options(name, **kwargs):
    util.helper_target(
        iree_runtime_cc_binary,
        name = name + "_subject",
        copts = ["-DUSER_COPT"] + select({
            "//conditions:default": ["-DUSER_SELECTED_COPT"],
        }),
        srcs = [name + "_subject.cc"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_runtime_cxx_binary_applies_cxx_options_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_runtime_cxx_binary_applies_cxx_options_impl(env, target):
    copts = target[TestingAspectInfo].attrs.copts
    cxxopts = target[TestingAspectInfo].attrs.cxxopts
    _expect_value(env, copts, "-Wall")
    _expect_value(env, copts, "-Werror")
    _expect_value(env, copts, "-Wno-unused-function")
    _expect_value(env, copts, "-DUSER_COPT")
    _expect_value(env, copts, "-DUSER_SELECTED_COPT")
    _expect_no_value(env, copts, "-Wno-invalid-offsetof")
    _expect_value(env, cxxopts, "-Wno-invalid-offsetof")
    _expect_value(env, cxxopts, "-std=c++17")

def _test_runtime_library_allows_configurable_srcs(name, **kwargs):
    util.helper_target(
        iree_runtime_cc_library,
        name = name + "_subject",
        srcs = [name + "_subject.c"] + select({
            "//conditions:default": [name + "_configured.c"],
        }),
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_runtime_library_allows_configurable_srcs_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_runtime_library_allows_configurable_srcs_impl(env, target):
    copts = target[TestingAspectInfo].attrs.copts
    _expect_value(env, copts, "-Wall")

def _test_runtime_test_adds_runtime_include_root(name, **kwargs):
    util.helper_target(
        iree_runtime_cc_test,
        name = name + "_subject",
        srcs = [name + "_subject.cc"],
        resource_group = "gpu",
        tags = [
            "requires-gpu-amd",
            "manual",
        ],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_runtime_test_adds_runtime_include_root_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_runtime_test_adds_runtime_include_root_impl(env, target):
    paths = _all_compilation_paths(target[CcInfo].compilation_context)
    _expect_path_suffix(env, paths, "runtime/src")
    tags = target[TestingAspectInfo].attrs.tags
    for expected_tag in [
        "requires-gpu-amd",
        "exclusive-if-local",
        "resource_group:gpu",
    ]:
        if expected_tag not in tags:
            env.fail("expected %r in test tags %r" % (expected_tag, tags))

def cc_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_runtime_library_adds_runtime_include_root,
            _test_runtime_binary_adds_runtime_include_root,
            _test_runtime_c_library_applies_c_options,
            _test_runtime_cxx_binary_applies_cxx_options,
            _test_runtime_library_allows_configurable_srcs,
            _test_runtime_test_adds_runtime_include_root,
        ],
    )
