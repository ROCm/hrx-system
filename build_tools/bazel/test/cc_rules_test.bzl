# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for shared C/C++ Bazel macros."""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo", "util")
load("//build_tools/bazel:cc.bzl", "iree_cc_binary", "iree_cc_library")
load("//build_tools/bazel:cc_test.bzl", "iree_cc_test")

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

def _test_cc_library_preserves_system_include_inputs(name, **kwargs):
    util.helper_target(
        iree_cc_library,
        name = name + "_subject",
        hdrs = [name + "_subject.h"],
        includes = ["public_include"],
        system_includes = ["system_include"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_library_preserves_system_include_inputs_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_library_preserves_system_include_inputs_impl(env, target):
    paths = _all_compilation_paths(target[CcInfo].compilation_context)
    _expect_path_suffix(env, paths, "public_include")
    _expect_path_suffix(env, paths, "system_include")

def _test_cc_library_preserves_language_compile_options(name, **kwargs):
    util.helper_target(
        iree_cc_library,
        name = name + "_subject",
        conlyopts = ["-DUSER_CONLYOPT"] + select({
            "//conditions:default": ["-DUSER_SELECTED_CONLYOPT"],
        }),
        cxxopts = ["-DUSER_CXXOPT"] + select({
            "//conditions:default": ["-DUSER_SELECTED_CXXOPT"],
        }),
        srcs = [
            name + "_subject.c",
            name + "_subject.cc",
        ],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_library_preserves_language_compile_options_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_library_preserves_language_compile_options_impl(env, target):
    attrs = target[TestingAspectInfo].attrs
    _expect_value(env, attrs.conlyopts, "-DUSER_CONLYOPT")
    _expect_value(env, attrs.conlyopts, "-DUSER_SELECTED_CONLYOPT")
    _expect_value(env, attrs.cxxopts, "-DUSER_CXXOPT")
    _expect_value(env, attrs.cxxopts, "-DUSER_SELECTED_CXXOPT")

def _test_cc_binary_preserves_system_include_inputs(name, **kwargs):
    util.helper_target(
        iree_cc_binary,
        name = name + "_subject",
        srcs = [name + "_subject.cc"],
        includes = ["binary_include"],
        system_includes = ["binary_system_include"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_binary_preserves_system_include_inputs_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_binary_preserves_system_include_inputs_impl(env, target):
    paths = _all_compilation_paths(target[CcInfo].compilation_context)
    _expect_path_suffix(env, paths, "binary_include")
    _expect_path_suffix(env, paths, "binary_system_include")

def _test_cc_binary_preserves_shared_library_mode(name, **kwargs):
    util.helper_target(
        iree_cc_binary,
        name = name + "_subject",
        linkshared = True,
        srcs = [name + "_subject.cc"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_binary_preserves_shared_library_mode_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_binary_preserves_shared_library_mode_impl(env, target):
    attrs = target[TestingAspectInfo].attrs
    if not attrs.linkshared:
        env.fail("expected shared-library binary mode")

def _test_cc_binary_links_statically_by_default(name, **kwargs):
    util.helper_target(
        iree_cc_binary,
        name = name + "_subject",
        srcs = [name + "_subject.cc"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_binary_links_statically_by_default_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_binary_links_statically_by_default_impl(env, target):
    attrs = target[TestingAspectInfo].attrs
    if not attrs.linkstatic:
        env.fail("expected C/C++ binaries to link statically by default")

def _test_cc_test_preserves_system_include_inputs(name, **kwargs):
    util.helper_target(
        iree_cc_test,
        name = name + "_subject",
        srcs = [name + "_subject.cc"],
        includes = ["test_include"],
        system_includes = ["test_system_include"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_test_preserves_system_include_inputs_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_test_preserves_system_include_inputs_impl(env, target):
    paths = _all_compilation_paths(target[CcInfo].compilation_context)
    _expect_path_suffix(env, paths, "test_include")
    _expect_path_suffix(env, paths, "test_system_include")

def _test_cc_test_applies_resource_group_tags(name, **kwargs):
    util.helper_target(
        iree_cc_test,
        name = name + "_subject",
        srcs = [name + "_subject.cc"],
        resource_group = "gpu",
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
        impl = _test_cc_test_applies_resource_group_tags_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_test_applies_resource_group_tags_impl(env, target):
    tags = target[TestingAspectInfo].attrs.tags
    for expected_tag in [
        "requires-gpu-vulkan",
        "exclusive-if-local",
        "resource_group:gpu",
    ]:
        if expected_tag not in tags:
            env.fail("expected %r in test tags %r" % (expected_tag, tags))

def _test_cc_test_links_statically_by_default(name, **kwargs):
    util.helper_target(
        iree_cc_test,
        name = name + "_subject",
        srcs = [name + "_subject.cc"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_test_links_statically_by_default_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_test_links_statically_by_default_impl(env, target):
    attrs = target[TestingAspectInfo].attrs
    if not attrs.linkstatic:
        env.fail("expected C/C++ tests to link statically by default")

def cc_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_cc_library_preserves_system_include_inputs,
            _test_cc_library_preserves_language_compile_options,
            _test_cc_binary_preserves_system_include_inputs,
            _test_cc_binary_preserves_shared_library_mode,
            _test_cc_binary_links_statically_by_default,
            _test_cc_test_preserves_system_include_inputs,
            _test_cc_test_applies_resource_group_tags,
            _test_cc_test_links_statically_by_default,
        ],
    )
