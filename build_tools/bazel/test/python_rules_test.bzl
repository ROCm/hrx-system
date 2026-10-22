# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for shared Python Bazel macros."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo", "util")
load("//build_tools/bazel:python.bzl", "iree_py_binary", "iree_py_test")

def _explicit_packages_test(name, macro, **kwargs):
    util.helper_target(
        macro,
        name = name + "_subject",
        srcs = [name + "_subject.py"],
    )
    analysis_test(
        name = name,
        config_settings = {
            str(Label("@rules_python//python/config_settings:incompatible_default_to_explicit_init_py")): False,
        },
        impl = _explicit_packages_test_impl,
        target = name + "_subject",
        **kwargs
    )

def _explicit_packages_test_impl(env, target):
    env.expect.that_collection(target[DefaultInfo].default_runfiles.empty_filenames.to_list()).contains_exactly([])

def _test_py_binary_owns_explicit_packages(name, **kwargs):
    _explicit_packages_test(name, iree_py_binary, **kwargs)

def _test_py_test_owns_explicit_packages(name, **kwargs):
    _explicit_packages_test(name, iree_py_test, **kwargs)

def _test_py_test_applies_resource_group_tags(name, **kwargs):
    util.helper_target(
        iree_py_test,
        name = name + "_subject",
        srcs = [name + "_subject.py"],
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
        impl = _test_py_test_applies_resource_group_tags_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_py_test_applies_resource_group_tags_impl(env, target):
    tags = target[TestingAspectInfo].attrs.tags
    for expected_tag in [
        "requires-gpu-vulkan",
        "manual",
        "exclusive-if-local",
        "resource_group:gpu",
    ]:
        if expected_tag not in tags:
            env.fail("expected %r in Python test tags %r" % (expected_tag, tags))

def python_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_py_binary_owns_explicit_packages,
            _test_py_test_owns_explicit_packages,
            _test_py_test_applies_resource_group_tags,
        ],
    )
