# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for shared Python Bazel macros."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo", "util")
load("//build_tools/bazel:python.bzl", "iree_py_test")

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
        tests = [_test_py_test_applies_resource_group_tags],
    )
