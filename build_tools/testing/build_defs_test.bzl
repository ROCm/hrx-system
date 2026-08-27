# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for shared execution-test Bazel macros."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo", "util")
load(":build_defs.bzl", "iree_execution_test_suite")

def _test_execution_test_suite_applies_resource_group_tags(name, **kwargs):
    util.helper_target(
        iree_execution_test_suite,
        name = name + "_subject",
        manifests = ["//build_tools/testing/test:smoke.test.json"],
        resource_group = "gpu",
        tags = ["manual", "user-supplied-tag"],
        tools = {
            "fixture": "//build_tools/testing/test:fixture_tool",
        },
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_execution_test_suite_applies_resource_group_tags_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_execution_test_suite_applies_resource_group_tags_impl(env, target):
    tags = target[TestingAspectInfo].attrs.tags
    for expected_tag in [
        "exclusive-if-local",
        "resource_group:gpu",
        "user-supplied-tag",
    ]:
        if expected_tag not in tags:
            env.fail("expected %r in test tags %r" % (expected_tag, tags))

def build_defs_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_execution_test_suite_applies_resource_group_tags,
        ],
    )
