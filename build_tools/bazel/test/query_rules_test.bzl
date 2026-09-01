# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for configured dependency checks."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test")
load("//build_tools/bazel:cc.bzl", "iree_cc_library")
load(
    "//build_tools/bazel:query.bzl",
    "iree_assert_dependency_boundary",
    "iree_assert_no_dependency",
)

def _expect_boundary_failure_impl(env, target):
    result = target[AnalysisTestResultInfo]
    env.expect.that_bool(result.success).equals(False)
    env.expect.that_str(result.message).contains("query_fixture/child:child_leaf")

def _expect_boundary_failure(name, target):
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _expect_boundary_failure_impl,
        target = target,
    )

def query_rules_test_suite(name):
    """Defines tests for configured dependency query rules.

    Args:
      name: Test suite target name.
    """

    leaf_name = name + "_leaf"
    root_name = name + "_root"
    unrelated_name = name + "_unrelated"
    absent_test_name = name + "_allows_absent_dependency"
    missing_test_name = name + "_allows_missing_dependency_label"
    allowlisted_test_name = name + "_allows_exact_subtree_exception"
    boundary_assertion_name = name + "_rejects_forbidden_subtree_assertion"
    boundary_test_name = name + "_rejects_forbidden_subtree"

    iree_cc_library(
        name = leaf_name,
        tags = ["manual"],
    )
    iree_cc_library(
        name = unrelated_name,
        tags = ["manual"],
    )
    iree_cc_library(
        name = root_name,
        deps = [
            ":" + leaf_name,
            "//build_tools/bazel/test/query_fixture:root_leaf",
            "//build_tools/bazel/test/query_fixture/child:child_leaf",
        ],
        tags = ["manual"],
    )

    iree_assert_no_dependency(
        name = absent_test_name,
        dependency = ":" + unrelated_name,
        target = ":" + root_name,
    )

    iree_assert_no_dependency(
        name = missing_test_name,
        dependency = "//missing/package:dependency",
        target = ":" + root_name,
    )

    iree_assert_dependency_boundary(
        name = allowlisted_test_name,
        allowed_dependencies = [
            "//build_tools/bazel/test/query_fixture:root_leaf",
            "//build_tools/bazel/test/query_fixture/child:child_leaf",
        ],
        forbidden_subtrees = ["//build_tools/bazel/test/query_fixture/..."],
        target = ":" + root_name,
    )

    iree_assert_dependency_boundary(
        name = boundary_assertion_name,
        forbidden_subtrees = ["//build_tools/bazel/test/query_fixture/..."],
        tags = ["manual"],
        target = ":" + root_name,
    )
    _expect_boundary_failure(
        name = boundary_test_name,
        target = ":" + boundary_assertion_name,
    )

    native.test_suite(
        name = name,
        tests = [
            ":" + absent_test_name,
            ":" + allowlisted_test_name,
            ":" + boundary_test_name,
            ":" + missing_test_name,
        ],
    )
