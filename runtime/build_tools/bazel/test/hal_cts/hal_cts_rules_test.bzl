# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for runtime HAL CTS Bazel macros."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo")
load("//build_tools/bazel:executable.bzl", "IreeExecutableInfo")
load("//runtime/build_tools/bazel:hal_cts.bzl", "iree_runtime_hal_cts_test_suite")

def _expect_value(env, values, expected_value):
    if expected_value not in values:
        env.fail("expected %r in %r" % (expected_value, values))

def _label_string(label):
    if label.package:
        return "//%s:%s" % (label.package, label.name)
    return "//:%s" % label.name

def _expect_label(env, labels, expected_label):
    if expected_label not in [_label_string(label) for label in labels]:
        env.fail("expected label %r in %r" % (expected_label, labels))

def _test_non_executable_suite_generates_wrapped_test(name, **kwargs):
    iree_runtime_hal_cts_test_suite(
        name = name + "_subject",
        args = ["--cts_fixture_flag=true"],
        backends = ":hal_cts_test_backends",
        env = {
            "IREE_HAL_CTS_FIXTURE": "1",
        },
        local = True,
        resource_group = "gpu",
        size = "medium",
        tags = [
            "fixture-driver",
            "manual",
        ],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_non_executable_suite_generates_wrapped_test_impl,
        target = name + "_subject_core_tests",
        **kwargs
    )

def _test_non_executable_suite_generates_wrapped_test_impl(env, target):
    executable_info = target[IreeExecutableInfo]
    env.expect.that_str(executable_info.output.basename).equals(target.label.name)
    env.expect.that_str(_label_string(executable_info.src)).equals(
        "//runtime/build_tools/bazel/test/hal_cts:%s_bin" % target.label.name,
    )
    env.expect.that_str(executable_info.env["IREE_HAL_CTS_FIXTURE"]).equals("1")

    attrs = target[TestingAspectInfo].attrs
    _expect_value(env, attrs.args, "--cts_fixture_flag=true")
    _expect_value(env, attrs.tags, "fixture-driver")
    _expect_value(env, attrs.tags, "exclusive-if-local")
    _expect_value(env, attrs.tags, "resource_group:gpu")
    env.expect.that_bool(attrs.local).equals(True)
    env.expect.that_str(attrs.size).equals("medium")

def _test_non_executable_suite_links_base_cts_deps(name, **kwargs):
    iree_runtime_hal_cts_test_suite(
        name = name + "_subject",
        backends = ":hal_cts_test_backends",
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_non_executable_suite_links_base_cts_deps_impl,
        target = name + "_subject_core_tests_bin",
        **kwargs
    )

def _test_non_executable_suite_links_base_cts_deps_impl(env, target):
    deps = [dep.label for dep in target[TestingAspectInfo].attrs.deps]
    _expect_label(env, deps, "//runtime/build_tools/bazel/test/hal_cts:hal_cts_test_backends")
    _expect_label(env, deps, "//runtime/src/iree/hal/cts/core:all_tests")
    _expect_label(env, deps, "//runtime/src/iree/hal/cts/util:registry")

def _test_executable_suite_links_prebuilt_testdata(name, **kwargs):
    iree_runtime_hal_cts_test_suite(
        name = name + "_subject",
        backends = ":hal_cts_test_backends",
        tags = ["manual"],
        testdata_libs = [":hal_cts_testdata"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_executable_suite_links_prebuilt_testdata_impl,
        target = name + "_subject_dispatch_tests_bin",
        **kwargs
    )

def _test_executable_suite_links_prebuilt_testdata_impl(env, target):
    deps = [dep.label for dep in target[TestingAspectInfo].attrs.deps]
    _expect_label(env, deps, "//runtime/build_tools/bazel/test/hal_cts:hal_cts_testdata")
    _expect_label(
        env,
        deps,
        "//runtime/src/iree/hal/cts/command_buffer:all_dispatch_tests",
    )

def hal_cts_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_non_executable_suite_generates_wrapped_test,
            _test_non_executable_suite_links_base_cts_deps,
            _test_executable_suite_links_prebuilt_testdata,
        ],
    )
