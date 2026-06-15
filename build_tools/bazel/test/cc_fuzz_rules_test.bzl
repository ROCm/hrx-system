# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for shared C/C++ fuzz Bazel macros."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo", "util")
load("//build_tools/bazel:cc_fuzz.bzl", "iree_cc_fuzz")

def _test_cc_fuzz_adds_fuzzer_contract(name, **kwargs):
    util.helper_target(
        iree_cc_fuzz,
        name = name + "_subject",
        defines = ["USER_DEFINE"],
        linkopts = ["-Wl,--user-linkopt"],
        srcs = [name + "_subject.cc"],
        tags = ["requires-gpu-vulkan"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_fuzz_adds_fuzzer_contract_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_fuzz_adds_fuzzer_contract_impl(env, target):
    attrs = target[TestingAspectInfo].attrs
    for expected_define in [
        "USER_DEFINE",
        "FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION",
    ]:
        if expected_define not in attrs.defines:
            env.fail("expected %r in fuzz defines %r" % (expected_define, attrs.defines))
    for expected_linkopt in [
        "-Wl,--user-linkopt",
        "-fsanitize=fuzzer",
    ]:
        if expected_linkopt not in attrs.linkopts:
            env.fail("expected %r in fuzz linkopts %r" % (expected_linkopt, attrs.linkopts))
    for expected_tag in [
        "requires-gpu-vulkan",
        "fuzz",
        "manual",
    ]:
        if expected_tag not in attrs.tags:
            env.fail("expected %r in fuzz tags %r" % (expected_tag, attrs.tags))
    if not attrs.testonly:
        env.fail("expected fuzz target to be testonly")

def cc_fuzz_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_cc_fuzz_adds_fuzzer_contract,
        ],
    )
