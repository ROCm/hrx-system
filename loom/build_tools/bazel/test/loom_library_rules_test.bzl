# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for reusable Loom library rules."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo")
load(
    "//loom/build_tools/bazel:defs.bzl",
    "LoomLibraryInfo",
)

def _find_action(env, actions, mnemonic):
    for action in actions:
        if action.mnemonic == mnemonic:
            return action
    env.fail("expected action with mnemonic %r in %r" % (mnemonic, actions))
    return None

def _expect_basename(env, files, expected_basename):
    for file in files:
        if file.basename == expected_basename:
            return
    env.fail("expected basename %r in %r" % (expected_basename, files))

def _expect_no_basename(env, files, unexpected_basename):
    for file in files:
        if file.basename == unexpected_basename:
            env.fail("unexpected basename %r in %r" % (unexpected_basename, files))

def _expect_arg_with_prefix_and_suffix(env, args, prefix, suffix):
    for arg in args:
        if arg.startswith(prefix) and arg.endswith(suffix):
            return
    env.fail("expected argument with prefix %r and suffix %r in %r" % (prefix, suffix, args))

def _test_library_consumes_dependency_archive(name, **kwargs):
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_library_consumes_dependency_archive_impl,
        target = ":archive_consumer",
        **kwargs
    )

def _test_library_consumes_dependency_archive_impl(env, target):
    env.expect.that_str(target[LoomLibraryInfo].module.basename).equals(
        "archive_consumer.loombc",
    )

    action = _find_action(env, target[TestingAspectInfo].actions, "LoomLibrary")
    _expect_arg_with_prefix_and_suffix(
        env,
        action.argv,
        "--library=",
        "archive_dependency.loombc",
    )
    _expect_arg_with_prefix_and_suffix(
        env,
        action.argv,
        "",
        "link_checks.loom",
    )

    inputs = action.inputs.to_list()
    _expect_basename(env, inputs, "archive_dependency.loombc")
    _expect_basename(env, inputs, "link_checks.loom")
    _expect_no_basename(env, inputs, "link_kernels.loom")

def _test_deps_only_library_consumes_archive(name, **kwargs):
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_deps_only_library_consumes_archive_impl,
        target = ":archive_aggregate",
        **kwargs
    )

def _test_deps_only_library_consumes_archive_impl(env, target):
    action = _find_action(env, target[TestingAspectInfo].actions, "LoomLibrary")
    _expect_arg_with_prefix_and_suffix(
        env,
        action.argv,
        "--library=",
        "archive_consumer.loombc",
    )

    inputs = action.inputs.to_list()
    _expect_basename(env, inputs, "archive_consumer.loombc")
    _expect_no_basename(env, inputs, "link_checks.loom")
    _expect_no_basename(env, inputs, "link_kernels.loom")

def loom_library_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_deps_only_library_consumes_archive,
            _test_library_consumes_dependency_archive,
        ],
    )
