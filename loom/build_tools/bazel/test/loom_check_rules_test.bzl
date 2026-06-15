# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for loom-check Bazel rules."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo")
load(
    "//loom/build_tools/bazel:loom_check.bzl",
    "LoomCheckTestInfo",
    "loom_check_test",
)

def _find_action_with_output(env, actions, expected_basename):
    for action in actions:
        for output in action.outputs.to_list():
            if output.basename == expected_basename:
                return action
    env.fail("expected action output basename %r in %r" % (expected_basename, actions))
    return None

def _test_loom_check_wrapper_declares_fixture(name, **kwargs):
    loom_check_test(
        name = name + "_subject",
        env = {
            "LOOM_CHECK_FIXTURE": "$(location :roundtrip.loom-test)",
        },
        runner = "loom_check_fixture_runner.sh",
        src = "roundtrip.loom-test",
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_loom_check_wrapper_declares_fixture_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_loom_check_wrapper_declares_fixture_impl(env, target):
    info = target[LoomCheckTestInfo]
    if not str(info.runner).endswith("//loom/build_tools/bazel/test:loom_check_fixture_runner.sh"):
        env.fail("unexpected runner %s" % info.runner)
    env.expect.that_str(info.fixture.basename).equals("roundtrip.loom-test")
    env.expect.that_str(info.output.basename).equals(target.label.name)
    env.expect.that_str(info.env["LOOM_CHECK_FIXTURE"]).contains("roundtrip.loom-test")

    action = _find_action_with_output(
        env,
        target[TestingAspectInfo].actions,
        target.label.name,
    )
    env.expect.that_str(action.mnemonic).equals("FileWrite")

def loom_check_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_loom_check_wrapper_declares_fixture,
        ],
    )
