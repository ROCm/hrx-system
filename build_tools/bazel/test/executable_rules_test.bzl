# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for executable wrapper Bazel rules."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo")
load(
    "//build_tools/bazel:executable.bzl",
    "IreeExecutableInfo",
    "iree_executable_alias",
    "iree_executable_test",
)
load(
    "//build_tools/bazel:runfiles.bzl",
    "IreeRunfilesArgumentsInfo",
    "RUNFILES_PATH_BEGIN",
    "RUNFILES_PATH_END",
)

def _expect_basename(env, files, expected_basename):
    for file in files:
        if file.basename == expected_basename:
            return
    env.fail("expected basename %r in %r" % (expected_basename, files))

def _find_action_with_output(env, actions, expected_basename):
    for action in actions:
        for output in action.outputs.to_list():
            if output.basename == expected_basename:
                return action
    env.fail("expected action output basename %r in %r" % (expected_basename, actions))
    return None

def _test_executable_alias_wraps_source(name, **kwargs):
    iree_executable_alias(
        name = name + "_subject",
        data = [":generate_rule_fixture.data"],
        src = ":generate_rule_fixture_tool",
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_executable_wrapper_contract_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_executable_test_wraps_source(name, **kwargs):
    iree_executable_test(
        name = name + "_subject",
        args = [
            "--smoke",
            "--fixture=$(rootpath :generate_rule_fixture.data)",
        ],
        data = [":generate_rule_fixture.data"],
        env = {
            "IREE_FIXTURE": "$(location :generate_rule_fixture.data)",
        },
        src = ":generate_rule_fixture_tool",
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_executable_wrapper_contract_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_executable_wrapper_contract_impl(env, target):
    info = target[IreeExecutableInfo]
    if not str(info.src).endswith("//build_tools/bazel/test:generate_rule_fixture_tool"):
        env.fail("unexpected source executable %s" % info.src)
    env.expect.that_str(info.output.basename).equals(target.label.name)
    _expect_basename(env, info.data.to_list(), "generate_rule_fixture.data")
    if target.label.name.endswith("_test_wraps_source_subject"):
        env.expect.that_str(info.env["IREE_FIXTURE"]).contains("generate_rule_fixture.data")
        runfiles_arguments = target[IreeRunfilesArgumentsInfo]
        env.expect.that_collection(runfiles_arguments.arguments).contains_exactly([
            "--smoke",
            "--fixture=build_tools/bazel/test/generate_rule_fixture.data",
        ]).in_order()
        env.expect.that_collection(runfiles_arguments.marked_arguments).contains_exactly([
            "--smoke",
            "--fixture=" +
            RUNFILES_PATH_BEGIN +
            "build_tools/bazel/test/generate_rule_fixture.data" +
            RUNFILES_PATH_END,
        ]).in_order()
    elif info.env:
        env.fail("expected no binary alias environment, got %r" % info.env)

    action = _find_action_with_output(
        env,
        target[TestingAspectInfo].actions,
        target.label.name,
    )
    env.expect.that_str(action.mnemonic).equals("ExecutableSymlink")

def executable_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_executable_alias_wraps_source,
            _test_executable_test_wraps_source,
        ],
    )
