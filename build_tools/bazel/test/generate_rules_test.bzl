# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for generated-file Bazel rules."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo", "util")
load("//build_tools/bazel:generate.bzl", "IreeGeneratedFilesInfo", "iree_generated_files")

def _expect_value(env, values, expected_value):
    if expected_value not in values:
        env.fail("expected %r in %r" % (expected_value, values))

def _expect_basename(env, files, expected_basename):
    for file in files:
        if file.basename == expected_basename:
            return
    env.fail("expected basename %r in %r" % (expected_basename, files))

def _expect_arg_with_prefix_and_suffix(env, args, prefix, suffix):
    for arg in args:
        if arg.startswith(prefix) and arg.endswith(suffix):
            return
    env.fail("expected argument with prefix %r and suffix %r in %r" % (prefix, suffix, args))

def _test_generated_files_declares_action_contract(name, **kwargs):
    util.helper_target(
        iree_generated_files,
        name = name + "_subject",
        srcs = [":generate_rule_fixture.in"],
        data = [":generate_rule_fixture.data"],
        outs = [
            name + "_generated.h",
            name + "_generated.c",
        ],
        args = [
            "--input",
            "$(location :generate_rule_fixture.in)",
            "--data=$(location :generate_rule_fixture.data)",
        ],
        mnemonic = "IreeGenerateTest",
        output_args = {
            name + "_generated.c": "--source",
            name + "_generated.h": "--header={path}",
        },
        progress_message = "Testing generated-file action %{label}",
        tags = ["manual"],
        tool = "//build_tools/bazel/test:generate_rule_fixture_tool",
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_generated_files_declares_action_contract_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_generated_files_declares_action_contract_impl(env, target):
    info = target[IreeGeneratedFilesInfo]
    _expect_basename(env, info.outputs.to_list(), "test_generated_files_declares_action_contract_generated.h")
    _expect_basename(env, info.outputs.to_list(), "test_generated_files_declares_action_contract_generated.c")
    _expect_basename(env, info.srcs.to_list(), "generate_rule_fixture.in")
    if not str(info.tool).endswith("//build_tools/bazel/test:generate_rule_fixture_tool"):
        env.fail("unexpected tool label %s" % info.tool)

    actions = target[TestingAspectInfo].actions
    env.expect.that_int(len(actions)).equals(1)
    action = actions[0]
    env.expect.that_str(action.mnemonic).equals("IreeGenerateTest")
    action_args = action.argv
    _expect_value(env, action_args, "--input")
    _expect_arg_with_prefix_and_suffix(env, action_args, "", "generate_rule_fixture.in")
    _expect_arg_with_prefix_and_suffix(env, action_args, "--data=", "generate_rule_fixture.data")
    _expect_arg_with_prefix_and_suffix(
        env,
        action_args,
        "--header=",
        "test_generated_files_declares_action_contract_generated.h",
    )
    _expect_value(env, action_args, "--source")
    _expect_arg_with_prefix_and_suffix(
        env,
        action_args,
        "",
        "test_generated_files_declares_action_contract_generated.c",
    )
    _expect_basename(env, action.inputs.to_list(), "generate_rule_fixture.in")
    _expect_basename(env, action.inputs.to_list(), "generate_rule_fixture.data")

def _test_generated_files_accepts_repeated_basenames(name, **kwargs):
    util.helper_target(
        iree_generated_files,
        name = name + "_subject",
        outs = [
            "first/shared.h",
            "second/shared.h",
        ],
        output_args = {
            "first/shared.h": "--first={path}",
            "second/shared.h": "--second={path}",
        },
        tags = ["manual"],
        tool = "//build_tools/bazel/test:generate_rule_fixture_tool",
    )
    analysis_test(
        name = name,
        impl = _test_generated_files_accepts_repeated_basenames_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_generated_files_accepts_repeated_basenames_impl(env, target):
    actions = target[TestingAspectInfo].actions
    env.expect.that_int(len(actions)).equals(1)
    action_args = actions[0].argv
    _expect_arg_with_prefix_and_suffix(
        env,
        action_args,
        "--first=",
        "first/shared.h",
    )
    _expect_arg_with_prefix_and_suffix(
        env,
        action_args,
        "--second=",
        "second/shared.h",
    )

def generate_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_generated_files_declares_action_contract,
            _test_generated_files_accepts_repeated_basenames,
        ],
    )
