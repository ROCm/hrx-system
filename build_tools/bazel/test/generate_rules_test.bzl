# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for generated-file Bazel rules."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:truth.bzl", "matching")
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
            "nested/" + name + "_generated.h",
            "nested/" + name + "_generated.c",
        ],
        output_directories = ["nested/" + name + "_assets"],
        args = [
            "--input",
            "$(location :generate_rule_fixture.in)",
            "--data=$(location :generate_rule_fixture.data)",
        ],
        mnemonic = "IreeGenerateTest",
        output_args = {
            "nested/" + name + "_assets": "--assets={path}",
            # A unique basename remains valid shorthand for a nested output.
            name + "_generated.c": "--source",
            "nested/" + name + "_generated.h": "--header={path}",
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

    directories = [output for output in info.outputs.to_list() if output.is_directory]
    env.expect.that_int(len(directories)).equals(1)
    env.expect.that_bool(directories[0].is_directory).equals(True)
    _expect_basename(env, directories, "test_generated_files_declares_action_contract_assets")

    actions = target[TestingAspectInfo].actions
    env.expect.that_int(len(actions)).equals(1)
    action = actions[0]
    env.expect.that_str(action.mnemonic).equals("IreeGenerateTest")
    action_args = action.argv
    _expect_arg_with_prefix_and_suffix(env, action_args, "--assets=", "nested/test_generated_files_declares_action_contract_assets")
    _expect_value(env, action_args, "--input")
    _expect_arg_with_prefix_and_suffix(env, action_args, "", "generate_rule_fixture.in")
    _expect_arg_with_prefix_and_suffix(env, action_args, "--data=", "generate_rule_fixture.data")
    _expect_arg_with_prefix_and_suffix(
        env,
        action_args,
        "--header=",
        "nested/test_generated_files_declares_action_contract_generated.h",
    )
    _expect_value(env, action_args, "--source")
    _expect_arg_with_prefix_and_suffix(
        env,
        action_args,
        "",
        "nested/test_generated_files_declares_action_contract_generated.c",
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

def _test_invalid_output_arguments(name, outs, output_args, impl, **kwargs):
    util.helper_target(
        iree_generated_files,
        name = name + "_subject",
        outs = outs,
        output_args = output_args,
        tags = ["manual"],
        tool = "//build_tools/bazel/test:generate_rule_fixture_tool",
    )
    analysis_test(
        name = name,
        expect_failure = True,
        impl = impl,
        target = name + "_subject",
        **kwargs
    )

def _test_generated_files_rejects_unknown_output(name, **kwargs):
    _test_invalid_output_arguments(
        name,
        outs = [name + "/declared.h"],
        output_args = {"undeclared.h": "--header={path}"},
        impl = _test_generated_files_rejects_unknown_output_impl,
        **kwargs
    )

def _test_generated_files_rejects_unknown_output_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(
        matching.contains("output_args contains undeclared outputs: undeclared.h"),
    )

def _test_generated_files_rejects_missing_output(name, **kwargs):
    _test_invalid_output_arguments(
        name,
        outs = [name + "/missing.h"],
        output_args = {},
        impl = _test_generated_files_rejects_missing_output_impl,
        **kwargs
    )

def _test_generated_files_rejects_missing_output_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(
        matching.contains("output_args missing declared outputs: test_generated_files_rejects_missing_output/missing.h"),
    )

def _test_generated_files_rejects_duplicate_selection(name, **kwargs):
    _test_invalid_output_arguments(
        name,
        outs = [name + "/duplicate.h"],
        output_args = {
            name + "/duplicate.h": "--first={path}",
            "duplicate.h": "--second={path}",
        },
        impl = _test_generated_files_rejects_duplicate_selection_impl,
        **kwargs
    )

def _test_generated_files_rejects_duplicate_selection_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(
        matching.contains("more than once"),
    )

def _test_generated_files_rejects_ambiguous_basename(name, **kwargs):
    _test_invalid_output_arguments(
        name,
        outs = [name + "/first/shared.h", name + "/second/shared.h"],
        output_args = {"shared.h": "--header={path}"},
        impl = _test_generated_files_rejects_ambiguous_basename_impl,
        **kwargs
    )

def _test_generated_files_rejects_ambiguous_basename_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(
        matching.contains("output_args contains undeclared outputs: shared.h"),
    )

def generate_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_generated_files_declares_action_contract,
            _test_generated_files_accepts_repeated_basenames,
            _test_generated_files_rejects_unknown_output,
            _test_generated_files_rejects_missing_output,
            _test_generated_files_rejects_duplicate_selection,
            _test_generated_files_rejects_ambiguous_basename,
        ],
    )
