# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for executable wrapper Bazel rules."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo", "util")
load("//build_tools/bazel:cc.bzl", "iree_cc_binary", "iree_cc_library")
load(
    "//build_tools/bazel:executable.bzl",
    "IreeExecutableInfo",
    "iree_executable_alias",
    "iree_executable_test",
    "iree_tool_test",
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

def _test_executable_test_applies_resource_group_tags(name, **kwargs):
    iree_executable_test(
        name = name + "_subject",
        src = ":generate_rule_fixture_tool",
        resource_group = "shared-device",
        tags = ["manual", "existing-tag"],
    )
    analysis_test(
        name = name,
        attr_values = {"timeout": "short"},
        impl = _test_executable_test_applies_resource_group_tags_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_executable_test_applies_resource_group_tags_impl(env, target):
    env.expect.that_collection(target[TestingAspectInfo].attrs.tags).contains_at_least([
        "existing-tag",
        "exclusive-if-local",
        "resource_group:shared-device",
    ])

def _test_cross_package_wrapper_launches_source(name, **kwargs):
    iree_executable_test(
        name = name + "_subject",
        src = "//build_tools/bazel/test/executable_fixture:binary",
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        impl = _test_cross_package_wrapper_launches_source_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cross_package_wrapper_launches_source_impl(env, target):
    action = _find_action_with_output(
        env,
        target[TestingAspectInfo].actions,
        target.label.name,
    )
    windows = bool(target[TestingAspectInfo].attrs.windows_launcher)
    if windows:
        env.expect.that_str(action.mnemonic).equals("ExecutableSymlink")
        env.expect.that_str(
            target[RunEnvironmentInfo].environment["IREE_BAZEL_EXECUTABLE_RUNFILE"],
        ).equals("_main/build_tools/bazel/test/executable_fixture/binary.exe")
    else:
        env.expect.that_str(action.mnemonic).equals("FileWrite")
        env.expect.that_str(action.content).contains(
            '"${RUNFILES_DIR:-${TEST_SRCDIR:-$0.runfiles}}"',
        )
        env.expect.that_str(action.content).contains("export RUNFILES_DIR")
        env.expect.that_str(action.content).contains(
            "_main/build_tools/bazel/test/executable_fixture/binary",
        )
        env.expect.that_str(action.content).contains('"$@"')
    _expect_basename(
        env,
        target[DefaultInfo].default_runfiles.files.to_list(),
        "binary.exe" if windows else "binary",
    )

def _test_executable_wrapper_composes_source_suppressions(name, **kwargs):
    iree_cc_library(
        name = name + "_first",
        sanitizer_suppressions = {"lsan": "//build_tools/sanitizer:lsan_suppressions_hsa.txt"},
        tags = ["manual"],
    )
    iree_cc_library(
        name = name + "_second",
        sanitizer_suppressions = {"lsan": "//build_tools/sanitizer:lsan_suppressions_vulkan.txt"},
        tags = ["manual"],
    )
    util.helper_target(
        iree_cc_binary,
        name = name + "_binary",
        deps = [":" + name + "_first"],
        srcs = [name + "_binary.cc"],
        tags = ["manual"],
    )
    iree_executable_alias(
        name = name + "_alias",
        src = ":" + name + "_binary",
        tags = ["manual"],
    )
    iree_executable_test(
        name = name + "_subject",
        src = ":" + name + "_alias",
        data = [":" + name + "_second"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        impl = _test_executable_wrapper_composes_source_suppressions_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_executable_wrapper_composes_source_suppressions_impl(env, target):
    actions = [action for action in target[TestingAspectInfo].actions if action.mnemonic == "SanitizerSuppressions"]
    env.expect.that_int(len(actions)).equals(1)
    env.expect.that_collection([file.basename for file in actions[0].inputs.to_list()]).contains_exactly([
        "lsan_suppressions_hsa.txt",
        "lsan_suppressions_vulkan.txt",
    ])
    output = actions[0].outputs.to_list()[0]
    env.expect.that_str(target[RunEnvironmentInfo].environment["LSAN_OPTIONS"]).equals(
        'suppressions="' + output.short_path + '":allow_addr2line=1',
    )
    env.expect.that_collection(target[DefaultInfo].default_runfiles.files.to_list()).contains(output)

def _execution_configuration_aspect_impl(target, ctx):
    return [TestingAspectInfo(
        actions = target.actions,
        attrs = ctx.rule.attr,
    )]

_execution_configuration_aspect = aspect(
    implementation = _execution_configuration_aspect_impl,
    attr_aspects = ["src", "data", "tools"],
)

def _external_tool_test_configuration():
    return {
        "//command_line_option:copt": ["-DIREE_TEST_TARGET_CONFIG"],
        "//command_line_option:extra_execution_platforms": [
            str(Label(":test_incompatible_execution_platform")),
            str(Label(":test_execution_platform")),
        ],
        "//command_line_option:host_copt": ["-DIREE_TEST_EXEC_CONFIG"],
        "//command_line_option:platforms": [Label(":test_execution_platform")],
        "@bazel_tools//tools/test:incompatible_use_default_test_toolchain": True,
    }

def _external_tool_fixture(name):
    util.helper_target(
        iree_cc_binary,
        name = name,
        srcs = [name + ".cc"],
        data = [":generate_rule_fixture.data"],
        copts = select({
            ":on_test_execution_platform": ["-DIREE_TEST_EXEC_PLATFORM"],
            "//conditions:default": [],
        }),
        tags = ["manual"],
    )

def _expect_compile_option(env, target, present, absent):
    actions = [action for action in target[TestingAspectInfo].actions if action.mnemonic == "CppCompile"]
    env.expect.that_int(len(actions)).equals(1)
    env.expect.that_collection(actions[0].argv).contains(present)
    env.expect.that_collection(actions[0].argv).not_contains(absent)

def _test_executable_helpers_use_test_execution_configuration(name, **kwargs):
    _external_tool_fixture(name + "_tool")
    _external_tool_fixture(name + "_runner")
    _external_tool_fixture(name + "_data")
    iree_executable_test(
        name = name + "_subject",
        src = ":" + name + "_runner",
        args = ["--tool=$(rootpath :" + name + "_tool)"],
        data = [":" + name + "_data"],
        env = {"IREE_HELPER": "$(rootpath :" + name + "_tool)"},
        tools = [":" + name + "_tool"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        config_settings = _external_tool_test_configuration(),
        impl = _test_executable_helpers_use_test_execution_configuration_impl,
        target = name + "_subject",
        testing_aspect = _execution_configuration_aspect,
        **kwargs
    )

def _expect_external_tool(env, target, tool):
    _expect_compile_option(env, tool, "-DIREE_TEST_EXEC_CONFIG", "-DIREE_TEST_TARGET_CONFIG")
    action = [action for action in tool[TestingAspectInfo].actions if action.mnemonic == "CppCompile"][0]
    env.expect.that_collection(action.argv).contains("-DIREE_TEST_EXEC_PLATFORM")
    runfiles = target[DefaultInfo].default_runfiles.files.to_list()
    env.expect.that_collection(runfiles).contains(tool[DefaultInfo].files_to_run.executable)
    _expect_basename(env, runfiles, "generate_rule_fixture.data")

def _test_executable_helpers_use_test_execution_configuration_impl(env, target):
    attrs = target[TestingAspectInfo].attrs
    _expect_compile_option(env, attrs.src, "-DIREE_TEST_TARGET_CONFIG", "-DIREE_TEST_EXEC_CONFIG")
    _expect_compile_option(env, attrs.data[0], "-DIREE_TEST_TARGET_CONFIG", "-DIREE_TEST_EXEC_CONFIG")
    tool = attrs.tools[0]
    _expect_external_tool(env, target, tool)
    path = tool[DefaultInfo].files_to_run.executable.short_path
    env.expect.that_str(target[IreeExecutableInfo].env["IREE_HELPER"]).equals(path)
    env.expect.that_collection(target[IreeRunfilesArgumentsInfo].arguments).contains_exactly([
        "--tool=" + path,
    ])

def _test_tool_checks_artifacts_in_execution_configuration(name, **kwargs):
    _external_tool_fixture(name + "_tool")
    iree_tool_test(
        name = name + "_subject",
        src = ":" + name + "_tool",
        data = [":generate_rule_fixture.in"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        config_settings = _external_tool_test_configuration(),
        impl = _test_tool_checks_artifacts_in_execution_configuration_impl,
        target = name + "_subject",
        testing_aspect = _execution_configuration_aspect,
        **kwargs
    )

def _test_tool_checks_artifacts_in_execution_configuration_impl(env, target):
    _expect_external_tool(env, target, target[TestingAspectInfo].attrs.src)
    _expect_basename(env, target[DefaultInfo].default_runfiles.files.to_list(), "generate_rule_fixture.in")

def executable_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_executable_alias_wraps_source,
            _test_executable_test_wraps_source,
            _test_executable_test_applies_resource_group_tags,
            _test_cross_package_wrapper_launches_source,
            _test_executable_wrapper_composes_source_suppressions,
            _test_executable_helpers_use_test_execution_configuration,
            _test_tool_checks_artifacts_in_execution_configuration,
        ],
    )
