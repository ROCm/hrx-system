# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for shared Python Bazel macros."""

load("@rules_python//python:py_runtime_info.bzl", "PyRuntimeInfo")
load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo", "util")
load("//build_tools/bazel:python.bzl", "iree_py_binary", "iree_py_test")

def _explicit_packages_test(name, macro, **kwargs):
    util.helper_target(
        macro,
        name = name + "_subject",
        srcs = [name + "_subject.py"],
    )
    analysis_test(
        name = name,
        config_settings = {
            str(Label("@rules_python//python/config_settings:incompatible_default_to_explicit_init_py")): False,
        },
        impl = _explicit_packages_test_impl,
        target = name + "_subject",
        **kwargs
    )

def _explicit_packages_test_impl(env, target):
    env.expect.that_collection(target[DefaultInfo].default_runfiles.empty_filenames.to_list()).contains_exactly([])

def _test_py_binary_owns_explicit_packages(name, **kwargs):
    _explicit_packages_test(name, iree_py_binary, **kwargs)

def _test_py_test_owns_explicit_packages(name, **kwargs):
    _explicit_packages_test(name, iree_py_test, **kwargs)

def _test_py_test_applies_resource_group_tags(name, **kwargs):
    util.helper_target(
        iree_py_test,
        name = name + "_subject",
        srcs = [name + "_subject.py"],
        resource_group = "gpu",
        tags = [
            "requires-gpu-vulkan",
            "manual",
        ],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_py_test_applies_resource_group_tags_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_py_test_applies_resource_group_tags_impl(env, target):
    tags = target[TestingAspectInfo].attrs.tags
    for expected_tag in [
        "requires-gpu-vulkan",
        "manual",
        "exclusive-if-local",
        "resource_group:gpu",
    ]:
        if expected_tag not in tags:
            env.fail("expected %r in Python test tags %r" % (expected_tag, tags))

def _build_data_test(name, stamp, impl):
    util.helper_target(
        iree_py_binary,
        name = name + "_subject",
        srcs = [name + "_subject.py"],
        stamp = stamp,
    )
    analysis_test(name = name, impl = impl, target = name + "_subject")

def _build_data_action(env, target):
    actions = [
        action
        for action in target[TestingAspectInfo].actions
        if any([output.basename.endswith(".build_data.txt") for output in action.outputs.to_list()])
    ]
    env.expect.that_int(len(actions)).equals(1)
    return actions[0]

def _test_unstamped_metadata_is_analysis_known(name):
    _build_data_test(name, stamp = 0, impl = _test_unstamped_metadata_is_analysis_known_impl)

def _test_unstamped_metadata_is_analysis_known_impl(env, target):
    action = _build_data_action(env, target)
    env.expect.that_str(action.mnemonic).equals("FileWrite")
    env.expect.that_collection(action.inputs.to_list()).contains_exactly([])
    env.expect.that_str(action.content).contains("STAMPED FALSE")

def _test_stamped_metadata_uses_workspace_status(name):
    _build_data_test(name, stamp = 1, impl = _test_stamped_metadata_uses_workspace_status_impl)

def _test_stamped_metadata_uses_workspace_status_impl(env, target):
    action = _build_data_action(env, target)
    env.expect.that_str(action.mnemonic).equals("PyWriteBuildData")
    env.expect.that_collection([file.basename for file in action.inputs.to_list()]).contains("stable-status.txt")

def _runtime_test(name, impl, debug = False, python_version = "3.12"):
    util.helper_target(
        iree_py_binary,
        name = name + "_subject",
        srcs = [name + "_subject.py"],
        python_version = python_version,
    )
    analysis_test(
        name = name,
        attr_values = {
            "target_compatible_with": ["@platforms//os:linux", "@platforms//cpu:x86_64"],
        },
        config_settings = {str(Label("//build_tools/bazel:python_runtime_debug")): debug},
        impl = impl,
        target = name + "_subject",
    )

def _expect_runtime_repository(env, target, repository, excluded_repository):
    owners = {file.owner.repo_name: True for file in target[DefaultInfo].default_runfiles.files.to_list()}
    env.expect.that_collection(owners.keys()).contains(repository)
    env.expect.that_collection(owners.keys()).not_contains(excluded_repository)

def _test_default_runtime_is_compact(name):
    _runtime_test(name, _test_default_runtime_is_compact_impl)

def _test_default_runtime_is_compact_impl(env, target):
    _expect_runtime_repository(
        env,
        target,
        Label("@hrx_python_linux_x86_64_stripped//:python_runtimes").repo_name,
        Label("@hrx_python_linux_x86_64_debug//:python_runtimes").repo_name,
    )

def _test_debug_runtime_retains_symbols(name):
    _runtime_test(name, _test_debug_runtime_retains_symbols_impl, debug = True)

def _test_debug_runtime_retains_symbols_impl(env, target):
    _expect_runtime_repository(
        env,
        target,
        Label("@hrx_python_linux_x86_64_debug//:python_runtimes").repo_name,
        Label("@hrx_python_linux_x86_64_stripped//:python_runtimes").repo_name,
    )

def _test_other_python_version_uses_upstream_runtime(name):
    _runtime_test(name, _test_other_python_version_uses_upstream_runtime_impl, python_version = "3.14")

def _test_other_python_version_uses_upstream_runtime_impl(env, target):
    owners = {file.owner.repo_name: True for file in target[DefaultInfo].default_runfiles.files.to_list()}
    for repository in [
        Label("@hrx_python_linux_x86_64_stripped//:python_runtimes").repo_name,
        Label("@hrx_python_linux_x86_64_debug//:python_runtimes").repo_name,
    ]:
        env.expect.that_collection(owners.keys()).not_contains(repository)

def _selected_python_runtime_impl(ctx):
    return [ctx.toolchains["@rules_python//python:toolchain_type"].py3_runtime]

_selected_python_runtime = rule(
    implementation = _selected_python_runtime_impl,
    toolchains = ["@rules_python//python:toolchain_type"],
)

def _test_windows_runtime_owns_source_files(name):
    util.helper_target(
        _selected_python_runtime,
        name = name + "_subject",
    )
    analysis_test(
        name = name,
        attr_values = {
            "target_compatible_with": ["@platforms//os:windows", "@platforms//cpu:x86_64"],
        },
        config_settings = {
            str(Label("//build_tools/bazel:sanitizer")): False,
            "//command_line_option:cc_output_directory_tag": "",
            "//command_line_option:platforms": [Label("//build_tools/windows:windows_x86_64")],
            str(Label("//build_tools/bazel:python_runtime_debug")): False,
        },
        impl = _test_windows_runtime_owns_source_files_impl,
        target = name + "_subject",
    )

def _test_windows_runtime_owns_source_files_impl(env, target):
    runtime = target[PyRuntimeInfo]

    # Configured outputs can be rematerialized between invocations while a
    # persistent worker still has the interpreter or its DLLs mapped.
    for file in [runtime.interpreter] + runtime.files.to_list():
        if not file.is_source:
            env.fail("Windows Python runtime contains a generated artifact: {}".format(file.path))

def python_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_py_binary_owns_explicit_packages,
            _test_py_test_owns_explicit_packages,
            _test_py_test_applies_resource_group_tags,
            _test_unstamped_metadata_is_analysis_known,
            _test_stamped_metadata_uses_workspace_status,
            _test_default_runtime_is_compact,
            _test_debug_runtime_retains_symbols,
            _test_other_python_version_uses_upstream_runtime,
            _test_windows_runtime_owns_source_files,
        ],
    )
