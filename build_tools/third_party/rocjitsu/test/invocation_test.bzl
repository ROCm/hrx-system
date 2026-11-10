# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis coverage of the declared invocation boundary."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:truth.bzl", "matching")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo")
load("//build_tools/bazel:executable.bzl", "IreeExecutableInfo", "iree_executable_alias")
load("//build_tools/third_party/rocjitsu:invocation.bzl", "rocjitsu_executable")
load("//build_tools/third_party/rocjitsu:runtime.bzl", "rocjitsu_runtime")

_CONFIGURATION = "//build_tools/third_party/rocjitsu/profiles:gfx942_two_gpu.json"

def _subject(name, sanitizer = "address", **kwargs):
    # Analysis consumes file identity only. Real runtime execution is covered by
    # the separately selected CTS invocation and its checked package repository.
    rocjitsu_runtime(
        name = name + "_runtime",
        library = ":launcher_fixture",
        memory_helper = ":launcher_fixture",
        manifest = _CONFIGURATION,
        host_cpu = "x86-64",
        sanitizer = sanitizer,
        testonly = True,
        tags = ["manual"],
    )
    rocjitsu_executable(
        name = name + "_subject",
        src = ":launcher_fixture",
        runtime = ":" + name + "_runtime",
        library_option = "--library",
        configuration_option = "--machine",
        testonly = True,
        tags = ["manual"],
        **kwargs
    )

def _test_machine_order_and_bootstrap(name, **kwargs):
    _subject(
        name,
        configurations = [_CONFIGURATION, _CONFIGURATION],
        bootstrap_args = ["--text=a b'$HOME", "--repeat=first", "--repeat=second"],
        bootstrap_files = {_CONFIGURATION: "--input"},
        bootstrap_lock_option = "--flag-lock",
    )
    analysis_test(
        name = name,
        target = name + "_subject",
        config_settings = {"//command_line_option:cc_output_directory_tag": "asan"},
        impl = _test_machine_order_and_bootstrap_impl,
        **kwargs
    )

def _test_machine_order_and_bootstrap_impl(env, target):
    actions = [action for action in target[TestingAspectInfo].actions if action.mnemonic == "FileWrite"]
    env.expect.that_int(len(actions)).equals(1)
    content = actions[0].content
    env.expect.that_int(content.count("--machine=")).equals(2)
    env.expect.that_int(content.count("--library=")).equals(1)
    env.expect.that_int(content.count("--repeat=")).equals(2)
    env.expect.that_str(content).contains("--text=a b")
    for option in ["text", "repeat", "input", "library", "machine"]:
        env.expect.that_int(content.count("--flag-lock=" + option)).equals(1)
    env.expect.that_str(content).contains(' -- "$@"')
    env.expect.that_str(content).contains("LD_LIBRARY_PATH=")

def _test_empty_topology(name, **kwargs):
    _subject(name, configurations = [])
    analysis_test(
        name = name,
        target = name + "_subject",
        config_settings = {"//command_line_option:cc_output_directory_tag": "asan"},
        expect_failure = True,
        impl = _test_empty_topology_impl,
        **kwargs
    )

def _test_empty_topology_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(matching.contains("requires at least one machine configuration"))

def _test_invocation_inherits_configuration(name, **kwargs):
    _subject(name, configurations = [_CONFIGURATION])
    iree_executable_alias(
        name = name + "_wrapper",
        src = ":" + name + "_subject",
        testonly = True,
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        target = name + "_wrapper",
        config_settings = {
            "//command_line_option:cc_output_directory_tag": "asan",
        },
        impl = _test_invocation_inherits_configuration_impl,
        **kwargs
    )

def _test_invocation_inherits_configuration_impl(env, target):
    files = target[DefaultInfo].default_runfiles.files.to_list()
    basenames = [target.label.name.removesuffix("_wrapper") + "_subject", "launcher_fixture"]
    for basename in basenames:
        matching_files = [file for file in files if file.basename == basename]
        env.expect.that_int(len(matching_files)).equals(1)
        for file in matching_files:
            env.expect.that_str(file.root.path).equals(target[IreeExecutableInfo].output.root.path)

def _test_conflicting_bootstrap(name, **kwargs):
    _subject(name, configurations = [_CONFIGURATION], bootstrap_args = ["--library=wrong"])
    analysis_test(
        name = name,
        target = name + "_subject",
        config_settings = {"//command_line_option:cc_output_directory_tag": "asan"},
        expect_failure = True,
        impl = _test_conflicting_bootstrap_impl,
        **kwargs
    )

def _test_conflicting_bootstrap_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(matching.contains("must have distinct bootstrap bindings"))

def _test_conflicting_lock_option(name, **kwargs):
    _subject(name, configurations = [_CONFIGURATION], bootstrap_lock_option = "--machine")
    analysis_test(
        name = name,
        target = name + "_subject",
        config_settings = {"//command_line_option:cc_output_directory_tag": "asan"},
        expect_failure = True,
        impl = _test_conflicting_lock_option_impl,
        **kwargs
    )

def _test_conflicting_lock_option_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(matching.contains("Bootstrap lock option must be distinct"))

def _test_incompatible_sanitizer(name, **kwargs):
    _subject(name, configurations = [_CONFIGURATION])
    analysis_test(
        name = name,
        target = name + "_subject",
        config_settings = {"//command_line_option:cc_output_directory_tag": ""},
        expect_failure = True,
        impl = _test_incompatible_sanitizer_impl,
        **kwargs
    )

def _test_incompatible_sanitizer_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(matching.contains("requires an address-sanitized consumer"))

def _test_ordinary_runtime(name, **kwargs):
    _subject(name, sanitizer = "none", configurations = [_CONFIGURATION])
    analysis_test(
        name = name,
        target = name + "_subject",
        config_settings = {"//command_line_option:cc_output_directory_tag": ""},
        impl = _test_ordinary_runtime_impl,
        **kwargs
    )

def _test_ordinary_runtime_impl(env, target):
    env.expect.that_str(target[IreeExecutableInfo].src.name).equals("launcher_fixture")

def rocjitsu_invocation_test_suite(name):
    """Declares analysis tests for simulator invocation policy.

    Args:
      name: Test suite name.
    """
    test_suite(
        name = name,
        tests = [
            _test_machine_order_and_bootstrap,
            _test_invocation_inherits_configuration,
            _test_empty_topology,
            _test_conflicting_bootstrap,
            _test_conflicting_lock_option,
            _test_incompatible_sanitizer,
            _test_ordinary_runtime,
        ],
    )
