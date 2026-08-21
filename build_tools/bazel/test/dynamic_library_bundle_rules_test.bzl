# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for runtime-only dynamic-library bundle rules."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:truth.bzl", "matching")
load(
    "//build_tools/bazel:dynamic_library_bundle.bzl",
    "IreeDynamicLibraryBindingsInfo",
    "IreeDynamicLibraryBundleInfo",
    "collect_dynamic_library_bundles",
    "iree_dynamic_library_bundle",
)

_TEST_ENVIRONMENT_NAME = "IREE_TEST_DYNAMIC_LIBRARY_PATH"

def _collect_bindings_impl(ctx):
    return [ctx.attr.target[IreeDynamicLibraryBindingsInfo]]

_collect_bindings = rule(
    implementation = _collect_bindings_impl,
    attrs = {
        "target": attr.label(
            aspects = [collect_dynamic_library_bundles],
            mandatory = True,
        ),
    },
)

def _expect_file_basename(env, files, expected_basename):
    for file in files:
        if file.basename == expected_basename:
            return
    env.fail("expected basename %r in %r" % (expected_basename, files))

def _test_bundle_is_runtime_only(name, **kwargs):
    bundle = name + "_subject"
    iree_dynamic_library_bundle(
        name = bundle,
        environment = {
            ":dynamic_library_root.so": _TEST_ENVIRONMENT_NAME,
        },
        srcs = [
            ":dynamic_library_dependency.so",
            ":dynamic_library_root.so",
        ],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {"timeout": "short"},
        impl = _test_bundle_is_runtime_only_impl,
        target = ":" + bundle,
        **kwargs
    )

def _test_bundle_is_runtime_only_impl(env, target):
    if CcInfo in target:
        env.fail("runtime-only dynamic-library bundle unexpectedly provides CcInfo")
    bundle = target[IreeDynamicLibraryBundleInfo]
    files = bundle.files.to_list()
    _expect_file_basename(env, files, "dynamic_library_root.so")
    _expect_file_basename(env, files, "dynamic_library_dependency.so")
    runfiles = target[DefaultInfo].default_runfiles.files.to_list()
    _expect_file_basename(env, runfiles, "dynamic_library_root.so")
    _expect_file_basename(env, runfiles, "dynamic_library_dependency.so")
    env.expect.that_str(bundle.environment[_TEST_ENVIRONMENT_NAME].basename).equals(
        "dynamic_library_root.so",
    )

def _test_aspect_collects_data_through_label_flag(name, **kwargs):
    bundle = name + "_bundle"
    iree_dynamic_library_bundle(
        name = bundle,
        environment = {
            ":dynamic_library_root.so": _TEST_ENVIRONMENT_NAME,
        },
        srcs = [
            ":dynamic_library_dependency.so",
            ":dynamic_library_root.so",
        ],
        tags = ["manual"],
    )
    selected_bundle = name + "_selected_bundle"
    native.label_flag(
        name = selected_bundle,
        build_setting_default = ":" + bundle,
        tags = ["manual"],
    )
    data_library = name + "_data_library"
    cc_library(
        name = data_library,
        data = [":" + selected_bundle],
        tags = ["manual"],
    )
    collector = name + "_subject"
    _collect_bindings(
        name = collector,
        target = ":" + data_library,
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {"timeout": "short"},
        impl = _test_aspect_collects_data_through_label_flag_impl,
        target = collector,
        **kwargs
    )

def _test_aspect_collects_data_through_label_flag_impl(env, target):
    bindings = target[IreeDynamicLibraryBindingsInfo]
    files = bindings.files.to_list()
    _expect_file_basename(env, files, "dynamic_library_root.so")
    _expect_file_basename(env, files, "dynamic_library_dependency.so")
    env.expect.that_str(bindings.environment[_TEST_ENVIRONMENT_NAME].basename).equals(
        "dynamic_library_root.so",
    )

def _test_aspect_deduplicates_identical_bindings(name, **kwargs):
    bundle = name + "_bundle"
    iree_dynamic_library_bundle(
        name = bundle,
        environment = {
            ":dynamic_library_root.so": _TEST_ENVIRONMENT_NAME,
        },
        srcs = [":dynamic_library_root.so"],
        tags = ["manual"],
    )
    first_library = name + "_first_library"
    cc_library(
        name = first_library,
        data = [":" + bundle],
        tags = ["manual"],
    )
    second_library = name + "_second_library"
    cc_library(
        name = second_library,
        data = [":" + bundle],
        tags = ["manual"],
    )
    combined_library = name + "_combined_library"
    cc_library(
        name = combined_library,
        data = [
            ":" + first_library,
            ":" + second_library,
        ],
        tags = ["manual"],
    )
    collector = name + "_subject"
    _collect_bindings(
        name = collector,
        target = ":" + combined_library,
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {"timeout": "short"},
        impl = _test_aspect_deduplicates_identical_bindings_impl,
        target = collector,
        **kwargs
    )

def _test_aspect_deduplicates_identical_bindings_impl(env, target):
    bindings = target[IreeDynamicLibraryBindingsInfo]
    env.expect.that_collection(bindings.files.to_list()).contains_exactly(
        [bindings.environment[_TEST_ENVIRONMENT_NAME]],
    )

def _test_conflicting_bindings_fail(name, **kwargs):
    first_bundle = name + "_first_bundle"
    iree_dynamic_library_bundle(
        name = first_bundle,
        environment = {
            ":dynamic_library_root.so": _TEST_ENVIRONMENT_NAME,
        },
        srcs = [":dynamic_library_root.so"],
        tags = ["manual"],
    )
    second_bundle = name + "_second_bundle"
    iree_dynamic_library_bundle(
        name = second_bundle,
        environment = {
            ":dynamic_library_other.so": _TEST_ENVIRONMENT_NAME,
        },
        srcs = [":dynamic_library_other.so"],
        tags = ["manual"],
    )
    conflicting_library = name + "_conflicting_library"
    cc_library(
        name = conflicting_library,
        data = [
            ":" + first_bundle,
            ":" + second_bundle,
        ],
        tags = ["manual"],
    )
    collector = name + "_subject"
    _collect_bindings(
        name = collector,
        target = ":" + conflicting_library,
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {"timeout": "short"},
        expect_failure = True,
        impl = _test_conflicting_bindings_fail_impl,
        target = collector,
        **kwargs
    )

def _test_conflicting_bindings_fail_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(
        matching.contains(
            "resolves dynamic-library environment variable %s to both" %
            _TEST_ENVIRONMENT_NAME,
        ),
    )

def dynamic_library_bundle_rules_test_suite(name):
    """Defines runtime-only dynamic-library bundle analysis tests.

    Args:
      name: Test suite target name.
    """
    test_suite(
        name = name,
        tests = [
            _test_bundle_is_runtime_only,
            _test_aspect_collects_data_through_label_flag,
            _test_aspect_deduplicates_identical_bindings,
            _test_conflicting_bindings_fail,
        ],
    )
