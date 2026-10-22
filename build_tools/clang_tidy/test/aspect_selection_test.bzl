# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for clang-tidy translation-unit selection."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test")
load("//build_tools/clang_tidy:clang_tidy.bzl", "IreeClangTidyInfo")

_SELECTED_FILES_FLAG = str(Label("//build_tools/clang_tidy:selected_files"))
_REPORT_PREFIX = "build_tools_clang_tidy_test_"
_TEST_ATTRIBUTES = {
    "tags": ["manual"],
    "timeout": "short",
}

def _expect_reports(env, target, expected_basenames):
    actual_basenames = sorted([
        report.basename.split(".", 1)[1]
        for report in target[IreeClangTidyInfo].local_reports.to_list()
    ])
    if actual_basenames != sorted(expected_basenames):
        env.fail("expected clang-tidy reports %r, got %r" % (expected_basenames, actual_basenames))

def _selected_source_test_impl(env, target):
    _expect_reports(env, target, [
        _REPORT_PREFIX + "action_smoke_c.clang_tidy.txt",
    ])

def _selected_header_test_impl(env, target):
    _expect_reports(env, target, [
        _REPORT_PREFIX + "action_smoke_c.clang_tidy.txt",
        _REPORT_PREFIX + "action_smoke_cc.clang_tidy.txt",
    ])

def _unrelated_file_test_impl(env, target):
    _expect_reports(env, target, [])

def clang_tidy_aspect_selection_test_suite(name, target):
    """Defines source, header, and unrelated-input selection tests.

    Args:
      name: Test-suite target name.
      target: Clang-tidy rule whose local reports are inspected.
    """
    target = Label(target)
    analysis_test(
        name = name + "_source",
        attr_values = _TEST_ATTRIBUTES,
        config_settings = {
            _SELECTED_FILES_FLAG: '["build_tools/clang_tidy/test/action_smoke.c"]',
        },
        impl = _selected_source_test_impl,
        target = target,
    )
    analysis_test(
        name = name + "_header",
        attr_values = _TEST_ATTRIBUTES,
        config_settings = {
            _SELECTED_FILES_FLAG: '["build_tools/clang_tidy/test/action_smoke.h"]',
        },
        impl = _selected_header_test_impl,
        target = target,
    )
    analysis_test(
        name = name + "_unrelated",
        attr_values = _TEST_ATTRIBUTES,
        config_settings = {
            _SELECTED_FILES_FLAG: '["runtime/src/iree/base/status.h"]',
        },
        impl = _unrelated_file_test_impl,
        target = target,
    )
    native.test_suite(
        name = name,
        tests = [
            name + "_header",
            name + "_source",
            name + "_unrelated",
        ],
    )
