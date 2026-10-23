# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for clang-tidy translation-unit selection."""

load("@iree_clang_tidy_llvm//:config.bzl", "CLANG_TIDY_LLVM_TARGET_COMPATIBLE_WITH")
load("@rules_testing//lib:analysis_test.bzl", "analysis_test")
load("//build_tools/clang_tidy:clang_tidy.bzl", "IreeClangTidyInfo", "make_clang_tidy_aspect")

_SELECTIONS = {
    "all": "",
    "empty": "[]",
    "header": '["build_tools/clang_tidy/test/action_smoke.h"]',
    "source": '["build_tools/clang_tidy/test/action_smoke.c"]',
    "unrelated": '["runtime/src/iree/base/status.h"]',
}
_selection_aspect = make_clang_tidy_aspect(selected_files_values = _SELECTIONS.values())
_REPORT_PREFIX = "build_tools_clang_tidy_test_"
_TEST_ATTRIBUTES = {
    "tags": ["manual"],
    # The aspect attaches directly to a C/C++ subject, so the test owns the
    # tool compatibility constraint normally supplied by iree_clang_tidy.
    "target_compatible_with": CLANG_TIDY_LLVM_TARGET_COMPATIBLE_WITH,
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
    """Defines source, header, unrelated, empty, and full-source selection tests.

    Args:
      name: Test-suite target name.
      target: C/C++ subject whose aspect reports are inspected.
    """
    implementations = {
        "all": _selected_header_test_impl,
        "empty": _unrelated_file_test_impl,
        "header": _selected_header_test_impl,
        "source": _selected_source_test_impl,
        "unrelated": _unrelated_file_test_impl,
    }
    for selection, encoded_files in _SELECTIONS.items():
        analysis_test(
            name = name + "_" + selection,
            attrs = {"selected_files": attr.string()},
            attr_values = dict(_TEST_ATTRIBUTES, selected_files = encoded_files),
            extra_target_under_test_aspects = [_selection_aspect],
            impl = implementations[selection],
            target = Label(target),
        )
    native.test_suite(
        name = name,
        tests = [name + "_" + selection for selection in _SELECTIONS],
    )
