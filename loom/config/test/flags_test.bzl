# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for public Loom capability selections."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@rules_testing//lib:analysis_test.bzl", "analysis_test")
load("@rules_testing//lib:truth.bzl", "matching")

_REPRESENTATIVE_VALUES = {
    "loom/config/arch": ["x86"],
    "loom/config/emit": ["elf"],
    "loom/config/execute": ["iree_hal"],
    "loom/config/import": ["cxx"],
}

def _expect_values(env, target):
    env.expect.that_collection(target[BuildSettingInfo].value).contains_exactly(
        _REPRESENTATIVE_VALUES[target.label.package],
    )

def _expect_empty(env, target):
    env.expect.that_collection(target[BuildSettingInfo].value).contains_exactly([])

def _expect_unknown(env, target):
    env.expect.that_target(target).failures().contains_predicate(
        matching.contains("Unknown value(s)"),
    )
    env.expect.that_target(target).failures().contains_predicate(
        matching.contains("unknown_option"),
    )

def loom_config_flag_test_suite(name):
    """Tests value admission and rejection through real build settings.

    Args:
      name: Aggregate test-suite name and prefix for the individual tests.
    """
    tests = []
    for package, values in _REPRESENTATIVE_VALUES.items():
        flag = Label("//" + package + ":enable")
        suffix = package.removeprefix("loom/config/").replace("/", "_")
        for selection_name, selection, impl in [
            ("selected", values, _expect_values),
            ("empty", [], _expect_empty),
            ("unknown", values[:1] + ["unknown_option"], _expect_unknown),
        ]:
            test_name = name + "_" + suffix + "_" + selection_name
            analysis_test(
                name = test_name,
                config_settings = {str(flag): selection},
                expect_failure = selection_name == "unknown",
                impl = impl,
                target = flag,
            )
            tests.append(test_name)
    native.test_suite(name = name, tests = tests)
