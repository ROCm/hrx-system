# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks the public execution selection settings at their analysis boundary."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@rules_testing//lib:analysis_test.bzl", "analysis_test")
load("@rules_testing//lib:truth.bzl", "matching")

_EXECUTION = str(Label("//build_tools/bazel:gpu_execution"))
_PROFILES = str(Label("//build_tools/bazel:rocjitsu_profiles"))

def _selection_test_impl(env, target):
    if env.ctx.attr.error:
        env.expect.that_target(target).failures().contains_predicate(matching.contains(env.ctx.attr.error))
    else:
        env.expect.that_collection(target[BuildSettingInfo].value).contains_exactly(env.ctx.attr.expected).in_order()

def execution_profiles_test_suite(name):
    """Declares checked-list and cross-setting selection tests.

    Args:
      name: Aggregate analysis test suite name.
    """
    tests = []
    for suffix, target, modes, profiles, expected, error in [
        ("native", _EXECUTION, ["native"], [], ["native"], ""),
        ("both", _EXECUTION, ["native", "rocjitsu"], ["gfx1151"], ["native", "rocjitsu"], ""),
        ("empty_modes", _EXECUTION, [], [], [], "nonempty list"),
        ("duplicate_modes", _EXECUTION, ["native", "native"], [], [], "without duplicates"),
        ("unknown_mode", _EXECUTION, ["automatic"], [], [], "Unknown gpu_execution value"),
        ("native_empty_profiles", _PROFILES, ["native"], [], [], ""),
        ("multiple_profiles", _PROFILES, ["rocjitsu"], ["gfx1151", "gfx1201"], ["gfx1151", "gfx1201"], ""),
        ("empty_profiles", _PROFILES, ["rocjitsu"], [], [], "requires a nonempty"),
        ("duplicate_profiles", _PROFILES, ["rocjitsu"], ["gfx1151", "gfx1151"], [], "must not contain duplicates"),
        ("unknown_profile", _PROFILES, ["rocjitsu"], ["gfx9999"], [], "Unknown rocjitsu profile"),
    ]:
        test_name = name + "_" + suffix
        analysis_test(
            name = test_name,
            target = target,
            impl = _selection_test_impl,
            expect_failure = bool(error),
            attrs = {"error": attr.string(), "expected": attr.string_list()},
            attr_values = {"error": error, "expected": expected},
            config_settings = {_EXECUTION: modes, _PROFILES: profiles},
        )
        tests.append(":" + test_name)
    native.test_suite(name = name, tests = tests)
