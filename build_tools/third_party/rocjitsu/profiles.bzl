# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualified functional topologies and explicit test-matrix selection."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")

# Each profile describes one VM containing multiple GPUs. Device identity
# drives the JIT; topology descriptions may contain heterogeneous devices.
ROCJITSU_PROFILES = {
    architecture: struct(
        configurations = ["//build_tools/third_party/rocjitsu/profiles:%s_%s_gpu.json" % (architecture, count)],
        gpu_target = architecture,
    )
    for architecture, count in [
        ("gfx942", "two"),
        ("gfx950", "two"),
        ("gfx1100", "four"),
        ("gfx1151", "four"),
        ("gfx1201", "four"),
        ("gfx1250", "two"),
    ]
}

def _rocjitsu_profiles_flag_impl(ctx):
    profiles = ctx.build_setting_value
    if len(profiles) != len(depset(profiles).to_list()):
        fail("rocjitsu_profiles must not contain duplicates")
    for profile in profiles:
        if profile not in ROCJITSU_PROFILES:
            fail("Unknown rocjitsu profile %r; expected one of %s" % (profile, ", ".join(ROCJITSU_PROFILES)))
    if "rocjitsu" in ctx.attr._gpu_execution[BuildSettingInfo].value and not profiles:
        fail("rocjitsu GPU test execution requires a nonempty --//build_tools/bazel:rocjitsu_profiles list")
    return BuildSettingInfo(value = profiles)

rocjitsu_profiles_flag = rule(
    implementation = _rocjitsu_profiles_flag_impl,
    build_setting = config.string_list(flag = True),
    attrs = {
        "_gpu_execution": attr.label(default = "//build_tools/bazel:gpu_execution"),
    },
    doc = "Selects known rocjitsu topology rows when a test matrix is requested.",
)

def rocjitsu_test_compatibility(profile):
    """Returns eligibility for one concrete simulator test leaf.

    Args:
      profile: Key in ROCJITSU_PROFILES.
    """
    return select({
        "//build_tools/third_party/rocjitsu/profiles:" + profile + "_selected": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })
