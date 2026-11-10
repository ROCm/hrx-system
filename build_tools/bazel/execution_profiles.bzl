# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Selection of independent native and simulated GPU invocations."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")

def _gpu_execution_flag_impl(ctx):
    values = ctx.build_setting_value
    if not values or len(values) != len(depset(values).to_list()):
        fail("gpu_execution requires a nonempty list without duplicates")
    for value in values:
        if value not in ["native", "rocjitsu"]:
            fail("Unknown gpu_execution value %r; expected native or rocjitsu" % value)
    return BuildSettingInfo(value = values)

gpu_execution_flag = rule(
    implementation = _gpu_execution_flag_impl,
    build_setting = config.string_list(flag = True),
    doc = "Selects independent native and/or rocjitsu GPU test invocations.",
)

def native_gpu_test_compatibility():
    """Returns eligibility for the native leaf of an admitted GPU matrix."""
    return select({
        "//build_tools/bazel:native_gpu_execution": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })
