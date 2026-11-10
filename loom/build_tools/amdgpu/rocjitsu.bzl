# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Concrete Loom tool invocations using the libamdf simulator backend."""

load("//build_tools/bazel:requirements.bzl", "apply_test_requirements")
load("//build_tools/third_party/rocjitsu:invocation.bzl", "rocjitsu_executable")
load("//build_tools/third_party/rocjitsu:profiles.bzl", "ROCJITSU_PROFILES")
load("//loom/target/amdgpu:execution_profiles.bzl", "AMD_GPU_HARDWARE_PROFILE")

def loom_rocjitsu_executable(name, src, profile, run_requirements = [], **kwargs):
    """Binds a complete simulated topology to an ordinary Loom tool.

    Device selection and workload options retain the tool's ordinary CLI.
    The topology is fixed by the invocation and locked in the IREE flag parser.

    Args:
      name: Concrete runnable name, independent of test matrix selection.
      src: Public executable label of the ordinary Loom tool.
      profile: Topology catalog key.
      run_requirements: Additional physical host requirements.
      **kwargs: Common rule attributes, such as tags and visibility.
    """
    if profile not in ROCJITSU_PROFILES:
        fail("Unknown rocjitsu profile %r" % profile)
    policy = apply_test_requirements(
        kwargs,
        build_requirements = AMD_GPU_HARDWARE_PROFILE.build_requirements,
        run_requirements = run_requirements,
    )
    policy["target_compatible_with"] = policy["target_compatible_with"] + select({
        "//libamdf/config/provider:rocjitsu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })
    rocjitsu_executable(
        name = name,
        src = src,
        configurations = ROCJITSU_PROFILES[profile].configurations,
        configuration_option = "--amd_rocjitsu_config",
        library_option = "--amd_rocjitsu_library",
        bootstrap_lock_option = "--flag-lock",
        **policy
    )
