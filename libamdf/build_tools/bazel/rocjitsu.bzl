# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Explicit simulator invocations of ordinary libamdf CTS binaries."""

load("//build_tools/bazel:executable.bzl", "iree_executable_test")
load("//build_tools/bazel:requirements.bzl", "apply_test_requirements")
load("//build_tools/third_party/rocjitsu:invocation.bzl", "rocjitsu_executable")

def amdf_rocjitsu_cts_invocation(
        name,
        src,
        configurations,
        gpu_target,
        required_tests = [],
        test_args = [],
        test_compatible_with = [],
        run_requirements = [],
        tags = [],
        visibility = None):
    """Declares a concrete runnable and a test of that same invocation.

    Args:
      name: Runnable target name; the test adds the suffix _test.
      src: Ordinary dynamic-provider CTS binary.
      configurations: Ordered machine configurations defining the complete fabric.
      gpu_target: Required canonical gfx target, checked by the ordinary CTS.
      required_tests: Cases required to execute successfully through run and test.
      test_args: Workload/case selection applied by the test wrapper.
      test_compatible_with: Test matrix selection; the concrete run stays available.
      run_requirements: Physical host resources, independent of simulation choice.
      tags: Tags applied to both runnable and test.
      visibility: Visibility of both targets.
    """
    policy = apply_test_requirements(
        {"tags": tags},
        run_requirements = run_requirements,
    )
    compatibility = select({
        "//libamdf/config/provider:rocjitsu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })
    policy["target_compatible_with"] = compatibility

    # Concrete runs are selected explicitly; test leaves own matrix selection.
    # Wildcard native builds must not acquire every simulator runtime.
    rocjitsu_executable(
        name = name,
        testonly = True,
        src = src,
        bootstrap_args = [
            "--amdf_native_lifetime=process",
            "--amdf_gpu_target=" + gpu_target,
        ] + ["--amdf_require_test=" + test for test in required_tests],
        bootstrap_files = {
            "//libamdf:amdf_shared_artifact": "--amdf_library",
        },
        library_option = "--amdf_rocjitsu_library",
        configuration_option = "--amdf_rocjitsu_config",
        configurations = configurations,
        visibility = visibility,
        **dict(policy, tags = depset(policy["tags"] + ["manual"]).to_list())
    )
    iree_executable_test(
        name = name + "_test",
        src = ":" + name,
        args = test_args,
        visibility = visibility,
        **dict(policy, target_compatible_with = compatibility + test_compatible_with)
    )
