# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cross-linkage conformance matrix for the public libamdf ABI."""

load("//build_tools/bazel:cc_attrs.bzl", "cc_attrs")
load("//build_tools/bazel:executable.bzl", "iree_executable_test")
load("//libamdf/requirements:package_policy.bzl", "apply_amdf_test_policy")
load(":cc.bzl", "amdf_cc_binary", "amdf_cc_library")

def amdf_cts_test_suite(
        name,
        srcs,
        deps,
        tags = None,
        resource_group = None,
        target_compatible_with = None,
        visibility = None):
    """Runs a CTS corpus through static, shared, and loaded providers.

    Args:
      name: Aggregate test-suite target name.
      srcs: Sources for one independently selectable conformance corpus.
      deps: Public API and test-helper dependencies of the corpus.
      tags: Additional tags applied to every generated test target.
      resource_group: Shared native resource used by the test invocations.
      target_compatible_with: Constraints required by every test mode.
      visibility: Visibility of the aggregate test suite.
    """
    policy = apply_amdf_test_policy({
        "tags": tags or [],
        "resource_group": resource_group,
        "target_compatible_with": target_compatible_with,
    })
    test_tags = cc_attrs.with_resource_group_tags(
        policy["tags"],
        policy.get("resource_group"),
    )
    corpus_name = name + "_cases"
    amdf_cc_library(
        name = corpus_name,
        testonly = True,
        srcs = srcs,
        deps = deps,
        alwayslink = True,
        target_compatible_with = target_compatible_with,
    )
    common_deps = [":" + corpus_name, "//libamdf/cts/util:test_main"]
    runtime_data = ["//libamdf:amdf_runtime"]
    tests = []
    for mode in ["static", "shared", "dynamic"]:
        binary_name = name + "_" + mode + "_bin"
        data = runtime_data
        if mode != "static":
            data = data + ["//libamdf:amdf_shared_artifact"]
        amdf_cc_binary(
            name = binary_name,
            testonly = True,
            data = data,
            deps = common_deps + ["//libamdf/cts/util:" + mode + "_provider"],
            target_compatible_with = target_compatible_with,
        )
        provider_args = []
        if mode == "dynamic":
            provider_args = [
                "--amdf_library=$(rootpath //libamdf:amdf_shared_artifact)",
            ]
        for lifetime, suffix in [("process", ""), ("instance", "_instance")]:
            test_name = name + "_" + mode + suffix
            iree_executable_test(
                name = test_name,
                src = ":" + binary_name,
                args = ["--amdf_native_lifetime=" + lifetime] + provider_args,
                data = data,
                tags = test_tags,
                target_compatible_with = policy["target_compatible_with"],
                visibility = visibility,
            )
            tests.append(":" + test_name)
    native.test_suite(
        name = name,
        tests = tests,
        visibility = visibility,
    )
