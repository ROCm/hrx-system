# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Grouped conformance corpora for the public libamdf ABI."""

load("//build_tools/bazel:cc_attrs.bzl", "cc_attrs")
load("//build_tools/bazel:executable.bzl", "iree_executable_test")
load("//build_tools/bazel:execution_profiles.bzl", "native_gpu_test_compatibility")
load("//build_tools/third_party/rocjitsu:profiles.bzl", "ROCJITSU_PROFILES", "rocjitsu_test_compatibility")
load("//libamdf/requirements:package_policy.bzl", "apply_amdf_test_policy")
load(":cc.bzl", "amdf_cc_binary", "amdf_cc_library")
load(":rocjitsu.bzl", "amdf_rocjitsu_cts_invocation")

def amdf_cts_test_suite(
        name,
        srcs,
        deps,
        defines = None,
        linkage_modes = ["dynamic"],
        linkopts = None,
        native_lifetimes = ["process"],
        rocjitsu_profiles = [],
        rocjitsu_required_tests = [],
        tags = None,
        resource_group = None,
        target_compatible_with = None,
        visibility = None):
    """Runs one CTS corpus through the selected library providers.

    Args:
      name: Aggregate test-suite target name.
      srcs: Sources for one independently selectable conformance corpus.
      deps: Public API and test-helper dependencies of the corpus.
      defines: Preprocessor definitions applied to the corpus sources.
      linkage_modes: Providers to link: dynamic (runtime-loaded, the default),
        shared (link-time shared), or static. Each mode adds one executable.
      linkopts: Native libraries required by the foreign API under test.
      native_lifetimes: Native ownership policies to exercise. Ordinary
        corpora use process lifetime; focused lifetime tests may also select
        instance lifetime.
      rocjitsu_profiles: Admitted functional topology keys for the dynamic corpus.
      rocjitsu_required_tests: Cases that must execute in every simulator invocation.
      tags: Additional tags applied to every generated test target.
      resource_group: Shared native resource used by the test invocations.
      target_compatible_with: Constraints required by every test mode.
      visibility: Visibility of the aggregate test suite.
    """
    policy = apply_amdf_test_policy({
        "resource_group": resource_group,
        "tags": tags or [],
        "target_compatible_with": target_compatible_with,
    })
    test_tags = cc_attrs.with_resource_group_tags(
        policy["tags"],
        policy.get("resource_group"),
    )
    corpus_tags = tags or []
    corpus_name = name + "_cases"
    amdf_cc_library(
        name = corpus_name,
        testonly = True,
        srcs = srcs,
        defines = defines,
        deps = deps,
        linkopts = linkopts,
        alwayslink = True,
        tags = corpus_tags,
        target_compatible_with = target_compatible_with,
    )
    common_deps = [":" + corpus_name, "//libamdf/cts/util:test_main"]
    runtime_data = ["//libamdf:amdf_runtime"]
    tests = []
    native_compatibility = policy["target_compatible_with"]
    if rocjitsu_profiles:
        if "dynamic" not in linkage_modes or native_lifetimes != ["process"]:
            fail("rocjitsu CTS matrices require dynamic linkage and process lifetime")
        native_compatibility = native_compatibility + native_gpu_test_compatibility()
    for mode in linkage_modes:
        binary_name = name + "_" + mode + "_bin"
        data = runtime_data
        if mode != "static":
            data = data + ["//libamdf:amdf_shared_artifact"]
        amdf_cc_binary(
            name = binary_name,
            testonly = True,
            data = data,
            deps = common_deps + ["//libamdf/cts/util:" + mode + "_provider"],
            tags = corpus_tags,
            target_compatible_with = target_compatible_with,
            # Invocation leaves reuse this corpus with their execution inputs.
            visibility = ["//libamdf/cts:__subpackages__"],
        )
        provider_args = []
        if mode == "dynamic":
            provider_args = [
                "--amdf_library=$(rootpath //libamdf:amdf_shared_artifact)",
            ]
        for lifetime in native_lifetimes:
            if lifetime not in ["process", "instance"]:
                fail("unsupported libamdf native lifetime: " + lifetime)
            suffix = "" if lifetime == "process" else "_" + lifetime
            test_name = name + "_" + mode + suffix
            iree_executable_test(
                name = test_name,
                src = ":" + binary_name,
                args = ["--amdf_native_lifetime=" + lifetime] + provider_args,
                data = data,
                tags = test_tags,
                target_compatible_with = native_compatibility,
                visibility = visibility,
            )
            tests.append(":" + test_name)
        if mode == "dynamic":
            for profile_name in rocjitsu_profiles:
                profile = ROCJITSU_PROFILES[profile_name]
                invocation_name = name + "_dynamic_rocjitsu_" + profile_name
                amdf_rocjitsu_cts_invocation(
                    name = invocation_name,
                    src = ":" + binary_name,
                    configurations = profile.configurations,
                    gpu_target = profile.gpu_target,
                    required_tests = rocjitsu_required_tests,
                    test_compatible_with = rocjitsu_test_compatibility(profile_name),
                    tags = corpus_tags,
                    visibility = visibility,
                )
                tests.append(":" + invocation_name + "_test")
    native.test_suite(
        name = name,
        tags = corpus_tags if rocjitsu_profiles else test_tags,
        tests = tests,
        visibility = visibility,
    )
