# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for prefix-based package policy."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test")
load(
    "//build_tools/bazel:package_policy.bzl",
    "collect_package_policy",
    "package_policy",
)

_BROAD_BUILD_REQUIREMENT = struct(id = "synthetic.broad")
_HOST_BUILD_REQUIREMENT = struct(id = "synthetic.host")
_GPU_RUN_REQUIREMENT = struct(id = "synthetic.gpu")

_PackagePolicyInfo = provider(
    fields = {
        "build_requirement_ids": "Collected build requirement identifiers.",
        "forbidden_deps": "Collected forbidden dependency patterns.",
        "resource_group": "Collected resource group or an empty string.",
        "run_requirement_ids": "Collected run requirement identifiers.",
    },
)

def _subject_impl(ctx):
    policy = collect_package_policy(
        ctx.attr.package_name,
        [
            package_policy(
                packages = ["synthetic/..."],
                excluded_packages = ["synthetic/host/..."],
                build_requirements = [_BROAD_BUILD_REQUIREMENT],
                forbidden_deps = ["//synthetic/forbidden/..."],
                run_requirements = [_GPU_RUN_REQUIREMENT],
                resource_group = "gpu",
            ),
            package_policy(
                packages = ["synthetic/host/..."],
                build_requirements = [_HOST_BUILD_REQUIREMENT],
            ),
        ],
    )
    return [_PackagePolicyInfo(
        build_requirement_ids = [
            requirement.id
            for requirement in policy.build_requirements
        ],
        forbidden_deps = policy.forbidden_deps,
        resource_group = policy.resource_group or "",
        run_requirement_ids = [
            requirement.id
            for requirement in policy.run_requirements
        ],
    )]

_subject = rule(
    implementation = _subject_impl,
    attrs = {
        "package_name": attr.string(mandatory = True),
    },
)

def _included_test_impl(env, target):
    policy = target[_PackagePolicyInfo]
    env.expect.that_collection(policy.build_requirement_ids).contains_exactly([
        "synthetic.broad",
    ])
    env.expect.that_collection(policy.forbidden_deps).contains_exactly([
        "//synthetic/forbidden/...",
    ])
    env.expect.that_collection(policy.run_requirement_ids).contains_exactly([
        "synthetic.gpu",
    ])
    env.expect.that_str(policy.resource_group).equals("gpu")

def _excluded_test_impl(env, target):
    policy = target[_PackagePolicyInfo]
    env.expect.that_collection(policy.build_requirement_ids).contains_exactly([
        "synthetic.host",
    ])
    env.expect.that_collection(policy.forbidden_deps).contains_exactly([])
    env.expect.that_collection(policy.run_requirement_ids).contains_exactly([])
    env.expect.that_str(policy.resource_group).equals("")

def package_policy_test_suite(name):
    """Defines package-policy analysis tests.

    Args:
      name: Test suite target name.
    """
    included_subject = name + "_included_subject"
    excluded_subject = name + "_excluded_subject"
    included_test = name + "_included"
    excluded_test = name + "_excluded"

    _subject(
        name = included_subject,
        package_name = "synthetic/device",
        tags = ["manual"],
    )
    _subject(
        name = excluded_subject,
        package_name = "synthetic/host/child",
        tags = ["manual"],
    )
    analysis_test(
        name = included_test,
        impl = _included_test_impl,
        target = ":" + included_subject,
    )
    analysis_test(
        name = excluded_test,
        impl = _excluded_test_impl,
        target = ":" + excluded_subject,
    )
    native.test_suite(
        name = name,
        tests = [
            ":" + excluded_test,
            ":" + included_test,
        ],
    )
