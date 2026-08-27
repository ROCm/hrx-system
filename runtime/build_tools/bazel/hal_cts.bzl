# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime HAL CTS Bazel macros.

CTS binaries link a driver-provided backend library and optional prebuilt
executable testdata registration libraries. Device artifact production is
backend-owned and remains outside this runtime composition rule.
"""

load("//build_tools/bazel:cc_attrs.bzl", "cc_attrs")
load("//build_tools/bazel:executable.bzl", "iree_executable_test")
load("//runtime/build_tools/bazel:cc.bzl", "iree_runtime_cc_binary")
load("//runtime/requirements:package_policy.bzl", "apply_runtime_test_policy")

_NON_EXECUTABLE_TEST_SUITES = [
    ("buffer_tests", "//runtime/src/iree/hal/cts/buffer:all_tests"),
    ("command_buffer_tests", "//runtime/src/iree/hal/cts/command_buffer:all_tests"),
    ("core_tests", "//runtime/src/iree/hal/cts/core:all_tests"),
    ("file_tests", "//runtime/src/iree/hal/cts/file:all_tests"),
    ("queue_tests", "//runtime/src/iree/hal/cts/queue:all_tests"),
]

_EXECUTABLE_TEST_SUITES = [
    ("dispatch_tests", "//runtime/src/iree/hal/cts/command_buffer:all_dispatch_tests"),
    ("executable_tests", "//runtime/src/iree/hal/cts/core:all_executable_tests"),
    ("queue_dispatch_tests", "//runtime/src/iree/hal/cts/queue:queue_dispatch_test"),
    ("sanitizer_tests", "//runtime/src/iree/hal/cts/sanitizer:all_tests"),
]

_COMMON_DEPS = [
    "//runtime/src/iree/base/tooling:flags",
    "//runtime/src/iree/hal/cts/util:registry",
    "//runtime/src/iree/hal/cts/util:test_base",
    "//runtime/src/iree/testing:gtest",
]

_TEST_ATTRS = [
    "data",
    "env",
    "env_inherit",
    "flaky",
    "local",
    "shard_count",
    "size",
    "timeout",
]

def _split_test_attrs(attrs):
    target_attrs = dict(attrs)
    test_attrs = {}
    for name in _TEST_ATTRS:
        if name in target_attrs:
            test_attrs[name] = target_attrs.pop(name)
    return struct(
        target = target_attrs,
        test = test_attrs,
    )

def _merge_dicts(*dicts):
    result = {}
    for source in dicts:
        result.update(source)
    return result

def _normalize_list(values):
    if values == None:
        return []
    return values

def _hal_cts_test(
        name,
        deps,
        args,
        resource_group,
        tags,
        testonly,
        target_attrs,
        test_attrs):
    binary_name = name + "_bin"
    iree_runtime_cc_binary(
        name = binary_name,
        testonly = testonly,
        srcs = ["//runtime/src/iree/hal/cts/util:test_main.cc"],
        deps = deps,
        **target_attrs
    )

    iree_executable_test(
        name = name,
        args = args,
        src = ":" + binary_name,
        tags = cc_attrs.with_resource_group_tags(tags, resource_group),
        testonly = testonly,
        **_merge_dicts(target_attrs, test_attrs)
    )

def iree_runtime_hal_cts_test_suite(
        backends,
        testdata_libs = None,
        name = "",
        args = None,
        resource_group = None,
        tags = None,
        testonly = True,
        **kwargs):
    """Generates HAL CTS test binaries for a runtime driver.

    Non-executable CTS binaries are always emitted. Executable-dependent CTS
    binaries are emitted when `testdata_libs` supplies libraries that register
    executable testdata with the CTS registry.

    Args:
      backends: Driver-specific library that registers CTS backends.
      testdata_libs: Prebuilt executable testdata registration libraries.
      name: Optional prefix for generated test targets.
      args: Command-line arguments passed to each generated test.
      resource_group: Local resource group used to serialize tests.
      tags: Tags applied to generated tests.
      testonly: Whether generated binaries and tests are test-only targets.
      **kwargs: Common Bazel attributes forwarded to generated targets. Test
        attributes such as `data`, `env`, `size`, and `timeout` apply only to
        the generated test wrapper.
    """
    policy_attrs = dict(kwargs)
    policy_attrs["resource_group"] = resource_group
    policy_attrs["tags"] = tags
    policy_attrs = apply_runtime_test_policy(policy_attrs)
    resource_group = policy_attrs.pop("resource_group", None)
    tags = policy_attrs.pop("tags", None)

    split_attrs = _split_test_attrs(policy_attrs)
    common_deps = [backends] + _COMMON_DEPS
    args = _normalize_list(args)
    tags = _normalize_list(tags)
    testdata_libs = _normalize_list(testdata_libs)
    prefix = (name + "_") if name else ""

    for suffix, suite_dep in _NON_EXECUTABLE_TEST_SUITES:
        _hal_cts_test(
            name = prefix + suffix,
            deps = common_deps + [suite_dep],
            args = args,
            resource_group = resource_group,
            tags = tags,
            testonly = testonly,
            target_attrs = split_attrs.target,
            test_attrs = split_attrs.test,
        )

    if testdata_libs:
        for suffix, suite_dep in _EXECUTABLE_TEST_SUITES:
            _hal_cts_test(
                name = prefix + suffix,
                deps = common_deps + testdata_libs + [suite_dep],
                args = args,
                resource_group = resource_group,
                tags = tags,
                testonly = testonly,
                target_attrs = split_attrs.target,
                test_attrs = split_attrs.test,
            )
