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
load("//build_tools/embed_data:build_defs.bzl", "iree_c_embed_data")
load("//runtime/build_tools/bazel:cc.bzl", "iree_runtime_cc_binary", "iree_runtime_cc_library")
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
    ("queue_dispatch_tests", "//runtime/src/iree/hal/cts/queue:all_executable_tests"),
    ("sanitizer_tests", "//runtime/src/iree/hal/cts/sanitizer:all_tests"),
]

_COMMON_DEPS = [
    "//runtime/src/iree/hal/cts/util:test_main",
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

def _testdata_registration_impl(ctx):
    ctx.actions.expand_template(
        template = ctx.file._template,
        output = ctx.outputs.out,
        substitutions = ctx.attr.substitutions,
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

_testdata_registration = rule(
    implementation = _testdata_registration_impl,
    attrs = {
        "out": attr.output(mandatory = True),
        "substitutions": attr.string_dict(mandatory = True),
        "_template": attr.label(
            default = Label("//runtime/src/iree/hal/cts/util:testdata_target.cc.tpl"),
            allow_single_file = True,
        ),
    },
)

def iree_runtime_hal_cts_testdata(
        name,
        srcs,
        backend_names,
        target_name,
        target_family,
        target_key,
        **kwargs):
    """Embeds one target's compiled corpus and registers it with CTS backends.

    Artifact production and configured-target selection belong to the caller.
    Source basenames are the logical names consumed by CTS executable lookups.

    Args:
      name: Registration library target and generated-file prefix.
      srcs: Compiled executable artifacts for one target.
      backend_names: Backend views receiving this target's registration.
      target_name: GTest-compatible executable target name.
      target_family: HAL executable target family.
      target_key: Canonical family-owned artifact identity.
      **kwargs: Common attributes, including configured-target compatibility.
    """
    identifier = name.replace("-", "_").replace(".", "_")
    data_name = name + "_data"
    header = data_name + ".h"
    iree_c_embed_data(
        name = data_name,
        srcs = srcs,
        c_file_output = data_name + ".c",
        h_file_output = header,
        identifier = identifier,
        flatten = True,
        testonly = True,
        **kwargs
    )
    registrations = []
    for ordinal, backend_name in enumerate(backend_names):
        registration = "%s_reg%d.cc" % (name, ordinal)
        _testdata_registration(
            name = "%s_reg%d" % (name, ordinal),
            out = registration,
            substitutions = {
                "{BACKEND_NAME}": backend_name,
                "{HEADER_PATH}": "%s/%s" % (native.package_name(), header),
                "{IDENTIFIER}": identifier,
                "{TARGET_FAMILY}": target_family,
                "{TARGET_FUNC_NAME}": identifier,
                "{TARGET_KEY}": target_key,
                "{TARGET_NAME}": target_name,
                "{TARGET_VAR_NAME}": identifier,
            },
            testonly = True,
            **kwargs
        )
        registrations.append(registration)
    iree_runtime_cc_library(
        name = name,
        srcs = registrations,
        deps = [
            ":" + data_name,
            "//runtime/src/iree/hal/cts/util:registry",
        ],
        alwayslink = True,
        testonly = True,
        **kwargs
    )

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
