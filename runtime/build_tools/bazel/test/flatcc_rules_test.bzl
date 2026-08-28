# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for runtime flatcc Bazel rules."""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "util")
load(
    "//runtime/build_tools/bazel:flatcc.bzl",
    "IreeFlatccInfo",
    "iree_runtime_flatbuffer_c_headers",
)

def _basenames(files):
    return [
        file.basename
        for file in files
    ]

def _expect_values(env, actual, expected):
    missing = [
        value
        for value in expected
        if value not in actual
    ]
    if missing:
        env.fail("expected %r in %r" % (missing, actual))

def _all_compilation_paths(compilation_context):
    return [
        str(path)
        for path in (
            compilation_context.includes.to_list() +
            compilation_context.quote_includes.to_list() +
            compilation_context.system_includes.to_list()
        )
    ]

def _test_flatcc_headers_declares_selected_outputs(name, **kwargs):
    util.helper_target(
        iree_runtime_flatbuffer_c_headers,
        name = name + "_subject",
        flatcc_args = [
            "--reader",
            "--builder",
            "--verifier",
        ],
        flatcc_includes = ["flatcc_test_include.fbs"],
        srcs = ["flatcc_test_schema.fbs"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_flatcc_headers_declares_selected_outputs_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_flatcc_headers_declares_selected_outputs_impl(env, target):
    info = target[IreeFlatccInfo]
    _expect_values(
        env,
        _basenames(info.headers.to_list()),
        [
            "flatcc_test_schema_reader.h",
            "flatcc_test_schema_builder.h",
            "flatcc_test_schema_verifier.h",
        ],
    )
    _expect_values(
        env,
        _basenames(info.schema_includes.to_list()),
        ["flatcc_test_include.fbs"],
    )
    _expect_values(
        env,
        info.args,
        [
            "--reader",
            "--builder",
            "--verifier",
        ],
    )

def _test_flatcc_library_adds_runtime_include_root(name, **kwargs):
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_flatcc_library_adds_runtime_include_root_impl,
        target = ":flatcc_test_schema",
        **kwargs
    )

def _test_flatcc_library_adds_runtime_include_root_impl(env, target):
    paths = _all_compilation_paths(target[CcInfo].compilation_context)
    for path in paths:
        if path.endswith("runtime/src"):
            return
    env.fail("expected one of %s to end with 'runtime/src'" % paths)

def flatcc_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_flatcc_headers_declares_selected_outputs,
            _test_flatcc_library_adds_runtime_include_root,
        ],
    )
