# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Macros for defining tests that run .loom-test files through loom-check."""

load("//build_tools/bazel:cc_attrs.bzl", "cc_attrs")
load(
    "//loom/requirements:package_policy.bzl",
    "apply_loom_test_policy",
)

_LOOM_CHECK_EXTENSION = ".loom-test"

LoomCheckTestInfo = provider(
    doc = "Metadata for a generated loom-check test wrapper.",
    fields = {
        "env": "Environment variables expanded against the wrapper runfiles.",
        "fixture": "Checked fixture file appended by the wrapper.",
        "output": "Executable test wrapper.",
        "runner": "loom-check compatible runner label.",
    },
)

def _loom_check_test_base_name(src):
    if not src.endswith(_LOOM_CHECK_EXTENSION):
        fail("loom_check_test source must use the .loom-test extension: %s" %
             src)
    return src[:-len(_LOOM_CHECK_EXTENSION)]

def _loom_check_expand_env(ctx):
    return {
        key: ctx.expand_location(
            value,
            ctx.attr.data + [ctx.attr.runner, ctx.attr.src],
        )
        for key, value in ctx.attr.env.items()
    }

def _loom_check_wrapper_content(ctx):
    return (
        "#!/usr/bin/env bash\n" +
        "set -euo pipefail\n" +
        "RUNFILES=\"${{RUNFILES_DIR:-$0.runfiles}}\"\n" +
        "cd \"${{RUNFILES}}/{workspace}\"\n" +
        "exec \"${{PWD}}/{runner}\" \"$@\" \"{fixture}\"\n"
    ).format(
        workspace = ctx.workspace_name,
        runner = ctx.executable.runner.short_path,
        fixture = ctx.file.src.short_path,
    )

def _loom_check_test_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(
        content = _loom_check_wrapper_content(ctx),
        is_executable = True,
        output = output,
    )

    runfiles = ctx.runfiles(
        files = [
            ctx.executable.runner,
            ctx.file.src,
        ] + ctx.files.data,
    )
    runfiles = runfiles.merge_all(
        [ctx.attr.runner[DefaultInfo].default_runfiles] +
        [target[DefaultInfo].default_runfiles for target in ctx.attr.data],
    )
    expanded_env = _loom_check_expand_env(ctx)
    return [
        DefaultInfo(
            executable = output,
            files = depset([output]),
            runfiles = runfiles,
        ),
        LoomCheckTestInfo(
            env = expanded_env,
            fixture = ctx.file.src,
            output = output,
            runner = ctx.attr.runner.label,
        ),
        testing.TestEnvironment(expanded_env),
    ]

_loom_check_executable_test = rule(
    implementation = _loom_check_test_impl,
    attrs = {
        "data": attr.label_list(
            allow_files = True,
            doc = "Runtime data dependencies available to loom-check.",
        ),
        "env": attr.string_dict(
            doc = (
                "Environment variables passed to loom-check. Values may use " +
                "$(location) for src, runner, or data labels."
            ),
        ),
        "runner": attr.label(
            allow_files = True,
            cfg = "target",
            default = "//loom/src/loom/tools/loom-check:loom-check",
            doc = "loom-check compatible runner binary.",
            executable = True,
        ),
        "src": attr.label(
            allow_single_file = [_LOOM_CHECK_EXTENSION],
            doc = "Source .loom-test file appended by the wrapper.",
            mandatory = True,
        ),
    },
    doc = "Runs a single .loom-test file through a loom-check compatible runner.",
    test = True,
)

def loom_check_test(
        name,
        src,
        size = "small",
        tags = [],
        data = [],
        env = {},
        runner = "//loom/src/loom/tools/loom-check:loom-check",
        **kwargs):
    """Creates a test that runs a single .loom-test file through loom-check.

    Args:
      name: Name of the generated test.
      src: Source .loom-test file containing the test cases.
      size: Test size (default: "small").
      tags: Additional tags to apply to the test.
      data: Additional runfiles made available to loom-check.
      env: Additional test environment variables.
      runner: loom-check compatible runner binary.
      **kwargs: Additional attributes passed to native_test.
    """
    _loom_check_test_base_name(src)
    kwargs = apply_loom_test_policy(kwargs, name = name)
    policy_tags = kwargs.pop("tags", [])
    resource_group = kwargs.pop("resource_group", None)
    _loom_check_executable_test(
        name = name,
        src = src,
        data = data,
        env = env,
        runner = runner,
        size = size,
        tags = cc_attrs.with_resource_group_tags(
            tags + ["loom-check"] + policy_tags,
            resource_group,
        ),
        **kwargs
    )

def loom_check_test_suite(
        name,
        srcs,
        size = "small",
        tags = [],
        data = [],
        env = {},
        runner = "//loom/src/loom/tools/loom-check:loom-check",
        test_name_prefix_to_strip = "",
        **kwargs):
    """Creates one test per .loom-test file, bundled into a test suite.

    Each .loom-test file becomes an independent test target. The test name
    is derived from the file path by replacing "/" with "_" and
    stripping the .loom-test extension.

    Args:
      name: Name of the generated test suite.
      srcs: List of .loom-test files to test.
      size: Test size (default: "small").
      tags: Additional tags to apply to each generated test and the suite.
      data: Additional runfiles made available to each generated test.
      env: Additional test environment variables for each generated test.
      runner: loom-check compatible runner binary.
      test_name_prefix_to_strip: Optional source path prefix to remove before
          deriving generated test names. This lets suites move files into a
          directory such as "test/" without changing the familiar test target
          names.
      **kwargs: Additional attributes passed to each loom_check_test
          and the test suite, when supported by both.
    """
    tests = []
    for src in srcs:
        test_name_src = _loom_check_test_base_name(src)
        if test_name_prefix_to_strip and test_name_src.startswith(test_name_prefix_to_strip):
            test_name_src = test_name_src[len(test_name_prefix_to_strip):]
        test_name = test_name_src.replace("/", "_")
        loom_check_test(
            name = test_name,
            src = src,
            size = size,
            tags = tags,
            data = data,
            env = env,
            runner = runner,
            **kwargs
        )
        tests.append(test_name)
    native.test_suite(name = name, tests = tests, tags = tags, **kwargs)
