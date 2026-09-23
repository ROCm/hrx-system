# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Macros for defining tests that run .<format>-test files through loom-check."""

load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//build_tools/bazel:cc_attrs.bzl", "cc_attrs")
load(
    "//loom/requirements:package_policy.bzl",
    "apply_loom_test_policy",
)
load(":loom_target_profile.bzl", "LoomTargetProfileInfo")

LoomCheckTestInfo = provider(
    doc = "Metadata for a generated loom-check test wrapper.",
    fields = {
        "compile_target": "Family-qualified compiler profile, or empty for RUN checks.",
        "env": "Environment variables expanded against the wrapper runfiles.",
        "fixture": "Checked fixture file appended by the wrapper.",
        "output": "Executable test wrapper.",
        "runner": "loom-check compatible runner label.",
    },
)

def _loom_check_test_base_name(src):
    base, separator, extension = src.rpartition(".")
    if not separator or not extension.endswith("-test") or extension == "-test":
        fail("loom_check_test source must use a .<format>-test extension: %s" % src)
    return base

def _loom_check_expand_env(ctx):
    return {
        key: ctx.expand_location(
            value,
            ctx.attr.data + [ctx.attr.runner, ctx.attr.src],
        )
        for key, value in ctx.attr.env.items()
    }

def _loom_check_wrapper_content(ctx):
    compile_argument = ""
    if ctx.attr.compile_target:
        profile = ctx.attr.compile_target[LoomTargetProfileInfo]
        value = ("--target=%s:%s" % (profile.family, profile.selector)).replace("'", "'\"'\"'")
        compile_argument = "'%s' " % value
    return (
        "#!/usr/bin/env bash\n" +
        "set -euo pipefail\n" +
        "RUNFILES=\"${{RUNFILES_DIR:-$0.runfiles}}\"\n" +
        "cd \"${{RUNFILES}}/{workspace}\"\n" +
        "exec \"${{PWD}}/{runner}\" \"$@\" " +
        "{compile_argument}\"{fixture}\"\n"
    ).format(
        workspace = ctx.workspace_name,
        runner = ctx.executable.runner.short_path,
        fixture = ctx.file.src.short_path,
        compile_argument = compile_argument,
    )

def _loom_check_test_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name + ".sh")
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
            compile_target = (
                "%s:%s" % (
                    ctx.attr.compile_target[LoomTargetProfileInfo].family,
                    ctx.attr.compile_target[LoomTargetProfileInfo].selector,
                ) if ctx.attr.compile_target else ""
            ),
            env = expanded_env,
            fixture = ctx.file.src,
            output = output,
            runner = ctx.attr.runner.label,
        ),
    ]

_loom_check_executable = rule(
    implementation = _loom_check_test_impl,
    attrs = {
        "compile_target": attr.label(
            providers = [LoomTargetProfileInfo],
            doc = "Optional offline compiler profile, independent of execution resources.",
        ),
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
            default = "//loom/src/loom/tools/loom-check:loom-check-test",
            doc = "loom-check compatible runner binary.",
            executable = True,
        ),
        "src": attr.label(
            allow_single_file = True,
            doc = "Source .<format>-test file appended by the wrapper.",
            mandatory = True,
        ),
    },
    doc = "Generates a shell launcher for one .<format>-test file.",
    executable = True,
)

def loom_check_compile_tests(name, src, targets, size = "small", tags = [], data = [], env = {}, **kwargs):
    """Adds compiler tests beside one source owner's RUN or execution test.

    Returns the generated test names so the owning suite includes every profile.
    The source is an ordinary fixture or the owner's linked test module; the
    same loom-check input admission and diagnostics apply to both.

    Args:
      name: Name prefix shared with the source owner's tests.
      src: Source fixture or linked test module.
      targets: Typed compiler profile labels.
      size: Bazel test size.
      tags: Additional test tags.
      data: Additional runtime data. TEMPLATE sources are not runtime inputs.
      env: Test environment variables.
      **kwargs: Additional sh_test attributes, including compiler arguments.

    Returns:
      Generated test names for inclusion in the owning suite.
    """
    tests = []
    for target in targets:
        label = native.package_relative_label(target)
        suffix = label.name.replace("-", "_").replace(".", "_").replace("+", "_")
        test_name = name + "_compile_" + suffix
        if test_name in tests:
            fail("compiler profiles for %s have colliding test names: %s" % (name, target))
        launcher_name = test_name + "_launcher"
        runner = Label("//loom/src/loom/tools/loom-check:loom-check")
        test_kwargs = apply_loom_test_policy(dict(kwargs), name = test_name)
        policy_tags = test_kwargs.pop("tags", [])
        resource_group = test_kwargs.pop("resource_group", None)
        _loom_check_executable(
            name = launcher_name,
            src = src,
            compile_target = target,
            runner = runner,
            data = data,
            env = env,
            tags = ["manual"],
            testonly = True,
        )
        sh_test(
            name = test_name,
            srcs = [":" + launcher_name],
            data = [":" + launcher_name],
            env = env,
            size = size,
            tags = cc_attrs.with_resource_group_tags(
                tags + policy_tags + ["loom-compile", "hostonly"],
                resource_group,
            ),
            **test_kwargs
        )
        tests.append(test_name)
    return tests

def loom_check_test(
        name,
        src,
        size = "small",
        tags = [],
        data = [],
        env = {},
        runner = Label("//loom/src/loom/tools/loom-check:loom-check-test"),
        compile_targets = [],
        **kwargs):
    """Creates a test that runs a single .<format>-test file through loom-check.

    Args:
      name: Name of the generated test.
      src: Source .<format>-test file containing the test cases.
      size: Test size (default: "small").
      tags: Additional tags to apply to the test.
      data: Additional runfiles made available to loom-check. TEMPLATE sources
          are checked by precommit and are not runtime inputs.
      env: Additional test environment variables.
      runner: loom-check compatible runner binary.
      compile_targets: Typed profiles checked independently of RUN goldens.
      **kwargs: Additional attributes passed to sh_test.
    """
    _loom_check_test_base_name(src)
    compile_tests = loom_check_compile_tests(
        name = name,
        src = src,
        targets = compile_targets,
        size = size,
        tags = tags,
        data = data,
        env = env,
        **kwargs
    )
    kwargs = apply_loom_test_policy(kwargs, name = name)
    policy_tags = kwargs.pop("tags", [])
    resource_group = kwargs.pop("resource_group", None)
    test_tags = cc_attrs.with_resource_group_tags(
        tags + ["loom-check"] + policy_tags,
        resource_group,
    )
    launcher_name = name + "_launcher"
    _loom_check_executable(
        name = launcher_name,
        src = src,
        data = data,
        env = env,
        runner = runner,
        tags = ["manual"],
        testonly = True,
    )
    run_name = name + "_run" if compile_tests else name
    sh_test(
        name = run_name,
        srcs = [":" + launcher_name],
        data = data + [runner, src],
        env = env,
        size = size,
        tags = test_tags,
        **kwargs
    )
    if compile_tests:
        native.test_suite(
            name = name,
            tests = [run_name] + compile_tests,
            tags = tags,
            visibility = kwargs.get("visibility"),
        )

def loom_check_test_suite(
        name,
        srcs,
        size = "small",
        tags = [],
        data = [],
        env = {},
        runner = Label("//loom/src/loom/tools/loom-check:loom-check-test"),
        compile_targets = {},
        test_name_prefix_to_strip = "",
        **kwargs):
    """Creates one test per .<format>-test file, bundled into a test suite.

    Each .<format>-test file becomes an independent test target. The test name
    is derived from the file path by replacing "/" with "_" and
    stripping the .<format>-test extension.

    Args:
      name: Name of the generated test suite.
      srcs: List of .<format>-test files to test.
      size: Test size (default: "small").
      tags: Additional tags to apply to each generated test and the suite.
      data: Additional runfiles made available to each generated test.
      env: Additional test environment variables for each generated test.
      runner: loom-check compatible runner binary.
      compile_targets: Source-to-profile lists for additional compiler checks.
          Keys must name sources in this suite; no separate case roster exists.
      test_name_prefix_to_strip: Optional source path prefix to remove before
          deriving generated test names. This lets suites move files into a
          directory such as "test/" without changing the familiar test target
          names.
      **kwargs: Additional attributes passed to each loom_check_test
          and the test suite, when supported by both.
    """
    for src in compile_targets:
        if src not in srcs:
            fail("compiler qualification source is not registered in %s: %s" % (name, src))
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
            compile_targets = compile_targets.get(src, []),
            **kwargs
        )
        tests.append(test_name)
    native.test_suite(name = name, tests = tests, tags = tags, **kwargs)
