# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native Bazel test leaf for a CAS-backed Windows test manifest."""

_MANIFEST_ENVIRONMENT_VARIABLE = "IREE_WINDOWS_CACHE_HANDOFF_MANIFEST"

def _runfile_path(ctx, file):
    if file.short_path.startswith("../"):
        return file.short_path[3:]
    return ctx.workspace_name + "/" + file.short_path

def _windows_cache_handoff_test_impl(ctx):
    runner = ctx.attr._runner[DefaultInfo]
    runner_executable = runner.files_to_run.executable
    if runner_executable == None:
        fail("cache handoff runner has no executable")
    executable = ctx.actions.declare_file(ctx.label.name + ".bat")
    runner_path = _runfile_path(ctx, runner_executable).replace("/", "\\")
    ctx.actions.write(
        output = executable,
        content = (
            "@echo off\r\n" +
            '"%%TEST_SRCDIR%%\\%s" run -- %%*\r\n' % runner_path +
            "exit /b %ERRORLEVEL%\r\n"
        ),
        is_executable = True,
    )
    runfiles = ctx.runfiles(
        files = [ctx.file.manifest, runner_executable],
    ).merge(runner.default_runfiles)
    return [
        DefaultInfo(
            executable = executable,
            runfiles = runfiles,
        ),
        testing.TestEnvironment(
            environment = {
                _MANIFEST_ENVIRONMENT_VARIABLE: ctx.file.manifest.short_path,
            },
            inherited_environment = ctx.attr.env_inherit,
        ),
    ]

windows_cache_handoff_test = rule(
    implementation = _windows_cache_handoff_test_impl,
    attrs = {
        "env_inherit": attr.string_list(
            doc = "Source-test environment variables inherited by the native test leaf.",
        ),
        "manifest": attr.label(
            allow_single_file = [".json"],
            mandatory = True,
            doc = "Verified CAS closure and source-test execution metadata.",
        ),
        "_runner": attr.label(
            default = "//build_tools/ci:windows_cache_handoff",
            executable = True,
            cfg = "target",
        ),
    },
    doc = "Runs one Bazel-selected test from a verified CAS closure.",
    test = True,
)
