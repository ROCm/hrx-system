# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native Bazel test leaf for a CAS-backed Windows test manifest."""

_MANIFEST_ENVIRONMENT_VARIABLE = "IREE_WINDOWS_CACHE_HANDOFF_MANIFEST"
_PYTHON_ENVIRONMENT_VARIABLE = "IREE_WINDOWS_CACHE_HANDOFF_PYTHON"

def _runfile_path(ctx, file):
    if file.short_path.startswith("../"):
        return file.short_path[3:]
    return ctx.workspace_name + "/" + file.short_path

def _windows_cache_handoff_test_impl(ctx):
    runner = ctx.file._runner
    executable = ctx.actions.declare_file(ctx.label.name + ".bat")
    runner_path = _runfile_path(ctx, runner).replace("/", "\\")
    ctx.actions.write(
        output = executable,
        content = (
            "@echo off\r\n" +
            '"%%%s%%" "%%TEST_SRCDIR%%\\%s" run -- %%*\r\n' % (_PYTHON_ENVIRONMENT_VARIABLE, runner_path) +
            "exit /b %ERRORLEVEL%\r\n"
        ),
        is_executable = True,
    )
    runfiles = ctx.runfiles(
        files = [ctx.file.manifest, runner],
    )
    return [
        DefaultInfo(
            executable = executable,
            runfiles = runfiles,
        ),
        testing.TestEnvironment(
            environment = {
                _MANIFEST_ENVIRONMENT_VARIABLE: ctx.file.manifest.short_path,
            },
            inherited_environment = ctx.attr.env_inherit + [_PYTHON_ENVIRONMENT_VARIABLE],
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
            allow_single_file = [".py"],
            default = "//handoff:runner.py",
        ),
    },
    doc = "Runs one Bazel-selected test from a verified CAS closure.",
    test = True,
)
