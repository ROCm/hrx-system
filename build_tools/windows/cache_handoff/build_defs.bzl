# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native Bazel test leaf for a CAS-backed Windows test manifest."""

_MANIFEST_ENVIRONMENT_VARIABLE = "IREE_WINDOWS_CACHE_HANDOFF_MANIFEST"
_PYTHON_ENVIRONMENT_VARIABLE = "IREE_WINDOWS_CACHE_HANDOFF_PYTHON"
_CAS_DIRECTORY_ENVIRONMENT_VARIABLE = "IREE_WINDOWS_CACHE_HANDOFF_CAS_DIR"

def _windows_cache_handoff_test_impl(ctx):
    runner = ctx.actions.declare_file(ctx.label.name + ".runner.py")
    ctx.actions.expand_template(
        template = ctx.file._runner,
        output = runner,
        substitutions = {},
    )
    executable = ctx.actions.declare_file(ctx.label.name + ".bat")
    ctx.actions.write(
        output = executable,
        content = (
            "@echo off\r\n" +
            '"%%%s%%" "%%~dp0%s" run -- %%*\r\n' % (_PYTHON_ENVIRONMENT_VARIABLE, runner.basename) +
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
            files = depset([executable, runner]),
            runfiles = runfiles,
        ),
        testing.TestEnvironment(
            environment = {
                _MANIFEST_ENVIRONMENT_VARIABLE: ctx.file.manifest.short_path,
            },
            inherited_environment = ctx.attr.env_inherit + [
                _CAS_DIRECTORY_ENVIRONMENT_VARIABLE,
                _PYTHON_ENVIRONMENT_VARIABLE,
            ],
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
