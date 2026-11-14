# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native Bazel test leaf for a CAS-backed Windows test manifest."""

_MANIFEST_ENVIRONMENT_VARIABLE = "IREE_WINDOWS_CACHE_HANDOFF_MANIFEST"

def _windows_cache_handoff_test_impl(ctx):
    runner = ctx.attr._runner[DefaultInfo]
    runner_executable = runner.files_to_run.executable
    if runner_executable == None:
        fail("cache handoff runner has no executable")
    extension = runner_executable.extension
    output_name = ctx.label.name + (("." + extension) if extension else "")
    executable = ctx.actions.declare_file(output_name)
    ctx.actions.symlink(
        is_executable = True,
        output = executable,
        target_file = runner_executable,
    )
    runfiles = ctx.runfiles(files = [ctx.file.manifest]).merge(runner.default_runfiles)
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
