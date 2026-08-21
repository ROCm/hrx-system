# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for exposing existing executables as binaries or tests."""

load(
    "//build_tools/wasm:build_defs.bzl",
    "collect_and_bundle_wasm",
    "collect_wasm_js",
    "discover_wasm_entry",
)
load(
    ":dynamic_library_bundle.bzl",
    "collect_dynamic_library_bundles",
    "inject_dynamic_library_bindings",
)

IreeExecutableInfo = provider(
    doc = "Metadata for an executable alias or test wrapper.",
    fields = {
        "data": "depset of additional runtime data files.",
        "env": "Environment variables expanded against the wrapper runfiles.",
        "output": "Executable symlink produced by the wrapper.",
        "src": "Wrapped executable label.",
    },
)

def _merge_runfiles(ctx):
    runfiles = ctx.runfiles(files = ctx.files.data)
    return runfiles.merge_all(
        [ctx.attr.src[DefaultInfo].default_runfiles] +
        [target[DefaultInfo].default_runfiles for target in ctx.attr.data],
    )

def _expand_env(ctx):
    return {
        key: ctx.expand_location(value, ctx.attr.data + [ctx.attr.src])
        for key, value in ctx.attr.env.items()
    }

def _native_executable_output(ctx):
    output_name = ctx.attr.out
    if not output_name:
        output_name = ctx.label.name
    output = ctx.actions.declare_file(output_name)
    ctx.actions.symlink(
        is_executable = True,
        output = output,
        target_file = ctx.executable.src,
    )
    return output

def _is_wasm_target(ctx):
    return ctx.target_platform_has_constraint(
        ctx.attr._wasm32_constraint[platform_common.ConstraintValueInfo],
    )

def _wasm_entry(ctx, allow_default_test_main):
    entry = discover_wasm_entry([ctx.attr.src])
    if entry != None:
        return struct(
            main = entry.main,
            srcs = list(entry.srcs),
        )
    if allow_default_test_main:
        return struct(
            main = ctx.file._wasm_test_main,
            srcs = [],
        )
    fail("%s needs an iree_wasm_entry dependency when wrapping wasm binaries" % ctx.label)

def _wasm_executable_output(ctx, allow_default_test_main):
    output_name = ctx.attr.out
    if not output_name:
        output_name = ctx.label.name
    output = ctx.actions.declare_file(output_name)
    entry = _wasm_entry(ctx, allow_default_test_main)
    output_mjs = collect_and_bundle_wasm(
        ctx = ctx,
        wasm_binary = ctx.executable.src,
        main_js = entry.main,
        cc_deps = [ctx.attr.src],
        bundler = ctx.executable._wasm_bundler,
        main_srcs = entry.srcs,
    )

    wrapper_content = (
        "#!/usr/bin/env bash\n" +
        "set -euo pipefail\n" +
        "RUNFILES=\"${{RUNFILES_DIR:-$0.runfiles}}\"\n" +
        "exec \"${{RUNFILES}}/{workspace}/{runner}\" " +
        "\"${{RUNFILES}}/{bundle}\" \"$@\"\n"
    ).format(
        workspace = ctx.workspace_name,
        runner = ctx.file._wasm_runner.short_path,
        bundle = output_mjs.short_path,
    )
    ctx.actions.write(
        content = wrapper_content,
        is_executable = True,
        output = output,
    )
    return struct(
        bundle = output_mjs,
        output = output,
        runfiles = ctx.runfiles(files = [
            ctx.executable.src,
            ctx.file._wasm_runner,
            output_mjs,
        ]),
    )

def _iree_executable_alias_impl(ctx):
    if _is_wasm_target(ctx):
        wasm_output = _wasm_executable_output(ctx, allow_default_test_main = False)
        output = wasm_output.output
        runfiles = _merge_runfiles(ctx).merge(wasm_output.runfiles)
    else:
        output = _native_executable_output(ctx)
        runfiles = _merge_runfiles(ctx)
    providers = [
        DefaultInfo(
            executable = output,
            files = depset([output]),
            runfiles = runfiles,
        ),
        IreeExecutableInfo(
            data = depset(ctx.files.data),
            env = {},
            output = output,
            src = ctx.attr.src.label,
        ),
    ]
    if RunEnvironmentInfo in ctx.attr.src:
        source_environment = ctx.attr.src[RunEnvironmentInfo]
        providers.append(RunEnvironmentInfo(
            environment = source_environment.environment,
            inherited_environment = source_environment.inherited_environment,
        ))
    return inject_dynamic_library_bindings(
        ctx,
        providers,
        ctx.attr.data + [ctx.attr.src],
        output,
    )

def _iree_executable_test_impl(ctx):
    if _is_wasm_target(ctx):
        wasm_output = _wasm_executable_output(ctx, allow_default_test_main = True)
        output = wasm_output.output
        runfiles = _merge_runfiles(ctx).merge(wasm_output.runfiles)
    else:
        output = _native_executable_output(ctx)
        runfiles = _merge_runfiles(ctx)
    expanded_env = {}
    inherited_environment = list(ctx.attr.env_inherit)
    if RunEnvironmentInfo in ctx.attr.src:
        source_environment = ctx.attr.src[RunEnvironmentInfo]
        expanded_env.update(source_environment.environment)
        inherited_environment.extend(source_environment.inherited_environment)
    expanded_env.update(_expand_env(ctx))
    providers = [
        DefaultInfo(
            executable = output,
            files = depset([output]),
            runfiles = runfiles,
        ),
        IreeExecutableInfo(
            data = depset(ctx.files.data),
            env = expanded_env,
            output = output,
            src = ctx.attr.src.label,
        ),
        testing.TestEnvironment(
            environment = expanded_env,
            inherited_environment = depset(inherited_environment).to_list(),
        ),
    ]
    return inject_dynamic_library_bindings(
        ctx,
        providers,
        ctx.attr.data + [ctx.attr.src],
        output,
    )

_SHARED_ATTRS = {
    "data": attr.label_list(
        allow_files = True,
        aspects = [collect_dynamic_library_bundles],
        doc = "Runtime data dependencies available to the wrapped executable.",
    ),
    "out": attr.string(
        doc = "Output executable filename. Defaults to the target name.",
    ),
    "src": attr.label(
        allow_files = True,
        aspects = [collect_wasm_js, collect_dynamic_library_bundles],
        cfg = "target",
        doc = "Executable target or file to expose.",
        executable = True,
        mandatory = True,
    ),
    "_wasm32_constraint": attr.label(
        default = "@platforms//cpu:wasm32",
    ),
    "_wasm_bundler": attr.label(
        cfg = "exec",
        default = "//build_tools/wasm:wasm_binary_bundler",
        executable = True,
    ),
    "_wasm_runner": attr.label(
        allow_single_file = True,
        default = "//build_tools/wasm:wasm_node_test_runner.sh",
    ),
    "_wasm_test_main": attr.label(
        allow_single_file = True,
        default = "//build_tools/wasm:wasm_test_main.mjs",
    ),
}

_TEST_ATTRS = dict(_SHARED_ATTRS)
_TEST_ATTRS["env"] = attr.string_dict(
    doc = "Environment variables passed to the wrapped executable. Values may use $(location) for src or data labels.",
)
_TEST_ATTRS["env_inherit"] = attr.string_list(
    doc = "Host environment variable names inherited by the wrapped test.",
)

iree_executable_alias = rule(
    implementation = _iree_executable_alias_impl,
    attrs = _SHARED_ATTRS,
    doc = "Exposes an executable target or file as another executable target.",
    executable = True,
)

iree_executable_test = rule(
    implementation = _iree_executable_test_impl,
    attrs = _TEST_ATTRS,
    doc = "Runs an executable target or file directly as a Bazel test.",
    test = True,
)
