# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for exposing existing executables as binaries or tests."""

load("@bazel_skylib//lib:shell.bzl", "shell")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
load(
    "//build_tools/wasm:build_defs.bzl",
    "collect_and_bundle_wasm",
    "collect_wasm_js",
    "discover_wasm_entry",
)
load(":cc_attrs.bzl", "cc_attrs")
load(
    ":execution_requirements.bzl",
    "collect_execution_requirements",
    "inject_execution_requirements",
)
load(":runfiles.bzl", "create_runfiles_arguments_info")

_WINDOWS_LAUNCH_RUNFILE_ENV = "IREE_BAZEL_EXECUTABLE_RUNFILE"

# The launcher executes on the destination platform. Build it independently of
# the wrapped binary's instrumentation and compile/link customization so its
# own startup does not require DLLs stored beside the wrapped binary.
_LAUNCHER_CONFIGURATION = {
    "//build_tools/bazel:sanitizer": False,
    "//command_line_option:cc_output_directory_tag": "",
    "//command_line_option:collect_code_coverage": False,
    "//command_line_option:compilation_mode": "opt",
    "//command_line_option:conlyopt": [],
    "//command_line_option:copt": [],
    "//command_line_option:cxxopt": [],
    "//command_line_option:features": [],
    "//command_line_option:linkopt": [],
    "//command_line_option:per_file_copt": [],
    "@rules_cc//:link_extra_libs": "@rules_cc//:empty_lib",
}

# A host wrapper may be requested from an instrumented native build, but the
# WASI SDK has its own target feature set and no native sanitizer runtimes.
# Retain semantic build settings while removing host compiler customization.
_WASI_CONFIGURATION = {
    "//build_tools/bazel:sanitizer": False,
    "//command_line_option:cc_output_directory_tag": "",
    "//command_line_option:collect_code_coverage": False,
    "//command_line_option:conlyopt": [],
    "//command_line_option:copt": [],
    "//command_line_option:cxxopt": [],
    "//command_line_option:features": [],
    "//command_line_option:linkopt": [],
    "//command_line_option:per_file_copt": [],
    "//command_line_option:platforms": [],
}

def _launcher_transition_impl(_settings, _attr):
    return _LAUNCHER_CONFIGURATION

executable_launcher_transition = transition(
    implementation = _launcher_transition_impl,
    inputs = [],
    outputs = _LAUNCHER_CONFIGURATION.keys(),
)

def _wasi_transition_impl(settings, attr):
    configuration = dict(_WASI_CONFIGURATION)
    configuration["//command_line_option:platforms"] = [attr.target_platform]
    configuration["//command_line_option:features"] = [
        feature
        for feature in settings["//command_line_option:features"]
        if feature in ["optimize_for_speed", "thin_lto_link", "-optimize_for_speed", "-thin_lto_link"]
    ]
    return configuration

_wasi_transition = transition(
    implementation = _wasi_transition_impl,
    inputs = ["//command_line_option:features"],
    outputs = _WASI_CONFIGURATION.keys(),
)

IreeExecutableInfo = provider(
    doc = "Metadata for an executable alias or test wrapper.",
    fields = {
        "data": "depset of additional runtime data files.",
        "env": "Environment variables expanded against the wrapper runfiles.",
        "output": "Executable produced by the wrapper.",
        "src": "Wrapped executable label.",
    },
)

def _source_target(ctx):
    source = ctx.attr.src
    if type(source) == type([]):
        if len(source) != 1:
            fail("%s expected one configured source executable" % ctx.label)
        source = source[0]
    return source

def _merge_runfiles(ctx, tools = []):
    source = _source_target(ctx)
    runfiles = ctx.runfiles(
        files = ctx.files.data,
        transitive_files = depset(transitive = [tool[DefaultInfo].files for tool in tools]),
    )
    return runfiles.merge_all(
        [source[DefaultInfo].default_runfiles] +
        [target[DefaultInfo].default_runfiles for target in ctx.attr.data + tools],
    )

def _expand_env(ctx, tools):
    source = _source_target(ctx)
    return {
        key: ctx.expand_location(value, ctx.attr.data + tools + [source])
        for key, value in ctx.attr.env.items()
    }

def _is_wasm_target(ctx):
    return ctx.target_platform_has_constraint(
        ctx.attr._wasm32_constraint[platform_common.ConstraintValueInfo],
    )

def _is_windows_target(ctx):
    return ctx.target_platform_has_constraint(
        ctx.attr._windows_constraint[platform_common.ConstraintValueInfo],
    )

def _runfile_path(ctx, file):
    if file.short_path.startswith("../"):
        return file.short_path[3:]
    return ctx.workspace_name + "/" + file.short_path

# The Windows loader resolves implicit imports beside the path used to launch
# the image. A cross-package wrapper changes that path and can therefore strand
# source-adjacent DLLs. Windows wrappers without these DLLs remain direct
# symlinks.
def _needs_windows_launcher(ctx, output):
    if not _is_windows_target(ctx):
        return False
    source_output_directory = ctx.executable.src.dirname
    if source_output_directory == output.dirname:
        return False
    for file in _source_target(ctx)[DefaultInfo].default_runfiles.files.to_list():
        if (
            file.dirname == source_output_directory and
            file.extension.lower() == "dll"
        ):
            return True
    return False

def _native_executable_output(ctx):
    output_name = ctx.attr.out
    if not output_name:
        output_name = ctx.label.name
    output = ctx.actions.declare_file(output_name)
    if not _is_windows_target(ctx) and ctx.executable.src.dirname != output.dirname:
        # A declared file symlink can become ordinary file content remotely.
        # Launch the source at its declared runfile path so loader-relative
        # library paths retain the source package's directory depth.
        ctx.actions.write(
            output = output,
            content = (
                "#!/bin/sh\n" +
                'RUNFILES_DIR="${RUNFILES_DIR:-${TEST_SRCDIR:-$0.runfiles}}"\n' +
                "export RUNFILES_DIR\n" +
                'exec "$RUNFILES_DIR"/%s "$@"\n' %
                shell.quote(_runfile_path(ctx, ctx.executable.src))
            ),
            is_executable = True,
        )
        return struct(
            # The script embeds its source runfile; no private environment.
            launch_environment = {},
            # Public executable retaining the wrapper target's identity.
            output = output,
            # The source must exist even if its own runfiles omit itself.
            runfiles = ctx.runfiles(files = [ctx.executable.src]),
        )
    needs_launcher = _needs_windows_launcher(ctx, output)
    ctx.actions.symlink(
        is_executable = True,
        output = output,
        target_file = ctx.executable.windows_launcher if needs_launcher else ctx.executable.src,
    )
    return struct(
        launch_environment = {
            _WINDOWS_LAUNCH_RUNFILE_ENV: _runfile_path(ctx, ctx.executable.src),
        } if needs_launcher else {},
        output = output,
        runfiles = ctx.runfiles(
            files = [ctx.executable.src] if needs_launcher else [],
        ),
    )

def _merge_launch_environment(
        ctx,
        environment,
        inherited_environment,
        launch_environment):
    for name, value in launch_environment.items():
        if name in environment or name in inherited_environment:
            fail("%s reserves environment variable %s" % (ctx.label, name))
        environment[name] = value

def _wasm_entry(ctx, allow_default_test_main):
    if ctx.file.wasm_main != None:
        return struct(
            main = ctx.file.wasm_main,
            srcs = [],
        )
    entry = discover_wasm_entry([_source_target(ctx)])
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
    wasm_bundle = collect_and_bundle_wasm(
        ctx = ctx,
        wasm_binary = ctx.executable.src,
        main_js = entry.main,
        cc_deps = [_source_target(ctx)],
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
        bundle = _runfile_path(ctx, wasm_bundle.main),
    )
    ctx.actions.write(
        content = wrapper_content,
        is_executable = True,
        output = output,
    )
    return struct(
        bundle = wasm_bundle.main,
        output = output,
        runfiles = ctx.runfiles(files = [
            wasm_bundle.binary,
            ctx.file._wasm_runner,
            wasm_bundle.main,
        ]),
    )

def _executable_alias_providers(ctx, output, runfiles, launch_environment):
    source = _source_target(ctx)
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
            src = source.label,
        ),
    ]
    environment = {}
    inherited_environment = []
    if RunEnvironmentInfo in source:
        source_environment = source[RunEnvironmentInfo]
        environment.update(source_environment.environment)
        inherited_environment.extend(source_environment.inherited_environment)
    _merge_launch_environment(
        ctx,
        environment,
        inherited_environment,
        launch_environment,
    )
    if environment or inherited_environment:
        providers.append(RunEnvironmentInfo(
            environment = environment,
            inherited_environment = inherited_environment,
        ))
    runfiles_arguments = create_runfiles_arguments_info(
        ctx,
        ctx.attr.data + [source],
    )
    if runfiles_arguments != None:
        providers.append(runfiles_arguments)
    return inject_execution_requirements(
        ctx,
        providers,
        ctx.attr.data + [source],
        output,
    )

def _iree_executable_alias_impl(ctx):
    if _is_wasm_target(ctx):
        wasm_output = _wasm_executable_output(ctx, allow_default_test_main = False)
        output = wasm_output.output
        runfiles = _merge_runfiles(ctx).merge(wasm_output.runfiles)
        launch_environment = {}
    else:
        native_output = _native_executable_output(ctx)
        output = native_output.output
        runfiles = _merge_runfiles(ctx).merge(native_output.runfiles)
        launch_environment = native_output.launch_environment
    return _executable_alias_providers(
        ctx,
        output,
        runfiles,
        launch_environment,
    )

def _iree_wasi_executable_wrapper_impl(ctx):
    wasm_output = _wasm_executable_output(ctx, allow_default_test_main = False)
    return _executable_alias_providers(
        ctx,
        wasm_output.output,
        _merge_runfiles(ctx).merge(wasm_output.runfiles),
        {},
    )

def _iree_executable_test_impl(ctx):
    if _is_wasm_target(ctx):
        wasm_output = _wasm_executable_output(ctx, allow_default_test_main = True)
        output = wasm_output.output
        runfiles = _merge_runfiles(ctx, ctx.attr.tools).merge(wasm_output.runfiles)
        launch_environment = {}
    else:
        native_output = _native_executable_output(ctx)
        output = native_output.output
        runfiles = _merge_runfiles(ctx, ctx.attr.tools).merge(native_output.runfiles)
        launch_environment = native_output.launch_environment
    expanded_env = {}
    inherited_environment = list(ctx.attr.env_inherit)
    source = _source_target(ctx)
    if RunEnvironmentInfo in source:
        source_environment = source[RunEnvironmentInfo]
        expanded_env.update(source_environment.environment)
        inherited_environment.extend(source_environment.inherited_environment)
    expanded_env.update(_expand_env(ctx, ctx.attr.tools))
    test_environment = dict(expanded_env)
    _merge_launch_environment(
        ctx,
        test_environment,
        inherited_environment,
        launch_environment,
    )
    providers = [
        DefaultInfo(
            executable = output,
            files = depset([output]),
            runfiles = runfiles,
        ),
        IreeExecutableInfo(
            data = depset(ctx.files.data + ctx.files.tools),
            env = expanded_env,
            output = output,
            src = source.label,
        ),
        testing.TestEnvironment(
            environment = test_environment,
            inherited_environment = depset(inherited_environment).to_list(),
        ),
    ]
    runfiles_arguments = create_runfiles_arguments_info(
        ctx,
        ctx.attr.data + ctx.attr.tools + [source],
    )
    if runfiles_arguments != None:
        providers.append(runfiles_arguments)
    return inject_execution_requirements(
        ctx,
        providers,
        ctx.attr.data + ctx.attr.tools + [source],
        output,
    )

_SHARED_ATTRS = {
    "data": attr.label_list(
        allow_files = True,
        aspects = [collect_execution_requirements],
        doc = "Runtime data dependencies available to the wrapped executable.",
    ),
    "out": attr.string(
        doc = "Output executable filename. Defaults to the target name.",
    ),
    "src": attr.label(
        allow_files = True,
        aspects = [collect_wasm_js, collect_execution_requirements],
        cfg = "target",
        doc = "Executable target or file to expose.",
        executable = True,
        mandatory = True,
    ),
    "wasm_main": attr.label(
        allow_single_file = [".js", ".mjs"],
        doc = "Explicit JavaScript entry point used when wrapping a wasm executable.",
    ),
    "windows_launcher": attr.label(
        cfg = executable_launcher_transition,
        executable = True,
    ),
    "_allowlist_function_transition": attr.label(
        default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
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
    "_windows_constraint": attr.label(
        default = "@platforms//os:windows",
    ),
}

_WASI_ALIAS_ATTRS = dict(_SHARED_ATTRS)
_WASI_ALIAS_ATTRS["src"] = attr.label(
    allow_files = True,
    aspects = [collect_wasm_js, collect_execution_requirements],
    cfg = _wasi_transition,
    doc = "Executable target cross-compiled to WASI and exposed through a host wrapper.",
    executable = True,
    mandatory = True,
)
_WASI_ALIAS_ATTRS["target_platform"] = attr.label(
    default = "//build_tools/wasm:wasm32_wasi",
    doc = "WASI target platform used to configure the source executable.",
)

_TEST_ATTRS = dict(_SHARED_ATTRS)
_TEST_ATTRS["tools"] = attr.label_list(
    allow_files = True,
    aspects = [collect_execution_requirements],
    cfg = config.exec("test"),
    doc = "External helper tools built for the test execution platform, independently of the test subject's target flags.",
)
_TEST_ATTRS["env"] = attr.string_dict(
    doc = "Environment variables passed to the wrapped executable. Values may use $(location) for src, data, or tools labels.",
)
_TEST_ATTRS["env_inherit"] = attr.string_list(
    doc = "Host environment variable names inherited by the wrapped test.",
)

_iree_executable_alias = rule(
    implementation = _iree_executable_alias_impl,
    attrs = _SHARED_ATTRS,
    doc = "Exposes an executable target or file as another executable target.",
    executable = True,
)

_iree_wasi_executable_wrapper = rule(
    implementation = _iree_wasi_executable_wrapper_impl,
    attrs = _WASI_ALIAS_ATTRS,
    doc = "Cross-compiles an executable to WASI and writes its host wrapper.",
    executable = True,
)

_iree_executable_test = rule(
    implementation = _iree_executable_test_impl,
    attrs = _TEST_ATTRS,
    doc = "Runs an executable target or file directly as a Bazel test.",
    test = True,
)

_TOOL_TEST_ATTRS = dict(_TEST_ATTRS)
_TOOL_TEST_ATTRS["src"] = attr.label(
    allow_files = True,
    aspects = [collect_execution_requirements],
    cfg = config.exec("test"),
    doc = "External tool used to check target-configured artifacts.",
    executable = True,
    mandatory = True,
)

_iree_tool_test = rule(
    implementation = _iree_executable_test_impl,
    attrs = _TOOL_TEST_ATTRS,
    doc = "Checks target artifacts using an external tool built for the test execution platform.",
    test = True,
)

def _windows_launcher():
    return select({
        Label("@platforms//os:windows"): Label("//build_tools/bazel:executable_launcher"),
        "//conditions:default": None,
    })

def _executable_alias_macro_impl(name, visibility, **kwargs):
    _iree_executable_alias(
        name = name,
        visibility = visibility,
        windows_launcher = _windows_launcher(),
        **kwargs
    )

iree_executable_alias = macro(
    implementation = _executable_alias_macro_impl,
    inherit_attrs = _iree_executable_alias,
    attrs = {"windows_launcher": None},
    doc = "Exposes an executable target or file as another executable target.",
)

def _wasi_executable_alias_macro_impl(
        name,
        visibility,
        data,
        src,
        tags,
        target_platform,
        wasm_main,
        **kwargs):
    wrapper_name = name + "_wrapper"
    tags = tags or []
    _iree_wasi_executable_wrapper(
        name = wrapper_name,
        data = data,
        src = src,
        tags = tags + ["manual"],
        target_platform = target_platform,
        visibility = ["//%s:__pkg__" % native.package_name()],
        wasm_main = wasm_main,
        **kwargs
    )

    # Let rules_shell provide the destination platform's launcher instead of
    # exposing the generated POSIX script as a native executable on Windows.
    # The inherited environment remains attached when another test rule wraps
    # this alias, so Node installed by the host stays visible in its sandbox.
    sh_binary(
        name = name,
        data = [":" + wrapper_name],
        env_inherit = [
            "IREE_WASM_NODE",
            "PATH",
        ],
        srcs = [":" + wrapper_name],
        tags = tags,
        visibility = visibility,
        **kwargs
    )

iree_wasi_executable_alias = macro(
    inherit_attrs = _iree_wasi_executable_wrapper,
    implementation = _wasi_executable_alias_macro_impl,
    attrs = {
        "out": None,
        "windows_launcher": None,
    },
    doc = "Cross-compiles an executable to WASI and exposes a host-launchable alias.",
)

def _executable_test_macro_impl(name, visibility, tags, resource_group, **kwargs):
    _iree_executable_test(
        name = name,
        visibility = visibility,
        tags = cc_attrs.with_resource_group_tags(tags, resource_group),
        windows_launcher = _windows_launcher(),
        **kwargs
    )

iree_executable_test = macro(
    implementation = _executable_test_macro_impl,
    inherit_attrs = _iree_executable_test,
    attrs = {
        "resource_group": attr.string(
            configurable = False,
            doc = "Local resource name used to serialize tests competing for the same host resource.",
        ),
        "windows_launcher": None,
    },
    doc = "Runs an executable target or file directly as a Bazel test.",
)

def _tool_test_macro_impl(name, visibility, tags, resource_group, **kwargs):
    _iree_tool_test(
        name = name,
        visibility = visibility,
        tags = cc_attrs.with_resource_group_tags(tags, resource_group),
        windows_launcher = _windows_launcher(),
        **kwargs
    )

iree_tool_test = macro(
    implementation = _tool_test_macro_impl,
    inherit_attrs = _iree_tool_test,
    attrs = {
        "resource_group": attr.string(
            configurable = False,
            doc = "Local resource name used to serialize tests competing for the same host resource.",
        ),
        "windows_launcher": None,
    },
    doc = "Checks target artifacts with an external tool instead of testing the tool itself.",
)
