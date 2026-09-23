# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exports and rebinds native executable tests for fixed CI configurations."""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_cc//cc/common:debug_package_info.bzl", "DebugPackageInfo")
load(":executable.bzl", "IreeExecutableInfo")
load(
    ":runfiles.bzl",
    "IreeRunfilesArgumentsInfo",
    "IreeRunfilesEnvironmentInfo",
    "RUNFILES_PATH_BEGIN",
    "RUNFILES_PATH_END",
)

IreeNativeArtifactInfo = provider(
    doc = "Files exported by the native executable artifact aspect.",
    fields = {
        "debug_files": "Transitive split-DWARF files for this configured target.",
        "metadata": "Per-executable metadata file, or None for non-executables.",
        "payload_files": "Executable runtime files, or an empty depset.",
    },
)

def _native_debug_files(target, ctx):
    transitive_files = []
    if DebugPackageInfo in target:
        transitive_files.append(target[DebugPackageInfo].dwo_files)
    if CcInfo in target:
        cc_info = target[CcInfo]

        # Match rules_cc while its public debug_context migration is in flight.
        debug_context = (
            cc_info._debug_context if hasattr(cc_info, "_debug_context") else cc_info.debug_context()
        )
        transitive_files.extend([debug_context.files, debug_context.pic_files])
    for attribute_name in [
        "data",
        "deps",
        "dynamic_deps",
        "implementation_deps",
        "src",
    ]:
        if not hasattr(ctx.rule.attr, attribute_name):
            continue
        value = getattr(ctx.rule.attr, attribute_name)
        dependencies = value if type(value) == "list" else [value]
        for dependency in dependencies:
            if type(dependency) == "Target" and IreeNativeArtifactInfo in dependency:
                transitive_files.append(
                    dependency[IreeNativeArtifactInfo].debug_files,
                )
    return depset(transitive = transitive_files)

def _native_executable_metadata(target, ctx, runner):
    metadata = {
        "arguments": [],
        "environment": {},
        "executable": runner.executable.path,
        "inherited_environment": [],
        "label": str(target.label),
        "repository_mapping": (
            runner.repo_mapping_manifest.path if runner.repo_mapping_manifest else None
        ),
        "rule_kind": ctx.rule.kind,
        "runfiles_manifest": (
            runner.runfiles_manifest.path if runner.runfiles_manifest else None
        ),
    }
    for name in [
        "exec_properties",
        "flaky",
        "shard_count",
        "size",
        "tags",
        "timeout",
    ]:
        if hasattr(ctx.rule.attr, name):
            metadata[name] = getattr(ctx.rule.attr, name)
    if IreeRunfilesArgumentsInfo in target:
        arguments = target[IreeRunfilesArgumentsInfo]
        metadata["arguments"] = arguments.arguments
        metadata["marked_arguments"] = arguments.marked_arguments
    else:
        for argument in getattr(ctx.rule.attr, "args", []):
            metadata["arguments"].extend(ctx.tokenize(argument))
    if RunEnvironmentInfo in target:
        environment = target[RunEnvironmentInfo]
        metadata["environment"] = environment.environment
        metadata["inherited_environment"] = environment.inherited_environment
    if IreeRunfilesEnvironmentInfo in target:
        metadata["marked_environment"] = (
            target[IreeRunfilesEnvironmentInfo].environment
        )
    return metadata

def _is_native_executable(target, ctx):
    if DebugPackageInfo in target:
        return True
    if (
        IreeExecutableInfo not in target or
        ctx.target_platform_has_constraint(
            ctx.attr._wasm32_constraint[platform_common.ConstraintValueInfo],
        )
    ):
        return False
    source = getattr(ctx.rule.attr, "src", None)
    return (
        type(source) == "Target" and
        IreeNativeArtifactInfo in source and
        source[IreeNativeArtifactInfo].metadata != None
    )

def _native_artifact_impl(target, ctx):
    debug_files = _native_debug_files(target, ctx)
    metadata_file = None
    payload_files = depset()
    if _is_native_executable(target, ctx):
        default_info = target[DefaultInfo]
        runner = default_info.files_to_run
        if runner == None or runner.executable == None:
            fail("%s exposes native executable metadata without an executable" % target.label)
        metadata_file = ctx.actions.declare_file(
            ctx.label.name + ".native-artifact.json",
        )
        ctx.actions.write(
            metadata_file,
            json.encode_indent(_native_executable_metadata(target, ctx, runner)) + "\n",
        )
        direct_files = [metadata_file, runner.executable]
        if runner.runfiles_manifest:
            direct_files.append(runner.runfiles_manifest)
        if runner.repo_mapping_manifest:
            direct_files.append(runner.repo_mapping_manifest)
        for entry in (
            default_info.default_runfiles.symlinks.to_list() +
            default_info.default_runfiles.root_symlinks.to_list()
        ):
            direct_files.append(entry.target_file)
        payload_files = depset(
            direct_files,
            transitive = [default_info.default_runfiles.files],
        )
    return [
        IreeNativeArtifactInfo(
            debug_files = debug_files,
            metadata = metadata_file,
            payload_files = payload_files,
        ),
        OutputGroupInfo(
            native_artifact_debug = debug_files,
            native_artifact_payload = payload_files,
        ),
    ]

native_test_artifact = aspect(
    implementation = _native_artifact_impl,
    attr_aspects = [
        "data",
        "deps",
        "dynamic_deps",
        "implementation_deps",
        "src",
    ],
    attrs = {
        "_wasm32_constraint": attr.label(
            default = "@platforms//cpu:wasm32",
        ),
    },
    doc = "Exports native executable metadata, runfiles, and split-DWARF closure.",
)

def _shell_quote(value):
    return "'" + value.replace("'", "'\"'\"'") + "'"

def _runtime_value(value, path_prefix):
    parts = value.split(RUNFILES_PATH_BEGIN)
    result = _shell_quote(parts[0])
    for part in parts[1:]:
        path, separator, suffix = part.partition(RUNFILES_PATH_END)
        if not separator:
            fail("Unterminated native artifact runfile marker")
        result += '"${%s}/_main/"' % path_prefix + _shell_quote(path + suffix)
    return result

def _imported_artifact_impl(ctx, is_test):
    output = ctx.actions.declare_file(ctx.label.name)
    nested_root = "/".join([
        ctx.workspace_name,
        ctx.label.package,
        "runfiles",
        ctx.label.name,
    ])
    script = """#!/usr/bin/env bash
set -euo pipefail
outer_root="${RUNFILES_DIR:-${TEST_SRCDIR:-$0.runfiles}}"
payload_root="$(cd "$outer_root/%s" && pwd)"
export TEST_SRCDIR="$payload_root"
export RUNFILES_DIR="$TEST_SRCDIR" JAVA_RUNFILES="$TEST_SRCDIR" PYTHON_RUNFILES="$TEST_SRCDIR"
unset RUNFILES_MANIFEST_FILE RUNFILES_MANIFEST_ONLY
""" % nested_root
    for name, value in sorted(ctx.attr.environment.items()):
        marked_value = ctx.attr.marked_environment.get(name, value)
        script += """if [[ ! -v %s || \"${%s}\" == %s ]]; then
  export %s=%s
fi
""" % (
            name,
            name,
            _shell_quote(value),
            name,
            _runtime_value(marked_value, "payload_root"),
        )
    if is_test:
        script += 'cd "$TEST_SRCDIR/_main"\n'
    script += 'exec "$TEST_SRCDIR/%s"' % ctx.attr.executable_path
    for argument in ctx.attr.source_arguments:
        script += " " + _runtime_value(argument, "payload_root")
    script += ' "$@"\n'
    ctx.actions.write(output, script, is_executable = True)

    providers = [
        DefaultInfo(
            executable = output,
            runfiles = ctx.runfiles(files = ctx.files.payload),
        ),
    ]
    if is_test:
        providers.append(testing.TestEnvironment(
            environment = ctx.attr.environment,
            inherited_environment = ctx.attr.inherited_environment,
        ))
    else:
        providers.append(RunEnvironmentInfo(
            environment = ctx.attr.environment,
            inherited_environment = ctx.attr.inherited_environment,
        ))
        runfiles_prefix = "/".join([
            ctx.label.package,
            "runfiles",
            ctx.label.name,
            "_main",
            "",
        ])
        marked_environment = {
            name: value.replace(
                RUNFILES_PATH_BEGIN,
                RUNFILES_PATH_BEGIN + runfiles_prefix,
            )
            for name, value in ctx.attr.marked_environment.items()
        }
        if marked_environment:
            providers.append(IreeRunfilesEnvironmentInfo(
                environment = marked_environment,
            ))
    return providers

def _imported_test_impl(ctx):
    return _imported_artifact_impl(ctx, is_test = True)

def _imported_tool_impl(ctx):
    return _imported_artifact_impl(ctx, is_test = False)

_IMPORTED_ATTRS = {
    "environment": attr.string_dict(),
    "executable_path": attr.string(mandatory = True),
    "inherited_environment": attr.string_list(),
    "marked_environment": attr.string_dict(),
    "payload": attr.label_list(allow_files = True),
    "source_arguments": attr.string_list(),
}

native_artifact_test = rule(
    implementation = _imported_test_impl,
    attrs = _IMPORTED_ATTRS,
    test = True,
)

native_artifact_tool = rule(
    implementation = _imported_tool_impl,
    attrs = _IMPORTED_ATTRS,
    executable = True,
)
