# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Configured runtime requirements carried from dependencies to executables."""

load("//build_tools/sanitizer:build_defs.bzl", "IreeSanitizerSuppressionsInfo")
load("//build_tools/sanitizer:suppressions.bzl", "sanitizer_suppression_options")
load(":dynamic_library_bundle.bzl", "IreeDynamicLibraryBindingsInfo", "IreeDynamicLibraryBundleInfo", "merge_dynamic_library_bindings")
load(":runfiles.bzl", "IreeRunfilesEnvironmentInfo", "RUNFILES_PATH_BEGIN", "RUNFILES_PATH_END")

IreeTransitiveSanitizerSuppressionsInfo = provider(
    doc = "Suppression inputs collected from the configured dependency graph.",
    fields = {"files": "Sanitizer names mapped to depsets of suppression files."},
)

_TRAVERSED_ATTRS = [
    "actual",
    "data",
    "deps",
    "implementation_deps",
    "runtime_deps",
    "src",
    "srcs",
]

def _dependency_values(ctx, attr_name):
    if not hasattr(ctx.rule.attr, attr_name):
        return []
    value = getattr(ctx.rule.attr, attr_name)
    if type(value) == type([]):
        return value
    return [value] if value else []

def _collect_execution_requirements_impl(target, ctx):
    bindings = []
    suppressions = {}
    if IreeDynamicLibraryBundleInfo in target:
        bundle = target[IreeDynamicLibraryBundleInfo]
        bindings.append(IreeDynamicLibraryBindingsInfo(
            environment = bundle.environment,
            files = bundle.files,
            has_bundles = True,
        ))
    if IreeSanitizerSuppressionsInfo in target:
        for sanitizer, files in target[IreeSanitizerSuppressionsInfo].files.items():
            suppressions.setdefault(sanitizer, []).append(files)
    for attr_name in _TRAVERSED_ATTRS:
        for dependency in _dependency_values(ctx, attr_name):
            if IreeDynamicLibraryBindingsInfo in dependency:
                bindings.append(dependency[IreeDynamicLibraryBindingsInfo])

            # An executable already owns its merged policy. Wrappers reuse
            # that result instead of rebuilding the same concatenation.
            if IreeSanitizerSuppressionsInfo not in target and IreeTransitiveSanitizerSuppressionsInfo in dependency:
                for sanitizer, files in dependency[IreeTransitiveSanitizerSuppressionsInfo].files.items():
                    suppressions.setdefault(sanitizer, []).append(files)
    providers = [merge_dynamic_library_bindings(bindings, target.label)]
    if suppressions:
        providers.append(IreeTransitiveSanitizerSuppressionsInfo(files = {
            sanitizer: depset(transitive = inputs)
            for sanitizer, inputs in suppressions.items()
        }))
    return providers

collect_execution_requirements = aspect(
    implementation = _collect_execution_requirements_impl,
    attr_aspects = _TRAVERSED_ATTRS,
    doc = "Collects runtime files and sanitizer policy through configured dependencies.",
)

def inject_execution_requirements(ctx, providers, dependencies, executable):
    """Adds dependency-owned runtime files and environment to an executable.

    Args:
      ctx: Executable rule context.
      providers: Complete provider list from the underlying executable rule.
      dependencies: Dependencies carrying the collection aspect.
      executable: Executable preserved in DefaultInfo.

    Returns:
      Providers with runtime files and environment composed at this boundary.
    """
    bindings = merge_dynamic_library_bindings([
        dependency[IreeDynamicLibraryBindingsInfo]
        for dependency in dependencies
        if IreeDynamicLibraryBindingsInfo in dependency
    ], ctx.label)
    suppressions = {}
    inherited_sanitizer_environment = {}
    for dependency in dependencies:
        if IreeTransitiveSanitizerSuppressionsInfo in dependency:
            for sanitizer, files in dependency[IreeTransitiveSanitizerSuppressionsInfo].files.items():
                suppressions.setdefault(sanitizer, []).append(files)
        if IreeSanitizerSuppressionsInfo in dependency and RunEnvironmentInfo in dependency:
            for sanitizer in dependency[IreeSanitizerSuppressionsInfo].files:
                name, _ = sanitizer_suppression_options(sanitizer, "")
                value = dependency[RunEnvironmentInfo].environment.get(name)
                if value != None:
                    inherited_sanitizer_environment.setdefault(name, []).append(value)
    if not bindings.has_bundles and not suppressions:
        return providers
    marked_environment = {
        name: RUNFILES_PATH_BEGIN + file.short_path + RUNFILES_PATH_END
        for name, file in bindings.environment.items()
    }
    suppression_files = []
    resolved_suppressions = {}
    for sanitizer, inputs in sorted(suppressions.items()):
        inputs = sorted(depset(transitive = inputs).to_list(), key = _file_path)
        if len(inputs) == 1:
            output = inputs[0]
        else:
            output = ctx.actions.declare_file(ctx.label.name + "." + sanitizer + "-suppressions.txt")

            # Shell builtins keep concatenation independent of host utilities or
            # an additional toolchain. Each input ends on its own line even if
            # its author omitted the final newline.
            ctx.actions.run_shell(
                inputs = inputs,
                outputs = [output],
                arguments = [output.path] + [file.path for file in inputs],
                command = """set -eu
output="$1"
shift
for source in "$@"; do
  while IFS= read -r line || [ -n "$line" ]; do
    printf '%s\n' "$line"
  done < "$source"
done > "$output"
""",
                mnemonic = "SanitizerSuppressions",
            )
        suppression_files.append(output)
        resolved_suppressions[sanitizer] = depset([output])
        name, options = sanitizer_suppression_options(
            sanitizer,
            RUNFILES_PATH_BEGIN + output.short_path + RUNFILES_PATH_END,
        )
        if name in marked_environment:
            fail("%s also binds sanitizer environment variable %s" % (ctx.label, name))
        marked_environment[name] = options
    files = depset(suppression_files, transitive = [bindings.files])
    default_info = None
    run_environment = None
    for value in providers:
        value_type = type(value)
        if value_type == "DefaultInfo":
            if default_info != None:
                fail("%s produced more than one DefaultInfo provider" % ctx.label)
            default_info = value
        elif value_type == "RunEnvironmentInfo":
            if run_environment != None:
                fail("%s produced more than one RunEnvironmentInfo provider" % ctx.label)
            run_environment = value

    if default_info == None:
        fail("%s cannot receive runtime bindings without DefaultInfo" % ctx.label)

    environment = dict(run_environment.environment if run_environment else {})
    inherited_environment = list(
        run_environment.inherited_environment if run_environment else [],
    )
    for name, marked_value in marked_environment.items():
        if name in inherited_environment:
            fail(
                "%s both inherits and binds runtime environment variable %s" %
                (ctx.label, name),
            )
        root_path = marked_value.replace(RUNFILES_PATH_BEGIN, "").replace(RUNFILES_PATH_END, "")
        existing = environment.get(name)
        if existing != None and existing != root_path and existing not in inherited_sanitizer_environment.get(name, []):
            fail(
                "%s sets runtime environment variable %s to both %r and %r" %
                (ctx.label, name, existing, root_path),
            )
        environment[name] = root_path

    added_runfiles = ctx.runfiles(transitive_files = files)
    default_runfiles = default_info.default_runfiles or ctx.runfiles()

    # The parent may use DefaultInfo(runfiles = ...), whose data_runfiles is
    # unset until Bazel installs the provider on the target.
    data_runfiles = default_info.data_runfiles or default_runfiles
    updated_default_info = DefaultInfo(
        data_runfiles = data_runfiles.merge(added_runfiles),
        default_runfiles = default_runfiles.merge(added_runfiles),
        executable = executable,
        files = default_info.files,
    )
    updated_run_environment = RunEnvironmentInfo(
        environment = environment,
        inherited_environment = inherited_environment,
    )

    result = []
    for value in providers:
        value_type = type(value)
        if value_type == "DefaultInfo":
            result.append(updated_default_info)
        elif value_type == "RunEnvironmentInfo":
            result.append(updated_run_environment)
        else:
            result.append(value)
    if run_environment == None:
        result.append(updated_run_environment)
    result.append(IreeRunfilesEnvironmentInfo(environment = marked_environment))
    if resolved_suppressions:
        result.append(IreeSanitizerSuppressionsInfo(files = resolved_suppressions))
    return result

def _file_path(file):
    return file.path
