# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime-only dynamic-library bundles and dependency-graph collection."""

IreeDynamicLibraryBundleInfo = provider(
    doc = "Runtime-only dynamic-library files and exact environment bindings.",
    fields = {
        "environment": "Environment variable names mapped to root library files.",
        "files": "depset containing the complete dynamic-library runtime closure.",
    },
)

IreeDynamicLibraryBindingsInfo = provider(
    doc = "Dynamic-library bundles collected from a configured dependency graph.",
    fields = {
        "environment": "Environment variable names mapped to root library files.",
        "files": "depset containing every collected dynamic-library runtime closure.",
        "has_bundles": "Whether the configured graph contains a bundle target.",
    },
)

_TRAVERSED_ATTRS = [
    "actual",
    "data",
    "deps",
    "implementation_deps",
    "runtime_deps",
    "src",
]

def _validate_environment_name(name):
    if not name:
        fail("Dynamic-library environment variable name must not be empty")
    initial_characters = "_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    remaining_characters = initial_characters + "0123456789"
    if name[0] not in initial_characters:
        fail("Invalid dynamic-library environment variable name %r" % name)
    for index in range(1, len(name)):
        if name[index] not in remaining_characters:
            fail("Invalid dynamic-library environment variable name %r" % name)

def _single_file(target, environment_name):
    files = target[DefaultInfo].files.to_list()
    if len(files) != 1:
        fail(
            "%s must provide exactly one file for environment variable %s, got %s" %
            (target.label, environment_name, files),
        )
    return files[0]

def _merge_environment(environment, additions, owner):
    for name, file in additions.items():
        existing = environment.get(name)
        if existing != None and existing.path != file.path:
            fail(
                "%s resolves dynamic-library environment variable %s to both %s and %s" %
                (owner, name, existing.path, file.path),
            )
        environment[name] = file

def merge_dynamic_library_bindings(bindings, owner):
    """Merges collected bundle providers with deterministic conflict handling.

    Args:
      bindings: IreeDynamicLibraryBindingsInfo providers to merge.
      owner: Label or description reported for conflicting bindings.

    Returns:
      One IreeDynamicLibraryBindingsInfo containing the union of the inputs.
    """
    environment = {}
    has_bundles = False
    transitive_files = []
    for binding in bindings:
        _merge_environment(environment, binding.environment, owner)
        has_bundles = has_bundles or binding.has_bundles
        transitive_files.append(binding.files)
    return IreeDynamicLibraryBindingsInfo(
        environment = environment,
        files = depset(transitive = transitive_files),
        has_bundles = has_bundles,
    )

def has_dynamic_library_bundles(dependencies):
    """Returns whether aspect-bearing dependencies contain a bundle target.

    Args:
      dependencies: Targets that may carry IreeDynamicLibraryBindingsInfo.

    Returns:
      True if any configured dependency graph contains a bundle target.
    """
    for dependency in dependencies:
        if (
            IreeDynamicLibraryBindingsInfo in dependency and
            dependency[IreeDynamicLibraryBindingsInfo].has_bundles
        ):
            return True
    return False

def inject_dynamic_library_bindings(
        ctx,
        providers,
        dependencies,
        executable):
    """Adds collected dynamic-library runfiles and environment to an executable.

    The input providers normally come from an existing executable rule. Every
    provider other than DefaultInfo and RunEnvironmentInfo is returned
    unchanged. If the configured dependency graph contains no bundles, the
    original provider list is returned without reconstruction.

    Args:
      ctx: Rule context for the executable receiving the bindings.
      providers: Providers produced for the executable before injection.
      dependencies: Aspect-bearing dependencies that may carry
        IreeDynamicLibraryBindingsInfo.
      executable: Executable file retained in the reconstructed DefaultInfo.

    Returns:
      The executable providers with conditional runfiles and environment
      injection.
    """
    collected_bindings = [
        dependency[IreeDynamicLibraryBindingsInfo]
        for dependency in dependencies
        if IreeDynamicLibraryBindingsInfo in dependency
    ]
    if not has_dynamic_library_bundles(dependencies):
        return providers
    bindings = merge_dynamic_library_bindings(collected_bindings, ctx.label)

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
        fail("%s cannot receive dynamic-library bindings without DefaultInfo" % ctx.label)

    environment = dict(run_environment.environment if run_environment else {})
    inherited_environment = list(
        run_environment.inherited_environment if run_environment else [],
    )
    for name, file in bindings.environment.items():
        if name in inherited_environment:
            fail(
                "%s both inherits and binds dynamic-library environment variable %s" %
                (ctx.label, name),
            )
        root_path = file.short_path
        existing = environment.get(name)
        if existing != None and existing != root_path:
            fail(
                "%s sets dynamic-library environment variable %s to both %r and %r" %
                (ctx.label, name, existing, root_path),
            )
        environment[name] = root_path

    added_runfiles = ctx.runfiles(transitive_files = bindings.files)
    default_runfiles = default_info.default_runfiles or ctx.runfiles()
    data_runfiles = default_info.data_runfiles or ctx.runfiles()
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
    return result

def _iree_dynamic_library_bundle_impl(ctx):
    environment = {}
    environment_files = []
    for target, environment_name in ctx.attr.environment.items():
        _validate_environment_name(environment_name)
        file = _single_file(target, environment_name)
        _merge_environment(environment, {environment_name: file}, ctx.label)
        environment_files.append(file)

    files = depset(ctx.files.srcs + environment_files)
    return [
        DefaultInfo(
            files = files,
            runfiles = ctx.runfiles(transitive_files = files),
        ),
        IreeDynamicLibraryBundleInfo(
            environment = environment,
            files = files,
        ),
    ]

iree_dynamic_library_bundle = rule(
    implementation = _iree_dynamic_library_bundle_impl,
    attrs = {
        "environment": attr.label_keyed_string_dict(
            allow_files = True,
            doc = "Root library labels mapped to environment variable names that receive their exact runfiles paths.",
        ),
        "srcs": attr.label_list(
            allow_files = True,
            doc = "Complete runtime-only dynamic-library closure carried in runfiles.",
        ),
    },
    doc = "Carries dynamic libraries as runtime data without exposing C/C++ linker inputs.",
)

def _dependency_values(ctx, attr_name):
    if not hasattr(ctx.rule.attr, attr_name):
        return []
    value = getattr(ctx.rule.attr, attr_name)
    if type(value) == type([]):
        return value
    if value:
        return [value]
    return []

def _collect_dynamic_library_bundles_impl(target, ctx):
    bindings = []

    if IreeDynamicLibraryBundleInfo in target:
        bundle = target[IreeDynamicLibraryBundleInfo]
        bindings.append(IreeDynamicLibraryBindingsInfo(
            environment = bundle.environment,
            files = bundle.files,
            has_bundles = True,
        ))

    for attr_name in _TRAVERSED_ATTRS:
        for dependency in _dependency_values(ctx, attr_name):
            if IreeDynamicLibraryBindingsInfo not in dependency:
                continue
            bindings.append(dependency[IreeDynamicLibraryBindingsInfo])

    return [merge_dynamic_library_bindings(bindings, target.label)]

collect_dynamic_library_bundles = aspect(
    implementation = _collect_dynamic_library_bundles_impl,
    attr_aspects = _TRAVERSED_ATTRS,
    doc = "Collects runtime-only dynamic-library bundles from dependency and data edges.",
)
