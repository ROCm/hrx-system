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
    transitive_files = []
    for binding in bindings:
        _merge_environment(environment, binding.environment, owner)
        transitive_files.append(binding.files)
    return IreeDynamicLibraryBindingsInfo(
        environment = environment,
        files = depset(transitive = transitive_files),
    )

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
