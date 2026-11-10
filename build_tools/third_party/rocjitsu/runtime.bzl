# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Declared runtime inputs for explicitly embedded rocjitsu machines."""

load("//build_tools/bazel:dynamic_library_bundle.bzl", "IreeDynamicLibraryBundleInfo")

RocjitsuRuntimeInfo = provider(
    doc = "An explicit simulator runtime and its host execution ABI.",
    fields = {
        "host_cpu": "Minimum x86-64 CPU instruction set required by the library.",
        "library": "Root shared library loaded by the consumer.",
        "sanitizer": "Runtime instrumentation: none or address (requires an ASan host).",
    },
)

def _rocjitsu_runtime_impl(ctx):
    if ctx.attr.error:
        fail(ctx.attr.error)
    files = depset(ctx.files.srcs + [ctx.file.library, ctx.file.memory_helper, ctx.file.manifest])
    return [
        DefaultInfo(files = files, runfiles = ctx.runfiles(transitive_files = files)),
        IreeDynamicLibraryBundleInfo(environment = {}, files = files),
        RocjitsuRuntimeInfo(
            host_cpu = ctx.attr.host_cpu,
            library = ctx.file.library,
            sanitizer = ctx.attr.sanitizer,
        ),
    ]

rocjitsu_runtime = rule(
    implementation = _rocjitsu_runtime_impl,
    attrs = {
        "error": attr.string(doc = "Unavailable package diagnostic, reported only when this runtime is selected."),
        "host_cpu": attr.string(values = ["x86-64", "x86-64-v3"]),
        "library": attr.label(allow_single_file = True),
        "manifest": attr.label(allow_single_file = True),
        "memory_helper": attr.label(allow_single_file = True),
        "sanitizer": attr.string(values = ["none", "address"]),
        "srcs": attr.label_list(allow_files = True),
    },
    doc = "Exposes a checked rocjitsu runtime package without C/C++ linker inputs.",
)
