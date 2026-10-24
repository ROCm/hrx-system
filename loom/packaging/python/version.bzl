# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Expose the checked-in SDK identity during Bazel wheel analysis."""

def _sdk_version_impl(repository_ctx):
    identity = json.decode(repository_ctx.read(repository_ctx.attr.version_file))
    if type(identity) != "dict" or sorted(identity.keys()) != ["model-catalog", "package-version"]:
        fail("SDK version.json requires package-version and model-catalog")
    version = identity["package-version"]
    catalog = identity["model-catalog"]
    if type(version) != "string" or not version:
        fail("SDK package-version must be a nonempty string")
    if catalog != None and (type(catalog) != "string" or not catalog):
        fail("SDK model-catalog must be a nonempty string or null")
    repository_ctx.file("BUILD.bazel", "exports_files([\"version.bzl\"])\n")
    repository_ctx.file(
        "version.bzl",
        "PACKAGE_VERSION = %s\nMODEL_CATALOG = %s\n" % (repr(version), repr(catalog)),
    )
    return repository_ctx.repo_metadata(reproducible = True)

sdk_version = repository_rule(
    implementation = _sdk_version_impl,
    attrs = {"version_file": attr.label(mandatory = True, allow_single_file = [".json"])},
)
