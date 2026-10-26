# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-neutral source catalogs for the Loom correctness corpus."""

load("//build_tools/bazel:glob.bzl", "iree_checked_glob")
load(":loom_library.bzl", "loom_test_module")

_MANIFEST_KIND = "loom_corpus_manifest"
_CATALOG_KIND = "loom_corpus_catalog"

LOOM_CORPUS_TAG = "loom-corpus"

def _target_stem(value):
    return value.replace("/", "_").replace(".", "_").replace("-", "_").replace("+", "_")

def _validate_manifest(manifest):
    if getattr(manifest, "kind", None) != _MANIFEST_KIND:
        fail("expected a Loom corpus manifest, got %r" % manifest)

def _validate_catalog(catalog):
    if getattr(catalog, "kind", None) != _CATALOG_KIND:
        fail("expected a Loom corpus catalog, got %r" % catalog)

def loom_corpus_manifest(name, package, srcs):
    """Describes one semantic package without target or execution policy.

    Args:
      name: Stable semantic package name used in source identities and targets.
      package: Absolute Bazel package containing the authored sources.
      srcs: Sources using the scenario correctness harness.

    Returns:
      An immutable manifest value suitable for a shared corpus catalog.
    """
    if type(name) != "string" or not name or _target_stem(name) != name:
        fail("Loom corpus manifest name %r must be a non-empty target stem" % name)
    if type(package) != "string" or not package.startswith("//") or ":" in package:
        fail("Loom corpus manifest package %r must be an absolute package label" % package)
    if not srcs:
        fail("Loom corpus manifest %s must contain at least one source" % name)

    programs = []
    seen_sources = {}
    seen_targets = {}
    for source in srcs:
        if (type(source) != "string" or not source.endswith(".loom") or
            source.startswith("/") or source.startswith(":")):
            fail("Loom corpus manifest %s has invalid source %r" % (name, source))
        if source in seen_sources:
            fail("Loom corpus manifest %s repeats source %r" % (name, source))
        seen_sources[source] = None
        source_stem = _target_stem(source[:-len(".loom")])
        target_name = name + "_" + source_stem
        if target_name in seen_targets:
            fail(
                "Loom corpus manifest %s sources %r and %r collide at target %r" %
                (name, seen_targets[target_name], source, target_name),
            )
        seen_targets[target_name] = source
        programs.append(struct(
            identity = name + "/" + source,
            label = package + ":" + source,
            manifest = name,
            module_label = package + ":" + target_name,
            source = source,
            target_name = target_name,
        ))

    return struct(
        kind = _MANIFEST_KIND,
        name = name,
        package = package,
        programs = programs,
        srcs = list(srcs),
    )

def loom_corpus_catalog(manifests):
    """Combines semantic manifests into the complete target-neutral catalog.

    Args:
      manifests: Semantic manifests to combine in catalog order.

    Returns:
      An immutable catalog containing every manifest and source program.
    """
    if not manifests:
        fail("Loom corpus catalog must contain at least one manifest")
    seen_names = {}
    seen_identities = {}
    seen_targets = {}
    programs = []
    for manifest in manifests:
        _validate_manifest(manifest)
        if manifest.name in seen_names:
            fail("Loom corpus catalog repeats manifest %r" % manifest.name)
        seen_names[manifest.name] = None
        for program in manifest.programs:
            if program.identity in seen_identities:
                fail("Loom corpus catalog repeats source identity %r" % program.identity)
            seen_identities[program.identity] = None
            if program.target_name in seen_targets:
                fail(
                    "Loom corpus programs %r and %r collide at target %r" %
                    (seen_targets[program.target_name], program.identity, program.target_name),
                )
            seen_targets[program.target_name] = program.identity
            programs.append(program)
    return struct(
        kind = _CATALOG_KIND,
        manifests = list(manifests),
        programs = programs,
    )

def loom_corpus_sources(name, manifest, visibility = ["//visibility:public"]):
    """Owns one reusable test module for each semantic corpus source.

    Args:
      name: Target-neutral filegroup name for the semantic package.
      manifest: Manifest owned by the current Bazel package.
      visibility: Visibility of the source modules and exported inventory.
    """
    _validate_manifest(manifest)
    current_package = "//" + native.package_name()
    if manifest.package != current_package:
        fail(
            "Loom corpus manifest %s belongs to %s, not %s" %
            (manifest.name, manifest.package, current_package),
        )
    srcs = iree_checked_glob(
        files = manifest.srcs,
        include = ["**/*.loom"],
        allow_empty = False,
    )
    native.exports_files(srcs, visibility = visibility)
    for program in manifest.programs:
        loom_test_module(
            name = program.target_name,
            srcs = [program.source],
            tags = [LOOM_CORPUS_TAG],
            visibility = visibility,
        )
    native.filegroup(
        name = name,
        srcs = srcs,
        visibility = visibility,
    )

def loom_corpus_validate_catalog(catalog):
    """Validates and returns a corpus catalog for consumer macros."""
    _validate_catalog(catalog)
    return catalog
