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

def _source_location(package, source):
    components = source.split("/")
    source_name = components[-1]
    source_directory = "/".join(components[:-1])
    source_package = package
    if source_directory:
        source_package += "/" + source_directory
    return source_package, source_name

def _validate_manifest(manifest):
    if getattr(manifest, "kind", None) != _MANIFEST_KIND:
        fail("expected a Loom corpus manifest, got %r" % manifest)

def _validate_catalog(catalog):
    if getattr(catalog, "kind", None) != _CATALOG_KIND:
        fail("expected a Loom corpus catalog, got %r" % catalog)

def loom_corpus_manifest(name, package, srcs, configs = {}):
    """Describes one semantic package without target or execution policy.

    Args:
      name: Stable semantic package name used in source identities and targets.
      package: Absolute Bazel package owning the semantic manifest. Source
        directories below it are independent Bazel packages.
      srcs: Sources using the scenario correctness harness.
      configs: Optional source paths mapped to string-valued configuration
        bindings required to execute and compile that source.

    Returns:
      An immutable manifest value suitable for a shared corpus catalog.
    """
    if type(name) != "string" or not name or _target_stem(name) != name:
        fail("Loom corpus manifest name %r must be a non-empty target stem" % name)
    if type(package) != "string" or not package.startswith("//") or ":" in package:
        fail("Loom corpus manifest package %r must be an absolute package label" % package)
    if not srcs:
        fail("Loom corpus manifest %s must contain at least one source" % name)
    for source, source_configs in configs.items():
        if source not in srcs:
            fail("Loom corpus manifest %s config names unknown source %r" % (name, source))
        if type(source_configs) != "dict":
            fail("Loom corpus manifest %s configs for %s must be a dict" % (name, source))
        for symbol, value in source_configs.items():
            if type(symbol) != "string" or not symbol:
                fail("Loom corpus manifest %s config symbols for %s must be non-empty strings" % (name, source))
            if type(value) != "string":
                fail("Loom corpus manifest %s config %s for %s must have a string value" % (name, symbol, source))

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
        source_package, source_name = _source_location(package, source)
        source_stem = _target_stem(source[:-len(".loom")])
        target_name = name + "_" + source_stem
        if target_name in seen_targets:
            fail(
                "Loom corpus manifest %s sources %r and %r collide at target %r" %
                (name, seen_targets[target_name], source, target_name),
            )
        seen_targets[target_name] = source
        programs.append(struct(
            configs = dict(configs.get(source, {})),
            identity = name + "/" + source,
            label = source_package + ":" + source_name,
            manifest = name,
            module_label = source_package + ":" + target_name,
            package = source_package,
            source = source,
            source_name = source_name,
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
    """Owns reusable test modules in one semantic corpus source package.

    A call in the manifest package also declares the complete source aggregate.
    Calls in descendant source packages declare their direct sources only. The
    manifest remains the single ordered inventory for both forms.

    Args:
      name: Target-neutral filegroup name for this source package. In the
        manifest package this aggregates the complete semantic family.
      manifest: Manifest owned by the current package or one of its ancestors.
      visibility: Visibility of the source modules and exported inventory.
    """
    _validate_manifest(manifest)
    current_package = "//" + native.package_name()
    if (current_package != manifest.package and
        not current_package.startswith(manifest.package + "/")):
        fail(
            "Loom corpus manifest %s owns %s and its descendants, not %s" %
            (manifest.name, manifest.package, current_package),
        )

    local_programs = [
        program
        for program in manifest.programs
        if program.package == current_package
    ]
    if current_package != manifest.package and not local_programs:
        fail(
            "Loom corpus package %s owns no sources in manifest %s" %
            (current_package, manifest.name),
        )
    local_sources = [program.source_name for program in local_programs]
    local_sources = iree_checked_glob(
        files = local_sources,
        include = ["*.loom"],
        allow_empty = True,
    )
    if local_sources:
        native.exports_files(local_sources, visibility = visibility)
    for program in local_programs:
        loom_test_module(
            name = program.target_name,
            srcs = [program.source_name],
            tags = [LOOM_CORPUS_TAG],
            visibility = visibility,
        )
    filegroup_sources = local_sources
    if current_package == manifest.package:
        filegroup_sources = [program.label for program in manifest.programs]
    native.filegroup(
        name = name,
        srcs = filegroup_sources,
        visibility = visibility,
    )

def loom_corpus_validate_catalog(catalog):
    """Validates and returns a corpus catalog for consumer macros."""
    _validate_catalog(catalog)
    return catalog
