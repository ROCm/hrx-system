# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-neutral source catalogs for the Loom correctness corpus."""

load("//build_tools/bazel:glob.bzl", "iree_checked_glob")

_MANIFEST_KIND = "loom_corpus_manifest"
_CATALOG_KIND = "loom_corpus_catalog"
_SCENARIO_HARNESS = "scenario"
_LEGACY_CASE_HARNESS = "legacy_case"

def _target_stem(value):
    return value.replace("/", "_").replace(".", "_").replace("-", "_").replace("+", "_")

def _validate_manifest(manifest):
    if getattr(manifest, "kind", None) != _MANIFEST_KIND:
        fail("expected a Loom corpus manifest, got %r" % manifest)

def _validate_catalog(catalog):
    if getattr(catalog, "kind", None) != _CATALOG_KIND:
        fail("expected a Loom corpus catalog, got %r" % catalog)

def loom_corpus_manifest(name, package, scenario_srcs, legacy_case_srcs):
    """Describes one semantic package without target or execution policy.

    Args:
      name: Stable semantic package name used in source identities and targets.
      package: Absolute Bazel package containing the authored sources.
      scenario_srcs: Sources using the scenario correctness harness.
      legacy_case_srcs: Quarantined sources using the legacy case harness.

    Returns:
      An immutable manifest value suitable for a shared corpus catalog.
    """
    if type(name) != "string" or not name or _target_stem(name) != name:
        fail("Loom corpus manifest name %r must be a non-empty target stem" % name)
    if type(package) != "string" or not package.startswith("//") or ":" in package:
        fail("Loom corpus manifest package %r must be an absolute package label" % package)
    if not scenario_srcs and not legacy_case_srcs:
        fail("Loom corpus manifest %s must contain at least one source" % name)

    programs = []
    seen_sources = {}
    seen_targets = {}
    programs_by_harness = {
        _SCENARIO_HARNESS: [],
        _LEGACY_CASE_HARNESS: [],
    }
    for harness, sources in [
        (_SCENARIO_HARNESS, scenario_srcs),
        (_LEGACY_CASE_HARNESS, legacy_case_srcs),
    ]:
        for source in sources:
            if (type(source) != "string" or not source.endswith(".loom") or
                source.startswith("/") or source.startswith(":")):
                fail("Loom corpus manifest %s has invalid source %r" % (name, source))
            if source in seen_sources:
                fail(
                    "Loom corpus manifest %s classifies source %r more than once" %
                    (name, source),
                )
            seen_sources[source] = None
            source_stem = _target_stem(source[:-len(".loom")])
            target_name = name + "_" + source_stem
            if target_name in seen_targets:
                fail(
                    "Loom corpus manifest %s sources %r and %r collide at target %r" %
                    (name, seen_targets[target_name], source, target_name),
                )
            seen_targets[target_name] = source
            program = struct(
                harness = harness,
                identity = name + "/" + source,
                label = package + ":" + source,
                manifest = name,
                source = source,
                target_name = target_name,
            )
            programs.append(program)
            programs_by_harness[harness].append(program)

    return struct(
        kind = _MANIFEST_KIND,
        legacy_case_programs = programs_by_harness[_LEGACY_CASE_HARNESS],
        legacy_case_srcs = list(legacy_case_srcs),
        name = name,
        package = package,
        programs = programs,
        scenario_programs = programs_by_harness[_SCENARIO_HARNESS],
        scenario_srcs = list(scenario_srcs),
        srcs = list(scenario_srcs) + list(legacy_case_srcs),
    )

def _loom_corpus_catalog(manifests, harness = None):
    """Combines semantic manifests, optionally selecting one harness."""
    if not manifests:
        fail("Loom corpus catalog must contain at least one manifest")
    seen_names = {}
    seen_identities = {}
    seen_targets = {}
    all_programs = []
    programs = []
    for manifest in manifests:
        _validate_manifest(manifest)
        if manifest.name in seen_names:
            fail("Loom corpus catalog repeats manifest %r" % manifest.name)
        seen_names[manifest.name] = None
        all_programs.extend(manifest.programs)
        manifest_programs = manifest.programs
        if harness == _SCENARIO_HARNESS:
            manifest_programs = manifest.scenario_programs
        elif harness == _LEGACY_CASE_HARNESS:
            manifest_programs = manifest.legacy_case_programs
        for program in manifest_programs:
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
        all_programs = all_programs,
        harness = harness,
        kind = _CATALOG_KIND,
        manifests = list(manifests),
        programs = programs,
    )

def loom_corpus_catalog(manifests):
    """Combines semantic manifests into the complete target-neutral catalog."""
    return _loom_corpus_catalog(manifests)

def loom_scenario_corpus(manifests):
    """Selects scenario sources from semantic manifests."""
    return _loom_corpus_catalog(manifests, _SCENARIO_HARNESS)

def loom_legacy_case_corpus(manifests):
    """Selects quarantined legacy case sources from semantic manifests."""
    return _loom_corpus_catalog(manifests, _LEGACY_CASE_HARNESS)

def loom_corpus_sources(name, manifest, visibility = ["//visibility:public"]):
    """Checks and exports one semantic package's authored source inventory.

    Args:
      name: Target-neutral filegroup name for the semantic package.
      manifest: Manifest owned by the current Bazel package.
      visibility: Visibility of the exported sources and filegroup.
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
    native.filegroup(
        name = name,
        srcs = srcs,
        visibility = visibility,
    )

def loom_corpus_validate_catalog(catalog):
    """Validates and returns a corpus catalog for consumer macros."""
    _validate_catalog(catalog)
    return catalog
