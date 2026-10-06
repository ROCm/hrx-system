# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-owned correctness execution for Loom corpus programs."""

load(":loom_corpus_catalog.bzl", "loom_corpus_validate_catalog")
load(":loom_library.bzl", "loom_test")

def _profile_with_runner_args(profile, runner_args):
    return struct(
        build_requirements = profile.build_requirements,
        executor = profile.executor,
        kind = profile.kind,
        name = profile.name,
        resource_group = profile.resource_group,
        run_requirements = profile.run_requirements,
        runner = profile.runner,
        runner_args = runner_args,
        tags = profile.tags,
        target_class = profile.target_class,
        target_family = profile.target_family,
    )

def _select_failure_qualifications(catalog, qualifications, field_name):
    selected_sources = {program.identity: None for program in catalog.programs}
    known_sources = {program.identity: None for program in catalog.all_programs}
    selected_qualifications = {}
    for profile_name, entries in qualifications.items():
        profile_qualifications = {}
        for identity, diagnostic in entries.items():
            separator = identity.find(":@")
            if separator == -1:
                fail(
                    "%s identity %r must use '<source>:@<record>'" %
                    (field_name, identity),
                )
            source_identity = identity[:separator]
            if source_identity not in known_sources:
                fail("%s names unknown source %r" % (field_name, source_identity))
            if source_identity in selected_sources:
                profile_qualifications[identity] = diagnostic
        if profile_qualifications:
            selected_qualifications[profile_name] = profile_qualifications
    return selected_qualifications

def _partition_failure_qualifications(
        catalog,
        profiles_by_name,
        selected_sources_by_profile,
        excludes,
        qualifications,
        field_name,
        entry_name):
    qualifications_by_source = {program.identity: {} for program in catalog.programs}
    for profile_name, entries in qualifications.items():
        if profile_name not in profiles_by_name:
            fail("loom_corpus_test declares %s for unknown profile %r" % (field_name, profile_name))
        if type(entries) != "dict":
            fail("loom_corpus_test %s for profile %s must be a dictionary" % (field_name, profile_name))
        for identity, diagnostic in entries.items():
            separator = identity.find(":@")
            if separator == -1:
                fail(
                    "loom_corpus_test %s identity %r must use '<source>:@<record>'" %
                    (entry_name, identity),
                )
            source_identity = identity[:separator]
            record = identity[separator + 1:]
            if source_identity not in qualifications_by_source:
                fail("loom_corpus_test %s names unknown source %r" % (entry_name, source_identity))
            if source_identity in excludes:
                fail(
                    "loom_corpus_test source %s cannot be both excluded and qualified by %s" %
                    (source_identity, entry_name),
                )
            selected_sources = selected_sources_by_profile.get(profile_name)
            if selected_sources != None and source_identity not in selected_sources:
                fail(
                    "loom_corpus_test profile %s applies %s to unselected source %s" %
                    (profile_name, entry_name, source_identity),
                )
            if not diagnostic:
                fail("loom_corpus_test %s %s must name a diagnostic" % (entry_name, identity))
            profile_qualifications = qualifications_by_source[source_identity].setdefault(profile_name, {})
            if record in profile_qualifications:
                fail(
                    "loom_corpus_test repeats %s %s for profile %s" %
                    (entry_name, identity, profile_name),
                )
            profile_qualifications[record] = diagnostic
    return qualifications_by_source

def loom_corpus_test(
        name,
        catalog,
        execution_profiles,
        allowed_failures = {},
        excludes = {},
        profile_sources = {},
        xfails = {},
        args = [],
        size = "small",
        tags = [],
        visibility = None,
        target_compatible_with = []):
    """Expands a source catalog into independent correctness tests.

    Each source is linked and executed independently so a source-only change
    invalidates only that program. Trials remain batched inside the authored
    scenario. Benchmark runners are deliberately absent from these CI tests.

    Args:
      name: Whole-catalog test suite name.
      catalog: Target-neutral catalog returned by `loom_corpus_catalog`.
      execution_profiles: Target-owned correctness execution environments.
      allowed_failures: Execution profile names mapped to
        '<source>:@<record>' diagnostic dictionaries. A named record may pass;
        if it fails, the failure must carry one of the named diagnostics.
      excludes: Source identities mapped to target-local exclusion reasons.
      profile_sources: Execution profile names mapped to the source identities
        that carry an authored witness for that profile. Profiles absent from
        this mapping apply to every non-excluded source. A source omitted from
        every explicitly selected profile is not part of this test suite.
      xfails: Execution profile names mapped to '<source>:@<record>' diagnostic
        dictionaries. Every expected failure is checked inside its source's
        existing execution action and fails on XPASS or diagnostic drift.
      args: Additional arguments passed to each correctness runner.
      size: Bazel test size applied to every source execution.
      tags: Additional tags applied to every generated target.
      visibility: Visibility of source, semantic, and whole-catalog suites.
      target_compatible_with: Build constraints applied to every source test.
    """
    catalog = loom_corpus_validate_catalog(catalog)
    if not execution_profiles:
        fail("loom_corpus_test requires at least one execution profile")
    if name in [manifest.name for manifest in catalog.manifests]:
        fail("loom_corpus_test aggregate %r collides with a semantic manifest" % name)

    programs_by_identity = {
        program.identity: program
        for program in catalog.programs
    }
    profiles_by_name = {}
    for profile in execution_profiles:
        if getattr(profile, "kind", None) != "loom_execution_profile":
            fail("%s execution profile was not created by loom_execution_profile" % name)
        if profile.name in profiles_by_name:
            fail("loom_corpus_test repeats execution profile %r" % profile.name)
        profiles_by_name[profile.name] = profile
    for identity, reason in excludes.items():
        if identity not in programs_by_identity:
            fail("loom_corpus_test exclusion names unknown source %r" % identity)
        if not reason:
            fail("loom_corpus_test exclusion for %s must include a reason" % identity)

    selected_sources_by_profile = {}
    for profile_name, source_identities in profile_sources.items():
        if profile_name not in profiles_by_name:
            fail("loom_corpus_test selects sources for unknown profile %r" % profile_name)
        if type(source_identities) != "list":
            fail("loom_corpus_test sources for profile %s must be a list" % profile_name)
        if not source_identities:
            fail("loom_corpus_test sources for profile %s must not be empty" % profile_name)
        selected_sources = {}
        for identity in source_identities:
            if identity not in programs_by_identity:
                fail(
                    "loom_corpus_test profile %s names unknown source %r" %
                    (profile_name, identity),
                )
            if identity in selected_sources:
                fail(
                    "loom_corpus_test profile %s repeats source %r" %
                    (profile_name, identity),
                )
            if identity in excludes:
                fail(
                    "loom_corpus_test profile %s selects excluded source %r" %
                    (profile_name, identity),
                )
            selected_sources[identity] = None
        selected_sources_by_profile[profile_name] = selected_sources

    xfails_by_source = _partition_failure_qualifications(
        catalog,
        profiles_by_name,
        selected_sources_by_profile,
        excludes,
        xfails,
        "xfails",
        "xfail",
    )
    allowed_failures_by_source = _partition_failure_qualifications(
        catalog,
        profiles_by_name,
        selected_sources_by_profile,
        excludes,
        allowed_failures,
        "allowed_failures",
        "allowed failure",
    )
    for program in catalog.programs:
        for profile_name, profile_allowed_failures in allowed_failures_by_source[program.identity].items():
            profile_xfails = xfails_by_source[program.identity].get(profile_name, {})
            for record in profile_allowed_failures:
                if record in profile_xfails:
                    fail(
                        "loom_corpus_test record %s:%s cannot be both xfailed and allowed to fail for profile %s" %
                        (program.identity, record, profile_name),
                    )

    tests = []
    tests_by_manifest = {manifest.name: [] for manifest in catalog.manifests}
    for program in catalog.programs:
        if program.identity in excludes:
            continue
        program_profiles = []
        for profile in execution_profiles:
            if (profile.name in selected_sources_by_profile and
                program.identity not in selected_sources_by_profile[profile.name]):
                continue
            qualification_args = []
            profile_xfails = xfails_by_source[program.identity].get(profile.name, {})
            if profile_xfails:
                qualification_args.extend([
                    "--xfail=%s=%s" % (record, profile_xfails[record])
                    for record in sorted(profile_xfails)
                ])
            profile_allowed_failures = allowed_failures_by_source[program.identity].get(profile.name, {})
            if profile_allowed_failures:
                qualification_args.extend([
                    "--allow-failure=%s=%s" % (record, profile_allowed_failures[record])
                    for record in sorted(profile_allowed_failures)
                ])
            if qualification_args:
                profile = _profile_with_runner_args(
                    profile,
                    profile.runner_args + qualification_args,
                )
            program_profiles.append(profile)
        if not program_profiles:
            continue
        test_name = program.target_name + "_test"
        loom_test(
            name = test_name,
            args = args,
            benchmark_smoke = False,
            execution_profiles = program_profiles,
            size = size,
            srcs = [program.label],
            tags = tags,
            target_compatible_with = target_compatible_with,
            visibility = visibility,
        )
        tests.append(":" + test_name)
        tests_by_manifest[program.manifest].append(":" + test_name)

    for manifest in catalog.manifests:
        native.test_suite(
            name = name + "_" + manifest.name,
            tags = tags,
            tests = tests_by_manifest[manifest.name],
            visibility = visibility,
        )
    native.test_suite(
        name = name,
        tags = tags,
        tests = tests,
        visibility = visibility,
    )

def loom_scenario_test(
        name,
        catalog,
        execution_profiles,
        allowed_failures = {},
        excludes = {},
        xfails = {},
        args = [],
        size = "small",
        tags = [],
        visibility = None,
        target_compatible_with = []):
    """Runs every scenario source against target-owned execution profiles."""
    if getattr(catalog, "harness", None) != "scenario":
        fail("loom_scenario_test requires a catalog created by loom_scenario_corpus")
    loom_corpus_test(
        name = name,
        allowed_failures = _select_failure_qualifications(
            catalog,
            allowed_failures,
            "loom_scenario_test allowed failure",
        ),
        args = args,
        catalog = catalog,
        excludes = excludes,
        execution_profiles = execution_profiles,
        size = size,
        tags = tags,
        target_compatible_with = target_compatible_with,
        visibility = visibility,
        xfails = _select_failure_qualifications(
            catalog,
            xfails,
            "loom_scenario_test xfail",
        ),
    )

def loom_legacy_case_test(
        name,
        catalog,
        execution_profiles,
        allowed_failures = {},
        excludes = {},
        profile_sources = {},
        xfails = {},
        args = [],
        size = "small",
        tags = [],
        visibility = None,
        target_compatible_with = []):
    """Runs the shrinking quarantine of direct check.case sources."""
    if getattr(catalog, "harness", None) != "legacy_case":
        fail("loom_legacy_case_test requires a catalog created by loom_legacy_case_corpus")
    loom_corpus_test(
        name = name,
        allowed_failures = _select_failure_qualifications(
            catalog,
            allowed_failures,
            "loom_legacy_case_test allowed failure",
        ),
        args = args,
        catalog = catalog,
        excludes = excludes,
        execution_profiles = execution_profiles,
        profile_sources = profile_sources,
        size = size,
        tags = tags,
        target_compatible_with = target_compatible_with,
        visibility = visibility,
        xfails = _select_failure_qualifications(
            catalog,
            xfails,
            "loom_legacy_case_test xfail",
        ),
    )
