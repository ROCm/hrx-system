# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bazel-to-CMake conversion for Loom corpus rules."""

import bazel_to_cmake_requirements


class LoomCorpusBuildFileFunctions:
    """Converts target-neutral catalogs and target-owned corpus rules."""

    def _loom_target_identity_label(self, target):
        if target.startswith("@"):
            return target
        return self._canonical_location_label(target)

    @staticmethod
    def _loom_corpus_target_stem(value):
        return (
            value.replace("/", "_")
            .replace(".", "_")
            .replace("-", "_")
            .replace("+", "_")
        )

    @staticmethod
    def _partition_corpus_test_failure_qualifications(
        name,
        profile_names,
        source_identities,
        excluded_sources,
        profile_sources,
        qualifications,
        field_name,
        entry_name,
    ):
        qualifications_by_profile_and_source = {}
        for profile_name, entries in qualifications.items():
            if profile_name not in profile_names:
                raise ValueError(
                    f"{name} declares {field_name} for unknown profile {profile_name}"
                )
            if not isinstance(entries, dict):
                raise ValueError(
                    f"{name} {field_name} for profile {profile_name} must be a "
                    "dictionary"
                )
            profile_qualifications = qualifications_by_profile_and_source.setdefault(
                profile_name, {}
            )
            for identity, diagnostic in entries.items():
                source_identity, separator, record = identity.partition(":@")
                if not separator:
                    raise ValueError(
                        f"{name} {entry_name} identity must use "
                        f"'<source>:@<record>': {identity}"
                    )
                record = "@" + record
                if source_identity not in source_identities:
                    raise ValueError(
                        f"{name} {entry_name} names unknown source {source_identity}"
                    )
                if source_identity in excluded_sources:
                    raise ValueError(
                        f"{name} source {source_identity} cannot be both excluded "
                        f"and qualified by {entry_name}"
                    )
                selected_sources = profile_sources.get(profile_name)
                if (
                    selected_sources is not None
                    and source_identity not in selected_sources
                ):
                    raise ValueError(
                        f"{name} profile {profile_name} applies {entry_name} to "
                        f"unselected source {source_identity}"
                    )
                if not diagnostic:
                    raise ValueError(
                        f"{name} {entry_name} {identity} must name a diagnostic"
                    )
                source_qualifications = profile_qualifications.setdefault(
                    source_identity, {}
                )
                if record in source_qualifications:
                    raise ValueError(
                        f"{name} repeats {entry_name} {identity} for profile "
                        f"{profile_name}"
                    )
                source_qualifications[record] = diagnostic
        return qualifications_by_profile_and_source

    def loom_corpus_manifest(self, name, package, scenario_srcs, legacy_case_srcs):
        programs = []
        programs_by_harness = {"scenario": [], "legacy_case": []}
        for harness, sources in (
            ("scenario", scenario_srcs),
            ("legacy_case", legacy_case_srcs),
        ):
            for source in sources:
                source_stem = self._loom_corpus_target_stem(
                    source.removesuffix(".loom")
                )
                program = {
                    "harness": harness,
                    "identity": f"{name}/{source}",
                    "label": f"{package}:{source}",
                    "manifest": name,
                    "source": source,
                    "target_name": f"{name}_{source_stem}",
                }
                programs.append(program)
                programs_by_harness[harness].append(program)
        return {
            "kind": "loom_corpus_manifest",
            "legacy_case_programs": programs_by_harness["legacy_case"],
            "legacy_case_srcs": list(legacy_case_srcs),
            "name": name,
            "package": package,
            "programs": programs,
            "scenario_programs": programs_by_harness["scenario"],
            "scenario_srcs": list(scenario_srcs),
            "srcs": [*scenario_srcs, *legacy_case_srcs],
        }

    @staticmethod
    def _loom_corpus_catalog(manifests, harness=None):
        all_programs = []
        programs = []
        for manifest in manifests:
            all_programs.extend(manifest["programs"])
            manifest_programs = manifest["programs"]
            if harness == "scenario":
                manifest_programs = manifest["scenario_programs"]
            elif harness == "legacy_case":
                manifest_programs = manifest["legacy_case_programs"]
            programs.extend(manifest_programs)
        return {
            "all_programs": all_programs,
            "harness": harness,
            "kind": "loom_corpus_catalog",
            "manifests": list(manifests),
            "programs": programs,
        }

    def loom_corpus_catalog(self, manifests):
        return self._loom_corpus_catalog(manifests)

    def loom_scenario_corpus(self, manifests):
        return self._loom_corpus_catalog(manifests, "scenario")

    def loom_legacy_case_corpus(self, manifests):
        return self._loom_corpus_catalog(manifests, "legacy_case")

    def loom_corpus_sources(self, name, manifest, visibility=None, **kwargs):
        self._check_no_unhandled_kwargs("loom_corpus_sources", kwargs)
        if manifest["kind"] != "loom_corpus_manifest":
            raise ValueError(f"{name} does not reference a Loom corpus manifest")
        self._converter.body += (
            "loom_corpus_sources(\n"
            + self._convert_string_arg_block("NAME", name)
            + self._convert_string_arg_block("MANIFEST", manifest["name"])
            + self._convert_string_list_block("SRCS", manifest["srcs"], sort=False)
            + ")\n\n"
        )

    def loom_corpus_build(
        self,
        name,
        catalog,
        profiles,
        xfails=None,
        all_roots_xfail=None,
        excludes=None,
        tags=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags):
            return
        self._check_no_unhandled_kwargs("loom_corpus_build", kwargs)
        if catalog["kind"] != "loom_corpus_catalog":
            raise ValueError(f"{name} does not reference a Loom corpus catalog")

        manifest_names = [manifest["name"] for manifest in catalog["manifests"]]

        xfail_values = []
        for target in sorted(xfails or {}):
            converted_targets = self._convert_target(
                self._loom_target_identity_label(target)
            )
            if len(converted_targets) != 1:
                raise NotImplementedError(f"loom_corpus_build xfail target: {target}")
            for identity in sorted(xfails[target]):
                source, separator, root = identity.partition(":@")
                if not separator:
                    raise ValueError(
                        "loom_corpus_build xfail identity must use "
                        f"'<source>:@<root>': {identity}"
                    )
                xfail_values.extend(
                    [
                        converted_targets[0],
                        source,
                        "@" + root,
                        xfails[target][identity],
                    ]
                )

        all_roots_xfail_values = []
        for target in sorted(all_roots_xfail or {}):
            converted_targets = self._convert_target(
                self._loom_target_identity_label(target)
            )
            if len(converted_targets) != 1:
                raise NotImplementedError(
                    f"loom_corpus_build all-roots xfail target: {target}"
                )
            for source in sorted(all_roots_xfail[target]):
                all_roots_xfail_values.extend([converted_targets[0], source])

        exclude_values = []
        for target in sorted(excludes or {}):
            converted_targets = self._convert_target(
                self._loom_target_identity_label(target)
            )
            if len(converted_targets) != 1:
                raise NotImplementedError(
                    f"loom_corpus_build exclusion target: {target}"
                )
            for source in sorted(excludes[target]):
                exclude_values.extend(
                    [
                        converted_targets[0],
                        source,
                        excludes[target][source],
                    ]
                )

        self._converter.body += (
            "loom_corpus_build(\n"
            + self._convert_string_arg_block("NAME", name)
            + self._convert_string_list_block("MANIFESTS", manifest_names, sort=False)
            + self._convert_target_list_block(
                "PROFILES",
                [self._loom_target_identity_label(profile) for profile in profiles],
            )
            + self._convert_string_list_block(
                "XFAILS", xfail_values or None, sort=False
            )
            + self._convert_string_list_block(
                "ALL_ROOTS_XFAIL", all_roots_xfail_values or None, sort=False
            )
            + self._convert_string_list_block(
                "EXCLUDES", exclude_values or None, sort=False
            )
            + ")\n\n"
        )

    def loom_corpus_test(
        self,
        name,
        catalog,
        execution_profiles,
        allowed_failures=None,
        excludes=None,
        profile_sources=None,
        xfails=None,
        args=None,
        size="small",
        tags=None,
        visibility=None,
        target_compatible_with=None,
    ):
        if catalog["kind"] != "loom_corpus_catalog":
            raise ValueError(f"{name} does not reference a Loom corpus catalog")
        if not execution_profiles:
            raise ValueError(f"{name} requires at least one execution profile")
        excluded_sources = excludes or {}
        source_identities = {program["identity"] for program in catalog["programs"]}
        for source_identity, reason in excluded_sources.items():
            if source_identity not in source_identities:
                raise ValueError(
                    f"{name} exclusion names unknown source {source_identity}"
                )
            if not reason:
                raise ValueError(
                    f"{name} exclusion for {source_identity} requires a reason"
                )
        profile_sources = profile_sources or {}
        profile_names = set()
        for profile in execution_profiles:
            if profile.kind != "loom_execution_profile":
                raise ValueError(
                    f"{name} execution profile was not created by "
                    "loom_execution_profile"
                )
            profile_names.add(profile.name)
        for profile_name, selected_sources in profile_sources.items():
            if profile_name not in profile_names:
                raise ValueError(
                    f"{name} selects sources for unknown profile {profile_name}"
                )
            if not isinstance(selected_sources, list):
                raise ValueError(
                    f"{name} sources for profile {profile_name} must be a list"
                )
            if not selected_sources:
                raise ValueError(
                    f"{name} sources for profile {profile_name} must not be empty"
                )
            if len(selected_sources) != len(set(selected_sources)):
                raise ValueError(f"{name} repeats a source for profile {profile_name}")
            for source_identity in selected_sources:
                if source_identity not in source_identities:
                    raise ValueError(
                        f"{name} profile {profile_name} names unknown source "
                        f"{source_identity}"
                    )
                if source_identity in excluded_sources:
                    raise ValueError(
                        f"{name} profile {profile_name} selects excluded source "
                        f"{source_identity}"
                    )
        xfails_by_profile_and_source = (
            self._partition_corpus_test_failure_qualifications(
                name,
                profile_names,
                source_identities,
                excluded_sources,
                profile_sources,
                xfails or {},
                "xfails",
                "xfail",
            )
        )
        allowed_failures_by_profile_and_source = (
            self._partition_corpus_test_failure_qualifications(
                name,
                profile_names,
                source_identities,
                excluded_sources,
                profile_sources,
                allowed_failures or {},
                "allowed_failures",
                "allowed failure",
            )
        )
        for (
            profile_name,
            profile_allowed_failures,
        ) in allowed_failures_by_profile_and_source.items():
            profile_xfails = xfails_by_profile_and_source.get(profile_name, {})
            for (
                source_identity,
                source_allowed_failures,
            ) in profile_allowed_failures.items():
                source_xfails = profile_xfails.get(source_identity, {})
                overlap = source_allowed_failures.keys() & source_xfails.keys()
                if overlap:
                    record = sorted(overlap)[0]
                    raise ValueError(
                        f"{name} record {source_identity}:{record} cannot be both "
                        f"xfailed and allowed to fail for profile {profile_name}"
                    )
        del size, visibility
        manifest_names = [manifest["name"] for manifest in catalog["manifests"]]

        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )
        execution_names = set()
        for profile in execution_profiles:
            if profile.kind != "loom_execution_profile":
                raise ValueError(
                    f"{name} execution profile was not created by "
                    "loom_execution_profile"
                )
            profile_suffix = self._loom_test_name_suffix(profile.name)
            if profile_suffix in execution_names:
                raise ValueError(
                    f"{name} has colliding execution profiles: {profile.name}"
                )
            execution_names.add(profile_suffix)

            profile_excludes = dict(excluded_sources)
            selected_sources = profile_sources.get(profile.name)
            profile_exclude_values = []
            for source_identity in sorted(profile_excludes):
                profile_exclude_values.extend(
                    [source_identity, profile_excludes[source_identity]]
                )
            profile_xfail_values = []
            for source_identity in sorted(
                xfails_by_profile_and_source.get(profile.name, {})
            ):
                source_xfails = xfails_by_profile_and_source[profile.name][
                    source_identity
                ]
                for record in sorted(source_xfails):
                    profile_xfail_values.extend(
                        [source_identity, record, source_xfails[record]]
                    )
            profile_allowed_failure_values = []
            for source_identity in sorted(
                allowed_failures_by_profile_and_source.get(profile.name, {})
            ):
                source_allowed_failures = allowed_failures_by_profile_and_source[
                    profile.name
                ][source_identity]
                for record in sorted(source_allowed_failures):
                    profile_allowed_failure_values.extend(
                        [
                            source_identity,
                            record,
                            source_allowed_failures[record],
                        ]
                    )

            policy = bazel_to_cmake_requirements.CollectedPackagePolicy(
                build_requirements=profile.build_requirements,
                run_requirements=profile.run_requirements,
                resource_group=profile.resource_group,
            )
            labels = list(tags or []) + profile.tags
            labels.extend(policy.tags(include_run_requirements=True))
            labels.extend(
                [
                    "loom-execution-profile=" + profile.name,
                    "loom-target-family=" + profile.target_family,
                    "loom-target-class=" + profile.target_class,
                    "loom-executor=" + profile.executor,
                ]
            )
            requirements = bazel_to_cmake_requirements.append_cmake_conditions(
                target_compatible_with,
                policy.cmake_conditions(),
            )
            condition = self._target_compatible_condition(requirements)
            self._converter.body += (
                "loom_corpus_test(\n"
                + self._convert_string_arg_block("NAME", name)
                + self._convert_string_arg_block("PROFILE", profile.name)
                + self._convert_string_list_block(
                    "MANIFESTS", manifest_names, sort=False
                )
                + self._convert_string_list_block(
                    "SOURCES", selected_sources, sort=False
                )
                + self._convert_string_list_block(
                    "EXCLUDES", profile_exclude_values or None, sort=False
                )
                + self._convert_string_list_block(
                    "XFAILS", profile_xfail_values or None, sort=False
                )
                + self._convert_string_list_block(
                    "ALLOWED_FAILURES",
                    profile_allowed_failure_values or None,
                    sort=False,
                )
                + self._convert_string_list_block(
                    "ARGS", self._convert_test_location_args(args), sort=False
                )
                + self._convert_string_list_block(
                    "RUNNER_ARGS",
                    self._convert_test_location_args(profile.runner_args),
                    sort=False,
                )
                + self._convert_string_list_block("LABELS", labels, sort=False)
                + self._convert_string_arg_block(
                    "RESOURCE_GROUP", policy.resource_group
                )
                + self._convert_string_arg_block("REQUIRES", condition)
                + ")\n\n"
            )

    @staticmethod
    def _select_corpus_failure_qualifications(catalog, qualifications):
        selected_sources = {program["identity"] for program in catalog["programs"]}
        known_sources = {program["identity"] for program in catalog["all_programs"]}
        selected_qualifications = {}
        for profile_name, entries in (qualifications or {}).items():
            profile_qualifications = {}
            for identity, diagnostic in entries.items():
                source_identity, separator, _ = identity.partition(":@")
                if not separator:
                    raise ValueError(
                        "failure qualification identity must use "
                        f"'<source>:@<record>': {identity}"
                    )
                if source_identity not in known_sources:
                    raise ValueError(
                        f"failure qualification names unknown source {source_identity}"
                    )
                if source_identity in selected_sources:
                    profile_qualifications[identity] = diagnostic
            if profile_qualifications:
                selected_qualifications[profile_name] = profile_qualifications
        return selected_qualifications

    def loom_scenario_test(self, catalog, **kwargs):
        if catalog.get("harness") != "scenario":
            raise ValueError(
                "loom_scenario_test requires a catalog created by loom_scenario_corpus"
            )
        kwargs["xfails"] = self._select_corpus_failure_qualifications(
            catalog, kwargs.get("xfails")
        )
        kwargs["allowed_failures"] = self._select_corpus_failure_qualifications(
            catalog, kwargs.get("allowed_failures")
        )
        excluded_sources = kwargs.get("excludes") or {}
        selected_sources = [
            program["identity"]
            for program in catalog["programs"]
            if program["identity"] not in excluded_sources
        ]
        kwargs["profile_sources"] = {
            profile.name: selected_sources for profile in kwargs["execution_profiles"]
        }
        kwargs["catalog"] = catalog
        self.loom_corpus_test(**kwargs)

    def loom_legacy_case_test(self, catalog, **kwargs):
        if catalog.get("harness") != "legacy_case":
            raise ValueError(
                "loom_legacy_case_test requires a catalog created by "
                "loom_legacy_case_corpus"
            )
        kwargs["xfails"] = self._select_corpus_failure_qualifications(
            catalog, kwargs.get("xfails")
        )
        kwargs["allowed_failures"] = self._select_corpus_failure_qualifications(
            catalog, kwargs.get("allowed_failures")
        )
        if not kwargs.get("profile_sources"):
            excluded_sources = kwargs.get("excludes") or {}
            selected_sources = [
                program["identity"]
                for program in catalog["programs"]
                if program["identity"] not in excluded_sources
            ]
            kwargs["profile_sources"] = {
                profile.name: selected_sources
                for profile in kwargs["execution_profiles"]
            }
        kwargs["catalog"] = catalog
        self.loom_corpus_test(**kwargs)
