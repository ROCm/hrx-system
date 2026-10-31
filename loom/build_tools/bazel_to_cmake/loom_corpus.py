# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bazel-to-CMake conversion for Loom corpus rules."""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import bazel_to_cmake_requirements
from conditions import Option, all_of, any_of
from native_cc import arguments, command


@dataclass
class Corpus:
    # Source-owned test module aliases, indexed by semantic program identity.
    modules: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Manifest owners establish the exhaustive inventory before consumers emit.
    manifests: dict[str, str] = field(default_factory=dict)

    def programs(self, catalog):
        for manifest in catalog["manifests"]:
            if self.manifests.get(manifest["name"]) != manifest["package"]:
                raise ValueError(f"Unknown Loom corpus manifest: {manifest['name']}")
        result = catalog["programs"]
        seen = set()
        for program in result:
            identity = program["identity"]
            if identity in seen:
                raise ValueError(f"Duplicate Loom corpus source: {identity}")
            seen.add(identity)
            if identity not in self.modules:
                raise ValueError(
                    f"Loom corpus source has no owning declaration: {identity}"
                )
        return result


class LoomCorpusBuildFileFunctions:
    """Converts target-neutral catalogs and target-owned corpus rules."""

    @property
    def _corpus(self):
        if "loom.corpus" not in self._declarations:
            self._declarations["loom.corpus"] = Corpus()
        return self._declarations["loom.corpus"]

    def _corpus_guard(self, predicate, text):
        if self._configuration is not None:
            return text if predicate.evaluate(self._configuration) else ""
        rendered = predicate.render()
        return f"if({rendered})\n{text}endif()\n" if rendered else text

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

    def loom_corpus_manifest(self, name, package, srcs, configs=None):
        if configs is None:
            configs = {}
        if not isinstance(configs, dict):
            raise ValueError(
                f"Loom corpus manifest {name} configs must be a dictionary"
            )
        for source, source_configs in configs.items():
            if source not in srcs:
                raise ValueError(
                    f"Loom corpus manifest {name} config names unknown source {source}"
                )
            if not isinstance(source_configs, dict):
                raise ValueError(
                    f"Loom corpus manifest {name} configs for {source} must be "
                    "a dictionary"
                )
            for symbol, value in source_configs.items():
                if not isinstance(symbol, str) or not symbol:
                    raise ValueError(
                        f"Loom corpus manifest {name} config symbols for "
                        f"{source} must be non-empty strings"
                    )
                if not isinstance(value, str):
                    raise ValueError(
                        f"Loom corpus manifest {name} config {symbol} for "
                        f"{source} must have a string value"
                    )
        programs = []
        for source in srcs:
            source_directory, separator, source_name = source.rpartition("/")
            source_package = package
            if separator:
                source_package += "/" + source_directory
            source_stem = self._loom_corpus_target_stem(source.removesuffix(".loom"))
            programs.append(
                {
                    "configs": dict(configs.get(source, {})),
                    "identity": f"{name}/{source}",
                    "label": f"{source_package}:{source_name}",
                    "manifest": name,
                    "package": source_package,
                    "source": source,
                    "source_name": source_name,
                    "target_name": f"{name}_{source_stem}",
                }
            )
        return {
            "kind": "loom_corpus_manifest",
            "name": name,
            "package": package,
            "programs": programs,
            "srcs": list(srcs),
        }

    def loom_corpus_catalog(self, manifests):
        programs = []
        for manifest in manifests:
            programs.extend(manifest["programs"])
        return {
            "kind": "loom_corpus_catalog",
            "manifests": list(manifests),
            "programs": programs,
        }

    def loom_corpus_sources(self, name, manifest, visibility=None, **kwargs):
        self._check_no_unhandled_kwargs("loom_corpus_sources", kwargs)
        if manifest["kind"] != "loom_corpus_manifest":
            raise ValueError(f"{name} does not reference a Loom corpus manifest")
        current_package = "//" + self._current_package()
        manifest_package = manifest["package"]
        if current_package == manifest_package:
            if manifest["name"] in self._corpus.manifests:
                raise ValueError(f"Duplicate Loom corpus manifest: {manifest['name']}")
            self._corpus.manifests[manifest["name"]] = manifest_package
        if current_package != manifest_package and not current_package.startswith(
            manifest_package + "/"
        ):
            raise ValueError(
                f"Loom corpus manifest {manifest['name']} owns {manifest_package} "
                f"and its descendants, not {current_package}"
            )
        local_programs = [
            program
            for program in manifest["programs"]
            if program["package"] == current_package
        ]
        if current_package != manifest_package and not local_programs:
            raise ValueError(
                f"Loom corpus package {current_package} owns no sources in "
                f"manifest {manifest['name']}"
            )
        filegroup_programs = (
            manifest["programs"]
            if current_package == manifest_package
            else local_programs
        )
        self._register_target_files(
            name,
            [
                self._cmake_source_location_path(program["label"])
                for program in filegroup_programs
            ],
        )
        if self._configuration is not None:
            inventory = sorted(
                path.name for path in Path(self._build_dir).glob("*.loom")
            )
            declared = sorted(program["source_name"] for program in local_programs)
            if inventory != declared:
                raise ValueError(
                    f"Loom corpus {current_package} inventory differs: "
                    f"declared {declared}, found {inventory}"
                )
        # Native regeneration observes additions/removals; source evaluation
        # owns the inventory and validates it before emitting any products.
        self._converter.body += command(
            "file",
            "GLOB _LOOM_CORPUS_INVENTORY CONFIGURE_DEPENDS",
            '"${_IREE_PACKAGE_SOURCE_DIR}/*.loom"',
        )
        for program in local_programs:
            identity = program["identity"]
            if identity in self._corpus.modules:
                raise ValueError(f"Duplicate Loom corpus source: {identity}")
            target = self._convert_single_target(
                self._current_target_label(program["target_name"])
            ).replace("::", "_")
            module = (
                "${IREE_BINARY_DIR}/"
                + self._current_package()
                + "/"
                + program["target_name"]
                + ".loombc"
            )
            self._corpus.modules[identity] = (target, module)
            self._converter.body += command(
                "loom_test_module",
                "NAME",
                arguments((program["target_name"],)),
                "SRCS",
                arguments((program["source_name"],)),
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

        self._converter.defer(
            lambda: self._render_corpus_build(
                name,
                catalog,
                profiles,
                xfails or {},
                all_roots_xfail or {},
                excludes or {},
            )
        )

    def _render_corpus_build(
        self, name, catalog, profiles, xfails, all_roots_xfail, excludes
    ):
        if self._configuration is None:
            raise ValueError(
                "Resolved corpus products need the native configuration; pass "
                "--configuration <build>/.iree/configuration.json when previewing"
            )
        programs = self._corpus.programs(catalog)
        identities = {program["identity"] for program in programs}
        selected = {}
        for reference in sorted(profiles):
            for profile, predicate in self._loom_profiles.expand(
                self._canonical_location_label(reference)
            ).items():
                selected[profile] = (
                    any_of(selected[profile], predicate)
                    if profile in selected
                    else predicate
                )
        stems = {
            profile: re.sub(
                r"[:/+.]", "-", self._loom_profiles.profiles[profile].compiler
            )
            for profile in selected
        }
        if len(stems) != len(set(stems.values())):
            raise ValueError(
                f"Loom corpus {name} has colliding compiler output identities"
            )
        available = [
            profile
            for profile, predicate in selected.items()
            if predicate.evaluate(self._configuration)
            and self._loom_profiles.profiles[profile].available.evaluate(
                self._configuration
            )
        ]

        def qualifications(entries, *, roots=False):
            for reference, sources in sorted(entries.items()):
                members = self._loom_profiles.expand(
                    self._canonical_location_label(reference)
                )
                for identity in sorted(sources):
                    source, separator, root = (
                        identity.partition(":@") if roots else (identity, "", "")
                    )
                    if source not in identities:
                        raise ValueError(
                            f"Loom corpus {name} qualifies unknown source {source}"
                        )
                    if roots and not separator:
                        raise ValueError(
                            f"Loom corpus xfail must use '<source>:@<root>': {identity}"
                        )
                    for profile, member_condition in members.items():
                        predicate = all_of(
                            member_condition,
                            self._loom_profiles.profiles[profile].available,
                        )
                        if not predicate.evaluate(self._configuration):
                            continue
                        if profile not in available:
                            raise ValueError(
                                f"Loom corpus {name} qualifies unselected profile {profile}"
                            )
                        yield (
                            profile,
                            source,
                            "@" + root if roots else None,
                            sources[identity] if isinstance(sources, dict) else None,
                        )

        failures = defaultdict(dict)
        for profile, source, root, diagnostic in qualifications(xfails, roots=True):
            previous = failures[profile, source]
            if not diagnostic or root in previous:
                raise ValueError(
                    f"Loom corpus {name} has empty or duplicate xfail {source}:{root} on {profile}"
                )
            previous[root] = diagnostic
        all_failed = set()
        for profile, source, _, _ in qualifications(all_roots_xfail):
            key = profile, source
            if key in all_failed or not failures[key]:
                raise ValueError(
                    f"Loom corpus {name} all-roots qualification requires unique diagnostic xfails: {key}"
                )
            all_failed.add(key)
        excluded = set()
        for profile, source, _, reason in qualifications(excludes):
            key = profile, source
            if not reason or key in excluded or failures[key]:
                raise ValueError(
                    f"Loom corpus {name} has empty, duplicate, or conflicting exclusion: {key}"
                )
            excluded.add(key)

        package = (
            self._convert_single_target(self._current_target_label(name))
            .rsplit("::", 1)[0]
            .replace("::", "_")
        )
        binary = "${IREE_BINARY_DIR}/" + self._current_package()
        lines = [
            "block(SCOPE_FOR VARIABLES)\nset(CMAKE_ADD_CUSTOM_COMMAND_DEPENDS_EXPLICIT_ONLY ON)\n"
        ]
        if available:
            lines.append(command("_loom_host_tool", "_LOOM_CORPUS_LINK_TOOL loom-link"))
        semantics = defaultdict(lambda: defaultdict(list))
        semantic_targets = defaultdict(list)
        # Ninja exposes file outputs as build entry points. Recursive/IDE
        # generators need native per-program targets to retain narrow builds.
        file_targets = self._configuration["CMAKE_GENERATOR"].startswith("Ninja")
        aggregate = defaultdict(list)
        output_kinds = ("ARTIFACTS", "COMPILE_REPORTS", "XFAIL_RESULTS")

        def properties(target, outputs):
            return command(
                "set_target_properties",
                target,
                "PROPERTIES",
                *(
                    "LOOM_CORPUS_" + kind + " " + arguments((";".join(outputs[kind]),))
                    for kind in output_kinds
                ),
            )

        def action(outputs, tool, args, dependencies, description, directory):
            return command(
                "add_custom_command",
                "OUTPUT",
                arguments(outputs),
                "COMMAND",
                '"${CMAKE_COMMAND}" -E make_directory',
                arguments((directory,)),
                "COMMAND",
                arguments(("$<TARGET_FILE:" + tool + ">", *args)),
                "DEPENDS",
                arguments(dependencies),
                "WORKING_DIRECTORY",
                '"${IREE_BINARY_DIR}"',
                "COMMENT",
                arguments((description,)),
                "VERBATIM",
            )

        for program in programs:
            identity, stem = program["identity"], program["target_name"]
            semantic = program["manifest"]
            outputs = defaultdict(list)
            directory = binary + "/" + stem
            subject = binary + "/" + stem + "_subjects.loombc"
            active = [
                profile for profile in available if (profile, identity) not in excluded
            ]
            if active:
                source = (
                    "${IREE_SOURCE_DIR}/"
                    + program["package"][2:]
                    + "/"
                    + program["source_name"]
                )
                lines.append(
                    command(
                        "add_custom_command",
                        "OUTPUT",
                        arguments((subject,)),
                        "COMMAND",
                        arguments(
                            (
                                "${_LOOM_CORPUS_LINK_TOOL}",
                                "--mode=link",
                                "--to=bc",
                                source,
                                "--include-input-tests",
                                "--strip-check=true",
                                "--output=" + subject,
                            )
                        ),
                        "DEPENDS",
                        arguments(("${_LOOM_CORPUS_LINK_TOOL}", source)),
                        'WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"',
                        "COMMENT",
                        arguments(
                            ("Linking Loom module " + stem + "_subjects.loombc",)
                        ),
                        "VERBATIM",
                    )
                )
            for profile in active:
                compiler = self._loom_profiles.profiles[profile].compiler
                prefix = directory + "/" + stems[profile]
                diagnostics = failures[profile, identity]
                require_all = (profile, identity) in all_failed
                if not require_all:
                    artifact, report = prefix + ".artifact", prefix + ".compile.json"
                    lines.append(
                        action(
                            (artifact, report),
                            "loom::tools::loom-compile",
                            (
                                subject,
                                "--target=" + compiler,
                                *("--exclude-root=" + root for root in diagnostics),
                                "--output=" + artifact,
                                "--compile-report=details",
                                "--compile-report-output=" + report,
                            ),
                            ("loom::tools::loom-compile", subject),
                            f"Compiling corpus program {identity} for {compiler}",
                            directory,
                        )
                    )
                    outputs["ARTIFACTS"].append(artifact)
                    outputs["COMPILE_REPORTS"].append(report)
                if diagnostics:
                    stamp = prefix + ".xfails"
                    args = [
                        "--compiler=$<TARGET_FILE:loom::tools::loom-compile>",
                        "--stamp-output=" + stamp,
                    ]
                    if require_all:
                        args.append("--require-all-roots")
                    for root, diagnostic in diagnostics.items():
                        args.extend(
                            (
                                "--expected-root=" + root,
                                "--expected-diagnostic=" + diagnostic,
                            )
                        )
                    args.extend((subject, "--target=" + compiler))
                    probe = "loom::build_tools::corpus::loom-corpus-compile-xfails"
                    lines.append(
                        action(
                            (stamp,),
                            probe,
                            args,
                            (probe, "loom::tools::loom-compile", subject),
                            f"Probing corpus diagnostic xfails in {identity} for {compiler}",
                            directory,
                        )
                    )
                    outputs["XFAIL_RESULTS"].append(stamp)
            if not file_targets:
                target = package + "_" + stem
                lines.append(
                    command(
                        "add_custom_target",
                        target,
                        "SOURCES",
                        arguments(
                            [value for kind in output_kinds for value in outputs[kind]]
                        ),
                    )
                )
                lines.append(properties(target, outputs))
                semantic_targets[semantic].append(target)
            for kind in output_kinds:
                semantics[semantic][kind].extend(outputs[kind])
        for semantic, outputs in semantics.items():
            target = package + "_" + semantic
            lines.append(
                command(
                    "add_custom_target",
                    target,
                    "DEPENDS",
                    arguments(
                        [value for kind in output_kinds for value in outputs[kind]]
                        if file_targets
                        else semantic_targets[semantic]
                    ),
                )
            )
            lines.append(properties(target, outputs))
            for kind in output_kinds:
                aggregate[kind].extend(outputs[kind])
        target = package + "_" + name
        lines.append(
            command(
                "add_custom_target",
                target,
                "DEPENDS",
                " ".join(package + "_" + semantic for semantic in semantics),
            )
        )
        lines.append(properties(target, aggregate))
        lines.append("endblock()\n")
        return "".join(lines)

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
        execution_names = set()
        for profile in execution_profiles:
            suffix = self._loom_test_name_suffix(profile.name)
            if suffix in execution_names:
                raise ValueError(
                    f"{name} has colliding execution profiles: {profile.name}"
                )
            execution_names.add(suffix)
            policy = bazel_to_cmake_requirements.CollectedPackagePolicy(
                build_requirements=profile.build_requirements,
                run_requirements=profile.run_requirements,
                resource_group=profile.resource_group,
            )
            labels = ["loom-corpus"] + list(tags or []) + profile.tags
            labels.extend(policy.tags(include_run_requirements=True))
            labels.extend(
                [
                    "loom-execution-profile=" + profile.name,
                    "loom-target-family=" + profile.target_family,
                    "loom-target-class=" + profile.target_class,
                    "loom-executor=" + profile.executor,
                ]
            )
            requirements = bazel_to_cmake_requirements.append_conditions(
                self._apply_loom_target_compatible_with(target_compatible_with),
                policy.conditions(),
            )
            predicate = all_of(
                Option("IREE_BUILD_TESTS"), self._loom_profile_condition(requirements)
            )
            self._converter.defer(
                lambda profile=profile, labels=labels, predicate=predicate, suffix=suffix, policy=policy: (
                    self._render_corpus_execution(
                        catalog,
                        profile,
                        suffix,
                        labels,
                        predicate,
                        policy.resource_group,
                        excluded_sources,
                        profile_sources.get(profile.name),
                        xfails_by_profile_and_source.get(profile.name, {}),
                        allowed_failures_by_profile_and_source.get(profile.name, {}),
                        args,
                    )
                )
            )

    def _render_corpus_execution(
        self,
        catalog,
        profile,
        suffix,
        labels,
        predicate,
        resource_group,
        excludes,
        selected,
        xfails,
        allowed_failures,
        args,
    ):
        lines = []
        for program in self._corpus.programs(catalog):
            identity = program["identity"]
            if identity in excludes or (
                selected is not None and identity not in selected
            ):
                continue
            target, module = self._corpus.modules[identity]
            name = program["target_name"] + "_test_execute_" + suffix + "_test"
            runner_args = list(profile.runner_args or [])
            for qualifications, flag in (
                (xfails, "--xfail"),
                (allowed_failures, "--allow-failure"),
            ):
                runner_args.extend(
                    flag + "=" + record + "=" + diagnostic
                    for record, diagnostic in sorted(
                        qualifications.get(identity, {}).items()
                    )
                )
            configurations = [
                "--config=" + symbol + "=" + value
                for symbol, value in sorted(program["configs"].items())
            ]
            lines.append(
                self._native_test(
                    name,
                    "loom::tools::loom-test",
                    args=[
                        "{{" + module + "}}",
                        *(self._convert_test_location_args(runner_args) or []),
                        *configurations,
                        *(self._convert_test_location_args(args) or []),
                    ],
                    labels=labels,
                    resource_group=resource_group,
                    working_directory="${IREE_BINARY_DIR}",
                    dependencies=(target,),
                )
            )
        return self._corpus_guard(predicate, "".join(lines))

    def loom_scenario_test(self, catalog, **kwargs):
        kwargs["catalog"] = catalog
        self.loom_corpus_test(**kwargs)
