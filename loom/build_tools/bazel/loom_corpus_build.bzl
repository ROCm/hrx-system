# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-owned build qualification for Loom corpus programs."""

load(":loom_corpus_catalog.bzl", "loom_corpus_validate_catalog")
load(
    ":loom_corpus_info.bzl",
    "LoomCorpusBuildInfo",
    "LoomCorpusProgramInfo",
)
load(":loom_linking.bzl", "loom_linking")
load(
    ":loom_target_profile.bzl",
    "LoomTargetProfileInfo",
    "LoomTargetSetInfo",
)

_LOOM_COMPILE_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:compile_toolchain_type")
_LOOM_LINK_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:link_toolchain_type")

def _target_profiles(target):
    if LoomTargetProfileInfo in target:
        return [target]
    if LoomTargetSetInfo in target:
        return target[LoomTargetSetInfo].profiles
    fail("%s is not a Loom target profile or set" % target.label)

def _collect_target_profiles(targets):
    profiles = []
    seen_labels = {}
    for target in targets:
        for profile in _target_profiles(target):
            label = str(profile.label)
            if label not in seen_labels:
                seen_labels[label] = None
                profiles.append(profile)
    return profiles

def _profile_stem(profile):
    info = profile[LoomTargetProfileInfo]
    return (info.family + "-" + info.selector).replace(":", "-").replace("/", "-").replace(".", "-").replace("+", "-")

def _display_label(label):
    repository = "@" + label.workspace_name if label.workspace_name else ""
    return "%s//%s:%s" % (repository, label.package, label.name)

def _decode_xfails(encoded_xfails, owner):
    xfails = json.decode(encoded_xfails)
    if type(xfails) != "dict":
        fail("%s has malformed diagnostic xfails" % owner)
    for root, diagnostic in xfails.items():
        if not root.startswith("@"):
            fail("%s diagnostic xfail root %r must begin with '@'" % (owner, root))
        if type(diagnostic) != "string" or not diagnostic:
            fail("%s diagnostic xfail %s must name a diagnostic" % (owner, root))
    return xfails

def _declare_subject_module(ctx, source):
    return loom_linking.declare_test_module(
        ctx = ctx,
        root_module = source,
        dependency_infos = [],
        output_stem = ctx.label.name + "/subjects",
        mnemonic = "LoomCorpusLink",
        progress_message = "Selecting tested corpus subjects from %s" % source.short_path,
        strip_check = True,
    )

def _add_config_args(args, configs):
    for symbol in sorted(configs):
        args.add("--config=%s=%s" % (symbol, configs[symbol]))

def _declare_positive_compile(ctx, source, profile, xfails):
    profile_info = profile[LoomTargetProfileInfo]
    output_stem = ctx.label.name + "/" + _profile_stem(profile)
    artifact = ctx.actions.declare_file(output_stem + ".artifact")
    compile_report = ctx.actions.declare_file(output_stem + ".compile.json")
    outputs = [artifact, compile_report]
    args = ctx.actions.args()
    args.add(source)
    args.add("--target=%s:%s" % (profile_info.family, profile_info.selector))
    _add_config_args(args, ctx.attr.configs)
    for root in sorted(xfails):
        args.add("--exclude-root=%s" % root)
    args.add("--output=%s" % artifact.path)
    args.add("--compile-report=details")
    args.add("--compile-report-output=%s" % compile_report.path)

    prepared_module = None
    launch_config = None
    if ctx.attr.prepare_execution_package:
        prepared_module = ctx.actions.declare_file(
            output_stem + ".prepared.loombc",
        )
        launch_config = ctx.actions.declare_file(
            output_stem + ".launch.loombc",
        )
        args.add("--emit-prepared-module=%s" % prepared_module.path)
        args.add("--emit-launch-config=%s" % launch_config.path)
        outputs.extend([prepared_module, launch_config])

    tool = ctx.toolchains[_LOOM_COMPILE_TOOLCHAIN_TYPE].tool
    ctx.actions.run(
        arguments = [args],
        executable = tool.files_to_run,
        inputs = [source],
        mnemonic = "LoomCorpusCompile",
        outputs = outputs,
        progress_message = "Compiling corpus program %s for %s" % (
            ctx.attr.source_identity,
            _display_label(profile.label),
        ),
    )
    return struct(
        artifact = artifact,
        compile_report = compile_report,
        launch_config = launch_config,
        prepared_module = prepared_module,
    )

def _declare_execution_package(ctx, profile, compile_outputs):
    profile_info = profile[LoomTargetProfileInfo]
    output_stem = ctx.label.name + "/" + _profile_stem(profile)
    output = ctx.actions.declare_directory(output_stem + ".package")
    args = ctx.actions.args()
    args.add(ctx.file.authored_module)
    args.add("--target=%s:%s" % (profile_info.family, profile_info.selector))
    args.add("--compile-report=%s" % compile_outputs.compile_report.path)
    args.add("--prepared-module=%s" % compile_outputs.prepared_module.path)
    args.add("--launch-config=%s" % compile_outputs.launch_config.path)
    args.add("--output=%s" % output.path)
    ctx.actions.run(
        arguments = [args],
        executable = ctx.executable._package_tool,
        inputs = [
            ctx.file.authored_module,
            compile_outputs.compile_report,
            compile_outputs.launch_config,
            compile_outputs.prepared_module,
        ],
        mnemonic = "LoomAssembleTestPackage",
        outputs = [output],
        progress_message = "Assembling corpus package %s for %s" % (
            ctx.attr.source_identity,
            _display_label(profile.label),
        ),
    )
    return output

def _declare_xfail_probes(ctx, source, profile, xfails, require_all_roots):
    profile_info = profile[LoomTargetProfileInfo]
    output_stem = ctx.label.name + "/" + _profile_stem(profile)
    result = ctx.actions.declare_file(output_stem + ".xfails")
    compile_tool = ctx.toolchains[_LOOM_COMPILE_TOOLCHAIN_TYPE].tool
    args = ctx.actions.args()
    args.add("--compiler=%s" % compile_tool.executable.path)
    args.add("--stamp-output=%s" % result.path)
    if require_all_roots:
        args.add("--require-all-roots")
    for root in sorted(xfails):
        args.add("--expected-root=%s" % root)
        args.add("--expected-diagnostic=%s" % xfails[root])
    args.add(source)
    args.add("--target=%s:%s" % (profile_info.family, profile_info.selector))
    _add_config_args(args, ctx.attr.configs)
    ctx.actions.run(
        arguments = [args],
        executable = ctx.executable._xfails_tool,
        inputs = [source],
        mnemonic = "LoomCorpusXfails",
        outputs = [result],
        progress_message = "Probing corpus diagnostic xfails in %s for %s" % (
            ctx.attr.source_identity,
            _display_label(profile.label),
        ),
        tools = [compile_tool.files_to_run],
    )
    return result

def _loom_corpus_program_impl(ctx):
    profiles = _collect_target_profiles(ctx.attr.profiles)
    selected_labels = {str(profile.label): None for profile in profiles}
    labels_by_output_stem = {}
    for profile in profiles:
        output_stem = _profile_stem(profile)
        prior_label = labels_by_output_stem.get(output_stem)
        if prior_label != None:
            fail(
                "%s target profiles %s and %s collide at output identity %r" %
                (ctx.label, prior_label, profile.label, output_stem),
            )
        labels_by_output_stem[output_stem] = profile.label

    xfails_by_label = {}
    for target, encoded_xfails in ctx.attr.xfails.items():
        decoded_xfails = _decode_xfails(encoded_xfails, ctx.label)
        for profile in _target_profiles(target):
            label = str(profile.label)
            if label not in selected_labels:
                fail("%s has diagnostic xfails for unselected profile %s" % (ctx.label, label))
            profile_xfails = xfails_by_label.setdefault(label, {})
            for root, diagnostic in decoded_xfails.items():
                if root in profile_xfails:
                    fail(
                        "%s declares diagnostic xfail %s for profile %s more than once" %
                        (ctx.label, root, label),
                    )
                profile_xfails[root] = diagnostic

    all_roots_xfail_by_label = {}
    for target in ctx.attr.all_roots_xfail:
        for profile in _target_profiles(target):
            label = str(profile.label)
            if label not in selected_labels:
                fail("%s marks all roots xfail for unselected profile %s" % (ctx.label, label))
            if label in all_roots_xfail_by_label:
                fail("%s marks all roots xfail for profile %s more than once" % (ctx.label, label))
            if label not in xfails_by_label:
                fail("%s marks all roots xfail for profile %s without diagnostic xfails" % (ctx.label, label))
            all_roots_xfail_by_label[label] = None

    excludes_by_label = {}
    for target, reason in ctx.attr.excludes.items():
        for profile in _target_profiles(target):
            label = str(profile.label)
            if label not in selected_labels:
                fail("%s has an exclusion for unselected profile %s" % (ctx.label, label))
            if not reason:
                fail("%s exclusion for %s must include a reason" % (ctx.label, label))
            if label in excludes_by_label:
                fail("%s declares an exclusion for profile %s more than once" % (ctx.label, label))
            if label in xfails_by_label:
                fail("%s cannot both exclude and xfail profile %s" % (ctx.label, label))
            if label in all_roots_xfail_by_label:
                fail("%s cannot both exclude and mark all roots xfail for profile %s" % (ctx.label, label))
            excludes_by_label[label] = reason

    active_profiles = [
        profile
        for profile in profiles
        if str(profile.label) not in excludes_by_label
    ]
    artifacts = []
    compile_reports = []
    prepared_packages = []
    qualification_results = []
    if active_profiles:
        subject_module = _declare_subject_module(ctx, ctx.file.src)
        for profile in active_profiles:
            label = str(profile.label)
            xfails = xfails_by_label.get(label, {})
            require_all_roots = label in all_roots_xfail_by_label
            if not require_all_roots:
                compile_outputs = _declare_positive_compile(
                    ctx,
                    subject_module,
                    profile,
                    xfails,
                )
                artifacts.append(compile_outputs.artifact)
                compile_reports.append(compile_outputs.compile_report)
                if ctx.attr.prepare_execution_package:
                    prepared_packages.append(_declare_execution_package(
                        ctx,
                        profile,
                        compile_outputs,
                    ))
            if xfails:
                qualification_results.append(_declare_xfail_probes(
                    ctx,
                    subject_module,
                    profile,
                    xfails,
                    require_all_roots,
                ))

    artifacts_depset = depset(artifacts)
    compile_reports_depset = depset(compile_reports)
    prepared_packages_depset = depset(prepared_packages)
    qualification_results_depset = depset(qualification_results)
    default_files = depset(transitive = [
        artifacts_depset,
        compile_reports_depset,
        prepared_packages_depset,
        qualification_results_depset,
    ])
    return [
        DefaultInfo(files = default_files),
        OutputGroupInfo(
            artifacts = artifacts_depset,
            compile_reports = compile_reports_depset,
            prepared_packages = prepared_packages_depset,
            xfail_results = qualification_results_depset,
        ),
        LoomCorpusProgramInfo(
            artifacts = artifacts_depset,
            compile_reports = compile_reports_depset,
            prepared_packages = prepared_packages_depset,
            qualification_results = qualification_results_depset,
            source_identity = ctx.attr.source_identity,
            sources = depset([ctx.file.src]),
        ),
    ]

_loom_corpus_program = rule(
    implementation = _loom_corpus_program_impl,
    attrs = {
        "all_roots_xfail": attr.label_list(
            providers = [
                [LoomTargetProfileInfo],
                [LoomTargetSetInfo],
            ],
            doc = "Target profiles or sets whose complete default root set is covered by diagnostic xfails.",
        ),
        "authored_module": attr.label(
            allow_single_file = [".loombc"],
            doc = "Linked authored test module retained beside prepared actuals.",
        ),
        "configs": attr.string_dict(
            doc = "String-valued configuration bindings required by this corpus source.",
        ),
        "excludes": attr.label_keyed_string_dict(
            doc = "Target profiles or sets mapped to whole-source exclusion reasons.",
        ),
        "prepare_execution_package": attr.bool(
            doc = "Whether positive compiles also produce hardware execution packages.",
        ),
        "profiles": attr.label_list(
            mandatory = True,
            providers = [
                [LoomTargetProfileInfo],
                [LoomTargetSetInfo],
            ],
            doc = "Target profiles or target-owned sets qualified by this program.",
        ),
        "source_identity": attr.string(
            mandatory = True,
            doc = "Canonical '<semantic-package>/<source>' catalog identity.",
        ),
        "src": attr.label(
            allow_single_file = [".loom"],
            mandatory = True,
            doc = "One independently compiled corpus source program.",
        ),
        "xfails": attr.label_keyed_string_dict(
            doc = "Target profiles or sets mapped to encoded root diagnostics.",
        ),
        "_package_tool": attr.label(
            cfg = "exec",
            default = Label("//loom/build_tools/corpus:loom-corpus-assemble-package"),
            executable = True,
        ),
        "_xfails_tool": attr.label(
            cfg = "exec",
            default = Label("//loom/build_tools/corpus:loom-corpus-compile-xfails"),
            executable = True,
        ),
    },
    doc = "Compiles one corpus program independently for target-owned profiles.",
    toolchains = [
        _LOOM_COMPILE_TOOLCHAIN_TYPE,
        _LOOM_LINK_TOOLCHAIN_TYPE,
    ],
)

def _loom_corpus_aggregate_impl(ctx):
    artifacts = depset(transitive = [
        dep[LoomCorpusProgramInfo].artifacts
        for dep in ctx.attr.programs
    ])
    compile_reports = depset(transitive = [
        dep[LoomCorpusProgramInfo].compile_reports
        for dep in ctx.attr.programs
    ])
    prepared_packages = depset(transitive = [
        dep[LoomCorpusProgramInfo].prepared_packages
        for dep in ctx.attr.programs
    ])
    qualification_results = depset(transitive = [
        dep[LoomCorpusProgramInfo].qualification_results
        for dep in ctx.attr.programs
    ])
    source_identities = depset([
        dep[LoomCorpusProgramInfo].source_identity
        for dep in ctx.attr.programs
    ])
    sources = depset(transitive = [
        dep[LoomCorpusProgramInfo].sources
        for dep in ctx.attr.programs
    ])
    return [
        DefaultInfo(files = depset(transitive = [
            artifacts,
            compile_reports,
            prepared_packages,
            qualification_results,
        ])),
        OutputGroupInfo(
            artifacts = artifacts,
            compile_reports = compile_reports,
            prepared_packages = prepared_packages,
            xfail_results = qualification_results,
        ),
        LoomCorpusBuildInfo(
            artifacts = artifacts,
            compile_reports = compile_reports,
            prepared_packages = prepared_packages,
            qualification_results = qualification_results,
            source_identities = source_identities,
            sources = sources,
        ),
    ]

_loom_corpus_aggregate = rule(
    implementation = _loom_corpus_aggregate_impl,
    attrs = {
        "programs": attr.label_list(
            mandatory = True,
            providers = [LoomCorpusProgramInfo],
        ),
    },
    doc = "Collects independently cacheable corpus program build outputs.",
)

def _partition_exceptions(catalog, exceptions, kind):
    by_source = {program.identity: {} for program in catalog.programs}
    for target, entries in exceptions.items():
        if type(entries) != "dict":
            fail("loom_corpus_build %s entries for %s must be a dictionary" % (kind, target))
        for identity, value in entries.items():
            if kind == "xfail":
                separator = identity.find(":@")
                if separator == -1:
                    fail(
                        "loom_corpus_build xfail identity %r must use '<source>:@<root>'" %
                        identity,
                    )
                source_identity = identity[:separator]
                key = identity[separator + 1:]
            else:
                source_identity = identity
                key = identity
            if source_identity not in by_source:
                fail("loom_corpus_build %s names unknown source %r" % (kind, source_identity))
            target_entries = by_source[source_identity].setdefault(target, {})
            if key in target_entries:
                fail(
                    "loom_corpus_build %s repeats %r for target %s" %
                    (kind, identity, target),
                )
            target_entries[key] = value
    return by_source

def _partition_all_roots_xfail(catalog, all_roots_xfail):
    by_source = {program.identity: [] for program in catalog.programs}
    for target, source_identities in all_roots_xfail.items():
        if type(source_identities) != "list":
            fail("loom_corpus_build all_roots_xfail entries for %s must be a list" % target)
        seen_sources = {}
        for source_identity in source_identities:
            if source_identity not in by_source:
                fail("loom_corpus_build all_roots_xfail names unknown source %r" % source_identity)
            if source_identity in seen_sources:
                fail(
                    "loom_corpus_build all_roots_xfail repeats %r for target %s" %
                    (source_identity, target),
                )
            seen_sources[source_identity] = None
            by_source[source_identity].append(target)
    return by_source

def loom_corpus_build(
        name,
        catalog,
        profiles,
        prepared_sources = [],
        xfails = {},
        all_roots_xfail = {},
        excludes = {},
        **kwargs):
    """Expands a shared source catalog into target-owned build actions.

    The macro declares one independently cacheable target per source, one
    aggregate per semantic manifest, and `name` as the whole-catalog aggregate.
    Product identity is inferred by `loom-compile`; target packages own every
    profile, diagnostic xfail, and whole-source exclusion.

    Args:
      name: Whole-catalog aggregate target name.
      catalog: Target-neutral catalog returned by `loom_corpus_catalog`.
      profiles: Target-owned compiler profiles or profile sets.
      prepared_sources: Source identities whose positive compiles also produce
        late-emission hardware execution packages.
      xfails: Profile or set keyed maps from '<source>:@<root>' to diagnostics.
      all_roots_xfail: Profile or set keyed lists of sources whose complete
        default root set is declared by xfails.
      excludes: Profile or set keyed maps from source identities to reasons.
      **kwargs: Common rule attributes applied to programs and aggregates.
    """
    catalog = loom_corpus_validate_catalog(catalog)
    if not profiles:
        fail("loom_corpus_build requires at least one profile or profile set")
    if name in [manifest.name for manifest in catalog.manifests]:
        fail("loom_corpus_build aggregate %r collides with a semantic manifest" % name)

    programs_by_identity = {
        program.identity: program
        for program in catalog.programs
    }
    prepared_source_set = {}
    for source_identity in prepared_sources:
        if source_identity not in programs_by_identity:
            fail("loom_corpus_build prepared source names unknown source %r" % source_identity)
        if source_identity in prepared_source_set:
            fail("loom_corpus_build repeats prepared source %r" % source_identity)
        prepared_source_set[source_identity] = None

    xfails_by_source = _partition_exceptions(catalog, xfails, "xfail")
    all_roots_xfail_by_source = _partition_all_roots_xfail(catalog, all_roots_xfail)
    excludes_by_source = _partition_exceptions(catalog, excludes, "exclusion")
    program_labels = []
    programs_by_manifest = {manifest.name: [] for manifest in catalog.manifests}
    for program in catalog.programs:
        source_xfails = {
            target: json.encode(entries)
            for target, entries in xfails_by_source[program.identity].items()
        }
        source_excludes = {}
        for target, entries in excludes_by_source[program.identity].items():
            reason = entries[program.identity]
            if not reason:
                fail(
                    "loom_corpus_build exclusion for %s on %s must include a reason" %
                    (program.identity, target),
                )
            if target in source_xfails:
                fail(
                    "loom_corpus_build source %s cannot both exclude and xfail target %s" %
                    (program.identity, target),
                )
            source_excludes[target] = reason

        program_kwargs = dict(kwargs)
        program_kwargs.update({
            "all_roots_xfail": all_roots_xfail_by_source[program.identity],
            "configs": program.configs,
            "excludes": source_excludes,
            "name": program.target_name,
            "prepare_execution_package": program.identity in prepared_source_set,
            "profiles": profiles,
            "source_identity": program.identity,
            "src": program.label,
            "testonly": True,
            "xfails": source_xfails,
        })
        if program.identity in prepared_source_set:
            program_kwargs["authored_module"] = program.module_label
        _loom_corpus_program(**program_kwargs)
        program_label = ":" + program.target_name
        program_labels.append(program_label)
        programs_by_manifest[program.manifest].append(program_label)

    aggregate_kwargs = dict(kwargs)
    aggregate_kwargs["testonly"] = True
    for manifest in catalog.manifests:
        _loom_corpus_aggregate(
            name = manifest.name,
            programs = programs_by_manifest[manifest.name],
            **aggregate_kwargs
        )
    _loom_corpus_aggregate(
        name = name,
        programs = program_labels,
        **aggregate_kwargs
    )
