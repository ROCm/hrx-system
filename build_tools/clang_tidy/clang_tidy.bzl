# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bazel actions for running IREE clang-tidy checks."""

load(
    "@iree_clang_tidy_llvm//:config.bzl",
    "CLANG_TIDY_CLANG_RESOURCE_DIR",
    "CLANG_TIDY_LLVM_IS_WINDOWS",
    "CLANG_TIDY_LLVM_TARGET_COMPATIBLE_WITH",
)
load("@rules_cc//cc:find_cc_toolchain.bzl", "CC_TOOLCHAIN_ATTRS", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load(
    "//build_tools/bazel:cc_introspection.bzl",
    "iree_cc_compile_command",
    "iree_cc_feature_configuration",
    "iree_cc_sanitize_label",
    "iree_cc_source_files",
)

# clang-tidy always parses through Clang even when Bazel's configured compile
# action uses GCC. Keep target semantic flags intact but omit GCC driver flags
# that Clang rejects before it can analyze the source.
_CLANG_TIDY_UNSUPPORTED_COMPILE_ARGS = {
    "-c": None,
    "-fno-canonical-system-headers": None,
    "/c": None,
    "/showIncludes": None,
}

def _clang_tidy_compile_args(compile_command, source):
    filtered_args = [
        arg
        for arg in compile_command.compile_args
        if arg not in _CLANG_TIDY_UNSUPPORTED_COMPILE_ARGS and arg.replace("\\", "/") != source.path
    ]

    # A fixed compilation database does not preserve the compiler executable
    # that normally selects Clang's driver dialect. Restore CL mode when the
    # configured Bazel compiler consumes MSVC-style arguments.
    compiler_name = compile_command.compiler.replace("\\", "/").split("/")[-1].lower()
    if compiler_name in ["cl", "cl.exe", "clang-cl", "clang-cl.exe"]:
        filtered_args.insert(0, "--driver-mode=cl")

    # Preserve the resource directory named by Bazel's crosstool module map.
    # The clang-tidy VFS overlay redirects this virtual path to the declared,
    # relocated LLVM tree.
    return filtered_args + [
        "-resource-dir=%s" % CLANG_TIDY_CLANG_RESOURCE_DIR,
    ]

IreeClangTidyInfo = provider(
    doc = "clang-tidy artifacts collected from configured C/C++ targets.",
    fields = {
        "fixes": "depset of per-translation-unit clang-tidy replacement YAML files.",
        "local_fixes": "depset of direct per-translation-unit clang-tidy replacement YAML files.",
        "local_reports": "depset of direct per-translation-unit clang-tidy report files.",
        "reports": "depset of per-translation-unit clang-tidy report files.",
    },
)

IreeRecursionInfo = provider(
    doc = "Cross-translation-unit recursion artifacts collected from C/C++ targets.",
    fields = {
        "local_reports": "depset of direct recursion summary clang-tidy reports.",
        "local_summaries": "depset of direct per-translation-unit call-graph summaries.",
        "reports": "depset of transitive recursion summary clang-tidy reports.",
        "summaries": "depset of transitive per-translation-unit call-graph summaries.",
    },
)

def _sanitize_path(path):
    result = path
    for old, new in [
        ("/", "_"),
        ("\\", "_"),
        (":", "_"),
        ("+", "_"),
        ("-", "_"),
        (".", "_"),
    ]:
        result = result.replace(old, new)
    return result

def _as_file_depset(value):
    if not value:
        return None
    if type(value) == "depset":
        return value
    return depset(value)

def _module_map_file(module_map):
    if hasattr(module_map, "file"):
        return module_map.file
    return module_map

def _compilation_input_depsets(compilation_context):
    inputs = []
    for field in [
        "headers",
        "direct_headers",
        "textual_headers",
        "direct_textual_headers",
    ]:
        if not hasattr(compilation_context, field):
            continue
        value = _as_file_depset(getattr(compilation_context, field))
        if value:
            inputs.append(value)
    if hasattr(compilation_context, "_module_map"):
        module_map = compilation_context._module_map
        if module_map:
            inputs.append(depset([_module_map_file(module_map)]))
    if hasattr(compilation_context, "_direct_module_maps"):
        module_maps = [
            _module_map_file(module_map)
            for module_map in compilation_context._direct_module_maps.to_list()
        ]
        if module_maps:
            inputs.append(depset(module_maps))
    return inputs

def _clang_tidy_report_path(target_label, source):
    return "%s.%s.clang_tidy.txt" % (
        iree_cc_sanitize_label(target_label),
        _sanitize_path(source.path),
    )

def _clang_tidy_fixes_report_path(target_label, source):
    return "%s.%s.clang_tidy_fixes.txt" % (
        iree_cc_sanitize_label(target_label),
        _sanitize_path(source.path),
    )

def _clang_tidy_fixes_path(target_label, source):
    return "%s.%s.clang_tidy_fixes.yaml" % (
        iree_cc_sanitize_label(target_label),
        _sanitize_path(source.path),
    )

def _recursion_report_path(target_label, source):
    return "%s.%s.recursion_clang_tidy.txt" % (
        iree_cc_sanitize_label(target_label),
        _sanitize_path(source.path),
    )

def _recursion_summary_path(target_label, source):
    return "%s.%s.recursion_summary.json" % (
        iree_cc_sanitize_label(target_label),
        _sanitize_path(source.path),
    )

def _run_clang_tidy_action(ctx, target, cc_toolchain, feature_configuration, source):
    compile_command = iree_cc_compile_command(
        ctx,
        target,
        cc_toolchain,
        feature_configuration,
        source,
    )
    emit_fixes = ctx.attr.emit_fixes == "true"
    report_path = (
        _clang_tidy_fixes_report_path(target.label, source) if emit_fixes else _clang_tidy_report_path(target.label, source)
    )
    report = ctx.actions.declare_file(report_path)
    fixes = None
    outputs = [report]
    tools = [ctx.executable._clang_tidy]
    args = ctx.actions.args()
    args.add("--clang-tidy", ctx.executable._clang_tidy)
    if not CLANG_TIDY_LLVM_IS_WINDOWS:
        args.add("--plugin", ctx.executable._plugin)
        tools.append(ctx.executable._plugin)
    args.add("--source", source)
    args.add("--output", report)
    args.add("--config-file", ctx.file._config)
    args.add("--vfsoverlay", ctx.file._clang_resource_overlay)
    if emit_fixes:
        fixes = ctx.actions.declare_file(_clang_tidy_fixes_path(target.label, source))
        outputs.append(fixes)
        args.add("--export-fixes", fixes)
        args.add("--allow-diagnostics")
    else:
        args.add("--warnings-as-errors=%s" % ctx.attr._warnings_as_errors)
    args.add("--")
    args.add_all(_clang_tidy_compile_args(compile_command, source))

    compilation_context = target[CcInfo].compilation_context
    inputs = depset(
        direct = [source, ctx.file._config, ctx.file._clang_resource_overlay],
        transitive = _compilation_input_depsets(compilation_context) + [
            depset(ctx.files._clang_resource_headers),
            cc_toolchain.all_files,
        ],
    )
    ctx.actions.run(
        executable = ctx.executable._runner,
        arguments = [args],
        env = compile_command.environment,
        inputs = inputs,
        outputs = outputs,
        tools = tools,
        mnemonic = "IreeClangTidy",
        progress_message = "Running clang-tidy on %s" % source.short_path,
    )
    return report, fixes

def _run_recursion_summary_action(ctx, target, cc_toolchain, feature_configuration, source):
    compile_command = iree_cc_compile_command(
        ctx,
        target,
        cc_toolchain,
        feature_configuration,
        source,
    )
    report = ctx.actions.declare_file(_recursion_report_path(target.label, source))
    summary = ctx.actions.declare_file(_recursion_summary_path(target.label, source))
    tools = [ctx.executable._clang_tidy]
    args = ctx.actions.args()
    args.add("--clang-tidy", ctx.executable._clang_tidy)
    if not CLANG_TIDY_LLVM_IS_WINDOWS:
        args.add("--plugin", ctx.executable._plugin)
        tools.append(ctx.executable._plugin)
    args.add("--source", source)
    args.add("--output", report)
    args.add("--config-file", ctx.file._config)
    args.add("--vfsoverlay", ctx.file._clang_resource_overlay)
    args.add("--checks=-*,iree-unbounded-recursion")
    args.add("--recursion-summary", summary)
    args.add("--suppress-recursion-diagnostics")
    args.add("--")
    args.add_all(_clang_tidy_compile_args(compile_command, source))

    compilation_context = target[CcInfo].compilation_context
    inputs = depset(
        direct = [source, ctx.file._config, ctx.file._clang_resource_overlay],
        transitive = _compilation_input_depsets(compilation_context) + [
            depset(ctx.files._clang_resource_headers),
            cc_toolchain.all_files,
        ],
    )
    ctx.actions.run(
        executable = ctx.executable._runner,
        arguments = [args],
        env = compile_command.environment,
        inputs = inputs,
        outputs = [report, summary],
        tools = tools,
        mnemonic = "IreeRecursionSummary",
        progress_message = "Extracting recursion call graph from %s" % source.short_path,
    )
    return report, summary

def _collect_clang_tidy_aspect_impl(target, ctx):
    transitive_fixes = []
    transitive_reports = []
    for attr_name in [
        "actual",
        "deps",
        "implementation_deps",
    ]:
        if not hasattr(ctx.rule.attr, attr_name):
            continue
        attr_value = getattr(ctx.rule.attr, attr_name)
        if type(attr_value) == type([]):
            deps = attr_value
        elif attr_value:
            deps = [attr_value]
        else:
            deps = []
        for dep in deps:
            if IreeClangTidyInfo in dep:
                transitive_fixes.append(dep[IreeClangTidyInfo].fixes)
                transitive_reports.append(dep[IreeClangTidyInfo].reports)

    local_fixes = []
    local_reports = []
    if CcInfo in target:
        cc_toolchain = find_cc_toolchain(ctx)
        feature_configuration = iree_cc_feature_configuration(ctx, cc_toolchain)
        for source in iree_cc_source_files(ctx):
            report, fixes = _run_clang_tidy_action(
                ctx,
                target,
                cc_toolchain,
                feature_configuration,
                source,
            )
            local_reports.append(report)
            if fixes:
                local_fixes.append(fixes)

    fixes = depset(local_fixes, transitive = transitive_fixes)
    reports = depset(local_reports, transitive = transitive_reports)
    return [
        IreeClangTidyInfo(
            fixes = fixes,
            local_fixes = depset(local_fixes),
            local_reports = depset(local_reports),
            reports = reports,
        ),
        OutputGroupInfo(
            iree_clang_tidy_fixes = fixes,
            iree_clang_tidy_local_fixes = depset(local_fixes),
            iree_clang_tidy_local_reports = depset(local_reports),
            iree_clang_tidy_reports = reports,
        ),
    ]

collect_clang_tidy_aspect = aspect(
    implementation = _collect_clang_tidy_aspect_impl,
    attr_aspects = [
        "actual",
        "deps",
        "implementation_deps",
    ],
    attrs = dict(CC_TOOLCHAIN_ATTRS, **{
        "emit_fixes": attr.string(
            default = "false",
            values = ["false", "true"],
            doc = "When true, emit clang-tidy replacement YAML files instead of failing on diagnostics.",
        ),
        "_clang_resource_headers": attr.label(
            allow_files = True,
            cfg = "exec",
            default = Label("@iree_clang_tidy_llvm//:clang_resource_headers"),
        ),
        "_clang_resource_overlay": attr.label(
            allow_single_file = True,
            cfg = "exec",
            default = Label("@iree_clang_tidy_llvm//:clang_resource_overlay"),
        ),
        "_clang_tidy": attr.label(
            cfg = "exec",
            default = Label("//build_tools/clang_tidy:clang-tidy-tool"),
            executable = True,
        ),
        "_config": attr.label(
            allow_single_file = True,
            default = Label("//build_tools/clang_tidy:clang_tidy_config.yaml"),
            doc = "Repository clang-tidy policy configuration.",
        ),
        "_plugin": attr.label(
            cfg = "exec",
            default = Label("//build_tools/clang_tidy:IREEClangTidyPlugin.so"),
            executable = True,
        ),
        "_runner": attr.label(
            cfg = "exec",
            default = Label("//build_tools/clang_tidy:run_clang_tidy_action"),
            executable = True,
        ),
        "_warnings_as_errors": attr.string(
            default = "*",
            doc = "clang-tidy warning globs promoted to errors.",
        ),
    }),
    fragments = ["cpp"],
    required_providers = [CcInfo],
    toolchains = use_cc_toolchain(),
)

def _collect_recursion_aspect_impl(target, ctx):
    transitive_reports = []
    transitive_summaries = []
    for attr_name in [
        "actual",
        "deps",
        "implementation_deps",
    ]:
        if not hasattr(ctx.rule.attr, attr_name):
            continue
        attr_value = getattr(ctx.rule.attr, attr_name)
        if type(attr_value) == type([]):
            deps = attr_value
        elif attr_value:
            deps = [attr_value]
        else:
            deps = []
        for dep in deps:
            if IreeRecursionInfo in dep:
                transitive_reports.append(dep[IreeRecursionInfo].reports)
                transitive_summaries.append(dep[IreeRecursionInfo].summaries)

    local_reports = []
    local_summaries = []
    if CcInfo in target:
        cc_toolchain = find_cc_toolchain(ctx)
        feature_configuration = iree_cc_feature_configuration(ctx, cc_toolchain)
        for source in iree_cc_source_files(ctx):
            report, summary = _run_recursion_summary_action(
                ctx,
                target,
                cc_toolchain,
                feature_configuration,
                source,
            )
            local_reports.append(report)
            local_summaries.append(summary)

    reports = depset(local_reports, transitive = transitive_reports)
    summaries = depset(local_summaries, transitive = transitive_summaries)
    return [
        IreeRecursionInfo(
            local_reports = depset(local_reports),
            local_summaries = depset(local_summaries),
            reports = reports,
            summaries = summaries,
        ),
        OutputGroupInfo(
            iree_recursion_reports = reports,
            iree_recursion_summaries = summaries,
        ),
    ]

collect_recursion_aspect = aspect(
    implementation = _collect_recursion_aspect_impl,
    attr_aspects = [
        "actual",
        "deps",
        "implementation_deps",
    ],
    attrs = dict(CC_TOOLCHAIN_ATTRS, **{
        "_clang_resource_headers": attr.label(
            allow_files = True,
            cfg = "exec",
            default = Label("@iree_clang_tidy_llvm//:clang_resource_headers"),
        ),
        "_clang_resource_overlay": attr.label(
            allow_single_file = True,
            cfg = "exec",
            default = Label("@iree_clang_tidy_llvm//:clang_resource_overlay"),
        ),
        "_clang_tidy": attr.label(
            cfg = "exec",
            default = Label("//build_tools/clang_tidy:clang-tidy-tool"),
            executable = True,
        ),
        "_config": attr.label(
            allow_single_file = True,
            default = Label("//build_tools/clang_tidy:clang_tidy_config.yaml"),
            doc = "Repository clang-tidy policy configuration.",
        ),
        "_plugin": attr.label(
            cfg = "exec",
            default = Label("//build_tools/clang_tidy:IREEClangTidyPlugin.so"),
            executable = True,
        ),
        "_runner": attr.label(
            cfg = "exec",
            default = Label("//build_tools/clang_tidy:run_clang_tidy_action"),
            executable = True,
        ),
    }),
    fragments = ["cpp"],
    required_providers = [CcInfo],
    toolchains = use_cc_toolchain(),
)

def _iree_clang_tidy_impl(ctx):
    transitive_fixes = []
    transitive_local_fixes = []
    transitive_local_reports = []
    transitive_reports = []
    for target in ctx.attr.targets:
        if IreeClangTidyInfo in target:
            transitive_fixes.append(target[IreeClangTidyInfo].fixes)
            transitive_local_fixes.append(target[IreeClangTidyInfo].local_fixes)
            transitive_local_reports.append(target[IreeClangTidyInfo].local_reports)
            transitive_reports.append(target[IreeClangTidyInfo].reports)
    fixes = depset(transitive = transitive_fixes)
    local_fixes = depset(transitive = transitive_local_fixes)
    local_reports = depset(transitive = transitive_local_reports)
    reports = depset(transitive = transitive_reports)
    return [
        DefaultInfo(files = reports),
        IreeClangTidyInfo(
            fixes = fixes,
            local_fixes = local_fixes,
            local_reports = local_reports,
            reports = reports,
        ),
        OutputGroupInfo(
            iree_clang_tidy_fixes = fixes,
            iree_clang_tidy_local_fixes = local_fixes,
            iree_clang_tidy_local_reports = local_reports,
            iree_clang_tidy_reports = reports,
        ),
    ]

_iree_clang_tidy_rule = rule(
    implementation = _iree_clang_tidy_impl,
    attrs = {
        "targets": attr.label_list(
            aspects = [collect_clang_tidy_aspect],
            doc = "C/C++ targets to analyze with clang-tidy.",
            mandatory = True,
            providers = [CcInfo],
        ),
    },
    doc = "Runs clang-tidy over configured C/C++ targets as Bazel actions.",
)

def iree_clang_tidy(name, **kwargs):
    """Runs clang-tidy over configured C/C++ targets as Bazel actions.

    Args:
      name: Target name.
      **kwargs: Attributes forwarded to the underlying clang-tidy rule.
    """
    target_compatible_with = kwargs.pop("target_compatible_with", [])
    _iree_clang_tidy_rule(
        name = name,
        target_compatible_with = CLANG_TIDY_LLVM_TARGET_COMPATIBLE_WITH + target_compatible_with,
        **kwargs
    )

def _iree_recursion_analysis_impl(ctx):
    transitive_reports = []
    transitive_summaries = []
    for target in ctx.attr.targets:
        if IreeRecursionInfo in target:
            transitive_reports.append(target[IreeRecursionInfo].reports)
            transitive_summaries.append(target[IreeRecursionInfo].summaries)
    source_reports = depset(transitive = transitive_reports)
    summaries = depset(transitive = transitive_summaries)
    report = ctx.actions.declare_file(ctx.label.name + ".txt")
    args = ctx.actions.args()
    args.add("--output", report)
    if ctx.attr.allow_diagnostics:
        args.add("--allow-diagnostics")
        args.add("--quiet")
    args.add_all(summaries)
    ctx.actions.run(
        executable = ctx.executable._aggregator,
        arguments = [args],
        inputs = summaries,
        outputs = [report],
        mnemonic = "IreeRecursionAnalysis",
        progress_message = "Checking cross-translation-unit recursion for %s" % ctx.label,
    )
    return [
        DefaultInfo(files = depset([report])),
        IreeRecursionInfo(
            local_reports = depset(),
            local_summaries = depset(),
            reports = depset([report], transitive = [source_reports]),
            summaries = summaries,
        ),
        OutputGroupInfo(
            iree_recursion_reports = depset([report], transitive = [source_reports]),
            iree_recursion_summaries = summaries,
        ),
    ]

_iree_recursion_analysis_rule = rule(
    implementation = _iree_recursion_analysis_impl,
    attrs = {
        "allow_diagnostics": attr.bool(
            default = False,
            doc = "When true, report recursive SCCs without failing the action.",
        ),
        "targets": attr.label_list(
            aspects = [collect_recursion_aspect],
            doc = "C/C++ targets to analyze for cross-translation-unit recursion.",
            mandatory = True,
            providers = [CcInfo],
        ),
        "_aggregator": attr.label(
            cfg = "exec",
            default = Label("//build_tools/clang_tidy:aggregate_recursion_summaries"),
            executable = True,
        ),
    },
    doc = "Extracts and checks a whole-target C/C++ call graph for recursion.",
)

def iree_recursion_analysis(name, **kwargs):
    """Checks configured C/C++ targets for cross-translation-unit recursion."""
    target_compatible_with = kwargs.pop("target_compatible_with", [])
    _iree_recursion_analysis_rule(
        name = name,
        target_compatible_with = CLANG_TIDY_LLVM_TARGET_COMPATIBLE_WITH + target_compatible_with,
        **kwargs
    )
