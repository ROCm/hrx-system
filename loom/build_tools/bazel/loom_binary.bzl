# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public rules for final Loom deployment products."""

load(
    ":loom_linking.bzl",
    "LoomLibraryInfo",
    "loom_linking",
)
load(
    ":loom_target_profile.bzl",
    "LoomTargetProfileInfo",
)

_LOOM_COMPILE_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:compile_toolchain_type")
_LOOM_LINK_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:link_toolchain_type")

LoomBinaryInfo = provider(
    doc = "One closed Loom kernel binary and its evidence artifacts.",
    fields = {
        "artifacts": "Depset of runtime artifacts produced by this binary.",
        "linked_module": "Closed Loom bytecode module used for emission.",
        "primary_artifact": "Primary runtime artifact produced by this binary.",
        "reports": "Depset of compile reports for emitted artifacts.",
        "target_profiles": "Ordered configured target-profile targets used for compilation.",
    },
)

def _require_binary_inputs(ctx):
    if not ctx.files.srcs and not ctx.attr.deps:
        fail("%s requires at least one source across srcs and deps" % ctx.label)
    if not ctx.files.srcs and (
        ctx.files.data or
        ctx.attr.input_format or
        ctx.attr.inputopts
    ):
        fail("%s source admission options require srcs" % ctx.label)

def _declare_binary_linked_module(ctx, target_profile):
    dependency_infos = [dep[LoomLibraryInfo] for dep in ctx.attr.deps]
    dependencies = loom_linking.collect_dependency_modules(dependency_infos)
    direct_modules = list(dependencies.direct)
    dependency_reports = []
    if ctx.files.srcs:
        source_library = loom_linking.declare_relocatable_module(
            ctx = ctx,
            sources = ctx.files.srcs,
            data = ctx.files.data,
            input_format = ctx.attr.input_format,
            inputopts = [
                ctx.expand_location(option, targets = ctx.attr.srcs + ctx.attr.data)
                for option in ctx.attr.inputopts
            ],
            dependency_infos = dependency_infos,
            output_stem = ctx.label.name + ".sources",
            mnemonic = "LoomBinarySources",
            progress_message = "Assembling direct sources for %s" % ctx.label,
        )
        direct_modules.append(source_library.module)
        dependency_reports.append(source_library.dependency_report)

    linked_module = loom_linking.declare_linked_module(
        ctx = ctx,
        direct_modules = direct_modules,
        transitive_modules = dependencies.transitive,
        roots = ctx.attr.roots,
        configs = ctx.attr.configs,
        target_profile = target_profile,
        output_stem = ctx.label.name + ".linked",
        mnemonic = "LoomBinaryLink",
        progress_message = "Linking kernel binary %s" % ctx.label,
    )
    return struct(
        dependency_reports = dependency_reports,
        linked_module = linked_module,
    )

def _declare_kernel_artifact(
        ctx,
        linked_module,
        target_profile,
        artifact):
    compile_report = ctx.actions.declare_file(ctx.label.name + ".compile.json")
    args = ctx.actions.args()
    args.add(linked_module)
    args.add("--target=%s:%s" % (
        target_profile.family,
        target_profile.selector,
    ))
    args.add("--output=%s" % artifact.path)
    args.add("--compile-report=details")
    args.add("--compile-report-output=%s" % compile_report.path)

    tool = ctx.toolchains[_LOOM_COMPILE_TOOLCHAIN_TYPE].tool
    ctx.actions.run(
        arguments = [args],
        executable = tool.files_to_run,
        inputs = depset(direct = [linked_module]),
        mnemonic = "LoomKernelBinary",
        outputs = [artifact, compile_report],
        progress_message = "Compiling kernel binary %s for %s" % (
            ctx.label,
            ctx.attr.target.label,
        ),
    )
    return struct(
        artifact = artifact,
        compile_report = compile_report,
    )

def _loom_kernel_binary_impl(ctx):
    _require_binary_inputs(ctx)
    target_profile = ctx.attr.target[LoomTargetProfileInfo]
    linked = _declare_binary_linked_module(
        ctx,
        target_profile = target_profile,
    )
    artifact = ctx.outputs.out
    if artifact == None:
        artifact = ctx.actions.declare_file(ctx.label.name)
    compiled = _declare_kernel_artifact(
        ctx = ctx,
        linked_module = linked.linked_module,
        target_profile = target_profile,
        artifact = artifact,
    )

    return [
        DefaultInfo(files = depset([compiled.artifact])),
        OutputGroupInfo(
            compile_reports = depset([compiled.compile_report]),
            dependency_reports = depset(linked.dependency_reports),
            linked_modules = depset([linked.linked_module]),
        ),
        LoomBinaryInfo(
            artifacts = depset([compiled.artifact]),
            linked_module = linked.linked_module,
            primary_artifact = compiled.artifact,
            reports = depset([compiled.compile_report]),
            target_profiles = [ctx.attr.target],
        ),
    ]

def _binary_link_attrs():
    return {
        "configs": attr.string_dict(
            doc = "Compile-time config symbol values keyed by declaration name.",
        ),
        "data": attr.label_list(
            allow_files = True,
            doc = "Declared source-admission inputs, such as included headers.",
        ),
        "deps": attr.label_list(
            providers = [LoomLibraryInfo],
            doc = "Direct Loom libraries contributing exported roots and closure.",
        ),
        "input_format": attr.string(
            doc = "Direct source provider override; empty selects each source by filename.",
        ),
        "inputopts": attr.string_list(
            doc = "Provider-scoped direct source options, with location expansion.",
        ),
        "roots": attr.string_list(
            doc = "Optional explicit roots replacing exports from direct inputs.",
        ),
        "srcs": attr.label_list(
            allow_files = True,
            doc = "Direct provider-owned sources assembled as an implicit relocatable library.",
        ),
    }

def _target_binary_attrs():
    attrs = _binary_link_attrs()
    attrs["target"] = attr.label(
        mandatory = True,
        providers = [LoomTargetProfileInfo],
        doc = "Immutable target profile used for every emitted kernel.",
    )
    return attrs

def _kernel_binary_attrs():
    attrs = _target_binary_attrs()
    attrs["out"] = attr.output(
        doc = "Optional kernel artifact path. Defaults to the extensionless rule name.",
    )
    return attrs

loom_kernel_binary = rule(
    implementation = _loom_kernel_binary_impl,
    attrs = _kernel_binary_attrs(),
    doc = "Links and emits one closed loader-ready kernel artifact.",
    toolchains = [
        _LOOM_COMPILE_TOOLCHAIN_TYPE,
        _LOOM_LINK_TOOLCHAIN_TYPE,
    ],
)
