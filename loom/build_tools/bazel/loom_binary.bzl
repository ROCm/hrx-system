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
    "LoomAmdgpuTargetProfileInfo",
    "LoomTargetProfileInfo",
)

_LOOM_COMPILE_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:compile_toolchain_type")
_LOOM_LINK_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:link_toolchain_type")

LoomBinaryInfo = provider(
    doc = "One closed Loom deployment product and its evidence artifacts.",
    fields = {
        "artifacts": "Depset of runtime artifacts produced by this binary.",
        "kind": "Deployment product kind: kernel, command, or vm.",
        "linked_module": "Closed Loom bytecode module used for emission.",
        "primary_artifact": "Primary runtime artifact produced by this binary.",
        "reports": "Depset of compile reports for emitted artifacts.",
        "target_profiles": "Ordered configured target-profile targets used for compilation.",
    },
)

def _loom_kernel_binary_impl(ctx):
    if not ctx.files.srcs and not ctx.attr.deps:
        fail("%s requires at least one source across srcs and deps" % ctx.label)

    target_profile = ctx.attr.target[LoomTargetProfileInfo]
    if target_profile.family != "amdgpu":
        fail(
            ("%s cannot emit a kernel binary for target profile family %r; " +
             "only amdgpu kernel products are available") % (
                ctx.label,
                target_profile.family,
            ),
        )
    if LoomAmdgpuTargetProfileInfo not in ctx.attr.target:
        fail(
            "%s target profile claims family amdgpu without an AMDGPU identity" %
            ctx.label,
        )
    amdgpu_profile = ctx.attr.target[LoomAmdgpuTargetProfileInfo]

    dependency_infos = [dep[LoomLibraryInfo] for dep in ctx.attr.deps]
    dependencies = loom_linking.collect_dependency_modules(dependency_infos)
    direct_modules = list(dependencies.direct)
    dependency_reports = []
    if ctx.files.srcs:
        source_library = loom_linking.declare_relocatable_module(
            ctx = ctx,
            sources = ctx.files.srcs,
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
        output_stem = ctx.label.name + ".linked",
        mnemonic = "LoomBinaryLink",
        progress_message = "Linking kernel binary %s" % ctx.label,
    )

    artifact = ctx.actions.declare_file(ctx.label.name + ".hsaco")
    compile_report = ctx.actions.declare_file(ctx.label.name + ".compile.json")
    args = ctx.actions.args()
    args.add(linked_module)
    args.add("--backend=amdgpu-hal")
    args.add("--target=%s" % amdgpu_profile.target)
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

    return [
        DefaultInfo(files = depset([artifact])),
        OutputGroupInfo(
            compile_reports = depset([compile_report]),
            dependency_reports = depset(dependency_reports),
            linked_modules = depset([linked_module]),
        ),
        LoomBinaryInfo(
            artifacts = depset([artifact]),
            kind = "kernel",
            linked_module = linked_module,
            primary_artifact = artifact,
            reports = depset([compile_report]),
            target_profiles = [ctx.attr.target],
        ),
    ]

loom_kernel_binary = rule(
    implementation = _loom_kernel_binary_impl,
    attrs = {
        "configs": attr.string_dict(
            doc = "Compile-time config symbol values keyed by declaration name.",
        ),
        "deps": attr.label_list(
            providers = [LoomLibraryInfo],
            doc = "Direct Loom libraries contributing exported roots and closure.",
        ),
        "roots": attr.string_list(
            doc = "Optional explicit roots replacing exports from direct inputs.",
        ),
        "srcs": attr.label_list(
            allow_files = [".loom", ".loombc"],
            doc = "Direct sources assembled as an implicit relocatable library.",
        ),
        "target": attr.label(
            mandatory = True,
            providers = [LoomTargetProfileInfo],
            doc = "Immutable target profile used for every emitted kernel.",
        ),
    },
    doc = "Links and emits one closed loader-ready kernel product.",
    toolchains = [
        _LOOM_COMPILE_TOOLCHAIN_TYPE,
        _LOOM_LINK_TOOLCHAIN_TYPE,
    ],
)
