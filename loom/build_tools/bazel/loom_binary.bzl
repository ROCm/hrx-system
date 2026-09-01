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
        "kind": "Deployment product kind: kernel, command, or VM.",
        "linked_module": "Closed Loom bytecode module used for emission.",
        "primary_artifact": "Primary runtime artifact produced by this binary.",
        "reports": "Depset of compile reports for emitted artifacts.",
        "target_profiles": "Ordered configured target-profile targets used for compilation.",
    },
)

def _require_binary_inputs(ctx):
    if not ctx.files.srcs and not ctx.attr.deps:
        fail("%s requires at least one source across srcs and deps" % ctx.label)

def _amdgpu_target_profile(ctx, product_kind):
    target_profile = ctx.attr.target[LoomTargetProfileInfo]
    if target_profile.family != "amdgpu":
        fail(
            ("%s cannot emit a %s binary for target profile family %r; " +
             "only amdgpu-backed %s products are available") % (
                ctx.label,
                product_kind,
                target_profile.family,
                product_kind,
            ),
        )
    if LoomAmdgpuTargetProfileInfo not in ctx.attr.target:
        fail(
            "%s target profile claims family amdgpu without an AMDGPU identity" %
            ctx.label,
        )
    return ctx.attr.target[LoomAmdgpuTargetProfileInfo]

def _declare_binary_linked_module(ctx, product_kind, target_profile = None):
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
        target_profile = target_profile,
        output_stem = ctx.label.name + ".linked",
        mnemonic = "LoomBinaryLink",
        progress_message = "Linking %s binary %s" % (product_kind, ctx.label),
    )
    return struct(
        dependency_reports = dependency_reports,
        linked_module = linked_module,
    )

def _declare_amdgpu_kernel_product(
        ctx,
        linked_module,
        amdgpu_profile,
        output_stem,
        mnemonic,
        progress_message):
    artifact = ctx.actions.declare_file(output_stem + ".hsaco")
    compile_report = ctx.actions.declare_file(output_stem + ".compile.json")
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
        mnemonic = mnemonic,
        outputs = [artifact, compile_report],
        progress_message = progress_message,
    )
    return struct(
        artifact = artifact,
        compile_report = compile_report,
    )

def _declare_command_product(ctx, linked_module):
    manifest = ctx.actions.declare_file(ctx.label.name + ".commands.json")
    artifacts = ctx.actions.declare_directory(ctx.label.name + ".commands")
    compile_report = ctx.actions.declare_file(
        ctx.label.name + ".commands.compile.json",
    )
    args = ctx.actions.args()
    args.add(linked_module)
    args.add("--backend=command")
    args.add("--output=%s" % manifest.path)
    args.add("--emit-command-artifacts=%s" % artifacts.path)
    args.add("--compile-report=details")
    args.add("--compile-report-output=%s" % compile_report.path)

    tool = ctx.toolchains[_LOOM_COMPILE_TOOLCHAIN_TYPE].tool
    ctx.actions.run(
        arguments = [args],
        executable = tool.files_to_run,
        inputs = depset(direct = [linked_module]),
        mnemonic = "LoomCommandBinary",
        outputs = [manifest, artifacts, compile_report],
        progress_message = "Emitting portable command binary %s" % ctx.label,
    )
    return struct(
        artifacts = artifacts,
        compile_report = compile_report,
        manifest = manifest,
    )

def _declare_vm_product(ctx, linked_module):
    artifact = ctx.actions.declare_file(ctx.label.name + ".vmfb")
    compile_report = ctx.actions.declare_file(ctx.label.name + ".compile.json")
    args = ctx.actions.args()
    args.add(linked_module)
    args.add("--backend=vm")
    args.add("--output=%s" % artifact.path)
    args.add("--compile-report=details")
    args.add("--compile-report-output=%s" % compile_report.path)

    tool = ctx.toolchains[_LOOM_COMPILE_TOOLCHAIN_TYPE].tool
    ctx.actions.run(
        arguments = [args],
        executable = tool.files_to_run,
        inputs = depset(direct = [linked_module]),
        mnemonic = "LoomVmBinary",
        outputs = [artifact, compile_report],
        progress_message = "Compiling VM binary %s" % ctx.label,
    )
    return struct(
        artifact = artifact,
        compile_report = compile_report,
    )

def _loom_kernel_binary_impl(ctx):
    _require_binary_inputs(ctx)
    amdgpu_profile = _amdgpu_target_profile(ctx, "kernel")
    linked = _declare_binary_linked_module(
        ctx,
        "kernel",
        target_profile = struct(
            family = "amdgpu",
            selector = amdgpu_profile.target,
        ),
    )
    product = _declare_amdgpu_kernel_product(
        ctx = ctx,
        linked_module = linked.linked_module,
        amdgpu_profile = amdgpu_profile,
        output_stem = ctx.label.name,
        mnemonic = "LoomKernelBinary",
        progress_message = "Compiling kernel binary %s for %s" % (
            ctx.label,
            ctx.attr.target.label,
        ),
    )

    return [
        DefaultInfo(files = depset([product.artifact])),
        OutputGroupInfo(
            compile_reports = depset([product.compile_report]),
            dependency_reports = depset(linked.dependency_reports),
            linked_modules = depset([linked.linked_module]),
        ),
        LoomBinaryInfo(
            artifacts = depset([product.artifact]),
            kind = "kernel",
            linked_module = linked.linked_module,
            primary_artifact = product.artifact,
            reports = depset([product.compile_report]),
            target_profiles = [ctx.attr.target],
        ),
    ]

def _binary_link_attrs():
    return {
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
    }

def _target_binary_attrs(target_doc):
    attrs = _binary_link_attrs()
    attrs["target"] = attr.label(
        mandatory = True,
        providers = [LoomTargetProfileInfo],
        doc = target_doc,
    )
    return attrs

loom_kernel_binary = rule(
    implementation = _loom_kernel_binary_impl,
    attrs = _target_binary_attrs(
        "Immutable target profile used for every emitted kernel.",
    ),
    doc = "Links and emits one closed loader-ready kernel product.",
    toolchains = [
        _LOOM_COMPILE_TOOLCHAIN_TYPE,
        _LOOM_LINK_TOOLCHAIN_TYPE,
    ],
)

def _loom_command_binary_impl(ctx):
    _require_binary_inputs(ctx)
    amdgpu_profile = _amdgpu_target_profile(ctx, "command")
    linked = _declare_binary_linked_module(
        ctx,
        "command",
        target_profile = struct(
            family = "amdgpu",
            selector = amdgpu_profile.target,
        ),
    )
    command_product = _declare_command_product(ctx, linked.linked_module)
    kernel_product = _declare_amdgpu_kernel_product(
        ctx = ctx,
        linked_module = linked.linked_module,
        amdgpu_profile = amdgpu_profile,
        output_stem = ctx.label.name + ".kernels",
        mnemonic = "LoomCommandKernelBinary",
        progress_message = "Compiling command kernels for %s against %s" % (
            ctx.label,
            ctx.attr.target.label,
        ),
    )
    artifacts = [
        command_product.manifest,
        command_product.artifacts,
        kernel_product.artifact,
    ]
    reports = [
        command_product.compile_report,
        kernel_product.compile_report,
    ]

    return [
        DefaultInfo(files = depset(artifacts)),
        OutputGroupInfo(
            command_artifacts = depset([command_product.artifacts]),
            command_compile_reports = depset(
                [command_product.compile_report],
            ),
            command_manifests = depset([command_product.manifest]),
            compile_reports = depset(reports),
            dependency_reports = depset(linked.dependency_reports),
            kernel_artifacts = depset([kernel_product.artifact]),
            kernel_compile_reports = depset([kernel_product.compile_report]),
            linked_modules = depset([linked.linked_module]),
        ),
        LoomBinaryInfo(
            artifacts = depset(artifacts),
            kind = "command",
            linked_module = linked.linked_module,
            primary_artifact = command_product.manifest,
            reports = depset(reports),
            target_profiles = [ctx.attr.target],
        ),
    ]

loom_command_binary = rule(
    implementation = _loom_command_binary_impl,
    attrs = _target_binary_attrs(
        "Immutable target profile used for every emitted kernel entry.",
    ),
    doc = "Links and emits portable command programs with their kernel executable.",
    toolchains = [
        _LOOM_COMPILE_TOOLCHAIN_TYPE,
        _LOOM_LINK_TOOLCHAIN_TYPE,
    ],
)

def _loom_vm_binary_impl(ctx):
    _require_binary_inputs(ctx)
    linked = _declare_binary_linked_module(ctx, "VM")
    product = _declare_vm_product(ctx, linked.linked_module)

    return [
        DefaultInfo(files = depset([product.artifact])),
        OutputGroupInfo(
            compile_reports = depset([product.compile_report]),
            dependency_reports = depset(linked.dependency_reports),
            linked_modules = depset([linked.linked_module]),
        ),
        LoomBinaryInfo(
            artifacts = depset([product.artifact]),
            kind = "vm",
            linked_module = linked.linked_module,
            primary_artifact = product.artifact,
            reports = depset([product.compile_report]),
            target_profiles = [],
        ),
    ]

_vm_binary_attrs = _binary_link_attrs()

loom_vm_binary = rule(
    implementation = _loom_vm_binary_impl,
    attrs = _vm_binary_attrs,
    doc = "Links and emits one IREE VM bytecode module from authored VM targets.",
    toolchains = [
        _LOOM_COMPILE_TOOLCHAIN_TYPE,
        _LOOM_LINK_TOOLCHAIN_TYPE,
    ],
)
