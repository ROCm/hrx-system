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
load(":loom_product_format.bzl", "LoomProductFormatInfo")
load(
    ":loom_target_profile.bzl",
    "LoomTargetFormatSupportInfo",
    "LoomTargetProfileInfo",
)

_LOOM_COMPILE_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:compile_toolchain_type")
_LOOM_LINK_TOOLCHAIN_TYPE = Label("//loom/build_tools/bazel:link_toolchain_type")

LoomBinaryInfo = provider(
    doc = "One closed Loom deployment product and its evidence artifacts.",
    fields = {
        "artifacts": "Depset of runtime artifacts produced by this binary.",
        "kind": "Deployment product kind: kernel or command.",
        "linked_module": "Closed Loom bytecode module used for emission.",
        "primary_artifact": "Primary runtime artifact produced by this binary.",
        "product_formats": "Product-name keyed selected format targets.",
        "reports": "Depset of compile reports for emitted artifacts.",
        "target_profiles": "Ordered configured target-profile targets used for compilation.",
    },
)

def _require_binary_inputs(ctx):
    if not ctx.files.srcs and not ctx.attr.deps:
        fail("%s requires at least one source across srcs and deps" % ctx.label)

def _require_product_format(ctx, format_target, product, output_kind):
    format_info = format_target[LoomProductFormatInfo]
    if not format_info.format:
        fail("%s format %s has an empty public format name" % (ctx.label, format_target.label))
    if format_info.product != product:
        fail(
            "%s format %s describes product %r, expected %r" % (
                ctx.label,
                format_target.label,
                format_info.product,
                product,
            ),
        )
    if format_info.output_kind != output_kind:
        fail(
            "%s format %s has output kind %r, expected %r for product %r" % (
                ctx.label,
                format_target.label,
                format_info.output_kind,
                output_kind,
                product,
            ),
        )
    if not format_info.output_extension.startswith("."):
        fail(
            "%s format %s output extension must begin with '.'" %
            (ctx.label, format_target.label),
        )
    if "/" in format_info.output_extension or "\\" in format_info.output_extension:
        fail(
            "%s format %s output extension must not contain a path" %
            (ctx.label, format_target.label),
        )
    if output_kind == "file" and format_info.artifact_directory_extension:
        fail(
            "%s single-file format %s has an artifact directory extension" %
            (ctx.label, format_target.label),
        )
    if output_kind == "artifact_set":
        if not format_info.artifact_directory_extension.startswith("."):
            fail(
                "%s artifact-set format %s directory extension must begin with '.'" %
                (ctx.label, format_target.label),
            )
        if ("/" in format_info.artifact_directory_extension or
            "\\" in format_info.artifact_directory_extension):
            fail(
                "%s artifact-set format %s directory extension must not contain a path" %
                (ctx.label, format_target.label),
            )
    return format_info

def _resolve_target_product_format(ctx, product, output_kind, explicit_format):
    support = ctx.attr.target[LoomTargetFormatSupportInfo]
    selected_format = explicit_format
    if selected_format == None:
        selected_format = support.canonical_formats.get(product)
        if selected_format == None:
            fail(
                "%s target profile %s has no canonical format for product %r" %
                (ctx.label, ctx.attr.target.label, product),
            )

    match_count = 0
    for supported_format in support.formats:
        if supported_format.label == selected_format.label:
            match_count += 1
    if match_count == 0:
        fail(
            "%s target profile %s does not support product format %s" % (
                ctx.label,
                ctx.attr.target.label,
                selected_format.label,
            ),
        )
    if match_count != 1:
        fail(
            "%s target profile %s declares product format %s more than once" % (
                ctx.label,
                ctx.attr.target.label,
                selected_format.label,
            ),
        )

    return struct(
        info = _require_product_format(
            ctx,
            selected_format,
            product,
            output_kind,
        ),
        target = selected_format,
    )

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

def _declare_kernel_product(
        ctx,
        linked_module,
        product_format,
        target_profile,
        output_stem,
        mnemonic,
        progress_message):
    artifact = ctx.actions.declare_file(
        output_stem + product_format.info.output_extension,
    )
    compile_report = ctx.actions.declare_file(output_stem + ".compile.json")
    args = ctx.actions.args()
    args.add(linked_module)
    args.add("--product=kernel")
    args.add("--format=%s" % product_format.info.format)
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
        mnemonic = mnemonic,
        outputs = [artifact, compile_report],
        progress_message = progress_message,
    )
    return struct(
        artifact = artifact,
        compile_report = compile_report,
        format = product_format.target,
    )

def _declare_command_product(ctx, linked_module, product_format):
    format_info = product_format.info
    manifest = ctx.actions.declare_file(
        ctx.label.name + format_info.output_extension,
    )
    artifacts = ctx.actions.declare_directory(
        ctx.label.name + format_info.artifact_directory_extension,
    )
    compile_report = ctx.actions.declare_file(
        ctx.label.name + ".commands.compile.json",
    )
    args = ctx.actions.args()
    args.add(linked_module)
    args.add("--product=command")
    args.add("--format=%s" % format_info.format)
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
        format = product_format.target,
        manifest = manifest,
    )

def _loom_kernel_binary_impl(ctx):
    _require_binary_inputs(ctx)
    target_profile = ctx.attr.target[LoomTargetProfileInfo]
    product_format = _resolve_target_product_format(
        ctx,
        "kernel",
        "file",
        ctx.attr.format,
    )
    linked = _declare_binary_linked_module(
        ctx,
        "kernel",
        target_profile = target_profile,
    )
    product = _declare_kernel_product(
        ctx = ctx,
        linked_module = linked.linked_module,
        product_format = product_format,
        target_profile = target_profile,
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
            product_formats = {"kernel": product.format},
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

def _target_binary_attrs(target_doc, format_doc):
    attrs = _binary_link_attrs()
    attrs["format"] = attr.label(
        providers = [LoomProductFormatInfo],
        doc = format_doc,
    )
    attrs["target"] = attr.label(
        mandatory = True,
        providers = [LoomTargetFormatSupportInfo, LoomTargetProfileInfo],
        doc = target_doc,
    )
    return attrs

loom_kernel_binary = rule(
    implementation = _loom_kernel_binary_impl,
    attrs = _target_binary_attrs(
        "Immutable target profile used for every emitted kernel.",
        "Optional kernel format; the target's canonical format is used when omitted.",
    ),
    doc = "Links and emits one closed loader-ready kernel product.",
    toolchains = [
        _LOOM_COMPILE_TOOLCHAIN_TYPE,
        _LOOM_LINK_TOOLCHAIN_TYPE,
    ],
)

def _loom_command_binary_impl(ctx):
    _require_binary_inputs(ctx)
    target_profile = ctx.attr.target[LoomTargetProfileInfo]
    command_format = _resolve_target_product_format(
        ctx,
        "command",
        "artifact_set",
        ctx.attr.format,
    )
    kernel_format = _resolve_target_product_format(
        ctx,
        "kernel",
        "file",
        ctx.attr.kernel_format,
    )
    linked = _declare_binary_linked_module(
        ctx,
        "command",
        target_profile = target_profile,
    )
    command_product = _declare_command_product(
        ctx,
        linked.linked_module,
        command_format,
    )
    kernel_product = _declare_kernel_product(
        ctx = ctx,
        linked_module = linked.linked_module,
        product_format = kernel_format,
        output_stem = ctx.label.name + ".kernels",
        target_profile = target_profile,
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
            product_formats = {
                "command": command_product.format,
                "kernel": kernel_product.format,
            },
            reports = depset(reports),
            target_profiles = [ctx.attr.target],
        ),
    ]

_COMMAND_BINARY_ATTRS = _target_binary_attrs(
    "Immutable target profile used for every emitted kernel entry.",
    "Optional command format; the target's canonical format is used when omitted.",
)
_COMMAND_BINARY_ATTRS["kernel_format"] = attr.label(
    providers = [LoomProductFormatInfo],
    doc = "Optional kernel format; the target's canonical format is used when omitted.",
)

loom_command_binary = rule(
    implementation = _loom_command_binary_impl,
    attrs = _COMMAND_BINARY_ATTRS,
    doc = "Links and emits portable command programs with their kernel executable.",
    toolchains = [
        _LOOM_COMPILE_TOOLCHAIN_TYPE,
        _LOOM_LINK_TOOLCHAIN_TYPE,
    ],
)
