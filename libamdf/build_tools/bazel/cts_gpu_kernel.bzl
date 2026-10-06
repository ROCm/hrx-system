# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-built GPU images consumed by native libamdf conformance tests."""

load("//libamdf/requirements:package_policy.bzl", "apply_amdf_target_policy")
load("//loom/build_tools/amdgpu:descriptor_sets.bzl", "loom_amdgpu_selected_descriptor_set_values")
load("//loom/build_tools/amdgpu:target_config.bzl", "LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITY_BY_TARGET")
load("//loom/build_tools/bazel:defs.bzl", "loom_kernel_binary")
load(":cc.bzl", "amdf_cc_library")

def _embed_gpu_kernel_set_impl(ctx):
    args = ctx.actions.args()
    for selector, src in zip(ctx.attr.selectors, ctx.files.srcs):
        args.add("--variant", selector + "=" + src.path)
    args.add("--output", ctx.outputs.header)
    args.add("--implementation", ctx.outputs.implementation)
    args.add("--symbol", ctx.attr.entry_point)
    args.add("--namespace", ctx.attr.namespace)
    ctx.actions.run(
        executable = ctx.executable._embed,
        arguments = [args],
        inputs = ctx.files.srcs,
        outputs = [ctx.outputs.header, ctx.outputs.implementation],
        mnemonic = "AmdfEmbedGpuKernelSet",
        progress_message = "Embedding GPU kernel variants %s" % ctx.label,
    )
    return [DefaultInfo(files = depset([ctx.outputs.header, ctx.outputs.implementation]))]

_embed_gpu_kernel_set = rule(
    implementation = _embed_gpu_kernel_set_impl,
    attrs = {
        "entry_point": attr.string(mandatory = True),
        "header": attr.output(mandatory = True),
        "implementation": attr.output(mandatory = True),
        "namespace": attr.string(mandatory = True),
        "selectors": attr.string_list(mandatory = True),
        "srcs": attr.label_list(allow_files = [".hsaco"], mandatory = True),
        "_embed": attr.label(
            default = Label("//libamdf/cts/gpu/kernels:embed"),
            executable = True,
            cfg = "exec",
        ),
    },
)

def amdf_cts_gpu_kernel_set(
        name,
        srcs,
        targets,
        entry_point,
        namespace,
        data = [],
        input_format = "",
        inputopts = [],
        visibility = None,
        target_compatible_with = []):
    """Compiles one behavior for each target and embeds its immutable products.

    Args:
      name: Library and generated header/implementation stem.
      srcs: Authored Loom source files linked into each program.
      targets: Physical selectors, also naming package-local target profiles.
      entry_point: Exported kernel symbol to extract.
      namespace: C++ namespace containing the kKernels set.
      data: Declared inputs used while admitting the authored sources.
      input_format: Optional source provider override.
      inputopts: Provider-scoped source admission options.
      visibility: Visibility of the generated kernel library.
      target_compatible_with: Additional source-provider constraints applied to
          every generated target.
    """
    policy = apply_amdf_target_policy({})
    policy["target_compatible_with"] += target_compatible_with
    products = {}
    selectors = {}
    compatibility = {"//conditions:default": ["@platforms//:incompatible"]}
    for target in targets:
        capability = LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITY_BY_TARGET[target]
        compatibility["//loom/config/target/amdgpu:" + capability] = []
        product_name = name + "_" + target
        loom_kernel_binary(
            name = product_name,
            testonly = True,
            srcs = srcs,
            data = data,
            input_format = input_format,
            inputopts = inputopts,
            out = product_name + ".hsaco",
            roots = ["@" + entry_point],
            target = ":" + target,
            **policy
        )
        products.setdefault(capability, []).append(":" + product_name)
        selectors.setdefault(capability, []).append(target)
    selected_compatibility = select(compatibility)
    policy["target_compatible_with"] += selected_compatibility
    _embed_gpu_kernel_set(
        name = name + "_embed",
        testonly = True,
        srcs = loom_amdgpu_selected_descriptor_set_values(products),
        selectors = loom_amdgpu_selected_descriptor_set_values(selectors),
        header = name + ".h",
        implementation = name + ".cc",
        entry_point = entry_point,
        namespace = namespace,
        **policy
    )
    amdf_cc_library(
        name = name,
        testonly = True,
        srcs = [name + ".cc"],
        hdrs = [name + ".h"],
        deps = ["//libamdf/cts/gpu/kernels:kernel"],
        target_compatible_with = target_compatible_with + selected_compatibility,
        visibility = visibility,
    )
