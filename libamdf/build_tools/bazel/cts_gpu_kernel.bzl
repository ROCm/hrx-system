# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-built GPU images consumed by native libamdf conformance tests."""

load("//build_tools/bazel:generate.bzl", "iree_generated_files")
load("//libamdf/requirements:package_policy.bzl", "apply_amdf_target_policy")
load("//loom/build_tools/amdgpu:descriptor_sets.bzl", "loom_amdgpu_selected_descriptor_set_values")
load("//loom/build_tools/amdgpu:target_config.bzl", "LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITY_BY_TARGET")
load("//loom/build_tools/bazel:defs.bzl", "loom_kernel_binary", "loom_library")
load(":cc.bzl", "amdf_cc_library")

def amdf_cts_gpu_kernel_set(
        name,
        targets,
        entry_point,
        namespace,
        srcs = [],
        library = None,
        target_srcs = None,
        data = [],
        deps = [],
        visibility = None,
        target_compatible_with = []):
    """Compiles one source module for each target and embeds its products.

    Sources are admitted once into a private relocatable library. Each physical
    target then links that same library, keeping frontend work independent of
    the number of emitted products.

    Args:
      name: Library and generated header/implementation stem.
      srcs: Loom sources admitted together into the relocatable program library.
      library: Explicit source library, exclusive with srcs, data, and deps.
      targets: Physical selectors, also naming package-local target profiles.
      entry_point: Exported kernel symbol to extract.
      namespace: C++ namespace containing the kKernels set.
      target_srcs: Optional physical-selector to direct-source mapping. When
          present, it supplies one source for every target at link time.
      data: Runtime fixtures owned by the source library.
      deps: Reusable Loom libraries linked with the source module.
      visibility: Visibility of the generated kernel library.
      target_compatible_with: Source-provider constraints inherited by the
          relocatable library and every generated target.
    """
    if target_srcs != None and sorted(target_srcs.keys()) != sorted(targets):
        fail("target_srcs must name exactly the kernel set's physical targets")
    if library:
        if srcs or data or deps:
            fail("library owns its sources, dependencies, and runtime data")
    else:
        source_name = name + "_source"
        loom_library(
            name = source_name,
            srcs = srcs,
            deps = deps,
            data = data,
            tags = ["manual"],
            target_compatible_with = target_compatible_with,
            visibility = ["//visibility:private"],
        )
        library = ":" + source_name

    policy = apply_amdf_target_policy({})
    policy["target_compatible_with"] += target_compatible_with
    products = {}
    variant_args = {}
    compatibility = {"//conditions:default": ["@platforms//:incompatible"]}
    for target in targets:
        capability = LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITY_BY_TARGET[target]
        capability_config = "//loom/config/arch/amdgpu:" + capability
        compatibility[capability_config] = []
        product_name = name + "_" + target
        product_policy = dict(policy)
        product_policy["target_compatible_with"] = policy["target_compatible_with"] + select({
            capability_config: [],
            "//conditions:default": ["@platforms//:incompatible"],
        })
        loom_kernel_binary(
            name = product_name,
            testonly = True,
            srcs = [target_srcs[target]] if target_srcs != None else [],
            deps = [library],
            out = product_name + ".hsaco",
            roots = ["@" + entry_point],
            target = ":" + target,
            **product_policy
        )
        products.setdefault(capability, []).append(":" + product_name)
        variant_args.setdefault(capability, []).extend([
            "--variant",
            target + "=$(location :" + product_name + ")",
        ])
    selected_compatibility = select(compatibility)
    policy["target_compatible_with"] += selected_compatibility
    iree_generated_files(
        name = name + "_embed",
        testonly = True,
        srcs = loom_amdgpu_selected_descriptor_set_values(products),
        outs = [name + ".h", name + ".cc"],
        args = [
            "--symbol",
            entry_point,
            "--namespace",
            namespace,
        ] + loom_amdgpu_selected_descriptor_set_values(variant_args),
        mnemonic = "AmdfEmbedGpuKernelSet",
        output_args = {
            name + ".h": "--output",
            name + ".cc": "--implementation",
        },
        progress_message = "Embedding GPU kernel variants %{label}",
        tool = "//libamdf/cts/gpu/kernels:embed",
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
