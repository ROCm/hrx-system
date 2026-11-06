# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Configured Loom executable corpora for native AMD conformance tests."""

load("//loom/build_tools/amdgpu:descriptor_sets.bzl", "loom_amdgpu_selected_descriptor_set_values")
load("//loom/build_tools/amdgpu:target_config.bzl", "LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITY_BY_TARGET", "LOOM_AMDGPU_SUPPORTED_TARGETS")
load("//loom/build_tools/bazel:build_defs.bzl", "loom_config_compatible_with")
load("//loom/build_tools/bazel:defs.bzl", "loom_kernel_binary", "loom_library", "loom_target_profile")
load("//runtime/build_tools/bazel:cc.bzl", "iree_runtime_cc_library")
load("//runtime/build_tools/bazel:hal_cts.bzl", "iree_runtime_hal_cts_testdata")

def iree_hal_amd_gpu_cts_testdata(name, srcs, backend_names, target_compatible_with = []):
    """Compiles and registers a corpus for every configured descriptor target.

    Source basenames are logical program names. Each source is admitted once;
    products reuse package-local profiles owned by this macro across corpora.

    Args:
      name: Aggregate testdata library.
      srcs: Loom sources, each defining one logical executable.
      backend_names: CTS backend views receiving the executable registrations.
      target_compatible_with: Additional constraints on the entire corpus.
    """
    compatibility = target_compatible_with + loom_config_compatible_with([
        "//loom/config/emit:amdgpu_elf",
        "//runtime/config/hal:driver_amd_gpu",
    ])
    programs = {}
    for src in srcs:
        program = src.split(":")[-1].split("/")[-1].removesuffix(".loom")
        if program in programs:
            fail("CTS sources must have distinct basenames: " + program)
        programs[program] = name + "_source_" + program
        loom_library(
            name = programs[program],
            srcs = [src],
            tags = ["manual"],
            target_compatible_with = compatibility,
        )

    registrations = {}
    for target in LOOM_AMDGPU_SUPPORTED_TARGETS:
        capability = LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITY_BY_TARGET[target]
        profile = "cts_profile_" + target
        profile_compatibility = loom_config_compatible_with([
            "//loom/config/emit:amdgpu_elf",
            "//loom/config/arch/amdgpu:" + capability,
            "//runtime/config/hal:driver_amd_gpu",
        ])
        if not native.existing_rule(profile):
            loom_target_profile(
                name = profile,
                family = "amdgpu",
                selector = target,
                target_compatible_with = profile_compatibility,
            )
        product_compatibility = target_compatible_with + profile_compatibility
        products = []
        for program, source in programs.items():
            product = name + "_" + target + "_" + program
            loom_kernel_binary(
                name = product,
                testonly = True,
                out = name + "_" + target + "/" + program + ".bin",
                target = ":" + profile,
                target_compatible_with = product_compatibility,
                deps = [":" + source],
            )
            products.append(":" + product)
        registration = name + "_" + target
        iree_runtime_hal_cts_testdata(
            name = registration,
            srcs = products,
            backend_names = backend_names,
            target_compatible_with = product_compatibility,
            target_family = "amdgpu",
            target_key = target,
            target_name = target.replace("-", "_"),
        )
        registrations.setdefault(capability, []).append(":" + registration)

    iree_runtime_cc_library(
        name = name,
        testonly = True,
        target_compatible_with = compatibility + select(dict(
            [("//loom/config/arch/amdgpu:" + capability, []) for capability in registrations] +
            [("//conditions:default", ["@platforms//:incompatible"])],
        )),
        deps = loom_amdgpu_selected_descriptor_set_values(registrations),
        alwayslink = True,
    )
