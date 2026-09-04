# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU target-profile declarations and compatibility policy."""

load(
    "//loom/build_tools/amdgpu:descriptor_sets.bzl",
    "loom_amdgpu_descriptor_set_compatible_with",
)
load(
    "//loom/build_tools/amdgpu:target_config.bzl",
    "LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITY_BY_TARGET",
)
load("//loom/build_tools/bazel:loom_target_profile.bzl", "loom_target_profile")

def loom_amdgpu_target_profile(
        name,
        target,
        canonical_formats = [
            "//loom/product/formats:amdgpu_hsaco",
            "//loom/product/formats:loom_command",
        ],
        formats = [],
        **kwargs):
    """Declares an AMDGPU profile available with its descriptor contract.

    Args:
      name: Bazel target name.
      target: Exact, generic, or overlay AMDGPU target selector.
      canonical_formats: Canonical product formats for the profile.
      formats: Additional noncanonical product formats for the profile.
      **kwargs: Common rule attributes forwarded to the profile target.
    """
    descriptor_set_capability = (
        LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITY_BY_TARGET.get(target)
    )
    if descriptor_set_capability == None:
        fail("Unknown Loom AMDGPU target profile: %s" % target)
    target_compatible_with = kwargs.pop("target_compatible_with", [])
    loom_target_profile(
        name = name,
        canonical_formats = canonical_formats,
        family = "amdgpu",
        formats = formats,
        selector = target,
        target_compatible_with = target_compatible_with +
                                 loom_amdgpu_descriptor_set_compatible_with(
                                     descriptor_set_capability,
                                 ),
        **kwargs
    )
