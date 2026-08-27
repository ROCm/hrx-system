# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public rules for immutable Loom target profiles."""

load(
    "//loom/build_tools/amdgpu:descriptor_sets.bzl",
    "loom_amdgpu_descriptor_set_compatible_with",
)
load(
    "//loom/build_tools/amdgpu:target_config.bzl",
    "LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITY_BY_TARGET",
)

LoomTargetProfileInfo = provider(
    doc = "Family identity shared by every immutable Loom target profile.",
    fields = {
        "family": "Target fact family, such as amdgpu or spirv.",
    },
)

LoomAmdgpuTargetProfileInfo = provider(
    doc = "Structured AMDGPU identity used to construct a target profile.",
    fields = {
        "target": "Exact, generic, or overlay AMDGPU target selector.",
    },
)

def _loom_amdgpu_target_profile_impl(ctx):
    return [
        LoomTargetProfileInfo(family = "amdgpu"),
        LoomAmdgpuTargetProfileInfo(target = ctx.attr.target),
    ]

_loom_amdgpu_target_profile = rule(
    implementation = _loom_amdgpu_target_profile_impl,
    attrs = {
        "target": attr.string(
            mandatory = True,
            doc = "Exact, generic, or overlay AMDGPU target selector.",
        ),
    },
    doc = "Declares one immutable AMDGPU target identity profile.",
)

def loom_amdgpu_target_profile(name, target, **kwargs):
    """Declares an AMDGPU profile available with its descriptor contract.

    Args:
      name: Bazel target name.
      target: Exact, generic, or overlay AMDGPU target selector.
      **kwargs: Common rule attributes forwarded to the profile target.
    """
    descriptor_set_capability = (
        LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITY_BY_TARGET.get(target)
    )
    if descriptor_set_capability == None:
        fail("Unknown Loom AMDGPU target profile: %s" % target)
    target_compatible_with = kwargs.pop("target_compatible_with", [])
    _loom_amdgpu_target_profile(
        name = name,
        target = target,
        target_compatible_with = target_compatible_with +
                                 loom_amdgpu_descriptor_set_compatible_with(
                                     descriptor_set_capability,
                                 ),
        **kwargs
    )
