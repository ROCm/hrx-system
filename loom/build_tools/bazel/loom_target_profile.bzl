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
    doc = "Immutable Loom target identity shared by all target families.",
    fields = {
        "family": "Target fact family, such as amdgpu or spirv.",
        "selector": "Family-owned exact, generic, or overlay target selector.",
    },
)

def _loom_target_profile_impl(ctx):
    return [
        LoomTargetProfileInfo(
            family = ctx.attr.family,
            selector = ctx.attr.selector,
        ),
    ]

_loom_target_profile = rule(
    implementation = _loom_target_profile_impl,
    attrs = {
        "family": attr.string(
            mandatory = True,
            doc = "Target fact family accepted by the compiler.",
        ),
        "selector": attr.string(
            mandatory = True,
            doc = "Family-owned exact, generic, or overlay selector.",
        ),
    },
    doc = "Declares one immutable Loom target identity profile.",
)

def loom_target_profile(name, family, selector, **kwargs):
    """Declares an immutable target profile understood by the compiler.

    Args:
      name: Bazel target name.
      family: Target fact family accepted by the compiler.
      selector: Family-owned exact, generic, or overlay selector.
      **kwargs: Common rule attributes forwarded to the profile target.
    """
    if not family:
        fail("Loom target profile family must not be empty")
    if ":" in family:
        fail("Loom target profile family must not contain ':'")
    if not selector:
        fail("Loom target profile selector must not be empty")
    _loom_target_profile(
        name = name,
        family = family,
        selector = selector,
        **kwargs
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
    loom_target_profile(
        name = name,
        family = "amdgpu",
        selector = target,
        target_compatible_with = target_compatible_with +
                                 loom_amdgpu_descriptor_set_compatible_with(
                                     descriptor_set_capability,
                                 ),
        **kwargs
    )
