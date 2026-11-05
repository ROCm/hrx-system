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
load("//loom/build_tools/bazel:build_defs.bzl", "loom_config_compatible_with")

LoomTargetProfileInfo = provider(
    doc = "Immutable Loom target identity shared by all target families.",
    fields = {
        "family": "Target fact family, such as amdgpu or spirv.",
        "selector": "Family-owned exact or generic target selector.",
    },
)

LoomTargetSetInfo = provider(
    doc = "Ordered collection of immutable Loom target profiles.",
    fields = {
        "profiles": "Ordered, deduplicated list of target-profile targets.",
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
            doc = "Family-owned exact or generic selector.",
        ),
    },
    doc = "Declares one immutable Loom target identity profile.",
)

def loom_target_profile(name, family, selector, **kwargs):
    """Declares an immutable target profile understood by the compiler.

    Args:
      name: Bazel target name.
      family: Target fact family accepted by the compiler.
      selector: Family-owned exact or generic selector.
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

def _loom_target_set_impl(ctx):
    profiles = []
    seen_labels = {}
    for target in ctx.attr.targets:
        target_profiles = (
            [target] if LoomTargetProfileInfo in target else target[LoomTargetSetInfo].profiles
        )
        for profile in target_profiles:
            label = str(profile.label)
            if label not in seen_labels:
                seen_labels[label] = None
                profiles.append(profile)
    if not profiles and not ctx.attr.allow_empty:
        fail("%s must contain at least one target profile" % ctx.label)
    return [LoomTargetSetInfo(profiles = profiles)]

_loom_target_set = rule(
    implementation = _loom_target_set_impl,
    attrs = {
        "allow_empty": attr.bool(
            doc = "Whether configuration may select no target profiles.",
        ),
        "targets": attr.label_list(
            mandatory = True,
            providers = [
                [LoomTargetProfileInfo],
                [LoomTargetSetInfo],
            ],
            doc = "Target profiles or target sets collected in declaration order.",
        ),
    },
    doc = "Collects immutable target profiles for exhaustive build fanout.",
)

def loom_target_set(name, targets, allow_empty = False, **kwargs):
    """Declares an ordered set of target profiles.

    Nested target sets are flattened and duplicate profiles retain their first
    position. Adding a profile to a central set therefore extends every corpus
    that consumes the set without changing those corpus packages.

    Args:
      name: Bazel target name.
      targets: Target profile or target set labels.
      allow_empty: Whether configuration may select no target profiles.
      **kwargs: Common rule attributes forwarded to the target-set rule.
    """
    if not targets and not allow_empty:
        fail("Loom target set must contain at least one target")
    _loom_target_set(
        name = name,
        allow_empty = allow_empty,
        targets = targets,
        **kwargs
    )

def loom_amdgpu_target_profile(name, target, **kwargs):
    """Declares an AMDGPU profile requiring its architecture and descriptors.

    Args:
      name: Bazel target name.
      target: Exact or generic AMDGPU target selector.
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
                                 loom_config_compatible_with([
                                     "//loom/config/arch:amdgpu",
                                 ]) +
                                 loom_amdgpu_descriptor_set_compatible_with(
                                     descriptor_set_capability,
                                 ),
        **kwargs
    )
