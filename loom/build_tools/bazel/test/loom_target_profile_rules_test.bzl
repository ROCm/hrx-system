# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for immutable Loom target profiles."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load(
    "//loom/build_tools/bazel:defs.bzl",
    "LoomTargetProfileInfo",
    "LoomTargetSetInfo",
)

def _test_amdgpu_profile_is_family_typed(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_amdgpu_profile_is_family_typed_impl,
        target = ":test_amdgpu_profile",
        **kwargs
    )

def _test_amdgpu_profile_is_family_typed_impl(env, target):
    profile = target[LoomTargetProfileInfo]
    env.expect.that_str(profile.family).equals("amdgpu")
    env.expect.that_str(profile.selector).equals("gfx942")

    if hasattr(profile, "backend"):
        env.fail("target profile must not select a compiler backend")
    if hasattr(profile, "artifact_extension"):
        env.fail("target profile must not select an artifact extension")

def _test_builtin_profile_preserves_overlay_identity(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_builtin_profile_preserves_overlay_identity_impl,
        target = "//loom/target/amdgpu:gfx1250-strict",
        **kwargs
    )

def _test_builtin_profile_preserves_overlay_identity_impl(env, target):
    profile = target[LoomTargetProfileInfo]
    env.expect.that_str(profile.family).equals("amdgpu")
    env.expect.that_str(profile.selector).equals("gfx1250-strict")

def _test_amdgpu_family_set_selects_compatible_profiles(name, **kwargs):
    analysis_test(
        name = name,
        config_settings = {
            str(Label("//loom/config/target/amdgpu:targets")): ["gfx1201"],
            str(Label("//loom/config/target:enable")): ["amdgpu"],
        },
        impl = _test_amdgpu_family_set_selects_compatible_profiles_impl,
        target = "//loom/target/amdgpu:all",
        **kwargs
    )

def _test_amdgpu_family_set_selects_compatible_profiles_impl(env, target):
    profiles = target[LoomTargetSetInfo].profiles
    env.expect.that_collection([
        profile[LoomTargetProfileInfo].selector
        for profile in profiles
    ]).contains_exactly([
        "gfx1200",
        "gfx1201",
    ])

def _test_generic_profile_preserves_family_identity(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_generic_profile_preserves_family_identity_impl,
        target = ":test_fake_profile",
        **kwargs
    )

def _test_generic_profile_preserves_family_identity_impl(env, target):
    profile = target[LoomTargetProfileInfo]
    env.expect.that_str(profile.family).equals("FakeTargetFamily123")
    env.expect.that_str(profile.selector).equals("FakeTargetSelector123")

def _test_target_set_flattens_profiles_in_order(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_target_set_flattens_profiles_in_order_impl,
        target = ":test_nested_target_set",
        **kwargs
    )

def _test_target_set_flattens_profiles_in_order_impl(env, target):
    profiles = target[LoomTargetSetInfo].profiles
    if len(profiles) != 2:
        env.fail("expected two deduplicated profiles, got %r" % profiles)
        return
    env.expect.that_str(profiles[0].label.name).equals("test_fake_profile")
    env.expect.that_str(profiles[1].label.name).equals("test_amdgpu_profile")
    env.expect.that_str(profiles[0][LoomTargetProfileInfo].family).equals(
        "FakeTargetFamily123",
    )
    env.expect.that_str(profiles[1][LoomTargetProfileInfo].family).equals(
        "amdgpu",
    )

def _test_target_set_may_be_configuration_empty(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_target_set_may_be_configuration_empty_impl,
        target = ":test_empty_target_set",
        **kwargs
    )

def _test_target_set_may_be_configuration_empty_impl(env, target):
    env.expect.that_collection(target[LoomTargetSetInfo].profiles).contains_exactly([])

def loom_target_profile_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_amdgpu_family_set_selects_compatible_profiles,
            _test_amdgpu_profile_is_family_typed,
            _test_builtin_profile_preserves_overlay_identity,
            _test_generic_profile_preserves_family_identity,
            _test_target_set_may_be_configuration_empty,
            _test_target_set_flattens_profiles_in_order,
        ],
    )
