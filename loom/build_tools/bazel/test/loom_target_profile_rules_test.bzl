# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for immutable Loom target profiles."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:truth.bzl", "matching")
load(
    "//loom/build_tools/bazel:defs.bzl",
    "LoomProductFormatInfo",
    "LoomTargetFormatSupportInfo",
    "LoomTargetProfileInfo",
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

    format_support = target[LoomTargetFormatSupportInfo]
    env.expect.that_str(
        format_support.canonical_formats["kernel"].label.name,
    ).equals("amdgpu_hsaco")
    env.expect.that_str(
        format_support.canonical_formats["command"].label.name,
    ).equals("loom_command")
    if len(format_support.formats) != 2:
        env.fail("expected two AMDGPU product formats, got %r" % format_support.formats)

    if hasattr(profile, "backend"):
        env.fail("target profile must not select a compiler backend")
    if hasattr(profile, "artifact_extension"):
        env.fail("target profile must not select an artifact extension")

def _test_builtin_profile_preserves_overlay_identity(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_builtin_profile_preserves_overlay_identity_impl,
        target = "//loom/target/amdgpu:gfx1250-a0",
        **kwargs
    )

def _test_builtin_profile_preserves_overlay_identity_impl(env, target):
    profile = target[LoomTargetProfileInfo]
    env.expect.that_str(profile.family).equals("amdgpu")
    env.expect.that_str(profile.selector).equals("gfx1250-a0")

def _test_generic_profile_preserves_family_identity(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_generic_profile_preserves_family_identity_impl,
        target = ":test_target_profile",
        **kwargs
    )

def _test_generic_profile_preserves_family_identity_impl(env, target):
    profile = target[LoomTargetProfileInfo]
    env.expect.that_str(profile.family).equals("test-family")
    env.expect.that_str(profile.selector).equals("test-selector")
    support = target[LoomTargetFormatSupportInfo]
    env.expect.that_str(
        support.canonical_formats["kernel"].label.name,
    ).equals("test_kernel_format")
    env.expect.that_str(
        support.canonical_formats["command"].label.name,
    ).equals("test_command_format")
    if len(support.formats) != 3:
        env.fail("expected three synthetic formats, got %r" % support.formats)

def _test_builtin_spirv_profile_owns_canonical_format(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_builtin_spirv_profile_owns_canonical_format_impl,
        target = "//loom/target/spirv:vulkan1.3+bda",
        **kwargs
    )

def _test_builtin_spirv_profile_owns_canonical_format_impl(env, target):
    profile = target[LoomTargetProfileInfo]
    env.expect.that_str(profile.family).equals("spirv")
    env.expect.that_str(profile.selector).equals("vulkan1.3+bda")
    support = target[LoomTargetFormatSupportInfo]
    format_target = support.canonical_formats["kernel"]
    env.expect.that_str(format_target.label.name).equals("spirv_binary")
    format_info = format_target[LoomProductFormatInfo]
    env.expect.that_str(format_info.format).equals("spirv-binary")
    env.expect.that_str(format_info.output_extension).equals(".spv")
    env.expect.that_str(
        support.canonical_formats["command"].label.name,
    ).equals("loom_command")

def _test_duplicate_canonical_product_fails(name, **kwargs):
    analysis_test(
        name = name,
        expect_failure = True,
        impl = _test_duplicate_canonical_product_fails_impl,
        target = ":test_duplicate_canonical_profile",
        **kwargs
    )

def _test_duplicate_canonical_product_fails_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(
        matching.contains("declares multiple canonical formats for product \"kernel\""),
    )

def loom_target_profile_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_amdgpu_profile_is_family_typed,
            _test_builtin_profile_preserves_overlay_identity,
            _test_builtin_spirv_profile_owns_canonical_format,
            _test_duplicate_canonical_product_fails,
            _test_generic_profile_preserves_family_identity,
        ],
    )
