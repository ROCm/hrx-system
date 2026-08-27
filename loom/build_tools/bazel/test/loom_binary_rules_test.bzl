# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for final Loom deployment product rules."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:truth.bzl", "matching")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo", "util")
load(
    "//loom/build_tools/bazel:defs.bzl",
    "LoomBinaryInfo",
    "LoomTargetProfileInfo",
    "loom_command_binary",
    "loom_kernel_binary",
)

def _test_target_profile_impl(ctx):
    return [LoomTargetProfileInfo(family = ctx.attr.family)]

test_target_profile = rule(
    implementation = _test_target_profile_impl,
    attrs = {
        "family": attr.string(mandatory = True),
    },
)

def _find_action(env, actions, mnemonic):
    for action in actions:
        if action.mnemonic == mnemonic:
            return action
    env.fail("expected action with mnemonic %r in %r" % (mnemonic, actions))
    return None

def _expect_basename(env, files, expected_basename):
    for file in files:
        if file.basename == expected_basename:
            return
    env.fail("expected basename %r in %r" % (expected_basename, files))

def _expect_arg_with_suffix(env, args, prefix, suffix):
    for arg in args:
        if arg.startswith(prefix) and arg.endswith(suffix):
            return
    env.fail(
        "expected argument with prefix %r and suffix %r in %r" %
        (prefix, suffix, args),
    )

def _expect_no_arg_with_suffix(env, args, prefix, suffix):
    for arg in args:
        if arg.startswith(prefix) and arg.endswith(suffix):
            env.fail(
                "unexpected argument with prefix %r and suffix %r in %r" %
                (prefix, suffix, args),
            )

def _test_kernel_binary_roots_direct_library_exports(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_kernel_binary_roots_direct_library_exports_impl,
        target = ":kernel_binary_deps",
        **kwargs
    )

def _test_kernel_binary_roots_direct_library_exports_impl(env, target):
    binary = target[LoomBinaryInfo]
    env.expect.that_str(binary.kind).equals("kernel")
    env.expect.that_str(binary.primary_artifact.basename).equals(
        "kernel_binary_deps.hsaco",
    )
    env.expect.that_str(binary.linked_module.basename).equals(
        "kernel_binary_deps.linked.loombc",
    )
    if len(binary.target_profiles) != 1:
        env.fail("expected one kernel target profile, got %r" % binary.target_profiles)
    env.expect.that_str(
        binary.target_profiles[0][LoomTargetProfileInfo].family,
    ).equals("amdgpu")

    default_files = target[DefaultInfo].files.to_list()
    if len(default_files) != 1:
        env.fail("expected one default kernel artifact, got %r" % default_files)
    _expect_basename(env, default_files, "kernel_binary_deps.hsaco")

    actions = target[TestingAspectInfo].actions
    link_action = _find_action(env, actions, "LoomBinaryLink")
    for expected_arg in [
        "--mode=link",
        "--strip-check",
        "--require-resolved-config",
        "--to=bc",
    ]:
        if expected_arg not in link_action.argv:
            env.fail("expected %r in link arguments %r" % (expected_arg, link_action.argv))
    _expect_arg_with_suffix(
        env,
        link_action.argv,
        "--root-library=",
        "public_kernel_library.loombc",
    )

    compile_action = _find_action(env, actions, "LoomKernelBinary")
    for expected_arg in [
        "--backend=amdgpu-hal",
        "--target=gfx11-generic",
        "--compile-report=details",
    ]:
        if expected_arg not in compile_action.argv:
            env.fail(
                "expected %r in compile arguments %r" %
                (expected_arg, compile_action.argv),
            )

    _expect_basename(
        env,
        target[OutputGroupInfo].compile_reports.to_list(),
        "kernel_binary_deps.compile.json",
    )
    _expect_basename(
        env,
        target[OutputGroupInfo].linked_modules.to_list(),
        "kernel_binary_deps.linked.loombc",
    )

def _test_kernel_binary_sources_are_an_implicit_library(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_kernel_binary_sources_are_an_implicit_library_impl,
        target = ":kernel_binary_mixed",
        **kwargs
    )

def _test_kernel_binary_sources_are_an_implicit_library_impl(env, target):
    actions = target[TestingAspectInfo].actions
    source_action = _find_action(env, actions, "LoomBinarySources")
    _expect_arg_with_suffix(
        env,
        source_action.argv,
        "",
        "link_checks.loom",
    )
    _expect_arg_with_suffix(
        env,
        source_action.argv,
        "--library=",
        "library_dependency.loombc",
    )

    link_action = _find_action(env, actions, "LoomBinaryLink")
    _expect_arg_with_suffix(
        env,
        link_action.argv,
        "--root-library=",
        "library_dependency.loombc",
    )
    _expect_arg_with_suffix(
        env,
        link_action.argv,
        "--root-library=",
        "kernel_binary_mixed.sources.loombc",
    )
    first_config_index = link_action.argv.index("--config=a.value=1")
    second_config_index = link_action.argv.index("--config=z.value=2")
    if first_config_index >= second_config_index:
        env.fail("expected config arguments in key order: %r" % link_action.argv)

    _expect_basename(
        env,
        target[OutputGroupInfo].dependency_reports.to_list(),
        "kernel_binary_mixed.sources.dependencies.json",
    )

def _test_explicit_roots_replace_direct_exports(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_explicit_roots_replace_direct_exports_impl,
        target = ":kernel_binary_explicit_root",
        **kwargs
    )

def _test_explicit_roots_replace_direct_exports_impl(env, target):
    link_action = _find_action(
        env,
        target[TestingAspectInfo].actions,
        "LoomBinaryLink",
    )
    if "--root=@scale" not in link_action.argv:
        env.fail("expected explicit root in %r" % link_action.argv)
    _expect_arg_with_suffix(
        env,
        link_action.argv,
        "--library=",
        "public_kernel_library.loombc",
    )
    _expect_no_arg_with_suffix(
        env,
        link_action.argv,
        "--root-library=",
        "public_kernel_library.loombc",
    )

def _test_kernel_binary_rejects_incompatible_profile_family(name, **kwargs):
    subject = name + "_subject"
    util.helper_target(
        loom_kernel_binary,
        name = subject,
        deps = [":public_kernel_library"],
        tags = ["manual"],
        target = ":test_spirv_profile",
    )
    analysis_test(
        name = name,
        expect_failure = True,
        impl = _test_kernel_binary_rejects_incompatible_profile_family_impl,
        target = subject,
        **kwargs
    )

def _test_kernel_binary_rejects_incompatible_profile_family_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(
        matching.contains(
            "cannot emit a kernel binary for target profile family \"spirv\"",
        ),
    )

def _test_command_binary_emits_composite_product(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_command_binary_emits_composite_product_impl,
        target = "//loom/build_tools/bazel/test/testdata/command_binary:deps_subject",
        **kwargs
    )

def _test_command_binary_emits_composite_product_impl(env, target):
    binary = target[LoomBinaryInfo]
    env.expect.that_str(binary.kind).equals("command")
    env.expect.that_str(binary.primary_artifact.basename).equals(
        "deps_subject.commands.json",
    )
    env.expect.that_str(binary.linked_module.basename).equals(
        "deps_subject.linked.loombc",
    )
    if len(binary.target_profiles) != 1:
        env.fail("expected one command target profile, got %r" % binary.target_profiles)
    env.expect.that_str(
        binary.target_profiles[0][LoomTargetProfileInfo].family,
    ).equals("amdgpu")

    artifacts = binary.artifacts.to_list()
    if len(artifacts) != 3:
        env.fail("expected three command product artifacts, got %r" % artifacts)
    for basename in [
        "deps_subject.commands",
        "deps_subject.commands.json",
        "deps_subject.kernels.hsaco",
    ]:
        _expect_basename(env, artifacts, basename)

    actions = target[TestingAspectInfo].actions
    link_action = _find_action(env, actions, "LoomBinaryLink")
    _expect_arg_with_suffix(
        env,
        link_action.argv,
        "--root-library=",
        "interface.loombc",
    )
    _expect_arg_with_suffix(
        env,
        link_action.argv,
        "--transitive-library=",
        "implementation.loombc",
    )
    _expect_arg_with_suffix(
        env,
        link_action.argv,
        "--transitive-library=",
        "unused.loombc",
    )

    command_action = _find_action(env, actions, "LoomCommandBinary")
    for expected_arg in [
        "--backend=command",
        "--compile-report=details",
    ]:
        if expected_arg not in command_action.argv:
            env.fail(
                "expected %r in command arguments %r" %
                (expected_arg, command_action.argv),
            )
    _expect_arg_with_suffix(
        env,
        command_action.argv,
        "--output=",
        "deps_subject.commands.json",
    )
    _expect_arg_with_suffix(
        env,
        command_action.argv,
        "--emit-command-artifacts=",
        "deps_subject.commands",
    )

    kernel_action = _find_action(env, actions, "LoomCommandKernelBinary")
    for expected_arg in [
        "--backend=amdgpu-hal",
        "--target=gfx11-generic",
        "--compile-report=details",
    ]:
        if expected_arg not in kernel_action.argv:
            env.fail(
                "expected %r in kernel arguments %r" %
                (expected_arg, kernel_action.argv),
            )

    for basename in [
        "deps_subject.commands.compile.json",
        "deps_subject.kernels.compile.json",
    ]:
        _expect_basename(
            env,
            target[OutputGroupInfo].compile_reports.to_list(),
            basename,
        )

def _test_command_binary_sources_are_an_implicit_library(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_command_binary_sources_are_an_implicit_library_impl,
        target = "//loom/build_tools/bazel/test/testdata/command_binary:mixed_subject",
        **kwargs
    )

def _test_command_binary_sources_are_an_implicit_library_impl(env, target):
    actions = target[TestingAspectInfo].actions
    source_action = _find_action(env, actions, "LoomBinarySources")
    _expect_arg_with_suffix(env, source_action.argv, "", "interface.loom")
    _expect_arg_with_suffix(
        env,
        source_action.argv,
        "--library=",
        "implementation.loombc",
    )

    link_action = _find_action(env, actions, "LoomBinaryLink")
    for basename in [
        "implementation.loombc",
        "mixed_subject.sources.loombc",
        "unused.loombc",
    ]:
        _expect_arg_with_suffix(
            env,
            link_action.argv,
            "--root-library=",
            basename,
        )

def _test_command_binary_explicit_roots_replace_exports(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_command_binary_explicit_roots_replace_exports_impl,
        target = "//loom/build_tools/bazel/test/testdata/command_binary:explicit_subject",
        **kwargs
    )

def _test_command_binary_explicit_roots_replace_exports_impl(env, target):
    link_action = _find_action(
        env,
        target[TestingAspectInfo].actions,
        "LoomBinaryLink",
    )
    if "--root=@scale_once" not in link_action.argv:
        env.fail("expected explicit command root in %r" % link_action.argv)
    _expect_arg_with_suffix(
        env,
        link_action.argv,
        "--library=",
        "interface.loombc",
    )
    _expect_no_arg_with_suffix(
        env,
        link_action.argv,
        "--root-library=",
        "interface.loombc",
    )

def _test_command_binary_rejects_incompatible_profile_family(name, **kwargs):
    subject = name + "_subject"
    util.helper_target(
        loom_command_binary,
        name = subject,
        deps = [":public_kernel_library"],
        tags = ["manual"],
        target = ":test_spirv_profile",
    )
    analysis_test(
        name = name,
        expect_failure = True,
        impl = _test_command_binary_rejects_incompatible_profile_family_impl,
        target = subject,
        **kwargs
    )

def _test_command_binary_rejects_incompatible_profile_family_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(
        matching.contains(
            "cannot emit a command binary for target profile family \"spirv\"",
        ),
    )

def loom_binary_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_command_binary_emits_composite_product,
            _test_command_binary_explicit_roots_replace_exports,
            _test_command_binary_rejects_incompatible_profile_family,
            _test_command_binary_sources_are_an_implicit_library,
            _test_explicit_roots_replace_direct_exports,
            _test_kernel_binary_rejects_incompatible_profile_family,
            _test_kernel_binary_roots_direct_library_exports,
            _test_kernel_binary_sources_are_an_implicit_library,
        ],
    )
