# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for reusable Loom library rules."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo")
load(
    "//loom/build_tools/bazel:defs.bzl",
    "LoomCompilationInfo",
    "LoomExecutionTestInfo",
    "LoomLibraryInfo",
    "loom_execution_profile",
)

REFERENCE_PROFILE = loom_execution_profile(
    name = "reference",
    executor = "reference",
    runner_args = ["--case=@scalar_case"],
    target_class = "cpu",
    target_family = "test",
)

SERIALIZED_REFERENCE_PROFILE = loom_execution_profile(
    name = "serialized_reference",
    executor = "reference",
    resource_group = "loom-profile-analysis-tests",
    runner_args = ["--case=@scalar_case"],
    target_class = "cpu",
    target_family = "test",
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

def _expect_no_basename(env, files, unexpected_basename):
    for file in files:
        if file.basename == unexpected_basename:
            env.fail("unexpected basename %r in %r" % (unexpected_basename, files))

def _expect_arg_with_prefix_and_suffix(env, args, prefix, suffix):
    for arg in args:
        if arg.startswith(prefix) and arg.endswith(suffix):
            return
    env.fail("expected argument with prefix %r and suffix %r in %r" % (prefix, suffix, args))

def _test_library_keeps_dependency_module_separate(name, **kwargs):
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_library_keeps_dependency_module_separate_impl,
        target = ":library_consumer",
        **kwargs
    )

def _test_library_keeps_dependency_module_separate_impl(env, target):
    info = target[LoomLibraryInfo]
    env.expect.that_str(info.module.basename).equals(
        "library_consumer.loombc",
    )
    dependencies = info.transitive_dependencies.to_list()
    if len(dependencies) != 1:
        env.fail("expected one propagated dependency, got %r" % dependencies)
    _expect_basename(env, dependencies, "library_dependency.loombc")

    action = _find_action(env, target[TestingAspectInfo].actions, "LoomLibrary")
    if "--mode=merge" not in action.argv:
        env.fail("expected merge mode in %r" % action.argv)
    _expect_arg_with_prefix_and_suffix(
        env,
        action.argv,
        "--library=",
        "library_dependency.loombc",
    )
    _expect_arg_with_prefix_and_suffix(
        env,
        action.argv,
        "",
        "link_checks.loom",
    )

    inputs = action.inputs.to_list()
    _expect_basename(env, inputs, "library_dependency.loombc")
    _expect_basename(env, inputs, "link_checks.loom")
    _expect_no_basename(env, inputs, "link_kernels.loom")

def _test_deps_only_library_propagates_dependencies(name, **kwargs):
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_deps_only_library_propagates_dependencies_impl,
        target = ":library_aggregate",
        **kwargs
    )

def _test_deps_only_library_propagates_dependencies_impl(env, target):
    info = target[LoomLibraryInfo]
    dependencies = info.transitive_dependencies.to_list()
    if len(dependencies) != 2:
        env.fail("expected two propagated dependencies, got %r" % dependencies)
    _expect_basename(env, dependencies, "library_consumer.loombc")
    _expect_basename(env, dependencies, "library_dependency.loombc")

    action = _find_action(env, target[TestingAspectInfo].actions, "LoomLibrary")
    if "--mode=merge" not in action.argv:
        env.fail("expected merge mode in %r" % action.argv)
    _expect_arg_with_prefix_and_suffix(
        env,
        action.argv,
        "--library=",
        "library_consumer.loombc",
    )

    inputs = action.inputs.to_list()
    _expect_basename(env, inputs, "library_consumer.loombc")
    _expect_no_basename(env, inputs, "link_checks.loom")
    _expect_no_basename(env, inputs, "link_kernels.loom")

def _test_wrapper_library_module_is_testonly(name, **kwargs):
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_wrapper_library_module_is_testonly_impl,
        target = ":profiled_library",
        **kwargs
    )

def _test_wrapper_library_module_is_testonly_impl(env, target):
    env.expect.that_str(target[LoomLibraryInfo].module.basename).equals(
        "profiled_library.loombc",
    )
    if not target[TestingAspectInfo].attrs.testonly:
        env.fail("expected wrapper library module to be testonly")

def _test_wrapper_compilation_is_testonly(name, **kwargs):
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_wrapper_compilation_is_testonly_impl,
        target = ":profiled_library_compile_test_compile",
        **kwargs
    )

def _test_wrapper_compilation_is_testonly_impl(env, target):
    if not target[TestingAspectInfo].attrs.testonly:
        env.fail("expected wrapper library compilation to be testonly")

    compilation = target[LoomCompilationInfo]
    env.expect.that_str(compilation.artifact.basename).equals(
        "profiled_library_compile_test_compile.testbin",
    )
    env.expect.that_str(compilation.report.basename).equals(
        "profiled_library_compile_test_compile.compile.json",
    )

    action = _find_action(env, target[TestingAspectInfo].actions, "LoomCompile")
    for expected_arg in [
        "--backend=test",
        "--target=test-target",
    ]:
        if expected_arg not in action.argv:
            env.fail("expected %r in compilation arguments %r" % (expected_arg, action.argv))

def _test_execution_profile_consumes_source_and_dependency_module(name, **kwargs):
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_execution_profile_consumes_source_and_dependency_module_impl,
        target = ":profiled_library_execute_reference_test_launcher",
        **kwargs
    )

def _test_execution_profile_consumes_source_and_dependency_module_impl(env, target):
    info = target[LoomExecutionTestInfo]
    env.expect.that_str(info.source.basename).equals("profile_cases.loom")
    if len(info.libraries) != 1:
        env.fail("expected one execution library, got %r" % info.libraries)
    env.expect.that_str(info.libraries[0].module.basename).equals(
        "library_dependency.loombc",
    )
    env.expect.that_str(info.profile_name).equals("reference")
    if info.runner_args != ["--case=@scalar_case"]:
        env.fail("unexpected runner args %r" % info.runner_args)

    runfiles = target[DefaultInfo].default_runfiles.files.to_list()
    _expect_basename(env, runfiles, "profile_cases.loom")
    _expect_basename(env, runfiles, "library_dependency.loombc")
    _expect_no_basename(env, runfiles, "profiled_library.loombc")

    tags = target[TestingAspectInfo].attrs.tags
    for expected_tag in [
        "loom-execution-profile=reference",
        "loom-target-family=test",
        "loom-target-class=cpu",
        "loom-executor=reference",
    ]:
        if expected_tag not in tags:
            env.fail("expected %r in test tags %r" % (expected_tag, tags))

def _test_resource_profile_preserves_direct_execution(name, **kwargs):
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_resource_profile_preserves_direct_execution_impl,
        target = ":profiled_library_execute_serialized_reference_test_launcher",
        **kwargs
    )

def _test_resource_profile_preserves_direct_execution_impl(env, target):
    info = target[LoomExecutionTestInfo]
    env.expect.that_str(info.source.basename).equals("profile_cases.loom")
    if len(info.libraries) != 1:
        env.fail("expected one execution library, got %r" % info.libraries)
    env.expect.that_str(info.libraries[0].module.basename).equals(
        "library_dependency.loombc",
    )
    env.expect.that_str(info.profile_name).equals("serialized_reference")

    tags = target[TestingAspectInfo].attrs.tags
    for expected_tag in [
        "loom-execution-profile=serialized_reference",
        "exclusive-if-local",
        "resource_group:loom-profile-analysis-tests",
    ]:
        if expected_tag not in tags:
            env.fail("expected %r in test tags %r" % (expected_tag, tags))

def loom_library_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_deps_only_library_propagates_dependencies,
            _test_execution_profile_consumes_source_and_dependency_module,
            _test_library_keeps_dependency_module_separate,
            _test_resource_profile_preserves_direct_execution,
            _test_wrapper_compilation_is_testonly,
            _test_wrapper_library_module_is_testonly,
        ],
    )
