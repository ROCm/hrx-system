# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for reusable Loom library rules."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo")
load("//build_tools/bazel:executable.bzl", "IreeExecutableInfo")
load("//build_tools/bazel:runfiles.bzl", "IreeRunfilesArgumentsInfo")
load(
    "//loom/build_tools/bazel:defs.bzl",
    "LoomBinaryInfo",
    "LoomExecutionTestInfo",
    "LoomLibraryInfo",
    "loom_execution_profile",
    "loom_test_module",
)

REFERENCE_PROFILE = loom_execution_profile(
    name = "reference",
    executor = "reference",
    runner_args = ["--max-samples-per-case=1"],
    target_class = "cpu",
    target_family = "test",
)

SERIALIZED_REFERENCE_PROFILE = loom_execution_profile(
    name = "serialized_reference",
    executor = "reference",
    resource_group = "loom-profile-analysis-tests",
    runner_args = ["--max-samples-per-case=1"],
    target_class = "cpu",
    target_family = "test",
)

EXPLICIT_RUNNER_PROFILE = loom_execution_profile(
    name = "explicit_runner",
    executor = "hosted",
    runner = "//loom/src/loom/tools/loom-test",
    runner_args = ["--max-samples-per-case=1"],
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

def _expect_no_arg_with_prefix_and_suffix(env, args, prefix, suffix):
    for arg in args:
        if arg.startswith(prefix) and arg.endswith(suffix):
            env.fail("unexpected argument with prefix %r and suffix %r in %r" % (prefix, suffix, args))

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
    for expected_arg in [
        "--mode=merge",
        "--strict-deps",
    ]:
        if expected_arg not in action.argv:
            env.fail("expected %r in %r" % (expected_arg, action.argv))
    _expect_arg_with_prefix_and_suffix(
        env,
        action.argv,
        "--dependency-component=",
        "//loom/build_tools/bazel/test:library_consumer",
    )
    _expect_arg_with_prefix_and_suffix(
        env,
        action.argv,
        "--dependency-report=",
        "library_consumer.dependencies.json",
    )
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

    reports = target[OutputGroupInfo].dependency_reports.to_list()
    if len(reports) != 1:
        env.fail("expected one dependency report, got %r" % reports)
    _expect_basename(env, reports, "library_consumer.dependencies.json")

def _test_deps_only_library_propagates_dependencies(name, **kwargs):
    if native.existing_rule("library_aggregate_test"):
        fail("deps-only libraries must not create an implicit all-package test suite")
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
    _expect_arg_with_prefix_and_suffix(
        env,
        action.argv,
        "--transitive-library=",
        "library_dependency.loombc",
    )

    inputs = action.inputs.to_list()
    _expect_basename(env, inputs, "library_consumer.loombc")
    _expect_basename(env, inputs, "library_dependency.loombc")
    _expect_no_basename(env, inputs, "link_checks.loom")
    _expect_no_basename(env, inputs, "link_kernels.loom")

def _test_redundant_direct_dependency_is_not_transitive(name, **kwargs):
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_redundant_direct_dependency_is_not_transitive_impl,
        target = ":library_redundant_direct",
        **kwargs
    )

def _test_redundant_direct_dependency_is_not_transitive_impl(env, target):
    action = _find_action(env, target[TestingAspectInfo].actions, "LoomLibrary")
    for basename in [
        "library_consumer.loombc",
        "library_dependency.loombc",
    ]:
        _expect_arg_with_prefix_and_suffix(
            env,
            action.argv,
            "--library=",
            basename,
        )
        _expect_no_arg_with_prefix_and_suffix(
            env,
            action.argv,
            "--transitive-library=",
            basename,
        )

def _test_transitive_audit_universe_is_separate(name, **kwargs):
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_transitive_audit_universe_is_separate_impl,
        target = ":library_transitive_consumer",
        **kwargs
    )

def _test_transitive_audit_universe_is_separate_impl(env, target):
    action = _find_action(env, target[TestingAspectInfo].actions, "LoomLibrary")
    _expect_arg_with_prefix_and_suffix(
        env,
        action.argv,
        "--library=",
        "library_aggregate.loombc",
    )
    for basename in [
        "library_consumer.loombc",
        "library_dependency.loombc",
    ]:
        _expect_arg_with_prefix_and_suffix(
            env,
            action.argv,
            "--transitive-library=",
            basename,
        )
        _expect_no_arg_with_prefix_and_suffix(
            env,
            action.argv,
            "--library=",
            basename,
        )

def _test_generated_kernel_binary_is_testonly(name, **kwargs):
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_generated_kernel_binary_is_testonly_impl,
        target = ":public_kernel_library_kernel_binary_gfx11_generic",
        **kwargs
    )

def _test_generated_kernel_binary_is_testonly_impl(env, target):
    if not target[TestingAspectInfo].attrs.testonly:
        env.fail("expected generated kernel binary to be testonly")

    binary = target[LoomBinaryInfo]
    env.expect.that_str(binary.primary_artifact.basename).equals(
        "public_kernel_library_kernel_binary_gfx11_generic",
    )
    _expect_basename(
        env,
        binary.reports.to_list(),
        "public_kernel_library_kernel_binary_gfx11_generic.compile.json",
    )

    action = _find_action(env, target[TestingAspectInfo].actions, "LoomKernelBinary")
    for expected_arg in [
        "--target=amdgpu:gfx11-generic",
    ]:
        if expected_arg not in action.argv:
            env.fail("expected %r in kernel binary arguments %r" % (expected_arg, action.argv))
    if any([arg.startswith("--product=") for arg in action.argv]):
        env.fail("kernel binary compilation must infer its entry category: %r" % action.argv)
    for arg in action.argv:
        if arg.startswith("--format="):
            env.fail("unexpected exact format in kernel binary arguments %r" % action.argv)

def _test_execution_profile_contract(name, **kwargs):
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_execution_profile_contract_impl,
        target = ":profiled_test_selected_execute_reference_test_launcher",
        **kwargs
    )

def _test_execution_profile_contract_impl(env, target):
    info = target[LoomExecutionTestInfo]
    env.expect.that_str(info.module.basename).equals(
        "profiled_module.loombc",
    )
    env.expect.that_str(info.profile_name).equals("reference")
    if not info.test_runner.basename.startswith("loom-test"):
        env.fail("unexpected correctness runner %r" % info.test_runner)
    if info.test_runner_args != [
        "--max-samples-per-case=1",
        "--case=benchmark_case",
        "--sample=0",
    ]:
        env.fail("unexpected correctness runner args %r" % info.test_runner_args)
    if not info.benchmark_runner.basename.startswith("loom-benchmark"):
        env.fail("unexpected benchmark runner %r" % info.benchmark_runner)
    if info.benchmark_runner_args != [
        "--iterations=1",
        "--warmup-iterations=0",
        "--min-time-ms=0",
        "--warmup-time-ms=0",
        "--max-batches=1",
        "--output-format=jsonl",
        "--compile-report=none",
        "--max-samples-per-case=1",
        "--case=benchmark_case",
    ]:
        env.fail("unexpected benchmark runner args %r" % info.benchmark_runner_args)

    runfiles = target[DefaultInfo].default_runfiles.files.to_list()
    _expect_basename(env, runfiles, "profiled_module.loombc")
    _expect_basename(env, runfiles, info.test_runner.basename)
    _expect_basename(env, runfiles, info.benchmark_runner.basename)
    _expect_no_basename(env, runfiles, "profile_cases.loom")
    _expect_no_basename(env, runfiles, "profile_benchmarks.loom")
    _expect_no_basename(env, runfiles, "library_dependency.loombc")

    tags = target[TestingAspectInfo].attrs.tags
    for expected_tag in [
        "loom-execution-profile=reference",
        "loom-target-family=test",
        "loom-target-class=cpu",
        "loom-executor=reference",
    ]:
        if expected_tag not in tags:
            env.fail("expected %r in test tags %r" % (expected_tag, tags))

def _test_correctness_only_profile_omits_benchmark_runner(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_correctness_only_profile_omits_benchmark_runner_impl,
        target = ":correctness_only_test_execute_reference_test_launcher",
        **kwargs
    )

def _test_correctness_only_profile_omits_benchmark_runner_impl(env, target):
    info = target[LoomExecutionTestInfo]
    if info.benchmark_runner != None:
        env.fail("correctness-only execution retained benchmark runner %r" % info.benchmark_runner)
    if info.benchmark_runner_args:
        env.fail("correctness-only execution retained benchmark arguments %r" % info.benchmark_runner_args)
    if info.test_runner_args != [
        "--max-samples-per-case=1",
        "--case=benchmark_case",
    ]:
        env.fail("unexpected correctness runner args %r" % info.test_runner_args)
    runfiles = target[DefaultInfo].default_runfiles.files.to_list()
    _expect_basename(env, runfiles, info.test_runner.basename)
    for file in runfiles:
        if file.basename.startswith("loom-benchmark"):
            env.fail("correctness-only execution retained benchmark runfile %r" % file)

def _test_resource_profile_preserves_direct_execution(name, **kwargs):
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_resource_profile_preserves_direct_execution_impl,
        target = ":profiled_test_all_execute_serialized_reference_test_launcher",
        **kwargs
    )

def _test_resource_profile_preserves_direct_execution_impl(env, target):
    info = target[LoomExecutionTestInfo]
    env.expect.that_str(info.module.basename).equals(
        "profiled_module.loombc",
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

def _test_explicit_runner_owns_profile_execution(name, **kwargs):
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_explicit_runner_owns_profile_execution_impl,
        target = ":profiled_test_selected_execute_explicit_runner_test",
        **kwargs
    )

def _test_explicit_runner_owns_profile_execution_impl(env, target):
    info = target[IreeExecutableInfo]
    env.expect.that_str(str(info.src)).contains("loom-test")
    arguments = target[IreeRunfilesArgumentsInfo].arguments
    if not arguments[0].endswith("profiled_module.loombc"):
        env.fail("unexpected explicit runner module argument %r" % arguments[0])
    if arguments[1:] != [
        "--max-samples-per-case=1",
        "--case=benchmark_case",
        "--sample=0",
    ]:
        env.fail("unexpected explicit runner args %r" % arguments)

    runfiles = target[DefaultInfo].default_runfiles.files.to_list()
    _expect_basename(env, runfiles, "profiled_module.loombc")
    for expected_tag in [
        "loom-execution-profile=explicit_runner",
        "loom-executor=hosted",
    ]:
        if expected_tag not in target[TestingAspectInfo].attrs.tags:
            env.fail(
                "expected %r in test tags %r" %
                (expected_tag, target[TestingAspectInfo].attrs.tags),
            )
    for file in runfiles:
        if file.basename.startswith("loom-benchmark"):
            env.fail("explicit profile runner retained benchmark runfile %r" % file)

def _test_grouped_execution_root_sources(name, **kwargs):
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_grouped_execution_root_sources_impl,
        target = ":profiled_module_library",
        **kwargs
    )

def _test_grouped_execution_root_sources_impl(env, target):
    action = _find_action(env, target[TestingAspectInfo].actions, "LoomLibrary")
    inputs = action.inputs.to_list()
    _expect_basename(env, inputs, "profile_benchmarks.loom")
    _expect_basename(env, inputs, "profile_cases.loom")
    _expect_basename(env, inputs, "library_dependency.loombc")

def _test_execution_module_links_root_tests(name, **kwargs):
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_execution_module_links_root_tests_impl,
        target = ":profiled_module",
        **kwargs
    )

def _test_execution_module_links_root_tests_impl(env, target):
    files = target[DefaultInfo].files.to_list()
    _expect_basename(env, files, "profiled_module.loombc")
    action = _find_action(env, target[TestingAspectInfo].actions, "LoomTestModule")
    for expected_arg in [
        "--mode=link",
        "--include-input-tests",
        "--to=bc",
    ]:
        if expected_arg not in action.argv:
            env.fail("expected %r in %r" % (expected_arg, action.argv))
    _expect_arg_with_prefix_and_suffix(
        env,
        action.argv,
        "",
        "profiled_module_library.loombc",
    )
    _expect_arg_with_prefix_and_suffix(
        env,
        action.argv,
        "--library=",
        "library_dependency.loombc",
    )
    inputs = action.inputs.to_list()
    _expect_basename(env, inputs, "profiled_module_library.loombc")
    _expect_basename(env, inputs, "library_dependency.loombc")
    _expect_no_basename(env, inputs, "profile_cases.loom")
    _expect_no_basename(env, inputs, "profile_benchmarks.loom")

def _test_direct_sources_own_checks_once(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_direct_sources_own_checks_once_impl,
        targets = {
            "format": ":direct_source_test_format_0_test",
            "format_launcher": ":direct_source_test_format_0_test_launcher",
            "library": ":direct_source_test_module_library",
            "lint": ":direct_source_test_lint_0_test",
            "lint_launcher": ":direct_source_test_lint_0_test_launcher",
            "module": ":direct_source_test_module",
            "suite": ":direct_source_test",
        },
        **kwargs
    )

def _expect_source_check(env, target, source, manual):
    attributes = target[TestingAspectInfo].attrs
    env.expect.that_collection(attributes.tags).contains_exactly(
        ["hostonly", "manual"] if manual else ["hostonly"],
    )
    env.expect.that_collection([
        constraint.label.name
        for constraint in attributes.target_compatible_with
    ]).contains_exactly(["source_provider_available"])
    files = target[DefaultInfo].default_runfiles.files.to_list()
    _expect_basename(env, files, source)
    for file in files:
        if file.basename.endswith(".loombc"):
            env.fail("source check depends on a linked artifact: %s" % file)

def _test_direct_sources_own_checks_once_impl(env, targets):
    env.expect.that_collection([
        test.label.name
        for test in targets.suite[TestingAspectInfo].attrs.tests
    ]).contains_exactly([
        "direct_source_test_lint_0_test",
        "direct_source_test_format_0_test",
        "direct_source_test_all_execute_reference_test",
        "direct_source_test_all_execute_serialized_reference_test",
        "direct_source_test_selected_execute_reference_test",
        "direct_source_test_selected_execute_serialized_reference_test",
    ])
    for check in [targets.lint, targets.format]:
        _expect_source_check(env, check, "profile_cases.loom", manual = False)
    for check in [targets.lint_launcher, targets.format_launcher]:
        _expect_source_check(env, check, "profile_cases.loom", manual = True)
    for artifact in [targets.module, targets.library]:
        env.expect.that_collection(artifact[TestingAspectInfo].attrs.tags).contains("manual")

def _test_reusable_module_owns_source_checks(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_reusable_module_owns_source_checks_impl,
        targets = {
            "consumer": ":correctness_only_test",
            "format": ":profiled_module_format_1_test",
            "lint": ":profiled_module_lint_1_test",
            "suite": ":profiled_module_test",
        },
        **kwargs
    )

def _test_reusable_module_owns_source_checks_impl(env, targets):
    env.expect.that_collection([
        test.label.name
        for test in targets.suite[TestingAspectInfo].attrs.tests
    ]).contains_exactly([
        "profiled_module_lint_0_test",
        "profiled_module_format_0_test",
        "profiled_module_lint_1_test",
        "profiled_module_format_1_test",
    ])
    env.expect.that_collection([
        test.label.name
        for test in targets.consumer[TestingAspectInfo].attrs.tests
    ]).contains_exactly(["correctness_only_test_execute_reference_test"])
    for check in [targets.lint, targets.format]:
        _expect_source_check(env, check, "profile_cases.loom", manual = True)

def _test_nontext_module_has_no_source_suite(name, **kwargs):
    module_name = name + "_module"
    loom_test_module(
        name = module_name,
        srcs = [":profiled_module"],
        tags = ["manual"],
    )
    if native.existing_rule(module_name + "_test"):
        fail("non-text inputs must not create an implicit all-package test suite")
    analysis_test(
        name = name,
        impl = _test_nontext_module_has_no_source_suite_impl,
        target = ":" + module_name + "_library",
        **kwargs
    )

def _test_nontext_module_has_no_source_suite_impl(env, target):
    action = _find_action(env, target[TestingAspectInfo].actions, "LoomLibrary")
    _expect_basename(env, action.inputs.to_list(), "profiled_module.loombc")

def loom_library_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_correctness_only_profile_omits_benchmark_runner,
            _test_deps_only_library_propagates_dependencies,
            _test_direct_sources_own_checks_once,
            _test_execution_module_links_root_tests,
            _test_execution_profile_contract,
            _test_explicit_runner_owns_profile_execution,
            _test_generated_kernel_binary_is_testonly,
            _test_grouped_execution_root_sources,
            _test_library_keeps_dependency_module_separate,
            _test_nontext_module_has_no_source_suite,
            _test_redundant_direct_dependency_is_not_transitive,
            _test_resource_profile_preserves_direct_execution,
            _test_reusable_module_owns_source_checks,
            _test_transitive_audit_universe_is_separate,
        ],
    )
