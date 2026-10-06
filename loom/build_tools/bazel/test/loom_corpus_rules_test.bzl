# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for target-owned Loom corpus build qualification."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo")
load(
    "//loom/build_tools/bazel:defs.bzl",
    "LoomCorpusBuildInfo",
    "LoomExecutionTestInfo",
)

_FIXTURE = "//loom/build_tools/bazel/test/testdata/corpus/target"

def _actions_with_mnemonic(actions, mnemonic):
    return [action for action in actions if action.mnemonic == mnemonic]

def _expect_basename(env, files, expected_basename):
    for file in files:
        if file.basename == expected_basename:
            return
    env.fail("expected basename %r in %r" % (expected_basename, files))

def _action_with_argument(env, actions, argument):
    for action in actions:
        if argument in action.argv:
            return action
    env.fail("expected action containing %r in %r" % (argument, actions))
    return None

def _test_program_fans_out_by_profile(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_program_fans_out_by_profile_impl,
        target = _FIXTURE + ":sample_nested_fixture",
        **kwargs
    )

def _test_program_fans_out_by_profile_impl(env, target):
    actions = target[TestingAspectInfo].actions
    link_actions = _actions_with_mnemonic(actions, "LoomCorpusLink")
    if len(link_actions) != 1:
        env.fail("expected one subject-selection link action, got %r" % link_actions)
        return
    link_action = link_actions[0]
    for expected_arg in [
        "--mode=link",
        "--include-input-tests",
        "--strip-check",
        "--to=bc",
    ]:
        if expected_arg not in link_action.argv:
            env.fail("expected %r in link arguments %r" % (expected_arg, link_action.argv))

    compile_actions = _actions_with_mnemonic(actions, "LoomCorpusCompile")
    if len(compile_actions) != 2:
        env.fail("expected two profile compile actions, got %r" % compile_actions)
        return
    profile_a = _action_with_argument(env, compile_actions, "--target=fake:a")
    profile_b = _action_with_argument(env, compile_actions, "--target=fake:b")
    for action in [profile_a, profile_b]:
        _expect_basename(env, action.inputs.to_list(), "subjects.loombc")
        if any([arg.startswith("--product=") for arg in action.argv]):
            env.fail("corpus compilation must infer its product: %r" % action.argv)
        if "--compile-report=details" not in action.argv:
            env.fail("expected detailed compile report in %r" % action.argv)
        if "--target=fake:c" in action.argv:
            env.fail("excluded profile C produced an action: %r" % action.argv)
    if "--exclude-root=@unsupported" in profile_a.argv:
        env.fail("profile A inherited profile B's xfail: %r" % profile_a.argv)
    if "--exclude-root=@unsupported_common" not in profile_a.argv:
        env.fail("profile A did not inherit its target-set xfail: %r" % profile_a.argv)
    if "--exclude-root=@unsupported" not in profile_b.argv:
        env.fail("profile B did not exclude its xfail: %r" % profile_b.argv)
    if "--exclude-root=@unsupported_other" not in profile_b.argv:
        env.fail("profile B did not exclude its second xfail: %r" % profile_b.argv)
    if "--exclude-root=@unsupported_common" not in profile_b.argv:
        env.fail("profile B did not inherit its target-set xfail: %r" % profile_b.argv)

def _test_program_exposes_outputs_and_batched_xfails(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_program_exposes_outputs_and_batched_xfails_impl,
        target = _FIXTURE + ":sample_nested_fixture",
        **kwargs
    )

def _test_program_exposes_outputs_and_batched_xfails_impl(env, target):
    actions = target[TestingAspectInfo].actions
    xfail_actions = _actions_with_mnemonic(actions, "LoomCorpusXfails")
    if len(xfail_actions) != 2:
        env.fail("expected one batched xfail action per selected profile, got %r" % xfail_actions)
        return
    profile_a = _action_with_argument(env, xfail_actions, "--target=fake:a")
    for expected_arg in [
        "--expected-root=@unsupported_common",
        "--expected-diagnostic=TYPE/002",
    ]:
        if expected_arg not in profile_a.argv:
            env.fail("expected %r in profile A xfail arguments %r" % (expected_arg, profile_a.argv))

    profile_b = _action_with_argument(env, xfail_actions, "--target=fake:b")
    for expected_arg in [
        "--expected-root=@unsupported_common",
        "--expected-diagnostic=TYPE/002",
        "--expected-root=@unsupported",
        "--expected-diagnostic=TARGET/072",
        "--expected-root=@unsupported_other",
        "--expected-diagnostic=TYPE/001",
    ]:
        if expected_arg not in profile_b.argv:
            env.fail("expected %r in profile B xfail arguments %r" % (expected_arg, profile_b.argv))
    if any([arg.startswith("--product=") for arg in profile_b.argv]):
        env.fail("xfail probes must infer their product: %r" % profile_b.argv)

    default_files = target[DefaultInfo].files.to_list()
    if len(default_files) != 6:
        env.fail("expected two artifacts, two reports, and two xfail results: %r" % default_files)
    for basename in [
        "fake-a.artifact",
        "fake-a.compile.json",
        "fake-a.xfails",
        "fake-b.artifact",
        "fake-b.compile.json",
        "fake-b.xfails",
    ]:
        _expect_basename(env, default_files, basename)
    if len(target[OutputGroupInfo].artifacts.to_list()) != 2:
        env.fail("expected two artifact output-group files")
    if len(target[OutputGroupInfo].compile_reports.to_list()) != 2:
        env.fail("expected two compile-report output-group files")
    if len(target[OutputGroupInfo].xfail_results.to_list()) != 2:
        env.fail("expected two xfail-result output-group files")

def _test_program_supports_all_roots_xfail(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_program_supports_all_roots_xfail_impl,
        target = _FIXTURE + ":sample_other",
        **kwargs
    )

def _test_program_supports_all_roots_xfail_impl(env, target):
    actions = target[TestingAspectInfo].actions
    compile_actions = _actions_with_mnemonic(actions, "LoomCorpusCompile")
    if len(compile_actions) != 1:
        env.fail("expected only profile A to produce a positive compile: %r" % compile_actions)
        return
    _action_with_argument(env, compile_actions, "--target=fake:a")

    xfail_actions = _actions_with_mnemonic(actions, "LoomCorpusXfails")
    if len(xfail_actions) != 1:
        env.fail("expected one all-roots xfail action: %r" % xfail_actions)
        return
    profile_b = _action_with_argument(env, xfail_actions, "--target=fake:b")
    for expected_arg in [
        "--require-all-roots",
        "--expected-root=@other",
        "--expected-diagnostic=TARGET/033",
    ]:
        if expected_arg not in profile_b.argv:
            env.fail("expected %r in all-roots xfail arguments %r" % (expected_arg, profile_b.argv))

    default_files = target[DefaultInfo].files.to_list()
    if len(default_files) != 3:
        env.fail("expected one artifact, one report, and one xfail result: %r" % default_files)
    if len(target[OutputGroupInfo].artifacts.to_list()) != 1:
        env.fail("expected one artifact output-group file")
    if len(target[OutputGroupInfo].compile_reports.to_list()) != 1:
        env.fail("expected one compile-report output-group file")
    if len(target[OutputGroupInfo].xfail_results.to_list()) != 1:
        env.fail("expected one xfail-result output-group file")

def _test_aggregate_collects_program_outputs(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_aggregate_collects_program_outputs_impl,
        target = _FIXTURE + ":all",
        **kwargs
    )

def _test_aggregate_collects_program_outputs_impl(env, target):
    corpus = target[LoomCorpusBuildInfo]
    if len(corpus.sources.to_list()) != 3:
        env.fail("expected three catalog sources, got %r" % corpus.sources.to_list())
    if sorted(corpus.source_identities.to_list()) != [
        "sample/excluded.loom",
        "sample/nested/fixture.loom",
        "sample/other.loom",
    ]:
        env.fail("unexpected source identities %r" % corpus.source_identities.to_list())
    if len(corpus.artifacts.to_list()) != 3:
        env.fail("expected three corpus artifacts, got %r" % corpus.artifacts.to_list())
    if len(corpus.compile_reports.to_list()) != 3:
        env.fail("expected three corpus reports, got %r" % corpus.compile_reports.to_list())
    if len(corpus.qualification_results.to_list()) != 3:
        env.fail(
            "expected three corpus qualification results, got %r" %
            corpus.qualification_results.to_list(),
        )
    if len(target[DefaultInfo].files.to_list()) != 9:
        env.fail("aggregate default outputs must request every qualification action")

def _test_program_allows_explicit_full_exclusion(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_program_allows_explicit_full_exclusion_impl,
        target = _FIXTURE + ":sample_excluded",
        **kwargs
    )

def _test_program_allows_explicit_full_exclusion_impl(env, target):
    actions = target[TestingAspectInfo].actions
    if _actions_with_mnemonic(actions, "LoomCorpusLink"):
        env.fail("a fully excluded source must not produce a subject link action")
    if _actions_with_mnemonic(actions, "LoomCorpusCompile"):
        env.fail("a fully excluded source must not produce compiler actions")
    if target[DefaultInfo].files.to_list():
        env.fail("a fully excluded source must not claim qualification outputs")

def _test_execution_partitions_profiles_and_xfails(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_execution_partitions_profiles_and_xfails_impl,
        target = ":sample_nested_fixture_test_execute_reference_test_launcher",
        **kwargs
    )

def _test_execution_partitions_profiles_and_xfails_impl(env, target):
    info = target[LoomExecutionTestInfo]
    if info.test_runner_args != [
        "--max-samples-per-case=1",
        "--xfail=@fixture_scenario=TARGET/003,EXPECT/003",
    ]:
        env.fail("unexpected corpus correctness arguments %r" % info.test_runner_args)
    if info.benchmark_runner != None or info.benchmark_runner_args:
        env.fail("corpus execution must remain correctness-only")

def _test_execution_carries_allowed_failures(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_execution_carries_allowed_failures_impl,
        target = ":sample_other_test_execute_reference_test_launcher",
        **kwargs
    )

def _test_execution_carries_allowed_failures_impl(env, target):
    info = target[LoomExecutionTestInfo]
    if info.test_runner_args != [
        "--max-samples-per-case=1",
        "--allow-failure=@other_case=TARGET/003,EXPECT/003",
    ]:
        env.fail("unexpected corpus correctness arguments %r" % info.test_runner_args)

def loom_corpus_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_aggregate_collects_program_outputs,
            _test_execution_carries_allowed_failures,
            _test_execution_partitions_profiles_and_xfails,
            _test_program_allows_explicit_full_exclusion,
            _test_program_exposes_outputs_and_batched_xfails,
            _test_program_fans_out_by_profile,
            _test_program_supports_all_roots_xfail,
        ],
    )
