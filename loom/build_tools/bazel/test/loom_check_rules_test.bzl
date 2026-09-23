# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for loom-check Bazel rules."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo")
load(
    "//loom/build_tools/bazel:loom_check.bzl",
    "LoomCheckTestInfo",
    "loom_check_test",
)
load("//loom/build_tools/bazel:loom_library.bzl", "loom_test")
load("//loom/target/vm:execution_profiles.bzl", "VM_REFERENCE_PROFILE")

def _find_action_with_output(env, actions, expected_basename):
    for action in actions:
        for output in action.outputs.to_list():
            if output.basename == expected_basename:
                return action
    env.fail("expected action output basename %r in %r" % (expected_basename, actions))
    return None

def _test_loom_check_wrapper_declares_fixture(name, **kwargs):
    loom_check_test(
        name = name + "_subject",
        env = {
            "LOOM_CHECK_FIXTURE": "$(location :roundtrip.loom-test)",
        },
        runner = "loom_check_fixture_runner.sh",
        src = "roundtrip.loom-test",
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_loom_check_wrapper_declares_fixture_impl,
        target = name + "_subject_launcher",
        **kwargs
    )

def _test_loom_check_wrapper_declares_fixture_impl(env, target):
    info = target[LoomCheckTestInfo]
    if not str(info.runner).endswith("//loom/build_tools/bazel/test:loom_check_fixture_runner.sh"):
        env.fail("unexpected runner %s" % info.runner)
    env.expect.that_str(info.fixture.basename).equals("roundtrip.loom-test")
    env.expect.that_str(info.output.basename).equals(target.label.name + ".sh")
    env.expect.that_str(info.env["LOOM_CHECK_FIXTURE"]).contains("roundtrip.loom-test")

    action = _find_action_with_output(
        env,
        target[TestingAspectInfo].actions,
        target.label.name + ".sh",
    )
    env.expect.that_str(action.mnemonic).equals("FileWrite")

def _test_loom_check_wrapper_uses_test_runner(name, **kwargs):
    loom_check_test(
        name = name + "_subject",
        src = "roundtrip.loom-test",
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        impl = _test_loom_check_wrapper_uses_test_runner_impl,
        target = name + "_subject_launcher",
        **kwargs
    )

def _test_loom_check_wrapper_uses_test_runner_impl(env, target):
    info = target[LoomCheckTestInfo]
    if not str(info.runner).endswith("//loom/src/loom/tools/loom-check:loom-check-test"):
        env.fail("unexpected default runner %s" % info.runner)

def _test_template_fixture_has_no_corpus_runfiles(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_template_fixture_has_no_corpus_runfiles_impl,
        target = "//loom/src/loom/transforms/vector/test:packed_memory_consumers_launcher",
        **kwargs
    )

def _test_template_fixture_has_no_corpus_runfiles_impl(env, target):
    fixture = target[LoomCheckTestInfo].fixture
    files = target[DefaultInfo].default_runfiles.files.to_list()
    env.expect.that_bool(fixture in files).equals(True)
    for file in files:
        if file.short_path.startswith("loom/src/loom/test/corpus/source_low/"):
            env.fail("template source is a test runtime dependency: %s" % file)

def _test_compiler_profile_uses_typed_identity(name, **kwargs):
    loom_check_test(
        name = name + "_subject",
        src = "roundtrip.loom-test",
        compile_targets = [":test_fake_profile"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        impl = _test_compiler_profile_uses_typed_identity_impl,
        target = name + "_subject_compile_test_fake_profile_launcher",
        **kwargs
    )

def _test_compiler_profile_uses_typed_identity_impl(env, target):
    info = target[LoomCheckTestInfo]
    env.expect.that_str(info.compile_target).equals("FakeTargetFamily123:FakeTargetSelector123")
    env.expect.that_str(info.fixture.basename).equals("roundtrip.loom-test")
    env.expect.that_str(str(info.runner)).contains("//loom/src/loom/tools/loom-check:loom-check")

def _test_execution_and_compiler_share_module(name, **kwargs):
    loom_test(
        name = name + "_subject",
        srcs = ["profile_cases.loom"],
        compile_targets = [":test_fake_profile"],
        tags = ["manual"],
        deps = [":library_dependency"],
        execution_profiles = [VM_REFERENCE_PROFILE],
    )
    analysis_test(
        name = name,
        impl = _test_execution_and_compiler_share_module_impl,
        target = name + "_subject_compile_test_fake_profile_launcher",
        **kwargs
    )

def _test_execution_and_compiler_share_module_impl(env, target):
    info = target[LoomCheckTestInfo]
    env.expect.that_str(info.fixture.basename).equals(
        target.label.name.removesuffix("_compile_test_fake_profile_launcher") + "_module.loombc",
    )
    runfiles = target[DefaultInfo].default_runfiles.files.to_list()
    for file in runfiles:
        if file.basename in ["profile_cases.loom", "library_dependency.loombc"]:
            env.fail("compiler bypasses the linked root-owned test closure: %s" % file)

def loom_check_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_loom_check_wrapper_declares_fixture,
            _test_loom_check_wrapper_uses_test_runner,
            _test_template_fixture_has_no_corpus_runfiles,
            _test_compiler_profile_uses_typed_identity,
            _test_execution_and_compiler_share_module,
        ],
    )
