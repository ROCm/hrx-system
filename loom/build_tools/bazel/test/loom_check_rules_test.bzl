# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for loom-check Bazel rules."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("//build_tools/bazel:executable.bzl", "IreeExecutableInfo")
load(
    "//build_tools/bazel:runfiles.bzl",
    "IreeRunfilesArgumentsInfo",
    "RUNFILES_PATH_BEGIN",
    "RUNFILES_PATH_END",
)
load(
    "//loom/build_tools/bazel:loom_check.bzl",
    "LoomCheckTestInfo",
    "loom_check_test",
)
load("//loom/build_tools/bazel:loom_library.bzl", "loom_test")
load("//loom/target/vm:execution_profiles.bzl", "VM_REFERENCE_PROFILE")

def _test_loom_check_test_declares_fixture(name, **kwargs):
    loom_check_test(
        name = name + "_subject",
        env = {
            "LOOM_CHECK_FIXTURE": "$(location :roundtrip.loom-test)",
        },
        src = "roundtrip.loom-test",
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_loom_check_test_declares_fixture_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_loom_check_test_declares_fixture_impl(env, target):
    info = target[IreeExecutableInfo]
    if not str(info.src).endswith("//loom/src/loom/tools/loom-check:loom-check-test"):
        env.fail("unexpected runner %s" % info.src)
    env.expect.that_str(info.output.basename).equals(target.label.name)
    env.expect.that_str(info.env["LOOM_CHECK_FIXTURE"]).contains("roundtrip.loom-test")
    env.expect.that_collection([file.basename for file in info.data.to_list()]).contains(
        "roundtrip.loom-test",
    )
    runfiles_arguments = target[IreeRunfilesArgumentsInfo]
    env.expect.that_collection(runfiles_arguments.arguments).contains_exactly([
        "loom/build_tools/bazel/test/roundtrip.loom-test",
    ]).in_order()
    env.expect.that_collection(runfiles_arguments.marked_arguments).contains_exactly([
        RUNFILES_PATH_BEGIN +
        "loom/build_tools/bazel/test/roundtrip.loom-test" +
        RUNFILES_PATH_END,
    ]).in_order()

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
    env.expect.that_str(info.compile_target).equals("FakeTargetFamily123:FakeTargetSelector123")
    env.expect.that_str(info.fixture.basename).equals(
        target.label.name.removesuffix("_compile_test_fake_profile_launcher") + "_module.loombc",
    )
    runfiles = target[DefaultInfo].default_runfiles.files.to_list()
    for file in runfiles:
        if file.basename in ["profile_cases.loom", "library_dependency.loombc"]:
            env.fail("compiler bypasses the linked root-owned test closure: %s" % file)

def _test_wasm_disassembler_has_runtime_closure(name, **kwargs):
    analysis_test(
        name = name,
        impl = _test_wasm_disassembler_has_runtime_closure_impl,
        targets = {
            "fixture": "//loom/src/loom/target/emit/wasm/test:assembly",
            "runtime": "@wasi_sdk//:tool_runtime_files",
            "tool": "@wasi_sdk//:llvm-objdump",
        },
        **kwargs
    )

def _test_wasm_disassembler_has_runtime_closure_impl(env, targets):
    files = targets.fixture[DefaultInfo].default_runfiles.files.to_list()
    required = depset(transitive = [
        targets.tool[DefaultInfo].files,
        targets.runtime[DefaultInfo].files,
    ])
    for file in required.to_list():
        if file not in files:
            env.fail("Wasm disassembly fixture omits SDK runtime input: %s" % file.short_path)

def loom_check_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_loom_check_test_declares_fixture,
            _test_wasm_disassembler_has_runtime_closure,
            _test_execution_and_compiler_share_module,
        ],
    )
