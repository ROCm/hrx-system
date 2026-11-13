# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for shared C/C++ fuzz Bazel macros."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo", "util")
load("//build_tools/bazel:cc_fuzz.bzl", "iree_cc_fuzz")
load(
    "//build_tools/bazel:runfiles.bzl",
    "IreeRunfilesArgumentsInfo",
    "RUNFILES_PATH_BEGIN",
    "RUNFILES_PATH_END",
)

_TEST_DYNAMIC_LIBRARY_ENVIRONMENT = "IREE_TEST_DYNAMIC_LIBRARY_PATH"

def _test_cc_fuzz_adds_fuzzer_contract(name, **kwargs):
    util.helper_target(
        iree_cc_fuzz,
        name = name + "_subject",
        args = [
            "-seed_inputs=$(rootpath :generate_rule_fixture.data),$(rootpath :empty_suppressions.txt)",
        ],
        data = [
            ":empty_suppressions.txt",
            ":generate_rule_fixture.data",
        ],
        defines = ["USER_DEFINE"],
        deps = [":dynamic_library_environment_library"],
        linkopts = ["-Wl,--user-linkopt"],
        srcs = [name + "_subject.cc"],
        tags = ["requires-gpu-vulkan"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_fuzz_adds_fuzzer_contract_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_fuzz_adds_fuzzer_contract_impl(env, target):
    attrs = target[TestingAspectInfo].attrs
    for expected_define in [
        "USER_DEFINE",
        "FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION",
    ]:
        if expected_define not in attrs.defines:
            env.fail("expected %r in fuzz defines %r" % (expected_define, attrs.defines))
    for expected_linkopt in [
        "-Wl,--user-linkopt",
        "-fsanitize=fuzzer",
    ]:
        if expected_linkopt not in attrs.linkopts:
            env.fail("expected %r in fuzz linkopts %r" % (expected_linkopt, attrs.linkopts))
    for expected_tag in [
        "requires-gpu-vulkan",
        "fuzz",
        "manual",
    ]:
        if expected_tag not in attrs.tags:
            env.fail("expected %r in fuzz tags %r" % (expected_tag, attrs.tags))
    if not attrs.testonly:
        env.fail("expected fuzz target to be testonly")
    runfiles_arguments = target[IreeRunfilesArgumentsInfo]
    configured_argument = (
        "-seed_inputs=" +
        "build_tools/bazel/test/generate_rule_fixture.data," +
        "build_tools/bazel/test/empty_suppressions.txt"
    )
    env.expect.that_collection(runfiles_arguments.arguments).contains_exactly([
        configured_argument,
    ])
    env.expect.that_collection(runfiles_arguments.marked_arguments).contains_exactly([
        "-seed_inputs=" +
        RUNFILES_PATH_BEGIN +
        "build_tools/bazel/test/generate_rule_fixture.data" +
        RUNFILES_PATH_END +
        "," +
        RUNFILES_PATH_BEGIN +
        "build_tools/bazel/test/empty_suppressions.txt" +
        RUNFILES_PATH_END,
    ])
    library_path = target[RunEnvironmentInfo].environment[_TEST_DYNAMIC_LIBRARY_ENVIRONMENT]
    if not library_path.endswith("dynamic_library_root.so"):
        env.fail("unexpected fuzz dynamic-library path %r" % library_path)

def _test_cc_fuzz_accepts_external_engine(name, **kwargs):
    util.helper_target(
        iree_cc_fuzz,
        name = name + "_subject",
        linkopts = ["/fuzzing-engine/libFuzzingEngine.a", "-Wl,--user-linkopt"],
        srcs = [name + "_subject.cc"],
    )
    analysis_test(
        name = name,
        config_settings = {
            str(Label("//build_tools/bazel:external_fuzzing_engine")): True,
        },
        impl = _test_cc_fuzz_accepts_external_engine_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_fuzz_accepts_external_engine_impl(env, target):
    linkopts = target[TestingAspectInfo].attrs.linkopts
    engines = [value for value in linkopts if value == "/fuzzing-engine/libFuzzingEngine.a"]
    env.expect.that_collection(engines).contains_exactly(["/fuzzing-engine/libFuzzingEngine.a"])
    if "-Wl,--user-linkopt" not in linkopts:
        env.fail("expected the user link option in fuzz linkopts %r" % linkopts)
    if "-fsanitize=fuzzer" in linkopts:
        env.fail("unexpected default fuzz engine alongside the external engine: %r" % linkopts)

def cc_fuzz_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_cc_fuzz_adds_fuzzer_contract,
            _test_cc_fuzz_accepts_external_engine,
        ],
    )
