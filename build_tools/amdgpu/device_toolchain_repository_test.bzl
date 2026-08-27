# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests coherent AMDGPU device-toolchain capability discovery."""

load(
    "@bazel_skylib//lib:unittest.bzl",
    "analysistest",
    "asserts",
)
load(
    ":device_toolchain_repository.bzl",
    "discover_amdgpu_device_toolchain",
)

_DeviceToolchainInfo = provider(
    doc = "Resolved AMDGPU device-toolchain capability under test.",
    fields = {
        "available": "Whether the base AMDGPU device toolchain is complete.",
        "error": "Base toolchain discovery error.",
        "rocm_device_libraries": "Optional HIP device-library directory.",
        "tools": "Resolved tool map.",
    },
)

_BASE_PATHS = [
    "/sdk/amdgcn/bitcode",
    "/sdk/amdgcn/bitcode/hip.bc",
    "/sdk/amdgcn/bitcode/ockl.bc",
    "/sdk/amdgcn/bitcode/ocml.bc",
    "/sdk/bin/clang",
    "/sdk/bin/ld.lld",
    "/sdk/bin/llvm-ar",
    "/sdk/bin/llvm-link",
    "/sdk/bin/llvm-objcopy",
    "/sdk/lib/clang/22/include",
    "/sdk/lib/clang/22/include/stddef.h",
]

def _path_result(path, include_bundler = False):
    path = str(path)
    exists = path in _BASE_PATHS
    if include_bundler and path == "/sdk/bin/clang-offload-bundler":
        exists = True
    return struct(exists = exists, realpath = path)

def _path_without_bundler(path):
    return _path_result(path)

def _path_with_bundler(path):
    return _path_result(path, include_bundler = True)

def _watch(path):
    return path

def _which_none(name):
    return None

def _getenv_none(name):
    return ""

def _getenv_explicit_clang(name):
    if name == "IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_BINARY":
        return "/sdk/bin/clang"
    return ""

def _execute_result(stdout = "", stderr = "", return_code = 0):
    return struct(
        return_code = return_code,
        stderr = stderr,
        stdout = stdout,
    )

def _execute(args, environment = None, quiet = False):
    if not quiet:
        fail("device-tool discovery commands must be quiet")
    if args[0] == "test":
        return _execute_result()
    if environment == None:
        fail("device-tool queries require a controlled environment")
    if args[0] != "/sdk/bin/clang":
        fail("unexpected executable: {}".format(args[0]))
    if args[1] == "--print-targets":
        return _execute_result(stdout = "  amdgcn - AMD GCN GPUs\n")
    if args[1] == "-print-resource-dir":
        return _execute_result(stdout = "/sdk/lib/clang/22\n")
    tool_prefix = "--print-prog-name="
    if args[1].startswith(tool_prefix):
        return _execute_result(
            stdout = "/sdk/bin/{}\n".format(args[1][len(tool_prefix):]),
        )
    fail("unexpected clang query: {}".format(args))

def _execute_missing_llvm_link(args, environment = None, quiet = False):
    if args == ["/sdk/bin/clang", "--print-prog-name=llvm-link"]:
        return _execute_result(stdout = "/missing/llvm-link\n")
    return _execute(args, environment = environment, quiet = quiet)

def _fake_repository_ctx(scenario):
    if scenario == "unavailable":
        return struct(
            execute = _execute,
            getenv = _getenv_none,
            os = struct(name = "linux"),
            path = _path_without_bundler,
            watch = _watch,
            which = _which_none,
        )
    return struct(
        execute = _execute_missing_llvm_link if scenario == "missing_companion" else _execute,
        getenv = _getenv_explicit_clang,
        os = struct(name = "linux"),
        path = _path_with_bundler if scenario == "complete" else _path_without_bundler,
        watch = _watch,
        which = _which_none,
    )

def _subject_impl(ctx):
    toolchain = discover_amdgpu_device_toolchain(
        _fake_repository_ctx(ctx.attr.scenario),
        "auto",
    )
    return [_DeviceToolchainInfo(
        available = toolchain.available,
        error = toolchain.error,
        rocm_device_libraries = toolchain.rocm_device_libraries,
        tools = toolchain.tools,
    )]

_subject = rule(
    implementation = _subject_impl,
    attrs = {"scenario": attr.string(mandatory = True)},
)

def _unavailable_test_impl(ctx):
    env = analysistest.begin(ctx)
    toolchain = analysistest.target_under_test(env)[_DeviceToolchainInfo]
    asserts.equals(env, False, toolchain.available)
    asserts.equals(
        env,
        True,
        "could not find clang with AMDGPU support" in toolchain.error,
    )
    return analysistest.end(env)

_unavailable_test = analysistest.make(_unavailable_test_impl)

def _complete_test_impl(ctx):
    env = analysistest.begin(ctx)
    toolchain = analysistest.target_under_test(env)[_DeviceToolchainInfo]
    asserts.equals(env, True, toolchain.available)
    asserts.equals(env, "", toolchain.error)
    asserts.equals(env, "/sdk/amdgcn/bitcode", toolchain.rocm_device_libraries)
    asserts.equals(
        env,
        "/sdk/bin/clang-offload-bundler",
        toolchain.tools.get("clang-offload-bundler"),
    )
    return analysistest.end(env)

_complete_test = analysistest.make(_complete_test_impl)

def _missing_bundler_test_impl(ctx):
    env = analysistest.begin(ctx)
    toolchain = analysistest.target_under_test(env)[_DeviceToolchainInfo]
    asserts.equals(env, True, toolchain.available)
    asserts.equals(env, "", toolchain.error)
    asserts.equals(env, "", toolchain.rocm_device_libraries)
    asserts.equals(env, None, toolchain.tools.get("clang-offload-bundler"))
    asserts.equals(env, "/sdk/bin/clang", toolchain.tools.get("clang"))
    return analysistest.end(env)

_missing_bundler_test = analysistest.make(_missing_bundler_test_impl)

def _missing_companion_test_impl(ctx):
    env = analysistest.begin(ctx)
    toolchain = analysistest.target_under_test(env)[_DeviceToolchainInfo]
    asserts.equals(env, False, toolchain.available)
    asserts.equals(
        env,
        True,
        "reported a missing llvm-link executable" in toolchain.error,
    )
    return analysistest.end(env)

_missing_companion_test = analysistest.make(_missing_companion_test_impl)

def device_toolchain_repository_test_suite(name):
    """Defines AMDGPU device-toolchain capability discovery tests.

    Args:
      name: Test suite target name.
    """
    tests = []
    for scenario, test_rule in [
        ("complete", _complete_test),
        ("missing_bundler", _missing_bundler_test),
        ("missing_companion", _missing_companion_test),
        ("unavailable", _unavailable_test),
    ]:
        subject_name = name + "_" + scenario + "_subject"
        test_name = name + "_" + scenario + "_test"
        _subject(name = subject_name, scenario = scenario, tags = ["manual"])
        test_rule(name = test_name, target_under_test = ":" + subject_name)
        tests.append(":" + test_name)
    native.test_suite(name = name, tests = tests)
