# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for SPIR-V assembly module generation."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo")
load(":asm_module.bzl", "IreeSpirvAsmModuleInfo")

visibility("//build_tools/spirv/...")

def _expect_basename(env, files, expected_basename):
    for file in files:
        if file.basename == expected_basename:
            return
    env.fail("expected basename %r in %r" % (expected_basename, files))

def _expect_arg_with_suffix(env, args, suffix):
    for arg in args:
        if arg.endswith(suffix):
            return
    env.fail("expected argument ending with %r in %r" % (suffix, args))

def _test_spirv_asm_action_contract(name, **kwargs):
    analysis_test(
        name = name,
        attr_values = {
            "tags": ["manual"],
            "timeout": "short",
        },
        impl = _test_spirv_asm_action_contract_impl,
        target = ":minimal",
        **kwargs
    )

def _test_spirv_asm_action_contract_impl(env, target):
    info = target[IreeSpirvAsmModuleInfo]
    _expect_basename(env, [info.module], "minimal.spv")
    _expect_basename(env, [info.src], "minimal.spvasm")
    if info.tool.name != "spirv-as":
        env.fail("unexpected spirv-as tool %s" % info.tool)

    actions = target[TestingAspectInfo].actions
    env.expect.that_int(len(actions)).equals(1)
    action = actions[0]
    env.expect.that_str(action.mnemonic).equals("IreeSpirvAsmModule")
    _expect_arg_with_suffix(env, action.argv, "--target-env=vulkan1.3")
    _expect_arg_with_suffix(env, action.argv, "-o")
    _expect_arg_with_suffix(env, action.argv, "minimal.spv")
    _expect_arg_with_suffix(env, action.argv, "minimal.spvasm")
    _expect_basename(env, action.inputs.to_list(), "minimal.spvasm")

def spirv_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_spirv_asm_action_contract,
        ],
    )
