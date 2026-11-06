# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks the selected ELF toolchain's compile and link debug policy."""

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_testing//lib:analysis_test.bzl", "analysis_test")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo")

def _compression_impl(env, target):
    enabled = target.label.name.endswith("_enabled")
    actions = target[TestingAspectInfo].actions
    for mnemonic, flag in [
        ("CppCompile", "-gz=zlib"),
        ("CppLink", "-Wl,--compress-debug-sections=zlib"),
    ]:
        selected = [action for action in actions if action.mnemonic == mnemonic]

        # Toolchains may compile both PIC and non-PIC objects for the fixture.
        env.expect.that_int(len(selected)).is_greater_than(0)
        for action in selected:
            env.expect.that_bool(flag in action.argv).equals(enabled)

def debug_info_test_suite(name):
    """Registers compile/link coverage for compression and its explicit opt-out.

    Args:
        name: Test suite name and fixture prefix.
    """
    tests = []
    for mode in ["enabled", "disabled"]:
        target_name = name + "_" + mode
        cc_binary(
            name = target_name,
            srcs = ["debug_info_fixture.c"],
            tags = ["manual"],
            testonly = True,
        )
        test_name = target_name + "_test"
        analysis_test(
            name = test_name,
            impl = _compression_impl,
            target = target_name,
            config_settings = {
                "//command_line_option:features": [
                    "compressed_debug_info" if mode == "enabled" else "-compressed_debug_info",
                ],
            },
            attr_values = {"target_compatible_with": ["@platforms//os:linux"]},
        )
        tests.append(test_name)
    native.test_suite(name = name, tests = tests)
