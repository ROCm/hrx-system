# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Actual command and ownership contracts for exported Windows toolchains."""

load("@platforms//host:constraints.bzl", "HOST_CONSTRAINTS")
load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/toolchains:cc_toolchain_config_info.bzl", "CcToolchainConfigInfo")
load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:truth.bzl", "matching")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo")

# Tool-path expansion follows the host's native absolute-path syntax.
NATIVE_ROOT = "C:/" if "@platforms//os:windows" in HOST_CONSTRAINTS else "/"

_CommandInfo = provider("Expanded commands and enabled cross-toolchain capabilities.", fields = {
    # Fully expanded C++ compile arguments from the production configuration.
    "arguments": "Compile arguments.",
    # Selected compiler action tool after configuration composition.
    "compiler": "Compiler action executable path.",
    # Whether the unsupported native-Windows export helper is disabled.
    "explicit_exports": "Automatic exports disabled.",
    # Whether the ELF-style ThinLTO indexing/backend graph is enabled.
    "split_lto": "ELF ThinLTO enabled.",
})

def _command_impl(ctx):
    compiler = ctx.attr.compiler[cc_common.CcToolchainInfo]
    configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = compiler,
        requested_features = ctx.attr.profile,
    )
    variables = cc_common.create_compile_variables(
        cc_toolchain = compiler,
        feature_configuration = configuration,
        source_file = "probe.cc",
        output_file = "probe.obj",
        user_compile_flags = ctx.attr.copts,
    )
    return [_CommandInfo(
        arguments = cc_common.get_memory_inefficient_command_line(
            feature_configuration = configuration,
            action_name = ACTION_NAMES.cpp_compile,
            variables = variables,
        ),
        explicit_exports = cc_common.is_enabled(feature_configuration = configuration, feature_name = "no_windows_export_all_symbols"),
        split_lto = cc_common.is_enabled(feature_configuration = configuration, feature_name = "thin_lto"),
        compiler = cc_common.get_tool_for_action(feature_configuration = configuration, action_name = ACTION_NAMES.cpp_compile),
    )]

_command = rule(
    implementation = _command_impl,
    attrs = {
        "compiler": attr.label(default = ":compiler", providers = [cc_common.CcToolchainInfo]),
        "copts": attr.string_list(),
        "profile": attr.string_list(),
    },
    fragments = ["cpp"],
)

def _test_optimized(name, **kwargs):
    _command(
        name = name + "_subject",
        copts = ["/O1"],
        profile = ["opt", "optimize_for_speed", "thin_lto_link"],
    )
    analysis_test(name = name, target = name + "_subject", impl = _test_optimized_impl, **kwargs)

def _test_optimized_impl(env, target):
    command = target[_CommandInfo]
    for flag in ["/clang:-O3", "/clang:-flto=thin", "/O1"]:
        env.expect.that_collection(command.arguments).contains(flag)
    if command.arguments.index("/clang:-O3") > command.arguments.index("/O1"):
        env.fail("toolchain profile overrides explicit caller optimization")
    env.expect.that_bool(command.explicit_exports).equals(True)
    env.expect.that_bool(command.split_lto).equals(False)

def _test_sanitizer(name, **kwargs):
    _command(name = name + "_subject", profile = ["fastbuild", "sanitizer_debug_info"])
    analysis_test(name = name, target = name + "_subject", impl = _test_sanitizer_impl, **kwargs)

def _test_sanitizer_impl(env, target):
    command = target[_CommandInfo]
    env.expect.that_collection(command.arguments).contains("/Z7")
    env.expect.that_collection(command.arguments).not_contains("/clang:-flto=thin")
    env.expect.that_collection(command.arguments).not_contains("/clang:-O3")

def _test_static_release_crt(name, **kwargs):
    _command(name = name + "_subject", profile = ["dbg", "static_link_msvcrt_release"])
    analysis_test(name = name, target = name + "_subject", impl = _test_static_release_crt_impl, **kwargs)

def _test_static_release_crt_impl(env, target):
    arguments = target[_CommandInfo].arguments
    env.expect.that_collection(arguments).contains("/MT")
    env.expect.that_collection(arguments).not_contains("/MTd")

def _test_static_debug_crt(name, **kwargs):
    _command(name = name + "_subject", profile = ["dbg", "static_link_msvcrt"])
    analysis_test(name = name, target = name + "_subject", impl = _test_static_debug_crt_impl, **kwargs)

def _test_static_debug_crt_impl(env, target):
    arguments = target[_CommandInfo].arguments
    env.expect.that_collection(arguments).contains("/MTd")
    env.expect.that_collection(arguments).not_contains("/MT")

def _test_native_configuration(name, **kwargs):
    analysis_test(name = name, target = ":native_configuration", impl = _test_native_configuration_impl, **kwargs)

def _test_native_configuration_impl(env, target):
    configuration = target[CcToolchainConfigInfo]
    tools = {item.name: item.path for item in configuration.tool_paths}
    env.expect.that_str(tools["gcc"]).equals(NATIVE_ROOT + "configuration/wrapper/clang-cl.exe")
    env.expect.that_str(tools["nm"]).equals(NATIVE_ROOT + "configuration/wrapper/msvc_nop.bat")
    env.expect.that_collection(configuration.cxx_builtin_include_directories).contains("C:/LLVM/lib/clang/23/share")

def _test_native_command(name, **kwargs):
    _command(
        name = name + "_subject",
        compiler = ":native_wrapped_compiler",
        copts = ["/O1"],
        profile = ["dbg", "static_link_msvcrt_release", "optimize_for_speed", "sanitizer_debug_info"],
    )
    analysis_test(name = name, target = name + "_subject", impl = _test_native_command_impl, **kwargs)

def _test_native_command_impl(env, target):
    command = target[_CommandInfo]
    env.expect.that_str(command.compiler).equals(NATIVE_ROOT + "configuration/wrapper/clang-cl.exe")
    for flag in ["/MT", "/Z7", "/clang:-O3", "/O1"]:
        env.expect.that_collection(command.arguments).contains(flag)
    env.expect.that_collection(command.arguments).not_contains("/MTd")
    env.expect.that_bool(command.explicit_exports).equals(False)
    if command.arguments.index("/clang:-O3") > command.arguments.index("/O1"):
        env.fail("native toolchain profile overrides explicit caller optimization")

def _test_native_compiler_mismatch(name, **kwargs):
    analysis_test(name = name, target = ":wrong_compiler_binding", impl = _test_native_compiler_mismatch_impl, expect_failure = True, **kwargs)

def _test_native_compiler_mismatch_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(matching.contains("does not match the selected C/C++ compiler"))

def _test_native_linker_inputs(name, **kwargs):
    analysis_test(
        name = name,
        target = "//build_tools/windows:compiler_runtime_test",
        impl = _test_native_linker_inputs_impl,
        config_settings = {str(Label("//build_tools/windows:compiler")): "clang-cl"},
        **kwargs
    )

def _test_native_linker_inputs_impl(env, target):
    link = [action for action in target[TestingAspectInfo].actions if action.mnemonic == "CppLink"][0]
    builtins = [file for file in link.inputs.to_list() if file.basename == "clang_rt.builtins-x86_64.lib"]
    env.expect.that_int(len(builtins)).equals(1)
    if builtins:
        env.expect.that_collection(link.argv).contains(builtins[0].path)

def toolchain_config_test_suite(name):
    tests = [_test_optimized, _test_sanitizer, _test_static_release_crt, _test_static_debug_crt, _test_native_configuration, _test_native_command, _test_native_compiler_mismatch]
    if "@platforms//os:windows" in HOST_CONSTRAINTS:
        tests.append(_test_native_linker_inputs)
    test_suite(
        name = name,
        tests = tests,
    )
