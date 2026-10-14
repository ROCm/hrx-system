# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for shared C/C++ Bazel macros."""

load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_cc//cc/common:debug_package_info.bzl", "DebugPackageInfo")
load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:truth.bzl", "matching")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo", "util")
load("//build_tools/bazel:cc.bzl", "iree_cc_binary", "iree_cc_library")
load("//build_tools/bazel:cc_test.bzl", "iree_cc_test")
load(
    "//build_tools/bazel:runfiles.bzl",
    "IreeRunfilesArgumentsInfo",
    "IreeRunfilesEnvironmentInfo",
    "RUNFILES_PATH_BEGIN",
    "RUNFILES_PATH_END",
)

_TEST_DYNAMIC_LIBRARY_ENVIRONMENT = "IREE_TEST_DYNAMIC_LIBRARY_PATH"

def _execution_platform_runner(ctx, binary_info, processed_environment, runner_name):
    return [
        DefaultInfo(
            executable = binary_info.executable,
            files = binary_info.files,
            runfiles = binary_info.runfiles,
        ),
        RunEnvironmentInfo(
            environment = processed_environment | {"IREE_TEST_RUNNER": runner_name},
            inherited_environment = ctx.attr.env_inherit,
        ),
    ]

def _execution_platform_runner_impl(ctx):
    return [platform_common.ToolchainInfo(cc_test_info = struct(
        # Callback and immutable arguments consumed by the rules_cc test rule.
        get_runner = struct(
            # Identifies the selected platform through the runtime environment.
            args = {"runner_name": ctx.label.name},
            # Preserves the compiled executable and its declared runfiles.
            func = _execution_platform_runner,
        ),
        # This runner adds no linker flags to the test binary.
        linkopts = [],
        # Preserve the shared test macro's static-link policy.
        linkstatic = True,
    ))]

# The optional C++ runner makes the selected execution platform observable
# through the production cc_test rule's normal RunEnvironmentInfo provider.
execution_platform_runner = rule(implementation = _execution_platform_runner_impl)

def _all_compilation_paths(compilation_context):
    return [
        str(path)
        for path in (
            compilation_context.includes.to_list() +
            compilation_context.quote_includes.to_list() +
            compilation_context.system_includes.to_list()
        )
    ]

def _expect_path_suffix(env, paths, suffix):
    for path in paths:
        if path.endswith(suffix):
            return
    env.fail("expected one of %s to end with %r" % (paths, suffix))

def _expect_value(env, values, expected_value):
    if expected_value not in values:
        env.fail("expected %r in %r" % (expected_value, values))

def _expect_file_basename(env, files, expected_basename):
    for file in files:
        if file.basename == expected_basename:
            return
    env.fail("expected basename %r in %r" % (expected_basename, files))

def _expect_cc_executable_providers(env, target):
    expected_providers = [
        (CcInfo, "CcInfo"),
        (InstrumentedFilesInfo, "InstrumentedFilesInfo"),
        (DebugPackageInfo, "DebugPackageInfo"),
        (cc_common.launcher_provider, "CcLauncherInfo"),
        (OutputGroupInfo, "OutputGroupInfo"),
        (DefaultInfo, "DefaultInfo"),
        (RunEnvironmentInfo, "RunEnvironmentInfo"),
    ]
    for provider, name in expected_providers:
        if provider not in target:
            env.fail("expected C/C++ executable provider %s" % name)

def _test_cc_library_preserves_system_include_inputs(name, **kwargs):
    util.helper_target(
        iree_cc_library,
        name = name + "_subject",
        hdrs = [name + "_subject.h"],
        includes = ["public_include"],
        system_includes = ["system_include"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_library_preserves_system_include_inputs_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_library_preserves_system_include_inputs_impl(env, target):
    paths = _all_compilation_paths(target[CcInfo].compilation_context)
    _expect_path_suffix(env, paths, "public_include")
    _expect_path_suffix(env, paths, "system_include")

def _test_cc_library_preserves_language_compile_options(name, **kwargs):
    util.helper_target(
        iree_cc_library,
        name = name + "_subject",
        conlyopts = ["-DUSER_CONLYOPT"] + select({
            "//conditions:default": ["-DUSER_SELECTED_CONLYOPT"],
        }),
        cxxopts = ["-DUSER_CXXOPT"] + select({
            "//conditions:default": ["-DUSER_SELECTED_CXXOPT"],
        }),
        srcs = [
            name + "_subject.c",
            name + "_subject.cc",
        ],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_library_preserves_language_compile_options_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_library_preserves_language_compile_options_impl(env, target):
    attrs = target[TestingAspectInfo].attrs
    _expect_value(env, attrs.conlyopts, "-DUSER_CONLYOPT")
    _expect_value(env, attrs.conlyopts, "-DUSER_SELECTED_CONLYOPT")
    _expect_value(env, attrs.cxxopts, "-DUSER_CXXOPT")
    _expect_value(env, attrs.cxxopts, "-DUSER_SELECTED_CXXOPT")

def _test_cc_binary_preserves_system_include_inputs(name, **kwargs):
    util.helper_target(
        iree_cc_binary,
        name = name + "_subject",
        srcs = [name + "_subject.cc"],
        includes = ["binary_include"],
        system_includes = ["binary_system_include"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_binary_preserves_system_include_inputs_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_binary_preserves_system_include_inputs_impl(env, target):
    paths = _all_compilation_paths(target[CcInfo].compilation_context)
    _expect_path_suffix(env, paths, "binary_include")
    _expect_path_suffix(env, paths, "binary_system_include")

def _test_cc_binary_preserves_shared_library_mode(name, **kwargs):
    util.helper_target(
        iree_cc_binary,
        name = name + "_subject",
        deps = [":dynamic_library_environment_library"],
        linkshared = True,
        srcs = ["dynamic_library_environment_test.c"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_binary_preserves_shared_library_mode_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_binary_preserves_shared_library_mode_impl(env, target):
    attrs = target[TestingAspectInfo].attrs
    if not attrs.linkshared:
        env.fail("expected shared-library binary mode")
    library_path = target[RunEnvironmentInfo].environment[_TEST_DYNAMIC_LIBRARY_ENVIRONMENT]
    if not library_path.endswith("dynamic_library_root.so"):
        env.fail("unexpected shared-library dynamic-library path %r" % library_path)

def _test_cc_binary_links_statically_by_default(name, **kwargs):
    util.helper_target(
        iree_cc_binary,
        name = name + "_subject",
        srcs = [name + "_subject.cc"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_binary_links_statically_by_default_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_binary_links_statically_by_default_impl(env, target):
    attrs = target[TestingAspectInfo].attrs
    if not attrs.linkstatic:
        env.fail("expected C/C++ binaries to link statically by default")

def _test_cc_binary_preserves_rules_cc_providers(name, **kwargs):
    util.helper_target(
        iree_cc_binary,
        name = name + "_subject",
        args = ["--binary-argument"],
        env = {"IREE_TEST_BINARY_ENV": "binary"},
        srcs = [name + "_subject.cc"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {"timeout": "short"},
        impl = _test_cc_binary_preserves_rules_cc_providers_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_binary_preserves_rules_cc_providers_impl(env, target):
    _expect_cc_executable_providers(env, target)
    attrs = target[TestingAspectInfo].attrs
    if attrs.args != ["--binary-argument"]:
        env.fail("expected rules_cc binary arguments, got %r" % attrs.args)
    env.expect.that_str(
        target[RunEnvironmentInfo].environment["IREE_TEST_BINARY_ENV"],
    ).equals("binary")
    if _TEST_DYNAMIC_LIBRARY_ENVIRONMENT in target[RunEnvironmentInfo].environment:
        env.fail("binary without a bundle received a dynamic-library environment")
    if IreeRunfilesEnvironmentInfo in target:
        env.fail("binary without a bundle received runfiles environment metadata")
    if IreeRunfilesArgumentsInfo in target:
        env.fail("binary without runfile arguments received argument metadata")

def _test_cc_binary_injects_dynamic_library_bindings(name, **kwargs):
    util.helper_target(
        iree_cc_binary,
        name = name + "_subject",
        data = [":generate_rule_fixture.data"],
        deps = [":dynamic_library_environment_library"],
        args = ["--fixture=$(rootpath :generate_rule_fixture.data)"],
        srcs = [name + "_subject.cc"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {"timeout": "short"},
        impl = _test_cc_binary_injects_dynamic_library_bindings_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_binary_injects_dynamic_library_bindings_impl(env, target):
    _expect_cc_execution_dynamic_library_bindings(env, target)

def _test_cc_binary_describes_runfile_arguments(name, **kwargs):
    util.helper_target(
        iree_cc_binary,
        name = name + "_subject",
        args = ["--fixture=$(rootpath :generate_rule_fixture.data)"],
        data = [":generate_rule_fixture.data"],
        srcs = [name + "_subject.cc"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {"timeout": "short"},
        impl = _test_cc_binary_describes_runfile_arguments_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_binary_describes_runfile_arguments_impl(env, target):
    _expect_runfiles_arguments(env, target)
    if IreeRunfilesEnvironmentInfo in target:
        env.fail("binary without a bundle received runfiles environment metadata")

def _test_cc_test_preserves_system_include_inputs(name, **kwargs):
    util.helper_target(
        iree_cc_test,
        name = name + "_subject",
        srcs = [name + "_subject.cc"],
        includes = ["test_include"],
        system_includes = ["test_system_include"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_test_preserves_system_include_inputs_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_test_preserves_system_include_inputs_impl(env, target):
    paths = _all_compilation_paths(target[CcInfo].compilation_context)
    _expect_path_suffix(env, paths, "test_include")
    _expect_path_suffix(env, paths, "test_system_include")

def _test_cc_test_applies_resource_group_tags(name, **kwargs):
    util.helper_target(
        iree_cc_test,
        name = name + "_subject",
        srcs = [name + "_subject.cc"],
        resource_group = "gpu",
        tags = [
            "requires-gpu-vulkan",
            "manual",
        ],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_test_applies_resource_group_tags_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_test_applies_resource_group_tags_impl(env, target):
    tags = target[TestingAspectInfo].attrs.tags
    for expected_tag in [
        "requires-gpu-vulkan",
        "exclusive-if-local",
        "resource_group:gpu",
    ]:
        if expected_tag not in tags:
            env.fail("expected %r in test tags %r" % (expected_tag, tags))

def _test_cc_test_links_statically_by_default(name, **kwargs):
    util.helper_target(
        iree_cc_test,
        name = name + "_subject",
        srcs = [name + "_subject.cc"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_cc_test_links_statically_by_default_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_test_links_statically_by_default_impl(env, target):
    attrs = target[TestingAspectInfo].attrs
    if not attrs.linkstatic:
        env.fail("expected C/C++ tests to link statically by default")

def _test_cc_test_skips_incompatible_execution_platform(name, **kwargs):
    util.helper_target(
        iree_cc_test,
        name = name + "_subject",
        srcs = [name + "_subject.cc"],
    )
    analysis_test(
        name = name,
        config_settings = {
            "//command_line_option:extra_execution_platforms": [
                str(Label(":test_incompatible_execution_platform")),
                str(Label(":test_execution_platform")),
            ],
            "//command_line_option:extra_toolchains": [
                str(Label(":incompatible_test_runner_toolchain")),
                str(Label(":compatible_test_runner_toolchain")),
            ],
            "//command_line_option:platforms": [Label(":test_execution_platform")],
            "@bazel_tools//tools/test:incompatible_use_default_test_toolchain": True,
        },
        impl = _test_cc_test_skips_incompatible_execution_platform_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_test_skips_incompatible_execution_platform_impl(env, target):
    _expect_cc_executable_providers(env, target)
    env.expect.that_str(
        target[RunEnvironmentInfo].environment["IREE_TEST_RUNNER"],
    ).equals("compatible_test_runner")

def _test_cc_test_accepts_matching_execution_platform(name, **kwargs):
    util.helper_target(
        iree_cc_test,
        name = name + "_subject",
        srcs = [name + "_subject.cc"],
    )
    analysis_test(
        name = name,
        config_settings = {
            "//command_line_option:extra_execution_platforms": [str(Label(":test_execution_platform"))],
            "//command_line_option:platforms": [Label(":test_execution_platform")],
            "@bazel_tools//tools/test:incompatible_use_default_test_toolchain": True,
        },
        impl = _test_cc_test_preserves_execution_platform_providers_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_test_cross_builds_without_executor(name, **kwargs):
    util.helper_target(
        iree_cc_test,
        name = name + "_subject",
        srcs = [name + "_subject.cc"],
    )
    analysis_test(
        name = name,
        config_settings = {
            "//command_line_option:platforms": [Label(":test_execution_platform")],
            "@bazel_tools//tools/test:incompatible_use_default_test_toolchain": False,
        },
        impl = _test_cc_test_preserves_execution_platform_providers_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_test_preserves_execution_platform_providers_impl(env, target):
    _expect_cc_executable_providers(env, target)

def _test_cc_test_preserves_rules_cc_providers(name, **kwargs):
    util.helper_target(
        iree_cc_test,
        name = name + "_subject",
        args = ["--test-argument"],
        env = {"IREE_TEST_RULE_ENV": "test"},
        env_inherit = ["IREE_TEST_INHERITED_ENV"],
        shard_count = 2,
        srcs = [name + "_subject.cc"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {"timeout": "short"},
        impl = _test_cc_test_preserves_rules_cc_providers_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_test_preserves_rules_cc_providers_impl(env, target):
    _expect_cc_executable_providers(env, target)
    attrs = target[TestingAspectInfo].attrs
    if attrs.args != ["--test-argument"]:
        env.fail("expected rules_cc test arguments, got %r" % attrs.args)
    if attrs.shard_count != 2:
        env.fail("expected rules_cc test shard count, got %r" % attrs.shard_count)
    run_environment = target[RunEnvironmentInfo]
    env.expect.that_str(run_environment.environment["IREE_TEST_RULE_ENV"]).equals(
        "test",
    )
    if "IREE_TEST_INHERITED_ENV" not in run_environment.inherited_environment:
        env.fail(
            "expected inherited test environment, got %r" %
            run_environment.inherited_environment,
        )
    if _TEST_DYNAMIC_LIBRARY_ENVIRONMENT in target[RunEnvironmentInfo].environment:
        env.fail("test without a bundle received a dynamic-library environment")
    if IreeRunfilesEnvironmentInfo in target:
        env.fail("test without a bundle received runfiles environment metadata")
    if IreeRunfilesArgumentsInfo in target:
        env.fail("test without runfile arguments received argument metadata")

def _test_cc_test_injects_dynamic_library_bindings(name, **kwargs):
    util.helper_target(
        iree_cc_test,
        name = name + "_subject",
        data = [":generate_rule_fixture.data"],
        deps = [":dynamic_library_environment_library"],
        args = ["--fixture=$(rootpath :generate_rule_fixture.data)"],
        env = {"IREE_TEST_EXPLICIT_ENV": "explicit"},
        srcs = [name + "_subject.cc"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {"timeout": "short"},
        impl = _test_cc_test_injects_dynamic_library_bindings_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_test_injects_dynamic_library_bindings_impl(env, target):
    _expect_cc_execution_dynamic_library_bindings(env, target)
    env.expect.that_str(
        target[RunEnvironmentInfo].environment["IREE_TEST_EXPLICIT_ENV"],
    ).equals("explicit")

def _expect_cc_execution_dynamic_library_bindings(env, target):
    _expect_cc_executable_providers(env, target)

    _expect_runfiles_arguments(env, target)
    runfiles_environment = target[IreeRunfilesEnvironmentInfo].environment
    env.expect.that_str(
        runfiles_environment[_TEST_DYNAMIC_LIBRARY_ENVIRONMENT],
    ).equals(
        RUNFILES_PATH_BEGIN + target[RunEnvironmentInfo].environment[_TEST_DYNAMIC_LIBRARY_ENVIRONMENT] + RUNFILES_PATH_END,
    )
    run_environment = target[RunEnvironmentInfo].environment
    library_path = run_environment[_TEST_DYNAMIC_LIBRARY_ENVIRONMENT]
    if not library_path.endswith(
        "build_tools/bazel/test/dynamic_library_root.so",
    ):
        env.fail("unexpected dynamic-library root path %r" % library_path)
    for runfiles in [
        target[DefaultInfo].default_runfiles,
        target[DefaultInfo].data_runfiles,
    ]:
        files = runfiles.files.to_list()
        _expect_file_basename(env, files, "dynamic_library_root.so")
        _expect_file_basename(env, files, "dynamic_library_dependency.so")
        _expect_file_basename(env, files, "generate_rule_fixture.data")

def _expect_runfiles_arguments(env, target):
    runfiles_arguments = target[IreeRunfilesArgumentsInfo]
    env.expect.that_collection(runfiles_arguments.arguments).contains_exactly([
        "--fixture=build_tools/bazel/test/generate_rule_fixture.data",
    ])
    env.expect.that_collection(runfiles_arguments.marked_arguments).contains_exactly([
        "--fixture=" +
        RUNFILES_PATH_BEGIN +
        "build_tools/bazel/test/generate_rule_fixture.data" +
        RUNFILES_PATH_END,
    ])

def _test_cc_test_rejects_conflicting_dynamic_library_env(name, **kwargs):
    util.helper_target(
        iree_cc_test,
        name = name + "_subject",
        deps = [":dynamic_library_environment_library"],
        env = {_TEST_DYNAMIC_LIBRARY_ENVIRONMENT: "wrong"},
        srcs = [name + "_subject.cc"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {"timeout": "short"},
        expect_failure = True,
        impl = _test_cc_test_rejects_conflicting_dynamic_library_env_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_test_rejects_conflicting_dynamic_library_env_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(
        matching.contains(
            "sets runtime environment variable %s to both" %
            _TEST_DYNAMIC_LIBRARY_ENVIRONMENT,
        ),
    )

def _test_cc_binary_merges_dependency_suppressions(name, **kwargs):
    # A diamond reaches the same policy twice; a second runtime contributes
    # another LSan file and an independent sanitizer kind.
    iree_cc_library(
        name = name + "_first",
        sanitizer_suppressions = {"lsan": "//build_tools/sanitizer:lsan_suppressions_hsa.txt"},
        tags = ["manual"],
    )
    iree_cc_library(
        name = name + "_second",
        deps = [":" + name + "_first"],
        sanitizer_suppressions = {
            "lsan": "//build_tools/sanitizer:lsan_suppressions_vulkan.txt",
            "tsan": ":empty_suppressions.txt",
        },
        tags = ["manual"],
    )
    util.helper_target(
        iree_cc_binary,
        name = name + "_subject",
        deps = [":" + name + "_first", ":" + name + "_second"],
        srcs = [name + "_subject.cc"],
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        impl = _test_cc_binary_merges_dependency_suppressions_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_cc_binary_merges_dependency_suppressions_impl(env, target):
    options = target[RunEnvironmentInfo].environment
    env.expect.that_collection(options.keys()).contains_exactly(["LSAN_OPTIONS", "TSAN_OPTIONS"])
    env.expect.that_str(options["TSAN_OPTIONS"]).contains("empty_suppressions.txt")
    actions = [action for action in target[TestingAspectInfo].actions if action.mnemonic == "SanitizerSuppressions"]
    env.expect.that_int(len(actions)).equals(1)
    env.expect.that_collection([file.basename for file in actions[0].inputs.to_list()]).contains_exactly([
        "lsan_suppressions_hsa.txt",
        "lsan_suppressions_vulkan.txt",
    ])
    output = actions[0].outputs.to_list()[0]
    env.expect.that_str(options["LSAN_OPTIONS"]).equals('suppressions="' + output.short_path + '":allow_addr2line=1')
    env.expect.that_collection(target[DefaultInfo].default_runfiles.files.to_list()).contains(output)

def cc_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_cc_binary_merges_dependency_suppressions,
            _test_cc_library_preserves_system_include_inputs,
            _test_cc_library_preserves_language_compile_options,
            _test_cc_binary_preserves_system_include_inputs,
            _test_cc_binary_preserves_shared_library_mode,
            _test_cc_binary_links_statically_by_default,
            _test_cc_binary_preserves_rules_cc_providers,
            _test_cc_binary_injects_dynamic_library_bindings,
            _test_cc_binary_describes_runfile_arguments,
            _test_cc_test_preserves_system_include_inputs,
            _test_cc_test_applies_resource_group_tags,
            _test_cc_test_links_statically_by_default,
            _test_cc_test_skips_incompatible_execution_platform,
            _test_cc_test_accepts_matching_execution_platform,
            _test_cc_test_cross_builds_without_executor,
            _test_cc_test_preserves_rules_cc_providers,
            _test_cc_test_injects_dynamic_library_bindings,
            _test_cc_test_rejects_conflicting_dynamic_library_env,
        ],
    )
