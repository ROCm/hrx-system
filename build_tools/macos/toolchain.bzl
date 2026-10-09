# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""macOS destinations and C/C++/Objective-C actions for native and Linux hosts."""

load("@platforms//host:constraints.bzl", "HOST_CONSTRAINTS")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_cc//cc/toolchains:args.bzl", "cc_args")
load("@rules_cc//cc/toolchains:artifacts.bzl", "cc_artifact_name_pattern")
load("@rules_cc//cc/toolchains:feature.bzl", "cc_feature")
load("@rules_cc//cc/toolchains:tool.bzl", "cc_tool")
load("@rules_cc//cc/toolchains:tool_map.bzl", "cc_tool_map")
load("@rules_cc//cc/toolchains:toolchain.bzl", "cc_toolchain")

_ARCHITECTURES = {"arm64": "aarch64", "x86_64": "x86_64"}
_ACTIONS = "@rules_cc//cc/toolchains/actions:"
_STANDARD_FEATURES = "@rules_cc//cc/toolchains/args:experimental_replace_legacy_action_config_features"

def _address_sanitizer_runtime_impl(ctx):
    if len(ctx.files.runtime) != 1:
        fail("macOS AddressSanitizer requires matching libclang_rt.asan_osx_dynamic.dylib. Native builds use Xcode; cross builds require MACOS_COMPILER_RT_ROOT pointing to the matching LLVM lib/clang/<version>/lib/darwin directory. See BUILDING.md.")
    runtime = ctx.files.runtime[0]
    cc = find_cc_toolchain(ctx)
    configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    library = cc_common.create_library_to_link(
        actions = ctx.actions,
        cc_toolchain = cc,
        feature_configuration = configuration,
        dynamic_library = runtime,
    )
    linker_input = cc_common.create_linker_input(owner = ctx.label, libraries = depset([library]))
    return [
        CcInfo(linking_context = cc_common.create_linking_context(linker_inputs = depset([linker_input]))),
        DefaultInfo(runfiles = ctx.runfiles(files = [runtime])),
    ]

# cc_import would itself consume the runtimes toolchain, forming a cycle.
_address_sanitizer_runtime = rule(
    implementation = _address_sanitizer_runtime_impl,
    attrs = {"runtime": attr.label(allow_files = True)},
    fragments = ["cpp"],
    toolchains = use_cc_toolchain(),
)

def _runtimes_toolchain_impl(ctx):
    return [platform_common.ToolchainInfo(cc_runtimes_info = struct(
        # Compiler-owned dependencies and their execution runfiles.
        runtimes = ctx.attr.runtimes,
        # Instrumentation flags belong to the compiler's sanitizer feature.
        copts = [],
    ))]

_runtimes_toolchain = rule(
    implementation = _runtimes_toolchain_impl,
    attrs = {"runtimes": attr.label_list(providers = [CcInfo])},
)

# This package has one toolchain set selected by the local host platform.
# buildifier: disable=unnamed-macro
def macos_toolchains():
    """Declares destinations without configuring an unused macOS SDK."""

    # The host repository spells the macOS constraint with its osx alias.
    supported_host = "@platforms//os:linux" in HOST_CONSTRAINTS or "@platforms//os:osx" in HOST_CONSTRAINTS
    for architecture, constraint in _ARCHITECTURES.items():
        constraints = ["@platforms//os:macos", "@platforms//cpu:" + constraint]
        native.platform(name = "macos_" + architecture, constraint_values = constraints)
        if supported_host:
            native.toolchain(
                name = "cc_" + architecture,
                exec_compatible_with = HOST_CONSTRAINTS,
                target_compatible_with = constraints,
                toolchain = "@iree_macos_toolchain//:cc_" + architecture,
                toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
            )
    if supported_host:
        _address_sanitizer_runtime(name = "address_sanitizer_runtime", runtime = "@iree_macos_toolchain//:address_sanitizer_files", tags = ["manual"])
        _runtimes_toolchain(name = "address_sanitizer_runtimes", runtimes = [":address_sanitizer_runtime"], tags = ["manual"])
        native.toolchain(
            name = "address_sanitizer",
            exec_compatible_with = HOST_CONSTRAINTS,
            target_compatible_with = ["@platforms//os:macos"],
            target_settings = ["//build_tools/bazel:address_sanitizer_target"],
            toolchain = ":address_sanitizer_runtimes",
            toolchain_type = "@bazel_tools//tools/cpp:cc_runtimes_toolchain_type",
        )

def macos_cc_toolchains(name, repository_path, metadata, minimum_os, cross):
    """Defines the selected compiler, SDK dependencies, and macOS architectures.

    Args:
      name: Prefix for shared compiler tools and arguments.
      repository_path: Stable execroot-relative repository path.
      metadata: Selected SDK header inventories and linker input closures.
      minimum_os: Minimum supported macOS deployment version.
      cross: Whether tools execute on Linux instead of macOS.
    """
    native.filegroup(
        name = "host_dynamic_libraries",
        srcs = native.glob(["host_dynamic_libraries/*"], allow_empty = True),
    )
    for tool in ["clang", "clang++", "libtool", "strip"]:
        cc_tool(
            name = name + "_" + tool,
            src = "tools/" + tool,
            data = ["bin/" + tool, ":host_dynamic_libraries"] + (["bin/ld64.lld" if cross else "bin/ld"] if tool == "clang++" else []),
            capabilities = {
                "clang": ["@rules_cc//cc/toolchains/capabilities:supports_pic"],
                "clang++": ["@rules_cc//cc/toolchains/capabilities:supports_dynamic_linker"],
            }.get(tool, []),
        )
    cc_tool_map(
        name = name + "_tools",
        tools = {
            _ACTIONS + "compile_actions": ":" + name + "_clang",
            _ACTIONS + "link_actions": ":" + name + "_clang++",
            _ACTIONS + "ar_actions": ":" + name + "_libtool",
            _ACTIONS + "strip": ":" + name + "_strip",
        },
    )
    cc_args(
        name = "sdk",
        actions = [_ACTIONS + "compile_actions", _ACTIONS + "link_actions"],
        args = (["--no-default-config"] if cross else []) + [
            "-isysroot",
            repository_path + "/sysroot",
            "-resource-dir=" + repository_path + "/resource",
            "-no-canonical-prefixes",
        ],
    )
    native.config_setting(name = "opt", values = {"compilation_mode": "opt"})
    cc_args(
        name = "compile",
        actions = [_ACTIONS + "compile_actions"],
        args = ["-g", "-fdebug-compilation-dir=.", "-fblocks"] + select({
            ":opt": ["-O2", "-DNDEBUG"],
            "//conditions:default": ["-O0"],
        }),
        data = ["sysroot/" + path for path in metadata["c_headers"]] + native.glob(["resource/include/**"]),
    )
    cc_args(
        name = "libcxx",
        actions = [_ACTIONS + "cpp_compile_actions"],
        args = ["-std=c++20", "-nostdinc++", "-isystem", repository_path + "/cxx"],
        data = ["cxx/" + path for path in metadata["cxx_headers"]],
    )
    cc_args(
        name = "link",
        actions = [_ACTIONS + "link_actions"],
        args = (["-fuse-ld=lld"] if cross else []) + [
            "-B" + repository_path + "/bin",
            "-Wl,-oso_prefix,.",
            "-Wl,-rpath,@loader_path",
        ],
        data = native.glob(["resource/lib/darwin/libclang_rt.osx.a"], allow_empty = True),
    )

    # cc_library links its own objects into a dylib without its dependencies.
    # The final consumer supplies those imports; final links remain strict.
    cc_args(
        name = "nodeps_link",
        actions = [_ACTIONS + "nodeps_dynamic_library_link_actions"],
        args = ["-Wl,-undefined,dynamic_lookup"],
    )
    cc_artifact_name_pattern(
        name = "dylib",
        category = "@rules_cc//cc/toolchains/artifacts:dynamic_library",
        prefix = "lib",
        extension = ".dylib",
    )
    native.filegroup(
        name = "address_sanitizer_files",
        srcs = native.glob(["resource/lib/darwin/libclang_rt.asan_osx_dynamic.dylib"], allow_empty = True),
    )
    cc_args(
        name = "sanitizer_debug",
        actions = [_ACTIONS + "compile_actions"],
        args = ["-g"],
    )
    cc_feature(name = "sanitizer_debug_info", feature_name = "sanitizer_debug_info", args = [":sanitizer_debug"])
    cc_args(
        name = "speed_optimization",
        actions = [_ACTIONS + "compile_actions"],
        args = ["-O3"],
        requires_any_of = [":optimize_for_speed"],
    )
    cc_feature(name = "optimize_for_speed", feature_name = "optimize_for_speed")
    cc_args(
        name = "thin_lto_compile_link",
        actions = [_ACTIONS + "compile_actions", _ACTIONS + "link_actions"],
        args = ["-flto=thin"],
        requires_any_of = [":thin_lto_link"],
    )
    cc_feature(name = "thin_lto_link", feature_name = "thin_lto_link")
    cc_args(
        name = "asan_compile",
        actions = [_ACTIONS + "compile_actions"],
        args = ["-fsanitize=address", "-fno-omit-frame-pointer"],
    )
    cc_args(
        name = "asan_link",
        actions = [_ACTIONS + "link_actions"],
        args = ["-fsanitize=address"],
        data = select({
            Label("//build_tools/bazel:address_sanitizer_target"): [":address_sanitizer_files"],
            "//conditions:default": [],
        }),
    )
    cc_feature(name = "asan", feature_name = "asan", args = [":asan_compile", ":asan_link"])
    for architecture in _ARCHITECTURES:
        cc_args(
            name = "target_" + architecture,
            actions = [_ACTIONS + "compile_actions", _ACTIONS + "link_actions"],
            args = ["--target=" + architecture + "-apple-macos" + minimum_os],
        )
        cc_args(
            name = "libraries_" + architecture,
            actions = [_ACTIONS + "link_actions"],
            data = ["sysroot/" + path for path in metadata["libraries"][architecture][""]],
        )
        cc_toolchain(
            name = "cc_" + architecture,
            compiler = "clang",
            tool_map = ":" + name + "_tools",
            args = [":target_" + architecture, ":sdk", ":compile", ":speed_optimization", ":thin_lto_compile_link", ":libcxx", ":link", ":nodeps_link", ":libraries_" + architecture],
            artifact_name_patterns = [":dylib"],
            enabled_features = [_STANDARD_FEATURES],
            known_features = [_STANDARD_FEATURES, ":asan", ":sanitizer_debug_info", ":optimize_for_speed", ":thin_lto_link"],
            supports_param_files = True,
        )
    for framework, headers in metadata["framework_headers"].items():
        cc_library(
            name = framework,
            hdrs = ["sysroot/" + path for path in headers],
            additional_linker_inputs = select({
                "@platforms//cpu:" + constraint: ["sysroot/" + path for path in metadata["libraries"][architecture][framework]]
                for architecture, constraint in _ARCHITECTURES.items()
            }),
            linkopts = ["-framework", framework],
            target_compatible_with = ["@platforms//os:macos"],
        )
