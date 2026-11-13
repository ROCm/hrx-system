# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C/C++ toolchain backed by the pinned WASI SDK repository."""

load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@platforms//host:constraints.bzl", "HOST_CONSTRAINTS")
load("@rules_cc//cc/toolchains:args.bzl", "cc_args")
load("@rules_cc//cc/toolchains:artifacts.bzl", "cc_artifact_name_pattern")
load("@rules_cc//cc/toolchains:feature.bzl", "cc_feature")
load("@rules_cc//cc/toolchains:tool.bzl", "cc_tool")
load("@rules_cc//cc/toolchains:tool_map.bzl", "cc_tool_map")
load("@rules_cc//cc/toolchains:toolchain.bzl", "cc_toolchain")
load("//build_tools/wasm:wasi_sdk_version.bzl", "WASI_SDK_CLANG_RESOURCE_VERSION")

_ACTIONS = "@rules_cc//cc/toolchains/actions:"
_STANDARD_FEATURES = "@rules_cc//cc/toolchains/args:experimental_replace_legacy_action_config_features"
_CLANG_RESOURCE_PATH = "lib/clang/" + WASI_SDK_CLANG_RESOURCE_VERSION

def _copy_tool(name, source_name, executable_suffix, source_suffix, allow_symlink):
    copy_file(
        name = name,
        allow_symlink = allow_symlink,
        is_executable = True,
        out = "tools/bin/{}{}".format(name, executable_suffix),
        src = "bin/{}{}".format(source_name, source_suffix),
    )

def wasi_cc_toolchain(
        name,
        repository_path,
        executable_suffix = "",
        source_suffix = "",
        allow_symlink = True):
    """Defines a wasm32-wasip1 C/C++ toolchain from a WASI SDK archive.

    Args:
      name: Registered C/C++ toolchain target name.
      repository_path: Stable execroot-relative path of the SDK repository.
      executable_suffix: Host executable suffix for copied tool outputs.
      source_suffix: Suffix applied to archive tools renamed before extraction.
      allow_symlink: Whether copied tool outputs may symlink their inputs.
    """
    _copy_tool("clang", "clang", executable_suffix, source_suffix, allow_symlink)
    _copy_tool("clang++", "clang++", executable_suffix, source_suffix, allow_symlink)
    _copy_tool("llvm-ar", "llvm-ar", executable_suffix, source_suffix, allow_symlink)
    _copy_tool("llvm-objdump", "llvm-objdump", executable_suffix, source_suffix, allow_symlink)
    _copy_tool("llvm-strip", "llvm-strip", executable_suffix, source_suffix, allow_symlink)
    _copy_tool("wasm-ld", "wasm-ld", executable_suffix, source_suffix, allow_symlink)

    # Tools load SDK-native libraries relative to their executable location.
    # Preserve the SDK's bin/lib geometry when Bazel materializes copied tools
    # as regular files instead of resolving them through repository symlinks.
    runtime_files = []
    for source in native.glob([
        "bin/*.dll",
        "lib/*.dylib",
        "lib/*.so*",
    ], allow_empty = True):
        runtime_name = "tool_runtime_" + source.replace("/", "_")
        copy_file(
            name = runtime_name,
            allow_symlink = allow_symlink,
            out = "tools/" + source,
            src = source,
        )
        runtime_files.append(":" + runtime_name)
    native.filegroup(
        name = "tool_runtime_files",
        srcs = runtime_files,
    )

    native.filegroup(
        name = "compile_files",
        srcs = native.glob([
            _CLANG_RESOURCE_PATH + "/include/**",
            "share/wasi-sysroot/include/wasm32-wasip1/**",
        ]),
    )
    native.filegroup(
        name = "link_files",
        srcs = native.glob([
            _CLANG_RESOURCE_PATH + "/lib/wasm32-unknown-wasip1/**",
            "share/wasi-sysroot/lib/wasm32-wasip1/**",
        ]),
    )
    cc_tool(
        name = "c_compiler",
        src = ":clang",
        data = [
            ":compile_files",
            ":tool_runtime_files",
        ],
    )
    cc_tool(
        name = "cxx_linker",
        src = ":clang++",
        data = [
            ":link_files",
            ":tool_runtime_files",
            ":wasm-ld",
        ],
    )
    cc_tool(
        name = "archiver",
        src = ":llvm-ar",
        data = [":tool_runtime_files"],
    )
    cc_tool(
        name = "stripper",
        src = ":llvm-strip",
        data = [":tool_runtime_files"],
    )
    cc_tool_map(
        name = "tools",
        tools = {
            _ACTIONS + "compile_actions": ":c_compiler",
            _ACTIONS + "link_actions": ":cxx_linker",
            _ACTIONS + "ar_actions": ":archiver",
            _ACTIONS + "strip": ":stripper",
        },
    )

    cc_args(
        name = "target",
        actions = [
            _ACTIONS + "compile_actions",
            _ACTIONS + "link_actions",
        ],
        args = [
            "--target=wasm32-wasip1",
            "--sysroot=" + repository_path + "/share/wasi-sysroot",
            "-resource-dir=" + repository_path + "/" + _CLANG_RESOURCE_PATH,
            "-no-canonical-prefixes",
        ],
    )
    native.config_setting(
        name = "opt",
        values = {"compilation_mode": "opt"},
    )

    cc_args(
        name = "compile",
        actions = [_ACTIONS + "compile_actions"],
        args = [
            "-g",
            "-fdebug-compilation-dir=.",
        ] + select({
            ":opt": [
                "-O2",
                "-DNDEBUG",
            ],
            "//conditions:default": ["-O0"],
        }),
    )

    # The SDK's generic C++ include directory is empty, so Bazel cannot place
    # it in a sandbox. Name the populated target directory explicitly instead
    # of relying on Clang to discover it through that empty parent.
    cc_args(
        name = "cxx",
        actions = [_ACTIONS + "cpp_compile_actions"],
        args = [
            "-isystem",
            repository_path + "/share/wasi-sysroot/include/wasm32-wasip1/c++/v1",
        ],
    )
    cc_args(
        name = "link",
        actions = [_ACTIONS + "link_actions"],
        args = [
            "-Wl,--export-memory",
            # Keep the downward-growing stack below static data so exhaustion
            # traps at address zero instead of overwriting mutable globals.
            "-Wl,--stack-first",
            # Hosted compilers need room for recursive IR rewrites. LLD's
            # 64 KiB default exhausts on ordinary vector arithmetic graphs.
            "-Wl,-z,stack-size=1048576",
        ],
    )
    cc_artifact_name_pattern(
        name = "wasm_executable",
        category = "@rules_cc//cc/toolchains/artifacts:executable",
        extension = ".wasm",
        prefix = "",
    )
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
    cc_toolchain(
        name = "cc",
        args = [
            ":target",
            ":compile",
            ":speed_optimization",
            ":thin_lto_compile_link",
            ":cxx",
            ":link",
        ],
        artifact_name_patterns = [":wasm_executable"],
        compiler = "clang",
        enabled_features = [_STANDARD_FEATURES],
        known_features = [_STANDARD_FEATURES, ":optimize_for_speed", ":thin_lto_link"],
        supports_param_files = True,
        tool_map = ":tools",
    )
    native.toolchain(
        name = name,
        exec_compatible_with = HOST_CONSTRAINTS,
        target_compatible_with = [
            "@platforms//cpu:wasm32",
            "@platforms//os:wasi",
        ],
        toolchain = ":cc",
        toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
    )
