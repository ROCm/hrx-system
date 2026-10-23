# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Windows toolchain features owned by the exporting HRX module."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load("@rules_cc//cc:cc_toolchain_config_lib.bzl", "action_config", "feature", "flag_group", "flag_set", "tool", "tool_path", "with_feature_set")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

# buildifier: disable=bzl-visibility
load("@rules_cc//cc/private/toolchain:windows_cc_toolchain_config.bzl", "all_compile_actions", "all_link_actions", _cc_toolchain_config = "cc_toolchain_config")
load("@rules_cc//cc/toolchains:cc_toolchain_config_info.bzl", "CcToolchainConfigInfo")
load(":native_runtime.bzl", "NativeWindowsRuntimeInfo")

def _static_crt_feature():
    flag_sets = []
    for actions, release, debug in [
        ([ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile], "/MT", "/MTd"),
        (all_link_actions, "/DEFAULTLIB:libcmt.lib", "/DEFAULTLIB:libcmtd.lib"),
    ]:
        flag_sets.extend([
            flag_set(
                actions = actions,
                flag_groups = [flag_group(flags = [release])],
                with_features = [
                    with_feature_set(not_features = ["dbg"]),
                    with_feature_set(features = ["static_link_msvcrt_release"]),
                ],
            ),
            flag_set(
                actions = actions,
                flag_groups = [flag_group(flags = [debug])],
                with_features = [with_feature_set(
                    features = ["dbg"],
                    not_features = ["static_link_msvcrt_release"],
                )],
            ),
        ])
    return feature(name = "static_link_msvcrt", flag_sets = flag_sets)

def _native_tool_path(path, directory):
    # Windows drive paths are absolute even when analysis runs on Linux.
    if paths.is_absolute(path) or (len(path) > 1 and path[1] == ":"):
        return path
    return paths.normalize(paths.join(directory, path))

def _configuration_impl(ctx):
    base = ctx.attr.base[CcToolchainConfigInfo]
    if "thin_lto_link" in ctx.features and base.compiler != "clang-cl":
        fail("The optimized ThinLTO profile requires clang-cl on Windows. Select windows-clang-cl, or use -c opt for the selected compiler without ThinLTO.")
    names = {item.name: True for item in base._features_DO_NOT_USE}
    additions = [
        feature(
            name = "optimize_for_speed",
            flag_sets = [flag_set(
                actions = all_compile_actions,
                flag_groups = [flag_group(flags = ["/clang:-O3"] if base.compiler == "clang-cl" else ["/O2"])],
            )],
        ),
    ]
    if base.compiler == "clang-cl":
        additions.append(feature(
            name = "thin_lto_link",
            # COFF ThinLTO is performed by lld-link, not ELF index/backend actions.
            flag_sets = [flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["/clang:-flto=thin"])],
            )],
        ))
    additions.extend([
        feature(
            name = "sanitizer_debug_info",
            flag_sets = [flag_set(
                actions = all_compile_actions,
                flag_groups = [flag_group(flags = ["/Z7"])],
            )],
        ),
        feature(name = "static_link_msvcrt_release", implies = ["static_link_msvcrt"]),
        # The base configuration already expanded legacy features and actions.
        feature(name = "no_legacy_features"),
    ])
    replacements = {}
    if ctx.attr.cross_compile:
        # Bazel's automatic DLL export helper is a Windows executable. A Linux
        # cross toolchain uses explicit exports and cannot execute that helper.
        replacements["no_windows_export_all_symbols"] = feature(name = "no_windows_export_all_symbols", enabled = True)
    if "static_link_msvcrt_release" not in names:
        replacements["static_link_msvcrt"] = _static_crt_feature()
    features = [replacements.get(item.name, item) for item in base._features_DO_NOT_USE]
    position = [item.name for item in features].index("user_compile_flags")
    features = features[:position] + [item for item in additions if item.name not in names] + features[position:]

    action_configs = base._action_configs_DO_NOT_USE
    tool_paths = base.tool_paths
    include_directories = base.cxx_builtin_include_directories
    if ctx.attr.native_runtime:
        runtime = ctx.attr.native_runtime[NativeWindowsRuntimeInfo]
        if runtime.builtins:
            # Builtins belong to this compiler, not a separately resolved
            # runtime toolchain that could pair with a consumer's own compiler.
            features.append(feature(
                name = "compiler_builtins",
                enabled = True,
                flag_sets = [flag_set(
                    actions = all_link_actions,
                    flag_groups = [flag_group(flags = [runtime.builtins.path])],
                )],
            ))

        # Relative helper paths belong to the original configuration's package,
        # not to this module's package. Native tools already use absolute paths;
        # resolve helpers likewise rather than introducing forbidden '..' paths.
        tool_paths = [tool_path(
            name = item.name,
            path = runtime.nm_executable if item.name == "nm" and runtime.nm_executable else _native_tool_path(item.path, runtime.configuration_directory),
        ) for item in tool_paths]
        action_configs = [action_config(
            action_name = action.action_name,
            enabled = action.enabled,
            flag_sets = action.flag_sets,
            implies = action.implies,
            tools = [tool(
                path = _native_tool_path(item.path, runtime.configuration_directory),
                execution_requirements = item.execution_requirements,
                with_features = item.with_features,
            ) if item.path else item for item in action.tools],
        ) for action in action_configs]
        if runtime.resource_directory:
            # Clang records its implicit sanitizer policy as an absolute input.
            # The file is also declared in compiler_files for content tracking.
            include_directories = include_directories + [runtime.resource_directory + "/share"]

    # Keep the provider representation boundary here. Consumer rules and their
    # CcInfo/toolchain identities still belong to the selected rules_cc module.
    optional = {}
    if hasattr(base, "disallowed_copts_infos"):
        optional["disallowed_copts_infos"] = base.disallowed_copts_infos
    return [cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        action_configs = action_configs,
        artifact_name_patterns = base._artifact_name_patterns_DO_NOT_USE,
        toolchain_identifier = base.toolchain_id,
        compiler = base.compiler,
        target_cpu = base.target_cpu,
        target_libc = base.target_libc,
        target_system_name = base.target_system_name,
        abi_version = base.abi_version,
        abi_libc_version = base.abi_libc_version,
        cxx_builtin_include_directories = include_directories,
        tool_paths = tool_paths,
        make_variables = base.make_variables,
        builtin_sysroot = base.builtin_sysroot,
        **optional
    )]

_configuration = rule(
    implementation = _configuration_impl,
    attrs = {
        "base": attr.label(mandatory = True, providers = [CcToolchainConfigInfo]),
        "cross_compile": attr.bool(default = False),
        "native_runtime": attr.label(providers = [NativeWindowsRuntimeInfo]),
    },
    provides = [CcToolchainConfigInfo],
)

def windows_cross_toolchain_config(name, **kwargs):
    """Defines a clang-cl cross configuration without root-only rules_cc patches.

    Args:
      name: Name of the composed toolchain configuration target.
      **kwargs: Attributes of rules_cc's Windows cc_toolchain_config rule.
    """
    _cc_toolchain_config(name = name + "_base", **kwargs)
    _configuration(
        name = name,
        base = ":" + name + "_base",
        cross_compile = True,
        **{key: kwargs[key] for key in ["visibility", "tags", "testonly", "target_compatible_with"] if key in kwargs}
    )

windows_native_toolchain_config = _configuration
