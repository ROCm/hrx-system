# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Optional, exact-build compiler/runtime sharing for LoomC tools."""

load("@rules_cc//cc:cc_shared_library.bzl", "cc_shared_library")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//build_tools/bazel:cc_opts.bzl", "cc_opts")
load("//loom/build_tools/bazel:cc.bzl", "loom_cc_binary", "loom_cc_library")

_SHARED_TOOLS = "//loom/config:shared_tools_enabled"

def _exports_impl(ctx):
    toolchain = find_cc_toolchain(ctx)
    inputs = []
    for dependency in ctx.attr.deps:
        for linker_input in dependency[CcInfo].linking_context.linker_inputs.to_list():
            for library in linker_input.libraries:
                if library.pic_objects:
                    inputs.extend(library.pic_objects)
                elif library.objects:
                    inputs.extend(library.objects)
                elif library.pic_static_library:
                    inputs.append(library.pic_static_library)
                elif library.static_library:
                    inputs.append(library.static_library)
    inputs = depset(inputs)
    output = ctx.actions.declare_file(ctx.label.name + ".def")
    args = ctx.actions.args()
    if toolchain.compiler == "msvc-cl":
        args.add("--dumpbin", ctx.var["DUMPBIN"])
    else:
        args.add("--nm", toolchain.nm_executable)
    args.add("--output", output)
    args.add_all(ctx.attr.prefixes, before_each = "--prefix")
    args.add_all(inputs)
    args.use_param_file("@%s", use_always = True)
    args.set_param_file_format("multiline")
    ctx.actions.run(
        executable = ctx.executable._generator,
        arguments = [args],
        inputs = depset(transitive = [inputs, toolchain.all_files]),
        outputs = [output],
        mnemonic = "LoomCExports",
        progress_message = "Collecting shared exports for %{label}",
    )
    return [DefaultInfo(files = depset([output]))]

_exports = rule(
    implementation = _exports_impl,
    attrs = {
        "deps": attr.label_list(providers = [CcInfo]),
        "prefixes": attr.string_list(),
        "_generator": attr.label(
            default = "//loom/build_tools:shared_exports",
            executable = True,
            cfg = "exec",
        ),
    },
    toolchains = use_cc_toolchain(),
    fragments = ["cpp"],
)

def _loomc_cc_binary_impl(name, visibility, linkopts, **kwargs):
    loom_cc_binary(
        name = name,
        visibility = visibility,
        linkopts = (linkopts or []) + select({
            "//loom/config:shared_tools_linux": ["-Wl,-rpath,$ORIGIN", "-Wl,-rpath,$ORIGIN/../lib"],
            "//loom/config:shared_tools_macos": ["-Wl,-rpath,@loader_path", "-Wl,-rpath,@loader_path/../lib"],
            "//conditions:default": [],
        }),
        dynamic_deps = select({
            _SHARED_TOOLS: [
                "//loom/binding/c:loomc_shared",
                "//loom/binding/c:iree_shared",
            ],
            "//conditions:default": [],
        }),
        **kwargs
    )

loomc_cc_binary = macro(
    doc = "Defines a LoomC tool with static linkage unless shared tools are selected.",
    implementation = _loomc_cc_binary_impl,
    inherit_attrs = loom_cc_binary,
    attrs = {"dynamic_deps": None},
)

def _interop_headers_impl(ctx):
    info = cc_common.merge_cc_infos(cc_infos = [dep[CcInfo] for dep in ctx.attr.deps])
    if ctx.attr.shared:
        # Exact-version interop borrows native types. Their implementations stay
        # behind the compiler API when the tool uses the shared compiler.
        info = CcInfo(compilation_context = info.compilation_context)
    return [info]

_interop_headers = rule(
    implementation = _interop_headers_impl,
    attrs = {
        "deps": attr.label_list(providers = [CcInfo]),
        "shared": attr.bool(),
    },
)

def loomc_interop_headers(name, deps):
    """Provides native interop types without importing a second shared compiler."""
    _interop_headers(
        name = name,
        deps = deps,
        shared = select({_SHARED_TOOLS: True, "//conditions:default": False}),
    )

def loomc_shared_libraries(name, components, runtime_deps):
    """Composes enabled bindings and their exact-build runtime companion.

    The ordinary component targets remain the static API. Native dynamic_deps
    substitutes exported owners only for tools that select shared linkage.

    Args:
      name: Compiler shared-library target name.
      components: Enabled public binding implementations exported by the compiler.
      runtime_deps: Runtime roots shared by the compiler and its tools.
    """
    compatible_with = select({_SHARED_TOOLS: [], "//conditions:default": ["@platforms//:incompatible"]})
    loom_cc_library(
        name = "iree_shared_components",
        deps = runtime_deps,
        visibility = ["//visibility:private"],
        target_compatible_with = compatible_with,
    )
    for target_name, library_name, deps, dynamic_deps, exports_filter in [
        ("iree_shared", "iree", [":iree_shared_components"], [], [
            "//runtime/src:__subpackages__",
            "//build_tools/third_party:__subpackages__",
            "@mimalloc//:__pkg__",
        ]),
        (name, "loomc", components, [":iree_shared"], []),
    ]:
        _exports(
            name = target_name + "_exports",
            deps = deps,
            prefixes = ["iree_", "___tracy_", "mi_"] if library_name == "iree" else ["loomc_"],
            target_compatible_with = compatible_with,
        )
        export_file = "cmake/" + library_name + "_exports.lds"
        cc_shared_library(
            name = target_name,
            deps = deps,
            dynamic_deps = dynamic_deps,
            exports_filter = exports_filter,
            win_def_file = select({
                "@platforms//os:windows": ":" + target_name + "_exports",
                "//conditions:default": None,
            }),
            # Keep final artifacts separate from cc_library implicit outputs.
            shared_lib_name = select({
                "@platforms//os:macos": "shared/lib" + library_name + ".dylib",
                "@platforms//os:windows": "shared/" + library_name + ".dll",
                "//conditions:default": "shared/lib" + library_name + ".so",
            }),
            additional_linker_inputs = select({
                "@platforms//os:macos": ["cmake/" + library_name + "_exports.macos"],
                "@platforms//os:windows": [],
                "//conditions:default": [export_file],
            }),
            user_link_flags = cc_opts.iree_code_link_options() + select({
                "@platforms//os:macos": ["-Wl,-exported_symbols_list,$(location cmake/" + library_name + "_exports.macos)"],
                "@platforms//os:windows": [],
                "//conditions:default": ["-Wl,--version-script=$(location " + export_file + ")", "-Wl,-rpath,$ORIGIN"],
            }),
            target_compatible_with = compatible_with,
        )
