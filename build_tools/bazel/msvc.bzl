# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""MSVC-specific Bazel rules."""

load("@rules_cc//cc:action_names.bzl", "ASSEMBLE_ACTION_NAME")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:find_cc_toolchain.bzl", "CC_TOOLCHAIN_ATTRS", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

def _iree_msvc_masm_object_impl(ctx):
    cc_toolchain = find_cc_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    source_file = ctx.file.src

    # cc_library recognizes generated .o files as precompiled objects on every
    # platform. link.exe and lib.exe do not require a .obj suffix.
    object_file = ctx.actions.declare_file(ctx.label.name + ".o")
    compile_variables = cc_common.create_compile_variables(
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        output_file = object_file.path,
        source_file = source_file.path,
    )

    # rules_cc includes globally enabled C/C++ default compilation flags in its
    # assemble command line. Keep the toolchain environment but construct the
    # MASM-only command line explicitly. The clang-cl toolchain intentionally
    # selects clang-cl.exe for generic assembly, so invoke ml64.exe from the
    # MSVC environment PATH for this explicitly MASM-only rule.
    arguments = ctx.actions.args()
    arguments.add("/nologo")
    arguments.add("/Zi")
    arguments.add("/c")
    arguments.add("/Fo" + object_file.path)
    arguments.add(source_file.path)
    environment = cc_common.get_environment_variables(
        action_name = ASSEMBLE_ACTION_NAME,
        feature_configuration = feature_configuration,
        variables = compile_variables,
    )

    ctx.actions.run(
        arguments = [arguments],
        env = environment,
        executable = ctx.file._masm_wrapper,
        inputs = depset(
            direct = [source_file],
            transitive = [cc_toolchain.all_files],
        ),
        mnemonic = "IreeMsvcMasm",
        outputs = [object_file],
        progress_message = "Assembling MSVC object %{label}",
    )
    return [DefaultInfo(files = depset([object_file]))]

_iree_msvc_masm_object = rule(
    implementation = _iree_msvc_masm_object_impl,
    attrs = dict(CC_TOOLCHAIN_ATTRS, **{
        "src": attr.label(
            allow_single_file = [".asm"],
            mandatory = True,
        ),
        "_masm_wrapper": attr.label(
            allow_single_file = [".bat"],
            default = Label("//build_tools/bazel:msvc_masm_wrapper.bat"),
        ),
    }),
    fragments = ["cpp"],
    toolchains = use_cc_toolchain(mandatory = True),
)

def iree_msvc_masm_library(
        name,
        src,
        target_compatible_with,
        visibility = None):
    """Builds one MASM source file as an MSVC C/C++ library dependency.

    This invokes MASM from the configured C/C++ toolchain environment without
    inheriting C/C++ compilation flags. Native cc_library assembly handling
    includes those flags and causes ml.exe/ml64.exe to warn for each one, while
    the clang-cl toolchain selects clang-cl.exe for generic assembly inputs.

    Args:
      name: Name of the resulting cc_library target.
      src: Single .asm source file.
      target_compatible_with: Platform constraints for the MASM source.
      visibility: Optional visibility for the resulting library.
    """
    object_target = name + "_object"
    _iree_msvc_masm_object(
        name = object_target,
        src = src,
        target_compatible_with = target_compatible_with,
    )
    cc_library(
        name = name,
        srcs = [":" + object_target],
        target_compatible_with = target_compatible_with,
        visibility = visibility,
    )
