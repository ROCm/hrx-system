# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for assembling SPIR-V modules from SPIR-V assembly."""

load("@bazel_skylib//lib:selects.bzl", "selects")
load("//build_tools/embed_data:build_defs.bzl", "iree_c_embed_data")

visibility("//...")

IreeSpirvAsmModuleInfo = provider(
    doc = "Metadata for a SPIR-V module generated from SPIR-V assembly.",
    fields = {
        "args": "Command-line arguments passed to spirv-as.",
        "module": "Generated SPIR-V binary module file.",
        "src": "SPIR-V assembly source file.",
        "tool": "spirv-as executable label.",
    },
)

_SPIRV_AS_TOOL = Label("//third_party:spirv_as")
_LOOM_SPIRV_ARTIFACTS_CONFIG = Label("//loom/config/target:spirv_artifacts")
_VULKAN_DRIVER_CONFIG = Label("//runtime/config/hal:driver_vulkan")
_INCOMPATIBLE_TARGET = ["@platforms//:incompatible"]

def iree_spirv_tool_target_compatible_with():
    """Returns the platform compatibility gate for SPIR-V tool users."""
    return selects.with_or({
        (
            _LOOM_SPIRV_ARTIFACTS_CONFIG,
            _VULKAN_DRIVER_CONFIG,
        ): [],
        "//conditions:default": _INCOMPATIBLE_TARGET,
    })

def _iree_spirv_asm_module_impl(ctx):
    module = ctx.outputs.out
    src = ctx.file.src

    command_args = []
    if ctx.attr.target_env:
        command_args.append("--target-env=%s" % ctx.attr.target_env)
    command_args.extend(ctx.attr.spirv_as_args)
    command_args.extend([
        "-o",
        module.path,
        src.path,
    ])

    args = ctx.actions.args()
    args.add_all(command_args)

    ctx.actions.run(
        arguments = [args],
        executable = ctx.executable.assemble_tool,
        inputs = depset([src]),
        mnemonic = "IreeSpirvAsmModule",
        outputs = [module],
        progress_message = "Assembling SPIR-V module for %{label}",
    )

    return [
        DefaultInfo(files = depset([module])),
        IreeSpirvAsmModuleInfo(
            args = command_args,
            module = module,
            src = src,
            tool = ctx.attr.assemble_tool.label,
        ),
    ]

_iree_spirv_asm_module = rule(
    implementation = _iree_spirv_asm_module_impl,
    attrs = {
        "assemble_tool": attr.label(
            cfg = "exec",
            default = _SPIRV_AS_TOOL,
            executable = True,
            doc = "Executable tool used to assemble SPIR-V assembly.",
        ),
        "out": attr.output(
            mandatory = True,
            doc = "Generated SPIR-V binary module file.",
        ),
        "spirv_as_args": attr.string_list(
            doc = "Additional arguments passed to spirv-as before the output and source paths.",
        ),
        "src": attr.label(
            allow_single_file = [".spvasm"],
            mandatory = True,
            doc = "SPIR-V assembly source file.",
        ),
        "target_env": attr.string(
            doc = "Optional spirv-as --target-env value.",
        ),
    },
    doc = "Assembles SPIR-V assembly into a SPIR-V binary module.",
)

def iree_spirv_asm_module(
        name,
        src,
        out = None,
        target_env = None,
        spirv_as_args = [],
        assemble_tool = _SPIRV_AS_TOOL,
        c_identifier = None,
        deps = None,
        testonly = False,
        visibility = None,
        **kwargs):
    """Builds a SPIR-V binary module from textual SPIR-V assembly.

    The primary target produces a `.spv` file. Supplying `c_identifier` also
    creates a `<name>_c` C library embedding the generated module with
    `iree_c_embed_data`.

    Args:
      name: Generated SPIR-V module target name.
      src: SPIR-V assembly source file.
      out: Generated SPIR-V module filename. Defaults to `<name>.spv`.
      target_env: Optional `spirv-as --target-env` value.
      spirv_as_args: Additional arguments passed to `spirv-as`.
      assemble_tool: Executable SPIR-V assembly tool.
      c_identifier: Optional C identifier prefix for an embedded module library.
      deps: Additional dependencies for the optional embedded module library.
      testonly: Whether generated targets are test-only.
      visibility: Visibility for generated targets.
      **kwargs: Additional common attributes forwarded to generated targets.
    """
    if out == None:
        out = "%s.spv" % name
    if deps == None:
        deps = []
    if not out.endswith(".spv"):
        fail("out must end with .spv, got %r" % out)

    if "target_compatible_with" not in kwargs:
        kwargs["target_compatible_with"] = iree_spirv_tool_target_compatible_with()

    _iree_spirv_asm_module(
        name = name,
        assemble_tool = assemble_tool,
        out = out,
        spirv_as_args = spirv_as_args,
        src = src,
        target_env = target_env or "",
        testonly = testonly,
        visibility = visibility,
        **kwargs
    )

    if c_identifier:
        iree_c_embed_data(
            name = "%s_c" % name,
            c_file_output = "%s_c.c" % name,
            deps = deps,
            flatten = True,
            h_file_output = "%s_c.h" % name,
            identifier = c_identifier,
            srcs = [":" + name],
            testonly = testonly,
            visibility = visibility,
            **kwargs
        )
