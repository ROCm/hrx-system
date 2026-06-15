# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for compiling with clang to produce AMDGPU libraries."""

load(
    "@iree_amdgpu_device_toolchain//:paths.bzl",
    "AMDGPU_CLANG_RESOURCE_HEADERS",
    "AMDGPU_CLANG_RESOURCE_MARKER",
    "AMDGPU_CLANG_TOOL",
    "AMDGPU_DEVICE_TOOLCHAIN_AVAILABLE",
    "AMDGPU_LLD_TOOL",
    "AMDGPU_LLVM_LINK_TOOL",
    "AMDGPU_LLVM_OBJCOPY_TOOL",
)
load(
    "//build_tools/amdgpu:selectors.bzl",
    "iree_amdgpu_target_label_fragment",
    "iree_amdgpu_target_selector_config_settings",
)
load("//build_tools/amdgpu:target_map.bzl", "IREE_AMDGPU_CODE_OBJECT_TARGETS")
load("//build_tools/embed_data:build_defs.bzl", "iree_c_embed_data")

_INCOMPATIBLE_TARGET = ["@platforms//:incompatible"]
_CLANG_RESOURCE_INCLUDE_FLAG = "-isystem $$(dirname $(location %s))" % (
    AMDGPU_CLANG_RESOURCE_MARKER,
) if AMDGPU_CLANG_RESOURCE_MARKER else ""

def _incompatible_filegroup(name, kwargs):
    filegroup_kwargs = {}
    for attr in ["tags", "testonly", "visibility"]:
        if attr in kwargs:
            filegroup_kwargs[attr] = kwargs[attr]
    native.filegroup(
        name = name,
        srcs = [],
        target_compatible_with = _INCOMPATIBLE_TARGET,
        **filegroup_kwargs
    )

def iree_amdgpu_binary(
        name,
        target,
        arch,
        srcs,
        internal_hdrs = [],
        copts = [],
        linkopts = [],
        clang_tool = AMDGPU_CLANG_TOOL,
        link_tool = AMDGPU_LLVM_LINK_TOOL,
        lld_tool = AMDGPU_LLD_TOOL,
        objcopy_tool = AMDGPU_LLVM_OBJCOPY_TOOL,
        minimize = False,
        out = None,
        builtin_headers_dep = AMDGPU_CLANG_RESOURCE_HEADERS,
        builtin_headers_marker = AMDGPU_CLANG_RESOURCE_MARKER,
        builtin_headers_include_flag = _CLANG_RESOURCE_INCLUDE_FLAG,
        **kwargs):
    """Builds an LLVM shared library for AMDGPU from input files via clang.

    Args:
        name: Name of the target.
        target: LLVM `-target` flag.
        arch: LLVM `-march` flag.
        srcs: source files or filegroups to pass to clang.
        internal_hdrs: headers that should invalidate device compilation but
                       are not compiled as translation units or exposed as
                       interface headers.
        copts: additional flags to pass to clang.
        linkopts: additional flags to pass to lld.
        clang_tool: clang/amdclang executable target.
        link_tool: llvm-link executable target.
        lld_tool: lld executable target.
        objcopy_tool: optional llvm-objcopy executable target used for
                      `minimize`.
        minimize: whether to apply the optional post-link symbol-table
                  minimization pass. This is only valid for opaque code-object
                  data blobs whose kernels are not looked up by name.
        out: output binary path. Defaults to `<name>.so`.
        builtin_headers_dep: target containing clang builtin headers.
        builtin_headers_marker: optional single builtin header file used to
                                derive the builtin header include directory.
        builtin_headers_include_flag: shell fragment adding the builtin header
                                      include directory to clang.
        **kwargs: any additional attributes to pass to the underlying rules.
    """
    if not AMDGPU_DEVICE_TOOLCHAIN_AVAILABLE:
        _incompatible_filegroup(name, kwargs)
        return

    base_copts = [
        # C configuration.
        "-x c",
        "-std=c23",
        "-Xclang -finclude-default-header",
        "-nogpulib",
        "-fno-short-wchar",

        # Target architecture/machine.
        "-target %s" % (target),
        "-march=%s" % (arch),
        "-fgpu-rdc",  # NOTE: may not be required for all targets

        # Header paths for builtins and our own includes.
        builtin_headers_include_flag,
        "-I$(BINDIR)/runtime/src",
        "-Iruntime/src",

        # Avoid warnings about things we do that are not compatible across
        # compilers but are fine because we're only ever compiling with clang.
        "-Wno-gnu-pointer-arith",

        # Optimized.
        "-fno-ident",
        "-fvisibility=hidden",
        "-O3",

        # Object file only in bitcode format.
        "-c",
        "-emit-llvm",
    ]

    archive_out = "%s.a" % (name)
    source_locations = " ".join(["$(locations %s)" % (src,) for src in srcs])
    object_dir = "$(@D)/%s.objects" % (name,)
    compile_srcs = srcs + internal_hdrs
    if builtin_headers_dep:
        compile_srcs.append(builtin_headers_dep)
    if builtin_headers_marker:
        compile_srcs.append(builtin_headers_marker)
    native.genrule(
        name = "archive_%s" % (name),
        srcs = compile_srcs,
        outs = [archive_out],
        cmd = " && ".join([
            "set -e",
            "object_dir=\"%s\"" % (object_dir,),
            "rm -rf \"$${object_dir}\"",
            "mkdir -p \"$${object_dir}\"",
            "object_index=0",
            "for src in %s; do %s; object_index=$$((object_index + 1)); done" % (
                source_locations,
                " ".join([
                    "$(location %s)" % (clang_tool),
                    " ".join(base_copts + copts),
                    "-o \"$${object_dir}/$${object_index}.bc\"",
                    "\"$${src}\"",
                ]),
            ),
            " ".join([
                "$(location %s)" % (link_tool),
                "\"$${object_dir}\"/*.bc",
                "-o $(location %s)" % (archive_out),
            ]),
        ]),
        tools = [
            clang_tool,
            link_tool,
        ],
        message = "Compiling bitcode library %s to %s..." % (srcs, archive_out),
        output_to_bindir = 1,
        **kwargs
    )

    link_out = "%s.bc" % (name)
    native.genrule(
        name = "link_%s" % (name),
        srcs = [archive_out],
        outs = [link_out],
        cmd = " && ".join([
            " ".join([
                "$(location %s)" % (link_tool),
                "-internalize",
                "-only-needed",
                "$(locations %s)" % (archive_out),
                "-o $(location %s)" % (link_out),
            ]),
        ]),
        tools = [link_tool],
        message = "Linking bitcode library %s to %s..." % (name, link_out),
        output_to_bindir = 1,
        **kwargs
    )

    base_linkopts = [
        "-m elf64_amdgpu",
        "--build-id=none",
        "--no-undefined",
        "-shared",
        "-plugin-opt=mcpu=%s" % (arch),
        "-plugin-opt=O3",
        "--lto-CGO3",
        "--no-whole-archive",
        "--gc-sections",
        "--strip-debug",
        "--discard-all",
        "--discard-locals",
    ]

    out = out or ("%s.so" % (name))
    version_script = "$(@D)/%s.local.version" % (name,)
    link_output = "$(location %s)" % (out)
    lld_linkopts = base_linkopts + linkopts
    objcopy_command = None
    if minimize and objcopy_tool != None:
        link_output = "$(@D)/%s.linked.so" % (name,)
        lld_linkopts = lld_linkopts + [
            "--version-script=\"%s\"" % (version_script,),
        ]
        objcopy_command = " ".join([
            "$(location %s)" % (objcopy_tool),
            "-R .comment",
            "-R .AMDGPU.gpr_maximums",
            "--discard-all",
            "-N _DYNAMIC",
            "\"%s\"" % (link_output,),
            "$(location %s)" % (out),
        ])
    cmd = []
    if minimize and objcopy_tool != None:
        cmd.append("printf '{\\n  local:\\n    *;\\n};\\n' > \"%s\"" % (version_script,))
    cmd.append(" ".join([
        "$(location %s)" % (lld_tool),
        "-flavor gnu",
        " ".join(lld_linkopts),
        "$(location %s)" % (link_out),
        "-o \"%s\"" % (link_output,),
    ]))
    if objcopy_command:
        cmd.append(objcopy_command)
    tools = [lld_tool]
    if minimize and objcopy_tool != None:
        tools.append(objcopy_tool)
    native.genrule(
        name = name,
        srcs = [link_out],
        outs = [out],
        cmd = " && ".join(cmd),
        tools = tools,
        message = "Generating OpenCL binary %s to %s..." % (name, out),
        output_to_bindir = 1,
        **kwargs
    )

def iree_amdgpu_binary_variants(
        name,
        target,
        srcs,
        target_selectors_flag,
        binary_name_prefix = None,
        code_object_targets = IREE_AMDGPU_CODE_OBJECT_TARGETS,
        minimize = False,
        tags = [],
        **kwargs):
    """Builds code-object variants and exposes the selected outputs.

    Args:
      name: Aggregate filegroup name containing selected binaries.
      target: LLVM `-target` flag.
      srcs: source files or filegroups to pass to clang.
      target_selectors_flag: Label of an `iree_amdgpu_target_selectors_flag`
        build setting controlling which variants are selected.
      binary_name_prefix: Prefix for generated per-code-object binary targets.
        Defaults to `name`.
      code_object_targets: Code-object targets to build variants for.
      minimize: Whether generated binaries should apply post-link
        symbol-table minimization.
      tags: Tags applied to generated binary targets and the aggregate
        filegroup.
      **kwargs: Additional attributes forwarded to `iree_amdgpu_binary`.
    """
    if binary_name_prefix == None:
        binary_name_prefix = name

    target_selection = iree_amdgpu_target_selector_config_settings(
        name = "{}_target".format(name),
        code_object_targets = code_object_targets,
        flag = target_selectors_flag,
    )

    selected_srcs = []
    for code_object_target in code_object_targets:
        binary_name = "{}_{}".format(
            binary_name_prefix,
            iree_amdgpu_target_label_fragment(code_object_target),
        )
        iree_amdgpu_binary(
            name = binary_name,
            target = target,
            arch = code_object_target,
            srcs = srcs,
            minimize = minimize,
            tags = tags,
            **kwargs
        )
        if AMDGPU_DEVICE_TOOLCHAIN_AVAILABLE:
            # Keep no-toolchain aggregates empty while per-variant targets stay
            # incompatible for direct requests.
            selected_srcs += select({
                target_selection.requested[code_object_target]: [":" + binary_name],
                "//conditions:default": [],
            })

    filegroup_kwargs = {}
    if "visibility" in kwargs:
        filegroup_kwargs["visibility"] = kwargs["visibility"]
    native.filegroup(
        name = name,
        srcs = selected_srcs,
        tags = tags,
        testonly = kwargs.get("testonly", False),
        **filegroup_kwargs
    )

def iree_amdgpu_binary_variants_embed_data(
        name,
        target,
        srcs,
        target_selectors_flag,
        binary_name_prefix = None,
        c_file_output = None,
        h_file_output = None,
        identifier = None,
        flatten = True,
        code_object_targets = IREE_AMDGPU_CODE_OBJECT_TARGETS,
        minimize = False,
        tags = [],
        **kwargs):
    """Builds selected AMDGPU binaries and embeds them into a C library.

    Args:
      name: Generated `cc_library` target name.
      target: LLVM `-target` flag.
      srcs: source files or filegroups to pass to clang.
      target_selectors_flag: Label of an `iree_amdgpu_target_selectors_flag`
        build setting controlling which variants are selected.
      binary_name_prefix: Prefix for generated per-code-object binary targets.
        Defaults to `name`.
      c_file_output: Generated C implementation filename. Defaults to
        `<name>.c`.
      h_file_output: Generated C header filename. Defaults to `<name>.h`.
      identifier: C identifier prefix. Defaults to `name`.
      flatten: Whether embedded table-of-contents names drop directory
        components.
      code_object_targets: Code-object targets to build variants for.
      minimize: Whether generated binaries should apply post-link
        symbol-table minimization.
      tags: Tags applied to generated targets.
      **kwargs: Additional attributes forwarded to `iree_amdgpu_binary`.
    """
    binary_variants_name = name + "_binaries"
    iree_amdgpu_binary_variants(
        name = binary_variants_name,
        target = target,
        srcs = srcs,
        target_selectors_flag = target_selectors_flag,
        binary_name_prefix = binary_name_prefix,
        code_object_targets = code_object_targets,
        minimize = minimize,
        tags = tags,
        **kwargs
    )

    embed_kwargs = {}
    if "visibility" in kwargs:
        embed_kwargs["visibility"] = kwargs["visibility"]
    iree_c_embed_data(
        name = name,
        testonly = kwargs.get("testonly", False),
        srcs = [":" + binary_variants_name],
        c_file_output = c_file_output or (name + ".c"),
        h_file_output = h_file_output or (name + ".h"),
        identifier = identifier,
        flatten = flatten,
        tags = tags,
        **embed_kwargs
    )
