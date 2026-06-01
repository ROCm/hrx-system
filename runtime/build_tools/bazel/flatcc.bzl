# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime FlatBuffers schema generation with flatcc."""

load("//runtime/build_tools/bazel:cc.bzl", "iree_runtime_cc_library")

IreeFlatccInfo = provider(
    doc = "Generated C headers produced from FlatBuffers schemas.",
    fields = {
        "args": "flatcc arguments used by the generating action.",
        "headers": "depset of generated C header files.",
        "schema_includes": "depset of FlatBuffers schema files available for imports.",
        "srcs": "depset of FlatBuffers schema files compiled by the action.",
    },
)

_FLATCC_RUNTIME_DEP = Label("//runtime/build_tools/third_party/flatcc:runtime")
_FLATCC_TOOL = Label("//runtime/build_tools/third_party/flatcc:flatcc")
_FLATCC_INCLUDE_ROOT = "runtime/src"
_FLATCC_OUTPUTS_BY_ARG = {
    "--builder": ["_builder.h"],
    "--json": [
        "_json_parser.h",
        "_json_printer.h",
    ],
    "--reader": ["_reader.h"],
    "--verifier": ["_verifier.h"],
}
_FLATCC_PASSTHROUGH_ARGS = [
    "--common",
]

def _single_files(targets, attr_name):
    files = []
    for target in targets:
        target_files = target.files.to_list()
        if len(target_files) != 1:
            fail("%s entries must produce exactly one file, got %d from %s" % (
                attr_name,
                len(target_files),
                target.label,
            ))
        files.append(target_files[0])
    return files

def _validate_flatcc_args(flatcc_args):
    for arg in flatcc_args:
        if arg in _FLATCC_OUTPUTS_BY_ARG:
            continue
        if arg in _FLATCC_PASSTHROUGH_ARGS:
            continue
        fail("unsupported flatcc argument %r; update //runtime/build_tools/bazel:flatcc.bzl before using it" % arg)

def _generated_headers(srcs, flatcc_args):
    _validate_flatcc_args(flatcc_args)
    outs = []
    for src in srcs:
        if not src.basename.endswith(".fbs"):
            fail("flatcc source %s must end with .fbs" % src.path)
        stem = src.basename[:-4]
        for arg in flatcc_args:
            for suffix in _FLATCC_OUTPUTS_BY_ARG.get(arg, []):
                outs.append(stem + suffix)
    if not outs:
        fail("flatcc arguments %r do not declare any generated C headers" % flatcc_args)
    if len(outs) != len({out: None for out in outs}):
        fail("flatcc output names must be unique, got %r" % outs)
    return outs

def _iree_runtime_flatbuffer_c_headers_impl(ctx):
    srcs = _single_files(ctx.attr.srcs, "srcs")
    schema_includes = _single_files(ctx.attr.flatcc_includes, "flatcc_includes")
    output_names = _generated_headers(srcs, ctx.attr.flatcc_args)
    outputs = [
        ctx.actions.declare_file(output_name)
        for output_name in output_names
    ]

    args = ctx.actions.args()
    args.add("-o" + outputs[0].dirname)
    args.add("-I")
    args.add(_FLATCC_INCLUDE_ROOT)
    args.add_all(ctx.attr.flatcc_args)
    args.add_all(srcs)

    ctx.actions.run(
        arguments = [args],
        executable = ctx.executable._flatcc,
        inputs = depset(srcs + schema_includes),
        mnemonic = "IreeFlatcc",
        outputs = outputs,
        progress_message = "Generating FlatBuffers C headers for %{label}",
    )

    return [
        DefaultInfo(files = depset(outputs)),
        IreeFlatccInfo(
            args = ctx.attr.flatcc_args,
            headers = depset(outputs),
            schema_includes = depset(schema_includes),
            srcs = depset(srcs),
        ),
    ]

iree_runtime_flatbuffer_c_headers = rule(
    implementation = _iree_runtime_flatbuffer_c_headers_impl,
    attrs = {
        "flatcc_args": attr.string_list(
            default = [
                "--common",
                "--reader",
            ],
            doc = "flatcc generation arguments that select generated C header families.",
        ),
        "flatcc_includes": attr.label_list(
            allow_files = [".fbs"],
            doc = "Additional schema files available to flatcc imports; generated headers for those schemas must be supplied through deps.",
        ),
        "srcs": attr.label_list(
            allow_files = [".fbs"],
            mandatory = True,
            doc = "FlatBuffers schema files to compile.",
        ),
        "_flatcc": attr.label(
            cfg = "exec",
            default = _FLATCC_TOOL,
            executable = True,
            doc = "flatcc executable used to generate C headers.",
        ),
    },
    doc = "Generates C headers from FlatBuffers schemas with flatcc.",
)

def _with_flatcc_runtime_deps(deps):
    if deps == None:
        deps = []
    return deps + [_FLATCC_RUNTIME_DEP]

def iree_runtime_flatbuffer_c_library(
        name,
        srcs,
        flatcc_args = None,
        flatcc_includes = None,
        deps = None,
        testonly = False,
        visibility = None,
        **kwargs):
    """Generates a runtime C/C++ header library from FlatBuffers schemas.

    The macro invokes the repository-local flatcc tool and exposes generated C
    headers through a normal runtime C/C++ library target. Callers select the
    generated header families with flatcc's `--reader`, `--builder`,
    `--verifier`, and `--json` arguments.

    Args:
      name: Generated header library target name.
      srcs: FlatBuffers schema files to compile.
      flatcc_args: flatcc generation arguments that select generated header
        families.
      flatcc_includes: Additional schema files available to flatcc imports.
        This only supplies generator inputs; any generated headers included by
        flatcc output must also be supplied by `deps`.
      deps: Additional dependencies propagated by the generated header library,
        including generated header libraries for any imported schemas.
      testonly: Whether the generated schema library is only available to test
        targets.
      visibility: Visibility for the generated header library.
      **kwargs: Additional attributes forwarded to `iree_runtime_cc_library`.
    """
    if flatcc_args == None:
        flatcc_args = [
            "--common",
            "--reader",
        ]
    if flatcc_includes == None:
        flatcc_includes = []

    headers_name = name + "_headers"
    iree_runtime_flatbuffer_c_headers(
        name = headers_name,
        flatcc_args = flatcc_args,
        flatcc_includes = flatcc_includes,
        srcs = srcs,
        testonly = testonly,
        visibility = ["//visibility:private"],
    )
    iree_runtime_cc_library(
        name = name,
        visibility = visibility,
        hdrs = [":" + headers_name],
        deps = _with_flatcc_runtime_deps(deps),
        testonly = testonly,
        **kwargs
    )
