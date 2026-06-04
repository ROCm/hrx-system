# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""HAL CTS compatibility macros for the extracted runtime build.

This reduced repository does not contain the IREE compiler pipeline that the
monorepo used to lower CTS MLIR test inputs into executable flatbuffers.
"""

load("//runtime/build_tools/bazel:cc.bzl", "iree_runtime_cc_library")
load("//runtime/build_tools/bazel:hal_cts.bzl", "iree_runtime_hal_cts_test_suite")

def _camel_case(snake_str):
    result = ""
    for part in snake_str.split("_"):
        result += part.capitalize()
    return result

def _empty_testdata_impl(ctx):
    guard = "%s_H_" % ctx.attr.identifier.upper()
    header = "\n".join([
        "#ifndef %s" % guard,
        "#define %s" % guard,
        "#include <stddef.h>",
        "#if defined(__cplusplus)",
        "extern \"C\" {",
        "#endif",
        "#ifndef IREE_FILE_TOC",
        "#define IREE_FILE_TOC",
        "typedef struct iree_file_toc_t {",
        "  const char* name;",
        "  const char* data;",
        "  size_t size;",
        "} iree_file_toc_t;",
        "#endif",
        "const iree_file_toc_t* %s_create(void);" % ctx.attr.identifier,
        "static inline size_t %s_size(void) { return 0; }" % ctx.attr.identifier,
        "#if defined(__cplusplus)",
        "}  // extern \"C\"",
        "#endif",
        "#endif  // %s" % guard,
        "",
    ])
    source = "\n".join([
        "#include \"%s\"" % ctx.outputs.h.basename,
        "static const iree_file_toc_t kEmptyToc[] = {{NULL, NULL, 0}};",
        "const iree_file_toc_t* %s_create(void) { return kEmptyToc; }" % ctx.attr.identifier,
        "",
    ])
    ctx.actions.write(ctx.outputs.h, header)
    ctx.actions.write(ctx.outputs.c, source)
    return [DefaultInfo(files = depset([ctx.outputs.h, ctx.outputs.c]))]

_empty_testdata = rule(
    implementation = _empty_testdata_impl,
    attrs = {
        "c": attr.output(mandatory = True),
        "h": attr.output(mandatory = True),
        "identifier": attr.string(mandatory = True),
    },
)

def _registration_impl(ctx):
    ctx.actions.expand_template(
        template = ctx.file.template,
        output = ctx.outputs.out,
        substitutions = ctx.attr.substitutions,
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

_registration = rule(
    implementation = _registration_impl,
    attrs = {
        "out": attr.output(mandatory = True),
        "substitutions": attr.string_dict(),
        "template": attr.label(mandatory = True, allow_single_file = True),
    },
)

def iree_hal_cts_testdata(
        format_name,
        target_device,
        identifier,
        backend_name,
        format_string,
        testdata,
        flags = [],
        flag_values = {},
        cmake_format_variant_values = [],
        data = [],
        testonly = True,
        **kwargs):
    testdata_name = "testdata_%s" % format_name
    header = "%s.h" % testdata_name
    source = "%s.c" % testdata_name

    _empty_testdata(
        name = "%s_gen" % testdata_name,
        c = source,
        h = header,
        identifier = identifier,
        testonly = testonly,
    )

    iree_runtime_cc_library(
        name = testdata_name,
        srcs = [source],
        hdrs = [header],
        testonly = testonly,
        **kwargs
    )

    registration = "%s_registration.cc" % testdata_name
    _registration(
        name = "%s_registration_gen" % testdata_name,
        template = "//runtime/src/iree/hal/cts/util:testdata_format.cc.tpl",
        out = registration,
        substitutions = {
            "{BACKEND_NAME}": backend_name,
            "{FORMAT_FUNC_NAME}": _camel_case(format_name),
            "{FORMAT_NAME}": format_name,
            "{FORMAT_STRING}": format_string,
            "{FORMAT_VAR_NAME}": "%s_format" % format_name,
            "{HEADER_PATH}": "%s/%s" % (native.package_name(), header),
            "{IDENTIFIER}": identifier,
        },
        testonly = testonly,
    )

    iree_runtime_cc_library(
        name = "%s_lib" % testdata_name,
        srcs = [registration],
        deps = [
            ":%s" % testdata_name,
            "//runtime/src/iree/hal/cts/util:registry",
        ],
        testonly = testonly,
        alwayslink = True,
    )

def iree_hal_cts_test_suite(
        backends_lib,
        executable_formats = {},
        testdata_libs = [],
        testdata = None,
        flag_values = {},
        **kwargs):
    iree_runtime_hal_cts_test_suite(
        backends = backends_lib,
        **kwargs
    )
