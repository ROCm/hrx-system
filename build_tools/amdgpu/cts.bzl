# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU HAL CTS executable testdata rules."""

load("//build_tools/amdgpu:binary.bzl", "iree_amdgpu_binary")
load(
    "//build_tools/amdgpu:selectors.bzl",
    "iree_amdgpu_target_label_fragment",
    "iree_amdgpu_target_selector_config_settings",
)
load(
    "//build_tools/amdgpu:target_map.bzl",
    "IREE_AMDGPU_CODE_OBJECT_TARGETS",
)
load("//build_tools/embed_data:build_defs.bzl", "iree_c_embed_data")
load("//runtime/build_tools/bazel:cc.bzl", "iree_runtime_cc_library")

_INCOMPATIBLE_TARGET = ["@platforms//:incompatible"]

def _camel_case(snake_str):
    result = ""
    for part in snake_str.split("_"):
        result += part.capitalize()
    return result

def _source_stem(src):
    filename = src.split(":")[-1].split("/")[-1]
    if not filename.endswith(".c"):
        fail("AMDGPU CTS executable source must be a C file: {}".format(src))
    return filename[:-2]

def _target_deps(deps, code_object_target):
    code_object_target_fragment = iree_amdgpu_target_label_fragment(code_object_target)
    return [
        dep.replace("{AMDGPU_CODE_OBJECT_TARGET}", code_object_target)
            .replace("{AMDGPU_CODE_OBJECT_TARGET_FRAGMENT}", code_object_target_fragment)
        for dep in deps
    ]

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

def _generate_registration(
        name,
        header,
        target_name,
        target_family,
        target_key,
        identifier,
        backend_name,
        testonly):
    registration = "%s.cc" % name
    _registration(
        name = "%s_gen" % name,
        template = "//runtime/src/iree/hal/cts/util:testdata_target.cc.tpl",
        out = registration,
        substitutions = {
            "{BACKEND_NAME}": backend_name,
            "{HEADER_PATH}": "%s/%s" % (native.package_name(), header),
            "{IDENTIFIER}": identifier,
            "{TARGET_FAMILY}": target_family,
            "{TARGET_FUNC_NAME}": _camel_case(target_name),
            "{TARGET_KEY}": target_key,
            "{TARGET_NAME}": target_name,
            "{TARGET_VAR_NAME}": "%s_target" % target_name,
        },
        testonly = testonly,
    )
    return registration

def iree_amdgpu_hal_cts_testdata(
        name,
        srcs,
        target_selectors_flag,
        target_name,
        identifier,
        backend_name = "amdgpu",
        target_family = "amdgpu",
        target = "amdgcn-amd-amdhsa",
        deps = [],
        internal_hdrs = [],
        internalize = True,
        testonly = True,
        tags = []):
    """Builds and registers AMDGPU HAL CTS executable testdata.

    Args:
      name: Aggregate cc_library target exposing selected registrations.
      srcs: C sources. Each source basename maps to `<basename>.bin` in the CTS
        executable-data table of contents.
      target_selectors_flag: AMDGPU target selector build setting.
      target_name: CTS executable target prefix.
      identifier: C identifier prefix for generated TOC functions.
      backend_name: CTS backend name.
      target_family: HAL executable target family.
      target: LLVM target triple.
      deps: Bitcode archives passed to each generated executable. Labels may
        use `{AMDGPU_CODE_OBJECT_TARGET}` or
        `{AMDGPU_CODE_OBJECT_TARGET_FRAGMENT}` placeholders to refer to the
        code-object target being generated.
      internal_hdrs: Headers that should invalidate device compilation.
      internalize: whether to internalize linked dependency symbols after lazy
        archive extraction.
      testonly: Whether generated targets are test-only.
      tags: Tags applied to generated device binaries and libraries.
    """
    target_settings = iree_amdgpu_target_selector_config_settings(
        name = "%s_target" % name,
        flag = target_selectors_flag,
    )

    selected_target_libs = []
    for code_object_target in IREE_AMDGPU_CODE_OBJECT_TARGETS:
        target_fragment = iree_amdgpu_target_label_fragment(code_object_target)
        target_identifier = "%s_%s" % (identifier, target_fragment)
        target_compatible_with = select({
            target_settings.requested[code_object_target]: [],
            "//conditions:default": _INCOMPATIBLE_TARGET,
        })
        target_srcs = []
        for src in srcs:
            stem = _source_stem(src)
            binary_name = "%s_%s_%s" % (name, target_fragment, stem)
            binary_out = "%s_%s/%s.bin" % (name, target_fragment, stem)
            iree_amdgpu_binary(
                name = binary_name,
                target = target,
                arch = code_object_target,
                srcs = [src],
                deps = _target_deps(deps, code_object_target),
                internal_hdrs = internal_hdrs,
                internalize = internalize,
                out = binary_out,
                testonly = testonly,
                tags = tags,
                target_compatible_with = target_compatible_with,
            )
            target_srcs.append(":%s" % binary_name)

        data_name = "%s_%s_data" % (name, target_fragment)
        header = "%s.h" % data_name
        iree_c_embed_data(
            name = data_name,
            srcs = target_srcs,
            c_file_output = "%s.c" % data_name,
            h_file_output = header,
            identifier = target_identifier,
            flatten = True,
            testonly = testonly,
            tags = tags,
            target_compatible_with = target_compatible_with,
        )

        registration_target_name = "%s_%s" % (target_name, target_fragment)
        registration = _generate_registration(
            name = "%s_%s_registration" % (name, target_fragment),
            header = header,
            target_name = registration_target_name,
            target_family = target_family,
            target_key = code_object_target,
            identifier = target_identifier,
            backend_name = backend_name,
            testonly = testonly,
        )
        target_lib_name = "%s_%s_lib" % (name, target_fragment)
        iree_runtime_cc_library(
            name = target_lib_name,
            srcs = [registration],
            deps = [
                ":%s" % data_name,
                "//runtime/src/iree/hal/cts/util:registry",
            ],
            alwayslink = True,
            testonly = testonly,
            tags = tags,
            target_compatible_with = target_compatible_with,
        )
        selected_target_libs = selected_target_libs + select({
            target_settings.requested[code_object_target]: [":%s" % target_lib_name],
            "//conditions:default": [],
        })

    iree_runtime_cc_library(
        name = name,
        deps = selected_target_libs,
        alwayslink = True,
        testonly = testonly,
        tags = tags,
    )
