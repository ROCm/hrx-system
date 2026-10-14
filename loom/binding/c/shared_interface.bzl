# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Aggregates LoomC component interfaces and selects shared-library linkage."""

load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

def _loomc_shared_interface_impl(ctx):
    link_info = cc_common.merge_cc_infos(cc_infos = [dep[CcInfo] for dep in ctx.attr.deps])
    runfiles = ctx.runfiles()
    for dep in ctx.attr.deps:
        runfiles = runfiles.merge(dep[DefaultInfo].default_runfiles)
    return [
        CcInfo(
            compilation_context = ctx.attr.implementation[CcInfo].compilation_context,
            linking_context = link_info.linking_context,
        ),
        DefaultInfo(runfiles = runfiles),
    ]

loomc_shared_interface = rule(
    implementation = _loomc_shared_interface_impl,
    attrs = {
        "deps": attr.label_list(providers = [CcInfo]),
        "implementation": attr.label(mandatory = True, providers = [CcInfo]),
    },
)

def _loomc_static_library_impl(ctx):
    runfiles = ctx.runfiles()
    for dep in ctx.attr.deps:
        runfiles = runfiles.merge(dep[DefaultInfo].default_runfiles)
    return [
        cc_common.merge_cc_infos(direct_cc_infos = [dep[CcInfo] for dep in ctx.attr.deps]),
        DefaultInfo(runfiles = runfiles),
    ]

# Re-export component compilation contexts, including their direct module maps,
# so consumers can include component headers with layering_check enabled.
loomc_static_library = rule(
    implementation = _loomc_static_library_impl,
    attrs = {"deps": attr.label_list(providers = [CcInfo])},
)
