# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""HRX CTS Bazel macros."""

load("//libhrx/build_tools/bazel:cc.bzl", "hrx_cc_test")

def hrx_cts_test(name, srcs, deps = None, data = None, args = None, **kwargs):
    """Defines an HRX CTS executable with the standard runtime library argument.

    Args:
      name: Short CTS case name, without the `hrx_cts_` target prefix.
      srcs: Source files for the CTS executable.
      deps: Additional dependencies for the CTS executable.
      data: Additional runtime data files for the CTS executable.
      args: Additional command-line arguments passed to the CTS executable.
      **kwargs: Additional `hrx_cc_test` keyword arguments.
    """
    if deps == None:
        deps = []
    if data == None:
        data = []
    if args == None:
        args = []

    # Catch assertions throw through the test function. Its stack must unwind
    # resource owners before the harness catches the assertion failure.
    cxxopts = select({
        "//build_tools/bazel:cc_compiler_clang_cl": [],
        "//build_tools/bazel:cc_compiler_msvc": [],
        "//conditions:default": ["-fexceptions"],
    }) + kwargs.pop("cxxopts", [])
    hrx_cc_test(
        name = "hrx_cts_" + name,
        srcs = srcs,
        cxxopts = cxxopts,
        deps = [
            "//libhrx/cts/core:harness",
            "//third_party:catch2",
        ] + deps,
        data = [
            "//libhrx/src/libhrx:hrx",
        ] + data,
        args = [
            "--hrx-library",
            "$(location //libhrx/src/libhrx:hrx)",
        ] + args,
        **kwargs
    )
