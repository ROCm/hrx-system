# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private Windows DLLs that isolate binary-only C++ and CRT ABIs."""

load(":cc.bzl", "amdf_cc_binary")

def amdf_windows_sidecar_library(
        name,
        srcs,
        deps = None,
        linkopts = None,
        visibility = None):
    """Builds a private x86-64 Windows DLL with the release static CRT.

    The sidecar owns every object allocated by binary-only dependencies so no
    C++ or CRT ownership crosses into the public libamdf library.

    Args:
      name: Sidecar DLL target and artifact base name.
      srcs: C or C++ implementation sources.
      deps: Private link dependencies.
      linkopts: Private linker options.
      visibility: Bazel visibility of the sidecar artifact.
    """
    amdf_cc_binary(
        name = name,
        srcs = srcs,
        copts = ["/MT"],
        local_defines = [
            "_DISABLE_STRING_ANNOTATION",
            "_DISABLE_VECTOR_ANNOTATION",
            "_ITERATOR_DEBUG_LEVEL=0",
        ],
        deps = deps,
        linkopts = linkopts,
        linkshared = True,
        target_compatible_with = [
            "@platforms//cpu:x86_64",
            "@platforms//os:windows",
        ],
        visibility = visibility,
    )
