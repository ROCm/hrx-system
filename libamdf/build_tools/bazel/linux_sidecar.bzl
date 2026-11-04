# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private Linux DSOs that isolate binary-only C++ ABIs."""

load(":cc.bzl", "amdf_cc_binary")

_SIDECAR_ATTRIBUTES = {
    "cxxopts": [
        "-fexceptions",
        "-frtti",
        "-pthread",
        "-Wno-unknown-pragmas",
    ],
    "target_compatible_with": [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
}

_SIDECAR_LINKOPTS = [
    "-fuse-ld=lld",
    "-pthread",
    "-Wl,--exclude-libs,ALL",
    "-Wl,--gc-sections",
    "-Wl,--no-gnu-unique",
]

_SIDECAR_STRICT_LINKOPTS = select({
    Label("//build_tools/bazel:sanitizer_disabled"): ["-Wl,--no-undefined"],
    "//conditions:default": [],
})

def amdf_linux_sidecar_library(
        name,
        srcs,
        exports_file,
        deps = None,
        linkopts = None,
        visibility = None):
    """Builds an unloadable private x86-64 Linux DSO.

    Binary-only C++ dependencies remain inside the sidecar. The exports map
    defines the complete C surface, and LLD localizes archive symbols while
    demoting GNU-unique bindings that would otherwise pin the DSO.

    Args:
      name: Sidecar DSO target and artifact base name.
      srcs: C or C++ implementation sources.
      exports_file: Linker version script defining the public C surface.
      deps: Private link dependencies.
      linkopts: Additional private linker options.
      visibility: Bazel visibility of the sidecar artifact.
    """
    amdf_cc_binary(
        name = name,
        srcs = srcs,
        deps = deps,
        additional_linker_inputs = [exports_file],
        linkopts = _SIDECAR_LINKOPTS + _SIDECAR_STRICT_LINKOPTS + (linkopts or []) + [
            "-Wl,--version-script=$(location %s)" % exports_file,
        ],
        linkshared = True,
        visibility = visibility,
        **_SIDECAR_ATTRIBUTES
    )
