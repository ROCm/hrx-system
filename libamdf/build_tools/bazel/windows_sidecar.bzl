# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private Windows DLLs that isolate binary-only C++ and CRT ABIs."""

load(":cc.bzl", "amdf_cc_binary")
load(":cc_test.bzl", "amdf_cc_test")

# Binary-only C++ dependencies retain their release ABI in every build mode.
_SIDECAR_ATTRIBUTES = {
    "features": ["static_link_msvcrt_release"],
    "local_defines": [
        "_DISABLE_STRING_ANNOTATION",
        "_DISABLE_VECTOR_ANNOTATION",
        "_ITERATOR_DEBUG_LEVEL=0",
    ],
    "target_compatible_with": [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
}

def amdf_windows_sidecar_library(
        name,
        srcs,
        deps = None,
        linkopts = None,
        visibility = None,
        testonly = False):
    """Builds a private x86-64 Windows DLL with the release static CRT.

    The sidecar owns every object allocated by binary-only dependencies so no
    C++ or CRT ownership crosses into the public libamdf library.

    Args:
      name: Sidecar DLL target and artifact base name.
      srcs: C or C++ implementation sources.
      deps: Private link dependencies.
      linkopts: Private linker options.
      visibility: Bazel visibility of the sidecar artifact.
      testonly: Whether the private DLL is exclusively a test dependency.
    """
    amdf_cc_binary(
        name = name,
        srcs = srcs,
        deps = deps,
        linkopts = linkopts,
        linkshared = True,
        visibility = visibility,
        testonly = testonly,
        **_SIDECAR_ATTRIBUTES
    )

def amdf_windows_sidecar_test(
        name,
        srcs,
        deps = None,
        linkopts = None,
        tags = None,
        resource_group = None):
    """Tests private sidecar implementation with its production C++ and CRT ABI.

    Args:
      name: Test target name.
      srcs: Sidecar implementation and test sources.
      deps: Private link dependencies using the release static CRT.
      linkopts: Private linker options.
      tags: Test selection and runtime resource requirements.
      resource_group: Local resource exclusion group for hardware tests.
    """
    amdf_cc_test(
        name = name,
        srcs = srcs,
        deps = deps,
        linkopts = linkopts,
        tags = tags,
        resource_group = resource_group,
        **_SIDECAR_ATTRIBUTES
    )
