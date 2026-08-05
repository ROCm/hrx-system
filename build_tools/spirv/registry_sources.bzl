# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bazel-to-CMake declaration for the locked SPIR-V registry sources."""

def loom_spirv_registry_sources(name, **kwargs):
    """Declares the locked registry inputs consumed by SPIR-V generators.

    Bazel obtains these files from the repositories pinned in MODULE.bazel.
    The declaration gives bazel_to_cmake the equivalent locked CMake source
    paths before it converts generator arguments and inputs.

    Args:
      name: Package-local declaration name used in generated CMake diagnostics.
      **kwargs: Reserved build-conversion metadata. Ignored by Bazel.
    """

    _ = (name, kwargs)  # @unused
