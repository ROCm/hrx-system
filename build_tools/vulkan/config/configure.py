# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Vulkan API-client Bazel configuration policy."""

from build_tools.bazel.configure_api import boolean_option_contributor

NATIVE_ENABLED_FLAG = "--//build_tools/vulkan/config:enabled"

CONTRIBUTOR = boolean_option_contributor(
    name="vulkan_api",
    default=True,
    portable_option="IREE_ENABLE_VULKAN",
    native_option=NATIVE_ENABLED_FLAG,
    section_heading="Vulkan API client selection; this does not enable a HAL driver.",
)
