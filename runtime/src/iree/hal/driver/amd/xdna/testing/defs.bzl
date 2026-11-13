# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared build constraints for native XDNA queue tests."""

# These tests implement libamdf's native queue wake protocol, which uses an
# eventfd or Win32 Event rather than the pipe events used by other platforms.
NATIVE_QUEUE_TEST_PLATFORMS = select({
    "@platforms//os:android": [],
    "@platforms//os:linux": [],
    "@platforms//os:windows": [],
    "//conditions:default": ["@platforms//:incompatible"],
})
