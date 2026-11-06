# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared C source list for AMDGPU device bitcode generation."""

IREE_HAL_AMDGPU_DEVICE_BITCODE_SRCS = [
    "atomic.c",
    "blit.c",
    "dispatch.c",
    "grid_sync.c",
    "tsan.c",
    "timestamp.c",
]
