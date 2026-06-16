# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# This header licenses this build file only. The shim sources in this directory
# are vendored under their own per-file licenses (Apache-2.0; GPL-2.0 WITH
# Linux-syscall-note; dual Apache-2.0/GPL-2.0) -- see shim/README.md for
# provenance and the per-file license map.

iree_cc_library(
  NAME
    shim-xdna
  SRCS
    amdxdna_accel.h
    bo.cpp
    bo_flags.h
    bo.h
    device.cpp
    device.h
    fence.cpp
    fence.h
    hwctx.cpp
    hwctx.h
    hwq.cpp
    hwq.h
    kernel.cpp
    kernel.h
    shim_debug.cpp
    shim_debug.h
  DEFINES
    $<$<CONFIG:Debug>:SHIM_XDNA_DEBUG>
  PUBLIC
)
