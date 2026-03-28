# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

if(IREE_STREAMING_BUILD_HIP_BINDING)
  iree_cc_library(
    SHARED
    NAME
      hip
    HDRS
      "api.h"
    SRCS
      "api.c"
    COPTS
      -fPIC
    DEPS
      iree::base
      streaming::streaming
      iree::hal
    DEFINES
      "IREE_HAL_STREAMING_HIP_EXPORTS"
    PUBLIC
  )
endif()
