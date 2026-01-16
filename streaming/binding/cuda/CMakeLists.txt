# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

if(IREE_STREAMING_BUILD_CUDA_BINDING)
  iree_cc_library(
    NAME
      cuda
    HDRS
      "api.h"
      "driver.h"
      "runtime.h"
    SRCS
      "driver.c"
      "runtime.c"
    DEPS
      iree::base
      streaming::streaming
      iree::hal
    DEFINES
      "IREE_HAL_STREAMING_CUDA_EXPORTS"
    PUBLIC
  )
endif()
