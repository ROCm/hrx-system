# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Get the project root directory (parent of hsa_driver/) for includes.
get_filename_component(_HSA_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
get_filename_component(_HSA_PROJECT_ROOT "${_HSA_PROJECT_ROOT}" DIRECTORY)

iree_cc_library(
  NAME
    registration
  HDRS
    "driver_module.h"
  SRCS
    "driver_module.c"
  INCLUDES
    "${_HSA_PROJECT_ROOT}"
  DEPS
    iree::base
    iree::base::core_headers
    iree::base::tooling::flags
    hsa_driver::hsa
    iree::hal
  PUBLIC
)
