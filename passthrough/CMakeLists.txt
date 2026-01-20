# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#-------------------------------------------------------------------------------
# HIP Intercept Library (libhip_intercept)
#
# Main library that gets symlinked in place of the real HIP library.
# Loads the real HIP library and optionally an interception library.
#
# Usage:
#   1. Symlink this library in place of the real HIP library
#   2. Set HIP_PASSTHROUGH_BACKEND_LIB to the real HIP library path
#   3. Optionally set HIP_INTERCEPTION_LIBRARY to load an interception library
#
# Environment variables:
# - HIP_PASSTHROUGH_BACKEND_LIB: Path to real HIP library (default: /opt/rocm/lib/libamdhip64.so)
# - HIP_INTERCEPTION_LIBRARY: Path to interception library (optional)
#-------------------------------------------------------------------------------

add_library(hip_intercept SHARED
  hip_intercept.c
)

target_include_directories(hip_intercept
  PUBLIC
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(hip_intercept
  PRIVATE
    dl
    pthread
)

set_target_properties(hip_intercept PROPERTIES
  OUTPUT_NAME "hip_intercept"
  VERSION 1.0.0
  SOVERSION 1
)

#-------------------------------------------------------------------------------
# HIP Logging Library (libhip_logging)
#
# Interception library that logs all HIP API calls.
# Load via HIP_INTERCEPTION_LIBRARY environment variable.
#
# Environment variables:
# - HIP_LOG_FILE: Path to log file (default: stderr)
# - HIP_LOG_LEVEL: 0=off, 1=errors, 2=calls, 3=verbose (default: 2)
#-------------------------------------------------------------------------------

add_library(hip_logging SHARED
  hip_logging.c
)

target_include_directories(hip_logging
  PUBLIC
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(hip_logging
  PRIVATE
    pthread
)

set_target_properties(hip_logging PROPERTIES
  OUTPUT_NAME "hip_logging"
  VERSION 1.0.0
  SOVERSION 1
)

#-------------------------------------------------------------------------------
# Installation
#-------------------------------------------------------------------------------

install(TARGETS hip_intercept hip_logging
  LIBRARY DESTINATION lib
)

install(FILES hip_function_table.h
  DESTINATION include/passthrough
)
