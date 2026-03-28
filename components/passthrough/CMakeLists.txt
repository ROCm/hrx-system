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
    ${CMAKE_CURRENT_SOURCE_DIR}/..
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
    ${CMAKE_CURRENT_SOURCE_DIR}/..
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
# HIP No-Op Interceptor Library (libhip_noop)
#
# Interception library that does nothing — all HIP calls pass through directly.
# Use this when you want interception plumbing active but no logging overhead.
# Load via HIP_INTERCEPTION_LIBRARY environment variable.
#-------------------------------------------------------------------------------

add_library(hip_noop SHARED
  interceptors/passthrough_interceptor.c
)

target_include_directories(hip_noop
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_target_properties(hip_noop PROPERTIES
  OUTPUT_NAME "hip_noop"
  VERSION 1.0.0
  SOVERSION 1
)

#-------------------------------------------------------------------------------
# HIP Buffer Tracer Library (libhip_buffer_tracer)
#
# Interception library that tracks memory allocations and can dump buffer
# contents before and after kernel launches for debugging/comparison.
# Load via HIP_INTERCEPTION_LIBRARY environment variable.
#
# Environment variables:
# - HIP_TRACE_FILE: Path to trace output file (default: stderr)
# - HIP_TRACE_LEVEL: 0=off, 1=errors, 2=calls, 3=buffers, 4=verbose (default: 2)
# - HIP_TRACE_SYNC: 1=sync before/after kernel launches (default: 0)
# - HIP_TRACE_DUMP: 1=dump buffer contents, 2=dump hash only (default: 0)
# - HIP_TRACE_DUMP_MAX: Max bytes to dump per buffer (default: 1024)
# - HIP_TRACE_KERNEL_FILTER: Substring to match kernel names (empty=all)
# - HIP_TRACE_KERNEL_COUNT: Only trace first N kernels (default: 0=unlimited)
#-------------------------------------------------------------------------------

add_library(hip_buffer_tracer SHARED
  hip_buffer_tracer.c
)

target_include_directories(hip_buffer_tracer
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(hip_buffer_tracer
  PRIVATE
    pthread
)

set_target_properties(hip_buffer_tracer PROPERTIES
  OUTPUT_NAME "hip_buffer_tracer"
  VERSION 1.0.0
  SOVERSION 1
)

#-------------------------------------------------------------------------------
# Installation
#-------------------------------------------------------------------------------

install(TARGETS hip_intercept hip_logging hip_noop hip_buffer_tracer
  LIBRARY DESTINATION lib
)

install(FILES hip_function_table.h
  DESTINATION include/passthrough
)
