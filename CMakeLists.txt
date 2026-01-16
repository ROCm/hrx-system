# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.24)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(IREE_STREAMING C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Set default build type
set(DEFAULT_CMAKE_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, default to ${DEFAULT_CMAKE_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "${DEFAULT_CMAKE_BUILD_TYPE}" CACHE STRING "Build type (default ${DEFAULT_CMAKE_BUILD_TYPE})" FORCE)
endif()

#-------------------------------------------------------------------------------
# Project options
#-------------------------------------------------------------------------------

option(IREE_STREAMING_BUILD_TESTS "Build streaming tests and benchmarks" ON)
option(IREE_STREAMING_BUILD_CUDA_BINDING "Build CUDA binding" ON)
option(IREE_STREAMING_BUILD_HIP_BINDING "Build HIP binding" ON)
option(IREE_STREAMING_BUILD_HSA_DRIVER "Build external HSA HAL driver" ON)

#-------------------------------------------------------------------------------
# IREE configuration - we only need the runtime, not the compiler
#-------------------------------------------------------------------------------

# Disable the compiler build - we only need the runtime
set(IREE_BUILD_COMPILER OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_BINDINGS_TFLITE OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_BINDINGS_TFLITE_JAVA OFF CACHE BOOL "" FORCE)

# Enable HAL drivers we need
set(IREE_HAL_DRIVER_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_LOCAL_TASK ON CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_CUDA ON CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_HIP ON CACHE BOOL "" FORCE)

# Enable zstd for decompressing compressed HIP fat binaries (required for rocBLAS/hipBLAS)
set(IREE_HAL_DRIVER_ENABLE_ZSTD ON CACHE BOOL "" FORCE)

#-------------------------------------------------------------------------------
# External HSA HAL Driver Registration
# Must be done BEFORE add_subdirectory(iree) so the driver is discovered.
#-------------------------------------------------------------------------------
if(IREE_STREAMING_BUILD_HSA_DRIVER)
  # Register our external HSA driver with IREE's external driver system.
  # These properties are read by runtime/src/iree/hal/drivers/CMakeLists.txt.
  set_property(GLOBAL APPEND PROPERTY IREE_EXTERNAL_HAL_DRIVERS_AVAILABLE "hsa")
  set_property(GLOBAL PROPERTY "IREE_EXTERNAL_HSA_HAL_DRIVER_SOURCE_DIR"
    "${CMAKE_CURRENT_SOURCE_DIR}/hsa_driver")
  set_property(GLOBAL PROPERTY "IREE_EXTERNAL_HSA_HAL_DRIVER_BINARY_DIR"
    "${CMAKE_CURRENT_BINARY_DIR}/hsa_driver")
  set_property(GLOBAL PROPERTY "IREE_EXTERNAL_HSA_HAL_DRIVER_TARGET"
    "hsa_driver::registration")
  set_property(GLOBAL PROPERTY "IREE_EXTERNAL_HSA_HAL_DRIVER_REGISTER"
    "iree_hal_hsa_driver_module_register")
  set(IREE_EXTERNAL_HAL_DRIVERS "hsa" CACHE STRING "" FORCE)
endif()

# Add IREE as a subdirectory
add_subdirectory(third_party/iree EXCLUDE_FROM_ALL)

#-------------------------------------------------------------------------------
# Streaming library
#-------------------------------------------------------------------------------

# Set the package root for our streaming library. This tells IREE's cmake
# functions to compute package names relative to the streaming directory.
# The prefix "streaming" will be prepended to create package names like
# streaming::streaming, streaming::util, streaming::binding::cuda, etc.
set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/streaming")
set(IREE_PACKAGE_ROOT_PREFIX "streaming")

# Add project root to include path so #include "streaming/..." works
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add the streaming library
add_subdirectory(streaming)

# Clear the package root settings
unset(IREE_PACKAGE_ROOT_DIR)
unset(IREE_PACKAGE_ROOT_PREFIX)

add_subdirectory(third_party/hip-cts)