# Copyright 2026 The Pyre Authors
# SPDX-License-Identifier: Apache-2.0
#
# Pyre Runtime: standalone shared library wrapping IREE HAL.
# Produces libpyre.so (LTO, hidden visibility, only pyre_* exported)
# and pyre-info (CLI tool).

cmake_minimum_required(VERSION 3.20)
project(pyre-runtime VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#===------------------------------------------------------------------------===#
# Options
#===------------------------------------------------------------------------===#

set(PYRE_IREE_SOURCE_DIR "" CACHE PATH
    "Path to IREE source tree (required)")
option(PYRE_ENABLE_TRACY
    "Build with IREE Tracy tracing integration" OFF)
option(PYRE_BUILD_CTS
    "Build conformance test suite" ON)

# Validate IREE source directory.
if(NOT PYRE_IREE_SOURCE_DIR OR
   NOT EXISTS "${PYRE_IREE_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "PYRE_IREE_SOURCE_DIR must point to an IREE source tree. "
    "Set -DPYRE_IREE_SOURCE_DIR=/path/to/iree")
endif()
message(STATUS "pyre-runtime: using IREE at ${PYRE_IREE_SOURCE_DIR}")

#===------------------------------------------------------------------------===#
# HSA runtime (direct link, part of ROCm)
#===------------------------------------------------------------------------===#

find_package(hsa-runtime64 REQUIRED)
message(STATUS "pyre-runtime: found HSA runtime")

#===------------------------------------------------------------------------===#
# IREE as source dependency (static build, runtime-only)
#===------------------------------------------------------------------------===#

set(IREE_BUILD_COMPILER OFF)
set(IREE_BUILD_TESTS OFF)
set(IREE_BUILD_SAMPLES OFF)
set(IREE_ERROR_ON_MISSING_SUBMODULES OFF)
set(IREE_VISIBILITY_HIDDEN OFF)

# CPU drivers only (local-sync + local-task).
set(IREE_HAL_DRIVER_DEFAULTS OFF)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON)
set(IREE_HAL_DRIVER_LOCAL_TASK ON)

# Enable Tracy if requested.
if(PYRE_ENABLE_TRACY)
  set(IREE_ENABLE_RUNTIME_TRACING ON)
  message(STATUS "pyre-runtime: Tracy tracing ENABLED")
endif()

# Force IREE to build static libs.
set(_pyre_saved_bsl ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(${PYRE_IREE_SOURCE_DIR} pyre_iree SYSTEM EXCLUDE_FROM_ALL)
set(BUILD_SHARED_LIBS ${_pyre_saved_bsl})

#===------------------------------------------------------------------------===#
# HSA HAL driver (from iree-hal-streaming, plain cmake build)
#===------------------------------------------------------------------------===#

add_library(pyre_hsa_driver STATIC
    src/hsa_driver/status_util.c
    src/hsa_driver/fat_binary.c
    src/hsa_driver/hsa_allocator.c
    src/hsa_driver/hsa_buffer.c
    src/hsa_driver/hsa_driver.c
    src/hsa_driver/hsa_device.c
    src/hsa_driver/hsa_semaphore.c
    src/hsa_driver/native_executable.c
    src/hsa_driver/nop_executable_cache.c
    src/hsa_driver/stream_command_buffer.c
)
target_include_directories(pyre_hsa_driver PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(pyre_hsa_driver PUBLIC
    hsa-runtime64::hsa-runtime64
    iree_base_base
    iree_base_core_headers
    iree_base_internal_arena
    iree_base_internal_atomic_slist
    iree_base_internal_flatcc_parsing
    iree_base_threading_threading
    iree_base_threading_thread
    iree_async_async
    iree_async_util_proactor_pool
    iree_hal_hal
    iree_hal_utils_collective_batch
    iree_hal_utils_deferred_command_buffer
    iree_hal_utils_executable_debug_info
    iree_hal_utils_executable_header
    iree_hal_utils_file_transfer
    iree_hal_utils_files
    iree_hal_utils_queue_emulation
    iree_hal_utils_queue_host_call_emulation
    iree_hal_utils_resource_set
    iree_schemas_executable_debug_info_c_fbs
    iree_schemas_hip_executable_def_c_fbs
)
set_target_properties(pyre_hsa_driver PROPERTIES POSITION_INDEPENDENT_CODE ON)

# HSA driver registration module.
add_library(pyre_hsa_registration STATIC
    src/hsa_driver/registration/driver_module.c
)
target_include_directories(pyre_hsa_registration PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(pyre_hsa_registration PUBLIC
    pyre_hsa_driver
    iree_base_base
    iree_base_core_headers
    iree_base_tooling_flags
    iree_hal_hal
)
set_target_properties(pyre_hsa_registration PROPERTIES POSITION_INDEPENDENT_CODE ON)

#===------------------------------------------------------------------------===#
# libpyre.so
#===------------------------------------------------------------------------===#

add_subdirectory(src/libpyre)

#===------------------------------------------------------------------------===#
# pyre-info — CLI tool, dual-links libpyre.so + IREE static (for flags)
#===------------------------------------------------------------------------===#

add_executable(pyre-info tools/pyre_info.c)
target_include_directories(pyre-info PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/libpyre
)
# Link against libpyre.so (public API) — validates symbol hiding.
target_link_libraries(pyre-info PRIVATE pyre)
# Also link IREE flags directly (static, in the binary itself).
# This forces dual-linkage: IREE inside libpyre.so AND in the binary.
target_link_libraries(pyre-info PRIVATE iree_base_tooling_flags)

#===------------------------------------------------------------------------===#
# CTS (conformance test suite)
#===------------------------------------------------------------------------===#

if(PYRE_BUILD_CTS)
  enable_testing()
  add_subdirectory(cts)
endif()

#===------------------------------------------------------------------------===#
# Passthrough / interception tools
#===------------------------------------------------------------------------===#

option(PYRE_BUILD_PASSTHROUGH
    "Build HIP passthrough/interception tools" ON)
if(PYRE_BUILD_PASSTHROUGH)
  add_subdirectory(src/passthrough)
endif()

#===------------------------------------------------------------------------===#
# cmake config for find_package(pyre)
#===------------------------------------------------------------------------===#

include(CMakePackageConfigHelpers)
install(TARGETS pyre EXPORT pyre-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include/pyre)
install(FILES include/pyre_runtime.h DESTINATION include/pyre)
install(EXPORT pyre-targets
    NAMESPACE pyre::
    DESTINATION lib/cmake/pyre)
configure_package_config_file(
    cmake/pyre-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/pyre-config.cmake
    INSTALL_DESTINATION lib/cmake/pyre)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pyre-config.cmake
    DESTINATION lib/cmake/pyre)
