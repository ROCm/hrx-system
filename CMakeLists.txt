# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0
#
# HRX (HIP Runtime Extended): standalone shared library wrapping IREE HAL.
# Produces libhrx.so (LTO, hidden visibility, only hrx_* exported)
# and hrx-info (CLI tool).

cmake_minimum_required(VERSION 3.20)
project(hrx-runtime VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#===------------------------------------------------------------------------===#
# Options
#===------------------------------------------------------------------------===#

set(HRX_IREE_SOURCE_DIR "" CACHE PATH
    "Path to IREE source tree (required)")
option(HRX_ENABLE_TRACY
    "Build with IREE Tracy tracing integration" OFF)
option(HRX_BUILD_CTS
    "Build conformance test suite" ON)
option(HRX_ENABLE_ASAN "Enable address sanitizer" OFF)
option(HRX_ENABLE_MSAN "Enable memory sanitizer" OFF)
option(HRX_ENABLE_TSAN "Enable thread sanitizer" OFF)
option(HRX_ENABLE_UBSAN "Enable undefined behavior sanitizer" OFF)

# Validate IREE source directory.
if(NOT HRX_IREE_SOURCE_DIR OR
   NOT EXISTS "${HRX_IREE_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "HRX_IREE_SOURCE_DIR must point to an IREE source tree. "
    "Set -DHRX_IREE_SOURCE_DIR=/path/to/iree")
endif()
message(STATUS "hrx-runtime: using IREE at ${HRX_IREE_SOURCE_DIR}")

#===------------------------------------------------------------------------===#
# HSA runtime (direct link, part of ROCm)
#===------------------------------------------------------------------------===#

find_package(hsa-runtime64 REQUIRED)
message(STATUS "hrx-runtime: found HSA runtime")

#===------------------------------------------------------------------------===#
# IREE as source dependency (static build, runtime-only)
#===------------------------------------------------------------------------===#

set(IREE_BUILD_COMPILER OFF)
if(BUILD_TESTING)
  set(IREE_BUILD_TESTS ON)
else()
  set(IREE_BUILD_TESTS OFF)
endif()
set(IREE_BUILD_SAMPLES OFF)
set(IREE_ERROR_ON_MISSING_SUBMODULES OFF)
set(IREE_VISIBILITY_HIDDEN OFF)
set(IREE_ALLOCATOR_SYSTEM "mimalloc" CACHE STRING "" FORCE)

# CPU drivers only (local-sync + local-task).
set(IREE_HAL_DRIVER_DEFAULTS OFF)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON)
set(IREE_HAL_DRIVER_LOCAL_TASK ON)
set(IREE_HAL_DRIVER_REMOTE ON)

# Sanitizers — propagate flags to IREE (before add_subdirectory).
if(HRX_ENABLE_ASAN)
  set(IREE_ENABLE_ASAN ON)
endif()
if(HRX_ENABLE_MSAN)
  set(IREE_ENABLE_MSAN ON)
endif()
if(HRX_ENABLE_TSAN)
  set(IREE_ENABLE_TSAN ON)
endif()
if(HRX_ENABLE_UBSAN)
  set(IREE_ENABLE_UBSAN ON)
endif()

# Enable Tracy if requested.
if(HRX_ENABLE_TRACY)
  set(IREE_ENABLE_RUNTIME_TRACING ON)
  message(STATUS "hrx-runtime: Tracy tracing ENABLED")
endif()

# Force IREE to build static libs.
set(_hrx_saved_bsl ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(${HRX_IREE_SOURCE_DIR} hrx_iree SYSTEM EXCLUDE_FROM_ALL)
set(BUILD_SHARED_LIBS ${_hrx_saved_bsl})

# Sanitizer flags for hrx targets (after IREE, which manages its own).
if(HRX_ENABLE_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address -shared-libasan)
endif()
if(HRX_ENABLE_MSAN)
  add_compile_options(-fsanitize=memory -fno-omit-frame-pointer)
  add_link_options(-fsanitize=memory)
endif()
if(HRX_ENABLE_TSAN)
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()
if(HRX_ENABLE_UBSAN)
  add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=undefined)
endif()

#===------------------------------------------------------------------------===#
# HSA HAL driver (from iree-hal-streaming, plain cmake build)
#===------------------------------------------------------------------------===#

add_library(hrx_hsa_driver STATIC
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
target_include_directories(hrx_hsa_driver PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(hrx_hsa_driver PUBLIC
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
set_target_properties(hrx_hsa_driver PROPERTIES POSITION_INDEPENDENT_CODE ON)

# HSA driver registration module.
add_library(hrx_hsa_registration STATIC
    src/hsa_driver/registration/driver_module.c
)
target_include_directories(hrx_hsa_registration PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(hrx_hsa_registration PUBLIC
    hrx_hsa_driver
    iree_base_base
    iree_base_core_headers
    iree_base_tooling_flags
    iree_hal_hal
)
set_target_properties(hrx_hsa_registration PROPERTIES POSITION_INDEPENDENT_CODE ON)

#===------------------------------------------------------------------------===#
# libhrx.so
#===------------------------------------------------------------------------===#

add_subdirectory(src/libhrx)

#===------------------------------------------------------------------------===#
# Streaming HIP runtime (internal static library)
#===------------------------------------------------------------------------===#

add_subdirectory(src/streaming)

#===------------------------------------------------------------------------===#
# API bindings
#===------------------------------------------------------------------------===#

add_subdirectory(src/binding/hip)
add_subdirectory(src/binding/cuda)

#===------------------------------------------------------------------------===#
# hrx-info — CLI tool, dual-links libhrx.so + IREE static (for flags)
#===------------------------------------------------------------------------===#

add_executable(hrx-info tools/hrx_info.c)
target_include_directories(hrx-info PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/libhrx
)
# Link against libhrx.so (public API) — validates symbol hiding.
target_link_libraries(hrx-info PRIVATE hrx)
# Also link IREE flags directly (static, in the binary itself).
# This forces dual-linkage: IREE inside libhrx.so AND in the binary.
target_link_libraries(hrx-info PRIVATE iree_base_tooling_flags)

#===------------------------------------------------------------------------===#
# CTS (conformance test suite)
#===------------------------------------------------------------------------===#

if(HRX_BUILD_CTS)
  enable_testing()
  add_subdirectory(cts)
endif()

#===------------------------------------------------------------------------===#
# Passthrough / interception tools
#===------------------------------------------------------------------------===#

option(HRX_BUILD_PASSTHROUGH
    "Build HIP passthrough/interception tools" ON)
if(HRX_BUILD_PASSTHROUGH)
  add_subdirectory(src/passthrough)
endif()

#===------------------------------------------------------------------------===#
# cmake config for find_package(hrx)
#===------------------------------------------------------------------------===#

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

add_library(hrx::hrx ALIAS hrx)

set(_hrx_cmake_package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/hrx")
set(_hrx_build_package_dir "${CMAKE_CURRENT_BINARY_DIR}/cmake/hrx")

install(TARGETS hrx EXPORT hrx-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hrx)
install(FILES
    include/hrx_compiler.h
    include/hrx_compiler_cxx.h
    include/hrx_runtime.h
    include/hrx_runtime_cxx.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hrx)
install(EXPORT hrx-targets
    NAMESPACE hrx::
    DESTINATION ${_hrx_cmake_package_dir})

configure_package_config_file(
    cmake/hrx-config.cmake.in
    ${_hrx_build_package_dir}/hrx-config.cmake
    INSTALL_DESTINATION ${_hrx_cmake_package_dir})
write_basic_package_version_file(
    ${_hrx_build_package_dir}/hrx-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
export(EXPORT hrx-targets
    NAMESPACE hrx::
    FILE ${_hrx_build_package_dir}/hrx-targets.cmake)

install(FILES
    ${_hrx_build_package_dir}/hrx-config.cmake
    ${_hrx_build_package_dir}/hrx-config-version.cmake
    DESTINATION ${_hrx_cmake_package_dir})
