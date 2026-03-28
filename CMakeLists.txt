# Copyright 2026 The Pyre Authors
# SPDX-License-Identifier: Apache-2.0
#
# Pyre Runtime: standalone shared library wrapping IREE HAL.
# Produces libpyre.so (LTO, hidden visibility, only pyre_* exported),
# pyre_internal (static, all symbols), and pyre-info (CLI tool).

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
# Source files
#===------------------------------------------------------------------------===#

set(PYRE_SOURCES
    src/status.c
    src/runtime.c
    src/device.c
    src/semaphore.c
    src/stream.c
    src/buffer.c
    src/queue_ops.c
)

set(PYRE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(PYRE_IREE_INCLUDE_DIR ${PYRE_IREE_SOURCE_DIR}/runtime/src)

#===------------------------------------------------------------------------===#
# libpyre.so — LTO'd shared library, only pyre_* symbols exported
#===------------------------------------------------------------------------===#

add_library(pyre SHARED ${PYRE_SOURCES})
target_include_directories(pyre
    PUBLIC
        $<BUILD_INTERFACE:${PYRE_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include/pyre>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(pyre PRIVATE iree_runtime_unified hsa-runtime64::hsa-runtime64)
target_compile_definitions(pyre PRIVATE PYRE_BUILDING_SHARED)
set_target_properties(pyre PROPERTIES
    OUTPUT_NAME pyre
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    C_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)

# Version script to hide all non-pyre_* symbols.
set_target_properties(pyre PROPERTIES
    LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/pyre_exports.lds")

# LTO (interprocedural optimization).
include(CheckIPOSupported)
check_ipo_supported(RESULT _pyre_ipo_supported OUTPUT _pyre_ipo_error)
if(_pyre_ipo_supported)
  set_target_properties(pyre PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  message(STATUS "pyre-runtime: LTO enabled for libpyre.so")
else()
  message(WARNING "pyre-runtime: LTO not supported: ${_pyre_ipo_error}")
endif()

#===------------------------------------------------------------------------===#
# pyre_internal — static library, no LTO, no hidden visibility
# For in-tree tools that need both pyre API and IREE internals.
#===------------------------------------------------------------------------===#

add_library(pyre_internal STATIC ${PYRE_SOURCES})
target_include_directories(pyre_internal
    PUBLIC
        $<BUILD_INTERFACE:${PYRE_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include/pyre>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(pyre_internal PUBLIC iree_runtime_unified hsa-runtime64::hsa-runtime64)
target_compile_definitions(pyre_internal PUBLIC PYRE_STATIC)
set_target_properties(pyre_internal PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

#===------------------------------------------------------------------------===#
# pyre-info — CLI tool, dual-links libpyre.so + IREE static (for flags)
#===------------------------------------------------------------------------===#

add_executable(pyre-info tools/pyre_info.c)
target_include_directories(pyre-info PRIVATE
    ${PYRE_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
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
