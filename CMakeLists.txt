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
option(HRX_ENABLE_IREE_AMDGPU
    "Enable IREE's built-in AMDGPU HAL driver" ON)
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
if(HRX_ENABLE_IREE_AMDGPU)
  set(IREE_HAL_DRIVER_AMDGPU ON)
  if(NOT DEFINED IREE_ROCM_TEST_TARGET_CHIP)
    set(IREE_ROCM_TEST_TARGET_CHIP "" CACHE STRING
        "ROCm test target chip for the embedded IREE build" FORCE)
  endif()
  # Runtime-only IREE builds skip LLVM setup, but the AMDGPU driver still needs
  # host clang/llvm-link/lld to compile its embedded device libraries.
  if(NOT IREE_BUILD_COMPILER)
    if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
      set(IREE_CLANG_BINARY "${CMAKE_C_COMPILER}" CACHE FILEPATH
          "Host clang used by the embedded IREE AMDGPU build" FORCE)
    else()
      find_program(_hrx_clang_binary clang REQUIRED)
      set(IREE_CLANG_BINARY "${_hrx_clang_binary}" CACHE FILEPATH
          "Host clang used by the embedded IREE AMDGPU build" FORCE)
    endif()
    execute_process(
        COMMAND "${IREE_CLANG_BINARY}" -print-prog-name=llvm-link
        OUTPUT_VARIABLE _hrx_llvm_link_binary
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
        COMMAND "${IREE_CLANG_BINARY}" -print-prog-name=ld.lld
        OUTPUT_VARIABLE _hrx_lld_binary
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(EXISTS "${_hrx_llvm_link_binary}")
      set(IREE_LLVM_LINK_BINARY "${_hrx_llvm_link_binary}" CACHE FILEPATH
          "Host llvm-link used by the embedded IREE AMDGPU build" FORCE)
    else()
      find_program(_hrx_llvm_link_fallback llvm-link REQUIRED)
      set(IREE_LLVM_LINK_BINARY "${_hrx_llvm_link_fallback}" CACHE FILEPATH
          "Host llvm-link used by the embedded IREE AMDGPU build" FORCE)
    endif()
    if(EXISTS "${_hrx_lld_binary}")
      set(IREE_LLD_BINARY "${_hrx_lld_binary}" CACHE FILEPATH
          "Host lld used by the embedded IREE AMDGPU build" FORCE)
    else()
      find_program(_hrx_lld_fallback NAMES lld ld.lld REQUIRED)
      set(IREE_LLD_BINARY "${_hrx_lld_fallback}" CACHE FILEPATH
          "Host lld used by the embedded IREE AMDGPU build" FORCE)
    endif()
    if(NOT IREE_CLANG_BUILTIN_HEADERS_PATH)
      execute_process(
          COMMAND "${IREE_CLANG_BINARY}" -print-resource-dir
          OUTPUT_VARIABLE _hrx_clang_resource_dir
          OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(NOT _hrx_clang_resource_dir)
        message(FATAL_ERROR
          "Failed to determine clang resource dir for ${IREE_CLANG_BINARY}")
      endif()
      set(IREE_CLANG_BUILTIN_HEADERS_PATH
          "${_hrx_clang_resource_dir}/include")
    endif()
  endif()
endif()
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
