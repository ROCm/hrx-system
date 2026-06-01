# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26...4.2)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# HRX is the top-level build/product. The runtime subtree is still the IREE
# runtime subsystem, and intentionally keeps IREE_* options and iree:: targets.
project(HRX VERSION 0.1.0 LANGUAGES C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(HRX_IDE_FOLDER HRX)
set(IREE_IDE_FOLDER IREE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(DEFAULT_CMAKE_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, default to ${DEFAULT_CMAKE_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "${DEFAULT_CMAKE_BUILD_TYPE}" CACHE STRING
      "Build type (default ${DEFAULT_CMAKE_BUILD_TYPE})" FORCE)
endif()

include(CMakeDependentOption)
include(CTest)
include(GNUInstallDirs)
find_package(Python3 COMPONENTS Interpreter QUIET)

option(HRX_HERMETIC_BUILD
  "Require all dependencies to be provided by package discovery; never FetchContent."
  OFF)
option(IREE_HERMETIC_BUILD
  "Require IREE/shared dependencies to be provided by package discovery; never FetchContent."
  ${HRX_HERMETIC_BUILD})

option(BUILD_SHARED_LIBS
  "Instructs CMake to build libraries as shared if possible." OFF)
option(IREE_VISIBILITY_HIDDEN "Builds C/C++ libraries with hidden visibility." ON)
option(IREE_ENABLE_POSITION_INDEPENDENT_CODE
  "Enable position independent code." TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ${IREE_ENABLE_POSITION_INDEPENDENT_CODE})

option(IREE_ENABLE_ASAN "Enable address sanitizer." OFF)
option(IREE_ENABLE_MSAN "Enable memory sanitizer." OFF)
option(IREE_ENABLE_TSAN "Enable thread sanitizer." OFF)
option(IREE_ENABLE_UBSAN "Enable undefined behavior sanitizer." OFF)
option(IREE_ENABLE_FUZZING "Enable libFuzzer-based fuzz targets." OFF)
option(IREE_ENABLE_SPLIT_DWARF
  "Enable gsplit-dwarf for debug information if supported." OFF)
option(IREE_ENABLE_THIN_ARCHIVES
  "Enables thin ar archives on ELF systems." OFF)
option(IREE_ENABLE_WERROR_FLAG "Treat warnings as errors for IREE code." ON)
option(IREE_ENABLE_ASSERTIONS "Force unset of NDEBUG compile option." OFF)
option(IREE_DEV_MODE
  "Configure settings optimized for IREE development instead of CI/release."
  OFF)

set(IREE_USE_LINKER "" CACHE STRING "")
cmake_dependent_option(IREE_ENABLE_LLD
  "Override the system default linker to lld." OFF "NOT APPLE" OFF)

function(hrx_fix_ndebug)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _UPPERCASE_CMAKE_BUILD_TYPE)
  if(IREE_ENABLE_ASSERTIONS AND NOT "${_UPPERCASE_CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
    foreach(_FLAGS_VAR_TO_SCRUB
            CMAKE_CXX_FLAGS_${_UPPERCASE_CMAKE_BUILD_TYPE}
            CMAKE_C_FLAGS_${_UPPERCASE_CMAKE_BUILD_TYPE})
      set(_ORIGINAL_FLAGS "${${_FLAGS_VAR_TO_SCRUB}}")
      string(REGEX REPLACE "(^| )[/-]D *NDEBUG($| )" " " _ALTERED_FLAGS
             "${_ORIGINAL_FLAGS}")
      if(NOT "${_ORIGINAL_FLAGS}" STREQUAL "${_ALTERED_FLAGS}")
        message(STATUS
          "IREE_ENABLE_ASSERTIONS force disabled NDEBUG for ${_FLAGS_VAR_TO_SCRUB}: '${_ORIGINAL_FLAGS}' -> '${_ALTERED_FLAGS}'")
        set(${_FLAGS_VAR_TO_SCRUB} "${_ALTERED_FLAGS}" PARENT_SCOPE)
      endif()
    endforeach()
  endif()
endfunction()
hrx_fix_ndebug()

list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake"
)

include("${CMAKE_CURRENT_LIST_DIR}/runtime/project.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/libhrx/project.cmake")

# Shared repository CMake primitives.
include(iree_macros)
include(iree_copts)
include(iree_cc_binary)
include(iree_cc_library)
include(iree_cc_test)
include(iree_cc_fuzz)
include(iree_import_binary)
include(iree_install_support)
include(iree_c_embed_data)
include(iree_add_all_subdirs)
include(iree_configure_testing)
include(iree_native_test)
include(iree_cc_binary_benchmark)
include(iree_genrule)
include(iree_python)
include(iree_setup_toolchain)

include("${CMAKE_CURRENT_LIST_DIR}/build_tools/third_party/googletest/googletest.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/build_tools/third_party/google_benchmark/google_benchmark.cmake")

iree_setup_toolchain()
iree_configure_googletest()
iree_configure_google_benchmark()
iree_runtime_configure_project()
libhrx_configure_project()

message(STATUS "HRX/IREE HAL drivers:")
if(IREE_HAL_DRIVER_AMDGPU)
  message(STATUS "  - amdgpu")
endif()
if(IREE_HAL_DRIVER_LOCAL_SYNC)
  message(STATUS "  - local-sync")
endif()
if(IREE_HAL_DRIVER_LOCAL_TASK)
  message(STATUS "  - local-task")
endif()
if(IREE_HAL_DRIVER_NULL)
  message(STATUS "  - null")
endif()

add_subdirectory(build_tools/embed_data)
add_subdirectory(runtime/src)

if(LIBHRX_BUILD)
  add_subdirectory(libhrx)
endif()

if(IREE_BUILD_TESTS)
  iree_create_ctest_customization()
  hrx_create_installed_tests()
endif()
