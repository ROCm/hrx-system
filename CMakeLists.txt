# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26...4.2)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# HRX is the top-level build/product. The runtime subtree is still the IREE
# runtime subsystem, and intentionally keeps IREE_* options and iree:: targets.
project(HRX VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# Static and object libraries need compile-time prerequisites, not completed
# archives for every transitive link dependency. CMake retains explicit ordering
# and code-generation dependencies when pruning these unnecessary build edges.
option(CMAKE_OPTIMIZE_DEPENDENCIES
  "Prune unnecessary static and object library build dependencies." ON)

# Native runtime targets require thread creation. Browser WebAssembly uses its
# host event loop, while bare-metal WebAssembly has no thread creation service.
# Neither target links a thread library. This is a target capability, not a
# build option.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _IREE_TARGET_SYSTEM_PROCESSOR)
if(EMSCRIPTEN OR _IREE_TARGET_SYSTEM_PROCESSOR MATCHES "^wasm(32|64)$")
  set(IREE_TARGET_HAS_THREADS OFF)
else()
  set(IREE_TARGET_HAS_THREADS ON)
endif()
unset(_IREE_TARGET_SYSTEM_PROCESSOR)

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

set(IREE_DEPENDENCY_MODE "pinned" CACHE STRING
  "How source dependencies are resolved: pinned, package, or auto.")
set_property(CACHE IREE_DEPENDENCY_MODE PROPERTY STRINGS pinned package auto)
set(IREE_ROCM_DEPENDENCY_MODE "" CACHE STRING
  "How ROCm header dependencies are resolved; empty uses package with IREE_ROCM_PATH, otherwise inherits IREE_DEPENDENCY_MODE.")
set_property(CACHE IREE_ROCM_DEPENDENCY_MODE PROPERTY STRINGS "" pinned package auto)

set(IREE_ROCM_PATH "" CACHE PATH
  "ROCm or TheRock SDK root used for ROCm headers and device tooling.")
if(IREE_ROCM_PATH)
  list(PREPEND CMAKE_PREFIX_PATH "${IREE_ROCM_PATH}")
endif()

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
option(LIBHRX_BUILD
  "Build libhrx and HRX compatibility targets." ON)

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
  "${CMAKE_CURRENT_LIST_DIR}/build_tools/spirv"
  "${CMAKE_CURRENT_LIST_DIR}/build_tools/sanitizer"
)

include("${CMAKE_CURRENT_LIST_DIR}/MODULE.cmake.lock")

include("${CMAKE_CURRENT_LIST_DIR}/runtime/project.cmake")
if(NOT DEFINED AMDF_BUILD_DEFAULT)
  set(AMDF_BUILD_DEFAULT ${IREE_HAL_DRIVER_XDNA})
endif()
include("${CMAKE_CURRENT_LIST_DIR}/libamdf/project.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/libhrx/project.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/loom/project.cmake")

# Shared repository CMake primitives.
include(iree_macros)
include(iree_copts)
include(iree_sanitizer_suppressions)
include(iree_cc_binary)
include(iree_cc_library)
include(iree_cc_test)
include(iree_cc_fuzz)
include(iree_import_binary)
include(iree_install_support)
include(iree_c_embed_data)
include(iree_spirv_asm_module)
include(iree_spirv_registry_sources)
include(iree_add_all_subdirs)
include(iree_configure_testing)
include(iree_native_test)
include(iree_execution_test_suite)
include(iree_cc_binary_benchmark)
include(iree_genrule)
include(iree_python)
include(iree_setup_toolchain)
include(hrx_optimization)

include("${CMAKE_CURRENT_LIST_DIR}/build_tools/third_party/googletest/googletest.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/build_tools/third_party/google_benchmark/google_benchmark.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/build_tools/third_party/libbacktrace/libbacktrace.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/build_tools/third_party/zstd/zstd.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/build_tools/third_party/rocm/headers.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/build_tools/third_party/rdma_core_headers/rdma_core_headers.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/build_tools/third_party/spirv_tools/spirv_tools.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/build_tools/third_party/vulkan_headers/vulkan_headers.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/build_tools/third_party/catch2/catch2.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/build_tools/third_party/cxx/cxx.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/build_tools/third_party/mimalloc/mimalloc.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/build_tools/third_party/wkmi/wkmi.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/build_tools/vulkan/vulkan.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/build_tools/d3d12/d3d12.cmake")

iree_setup_toolchain()

# Sanitizer runtimes own allocation interposition. Preserve the requested cache
# value while overriding the active provider for all downstream consumers.
if(IREE_ALLOCATOR_SYSTEM STREQUAL "mimalloc" AND
   (IREE_ENABLE_ASAN OR IREE_ENABLE_MSAN OR IREE_ENABLE_TSAN OR
    IREE_ENABLE_UBSAN))
  message(STATUS
    "Sanitizer instrumentation overrides IREE_ALLOCATOR_SYSTEM=mimalloc with libc")
  set(IREE_ALLOCATOR_SYSTEM "libc")
endif()

hrx_configure_ipo()
set(IREE_DEFAULT_EXECUTABLE_LINK_LIBRARIES "")
if(IREE_ALLOCATOR_SYSTEM STREQUAL "mimalloc")
  iree_configure_mimalloc()
  list(APPEND IREE_DEFAULT_EXECUTABLE_LINK_LIBRARIES
    "$<$<LINK_LANGUAGE:CXX>:$<LINK_LIBRARY:WHOLE_ARCHIVE,iree::third_party::mimalloc_new_delete>>")
endif()
if(IREE_BUILD_TESTS)
  iree_configure_googletest()
endif()
if(IREE_TARGET_HAS_THREADS AND (IREE_BUILD_TESTS OR IREE_BUILD_BENCHMARKS))
  iree_configure_google_benchmark()
endif()
iree_configure_libbacktrace()
if(IREE_NET_RDMA)
  iree_configure_rdma_core_headers()
endif()
if(LIBHRX_BUILD AND LIBHRX_BUILD_HIP_BINDING)
  iree_configure_zstd()
endif()
if(IREE_HAL_DRIVER_AMDGPU)
  iree_configure_rocm_hsa_runtime_headers()
  iree_configure_rocm_aqlprofile_sdk_headers()
endif()
if(IREE_VULKAN_AVAILABLE)
  iree_configure_vulkan_headers()
endif()
if(AMDF_BUILD AND WIN32 AND IREE_ARCH STREQUAL "x86_64" AND
   (AMDF_FAMILY_RDNA OR AMDF_FAMILY_CDNA))
  iree_configure_amdf_wkmi()
endif()
if(IREE_BUILD_TESTS AND (IREE_HAL_DRIVER_VULKAN OR
    (LOOM_BUILD AND LOOM_TARGET_ARCH_SPIRV AND LOOM_EMIT_SPIRV)))
  iree_configure_spirv_tools()
endif()
if(LIBHRX_BUILD AND LIBHRX_BUILD_CTS)
  iree_configure_catch2()
endif()
if(LOOM_BUILD AND LOOM_IMPORT_CXX)
  iree_configure_cxx()
endif()
iree_runtime_configure_project()
libhrx_configure_project()
loom_configure_project()

message(STATUS "HRX/IREE HAL drivers:")
if(IREE_HAL_DRIVER_AMDGPU)
  message(STATUS "  - amdgpu")
endif()
if(IREE_HAL_DRIVER_TASK)
  message(STATUS "  - task")
endif()
if(IREE_HAL_DRIVER_VULKAN)
  message(STATUS "  - vulkan")
endif()
if(IREE_HAL_DRIVER_XDNA)
  message(STATUS "  - xdna")
endif()

add_subdirectory(build_tools/cmake)
add_subdirectory(build_tools/embed_data)
add_subdirectory(build_tools/spirv)
include("${PROJECT_SOURCE_DIR}/loom/build_tools/cmake/loom_target_profile.cmake")
include("${PROJECT_SOURCE_DIR}/loom/build_tools/cmake/loom_kernel_binary.cmake")
if(AMDF_BUILD)
  add_subdirectory(libamdf)
endif()
add_subdirectory(runtime/src)
add_subdirectory(experimental EXCLUDE_FROM_ALL)

if(LOOM_BUILD)
  include("${PROJECT_SOURCE_DIR}/loom/build_tools/cmake/loom_module.cmake")
  include("${PROJECT_SOURCE_DIR}/loom/build_tools/cmake/loom_test.cmake")
  include("${PROJECT_SOURCE_DIR}/loom/build_tools/cmake/loom_corpus.cmake")
  add_subdirectory(loom/target)
  add_subdirectory(loom/py/loom)
  add_subdirectory(loom/src)
  block(SCOPE_FOR VARIABLES)
    set(IREE_PACKAGE_ROOT_DIR "${PROJECT_SOURCE_DIR}/loom/build_tools")
    set(IREE_PACKAGE_ROOT_PREFIX "loom/build_tools")
    add_subdirectory(loom/build_tools/corpus)
  endblock()
  add_subdirectory(loom/binding/c)
  hrx_installed_tests_install_source_tree(
    "${PROJECT_SOURCE_DIR}/loom/src/loom" "loom/src/loom"
    _HRX_INSTALLED_LOOM_SOURCE_TREE)
endif()

if(LIBHRX_BUILD)
  add_subdirectory(libhrx)
endif()

set(IREE_CMAKE_TRY_FILE "" CACHE FILEPATH
  "Optional developer-tool CMake file included after repository targets exist.")
mark_as_advanced(IREE_CMAKE_TRY_FILE)
if(IREE_CMAKE_TRY_FILE)
  include("${IREE_CMAKE_TRY_FILE}")
endif()

if(LOOM_BUILD)
  loom_finalize_corpus()
  loom_finalize_module_libraries()
  loom_finalize_execution_tests()
endif()
iree_finalize_target_dependencies()
iree_finalize_sanitizer_suppressions()
iree_finalize_execution_tool_environments()
if(IREE_BUILD_TESTS)
  iree_finalize_test_build_targets()
  iree_create_ctest_customization()
  hrx_create_installed_tests()
endif()

iree_write_target_alias_map()
