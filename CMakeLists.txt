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

option(LIBHRX_BUILD
  "Build libhrx and HRX compatibility targets." ON)
option(LIBHRX_BUILD_PASSTHROUGH
  "Build libhrx HIP passthrough/interception tools." ON)
option(LIBHRX_BUILD_CUDA_BINDING
  "Build libhrx CUDA runtime/driver API compatibility binding." OFF)

option(IREE_BUILD_TESTS "Build IREE runtime unit tests and CTS targets." ON)
option(LIBHRX_BUILD_CTS
  "Build libhrx conformance tests." ${IREE_BUILD_TESTS})
option(HRX_INSTALL_TESTS
  "Install a relocatable CTest tree and test artifacts." ${IREE_BUILD_TESTS})
set(HRX_INSTALL_TESTS_COMPONENT "HrxSystemInstalledTests" CACHE STRING
  "Install component for the installed HRX system test suite.")
set(HRX_INSTALL_TESTS_DIR "${CMAKE_INSTALL_DATADIR}/hrx-system/tests" CACHE STRING
  "Install directory for the HRX system CTest tree and test artifacts.")
cmake_dependent_option(IREE_BUILD_BENCHMARKS
  "Build IREE runtime benchmarks." ON "NOT IREE_BUILD_TESTS" ON)

set(IREE_BUILD_COMPILER OFF CACHE BOOL
  "The reduced HRX runtime tree does not build the IREE compiler." FORCE)
option(IREE_MODULE_VMVX
  "Builds the VMVX runtime module. Disabled until builtins/ukernel is imported."
  OFF)
set(IREE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)

option(IREE_ENABLE_RUNTIME_TRACING "Enables instrumented runtime tracing." OFF)
set(IREE_TRACING_PROVIDER_DEFAULT "tracy" CACHE STRING
  "Default tracing implementation.")
set(IREE_TRACING_PROVIDER ${IREE_TRACING_PROVIDER_DEFAULT} CACHE STRING
  "Chooses which built-in tracing implementation is used when tracing is enabled.")
set(IREE_TRACING_PROVIDER_H "" CACHE STRING
  "Header file for custom tracing providers.")
set(IREE_TRACING_MODE_DEFAULT "2" CACHE STRING
  "Default tracing feature/verbosity mode. See iree/base/tracing.h for more.")
set(IREE_TRACING_MODE ${IREE_TRACING_MODE_DEFAULT} CACHE STRING
  "Tracing feature/verbosity mode. See iree/base/tracing.h for more.")

option(IREE_ENABLE_THREADING "Builds IREE with thread library support." ON)
option(IREE_SYNCHRONIZATION_DISABLE_UNSAFE
  "Disables synchronization primitives for single-threaded bare-metal targets only."
  OFF)
option(IREE_ENABLE_POSIX "Builds IREE with POSIX support." ON)
option(IREE_ENABLE_CPUINFO
  "Enables runtime use of cpuinfo for processor topology detection." OFF)
option(IREE_ENABLE_LIBBACKTRACE
  "Enables libbacktrace for Linux stack traces." OFF)

set(IREE_ALLOCATOR_SYSTEM "libc" CACHE STRING
  "Default named iree_allocator_t library and function base name.")

option(BUILD_SHARED_LIBS
  "Instructs CMake to build libraries as shared if possible." OFF)
set(IREE_RUNTIME_OPTIMIZATION_PROFILE "" CACHE STRING
  "Runtime optimization profile to apply. One of '', 'lto', 'size'.")
set(IREE_LTO_MODE "full" CACHE STRING
  "LTO type, 'thin' or 'full'. Only consulted on clang-like compilers.")
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
option(IREE_ENABLE_RUNTIME_COVERAGE
  "Enable LLVM code coverage of the runtime." OFF)
set(IREE_RUNTIME_COVERAGE_OBJECTS "" CACHE STRING
  "Archives, objects, or binaries to use as coverage data sources.")
set_property(GLOBAL PROPERTY IREE_RUNTIME_COVERAGE_TARGETS "")

set(IREE_USE_LINKER "" CACHE STRING "")
cmake_dependent_option(IREE_ENABLE_LLD
  "Override the system default linker to lld." OFF "NOT APPLE" OFF)

set(IREE_HOST_BIN_DIR "" CACHE PATH
  "Path to directory containing IREE host tools to use instead of building them.")

set(IREE_EXTERNAL_HAL_DRIVERS "" CACHE STRING "")
set(IREE_HAL_EXECUTABLE_LOADER_EXTRA_DEPS "" CACHE STRING "")
set(IREE_HAL_EXECUTABLE_PLUGIN_EXTRA_DEPS "" CACHE STRING "")

option(IREE_HAL_DRIVER_DEFAULTS
  "Sets the default value for all runtime HAL drivers." OFF)
option(IREE_HAL_DRIVER_AMDGPU "Enables the amdgpu runtime HAL driver." ON)
option(IREE_HAL_DRIVER_LOCAL_SYNC "Enables the local-sync runtime HAL driver." ON)
option(IREE_HAL_DRIVER_LOCAL_TASK "Enables the local-task runtime HAL driver." ON)
option(IREE_HAL_DRIVER_NULL "Enables the null runtime HAL driver." ON)
set(IREE_HAL_DRIVER_CUDA OFF CACHE BOOL
  "CUDA HAL driver is not wired in the reduced HRX runtime tree." FORCE)
set(IREE_HAL_DRIVER_HIP OFF CACHE BOOL
  "HIP HAL driver is not wired in the reduced HRX runtime tree." FORCE)
set(IREE_HAL_DRIVER_METAL OFF CACHE BOOL
  "Metal HAL driver is not wired in the reduced HRX runtime tree." FORCE)
set(IREE_HAL_DRIVER_VULKAN OFF CACHE BOOL
  "Vulkan HAL driver is not wired in the reduced HRX runtime tree." FORCE)
set(IREE_HAL_DRIVER_WEBGPU OFF CACHE BOOL
  "WebGPU HAL driver is not wired in the reduced HRX runtime tree." FORCE)

option(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS
  "Sets the default value for all runtime HAL executable loaders." ON)
set(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF_DEFAULT
  ${IREE_HAL_EXECUTABLE_LOADER_DEFAULTS})
set(IREE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY_DEFAULT
  ${IREE_HAL_EXECUTABLE_LOADER_DEFAULTS})
set(IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE_DEFAULT OFF)
option(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF
  "Enables the embedded dynamic library loader for local HAL drivers."
  ${IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF_DEFAULT})
option(IREE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY
  "Enables the system dynamic library loader for local HAL drivers."
  ${IREE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY_DEFAULT})
option(IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE
  "Enables the VMVX module loader for local HAL drivers."
  ${IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE_DEFAULT})

option(IREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS
  "Sets the default value for all runtime HAL executable plugin mechanisms." ON)
option(IREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF
  "Enables the embedded dynamic library plugin mechanism for local HAL drivers."
  ${IREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS})
option(IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY
  "Enables the system dynamic library plugin mechanism for local HAL drivers."
  ${IREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS})

set(IREE_ROCM_TEST_TARGET_CHIP "" CACHE STRING
  "Target chip for ROCm tests that need to compile device code. Empty disables live ROCm CTS.")
set(IREE_HIP_TEST_TARGET_CHIP "" CACHE STRING
  "Deprecated; use IREE_ROCM_TEST_TARGET_CHIP instead.")
if(IREE_HIP_TEST_TARGET_CHIP AND NOT IREE_ROCM_TEST_TARGET_CHIP)
  message(WARNING
    "IREE_HIP_TEST_TARGET_CHIP is deprecated; use IREE_ROCM_TEST_TARGET_CHIP instead.")
  set(IREE_ROCM_TEST_TARGET_CHIP "${IREE_HIP_TEST_TARGET_CHIP}" CACHE STRING "" FORCE)
endif()
set(IREE_ROCM_TEST_AMDGCNSPIRV OFF CACHE BOOL
  "Use amdgcnspirv for ROCm e2e tests.")

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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake")

include(iree_macros)
include(iree_copts)
include(iree_cc_binary)
include(iree_cc_library)
include(iree_cc_test)
include(iree_cc_fuzz)
include(iree_import_binary)
include(iree_install_support)
include(iree_c_embed_data)
include(iree_amdgpu_binary)
include(iree_bitcode_library)
include(iree_vmasm_module)
include(iree_add_all_subdirs)
include(iree_check_test)
include(iree_configure_testing)
include(iree_native_test)
include(iree_cc_binary_benchmark)
include(iree_hal_cts_test_suite)
include(iree_genrule)
include(iree_wasm_library)
include(iree_python)
include(iree_lit_test)
include(external_cc_library)
include(flatbuffer_c_library)
include(hrx_dependencies)
include(hrx_installed_tests)
include(iree_setup_toolchain)

iree_setup_toolchain()
hrx_configure_dependencies()

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
