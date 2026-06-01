# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake"
)

option(IREE_BUILD_TESTS "Build IREE runtime unit tests and CTS targets." ON)
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
set(IREE_RUNTIME_OPTIMIZATION_PROFILE "" CACHE STRING
  "Runtime optimization profile to apply. One of '', 'lto', 'size'.")
set(IREE_LTO_MODE "full" CACHE STRING
  "LTO type, 'thin' or 'full'. Only consulted on clang-like compilers.")
option(IREE_ENABLE_RUNTIME_COVERAGE
  "Enable LLVM code coverage of the runtime." OFF)
set(IREE_RUNTIME_COVERAGE_OBJECTS "" CACHE STRING
  "Archives, objects, or binaries to use as coverage data sources.")
set_property(GLOBAL PROPERTY IREE_RUNTIME_COVERAGE_TARGETS "")

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

include("${CMAKE_CURRENT_LIST_DIR}/build_tools/third_party/flatcc/flatcc.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/build_tools/third_party/libbacktrace/libbacktrace.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/build_tools/third_party/rocm_headers.cmake")

include(flatbuffer_c_library)
include(iree_amdgpu_binary)
include(iree_execution_test_suite)
include(iree_runtime_amdgpu_toolchain)
include(iree_vmasm_module)
include(iree_hal_cts_test_suite)
include(iree_wasm_library)

function(iree_runtime_configure_project)
  iree_runtime_configure_amdgpu_toolchain()
  iree_runtime_configure_rocm_headers()
  iree_runtime_configure_libbacktrace()
  iree_configure_flatcc()
endfunction()
