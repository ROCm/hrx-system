# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_LIST_DIR}/../build_tools/amdgpu"
  "${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake"
)

option(IREE_BUILD_TESTS "Build IREE runtime unit tests and CTS targets." ON)
option(IREE_BUILD_BENCHMARKS "Build IREE runtime benchmarks." ON)

option(IREE_MODULE_VMVX
  "Builds the VMVX runtime module boundary."
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
  "Default value for opt-in runtime HAL drivers." OFF)
if(NOT DEFINED IREE_HAL_DRIVER_AMDGPU_DEFAULT)
  set(IREE_HAL_DRIVER_AMDGPU_DEFAULT ${IREE_HAL_DRIVER_DEFAULTS})
endif()
if(NOT DEFINED IREE_HAL_DRIVER_CUDA_DEFAULT)
  set(IREE_HAL_DRIVER_CUDA_DEFAULT ${IREE_HAL_DRIVER_DEFAULTS})
endif()
if(NOT DEFINED IREE_HAL_DRIVER_HIP_DEFAULT)
  set(IREE_HAL_DRIVER_HIP_DEFAULT ${IREE_HAL_DRIVER_DEFAULTS})
endif()
if(NOT DEFINED IREE_HAL_DRIVER_LOCAL_SYNC_DEFAULT)
  set(IREE_HAL_DRIVER_LOCAL_SYNC_DEFAULT ON)
endif()
if(NOT DEFINED IREE_HAL_DRIVER_LOCAL_TASK_DEFAULT)
  set(IREE_HAL_DRIVER_LOCAL_TASK_DEFAULT ON)
endif()
if(NOT DEFINED IREE_HAL_DRIVER_METAL_DEFAULT)
  set(IREE_HAL_DRIVER_METAL_DEFAULT ${IREE_HAL_DRIVER_DEFAULTS})
endif()
if(NOT DEFINED IREE_HAL_DRIVER_NULL_DEFAULT)
  set(IREE_HAL_DRIVER_NULL_DEFAULT ON)
endif()
if(NOT DEFINED IREE_HAL_DRIVER_VULKAN_DEFAULT)
  set(IREE_HAL_DRIVER_VULKAN_DEFAULT ${IREE_HAL_DRIVER_DEFAULTS})
endif()
if(NOT DEFINED IREE_HAL_DRIVER_WEBGPU_DEFAULT)
  set(IREE_HAL_DRIVER_WEBGPU_DEFAULT ${IREE_HAL_DRIVER_DEFAULTS})
endif()
option(IREE_HAL_DRIVER_AMDGPU
  "Enables the amdgpu runtime HAL driver."
  ${IREE_HAL_DRIVER_AMDGPU_DEFAULT})
option(IREE_HAL_DRIVER_CUDA
  "Enables the CUDA runtime HAL driver."
  ${IREE_HAL_DRIVER_CUDA_DEFAULT})
option(IREE_HAL_DRIVER_HIP
  "Enables the HIP runtime HAL driver."
  ${IREE_HAL_DRIVER_HIP_DEFAULT})
option(IREE_HAL_DRIVER_LOCAL_SYNC
  "Enables the local-sync runtime HAL driver."
  ${IREE_HAL_DRIVER_LOCAL_SYNC_DEFAULT})
option(IREE_HAL_DRIVER_LOCAL_TASK
  "Enables the local-task runtime HAL driver."
  ${IREE_HAL_DRIVER_LOCAL_TASK_DEFAULT})
option(IREE_HAL_DRIVER_METAL
  "Enables the Metal runtime HAL driver."
  ${IREE_HAL_DRIVER_METAL_DEFAULT})
option(IREE_HAL_DRIVER_NULL
  "Enables the null runtime HAL driver."
  ${IREE_HAL_DRIVER_NULL_DEFAULT})
option(IREE_HAL_DRIVER_VULKAN
  "Enables the Vulkan runtime HAL driver."
  ${IREE_HAL_DRIVER_VULKAN_DEFAULT})
option(IREE_HAL_DRIVER_WEBGPU
  "Enables the WebGPU runtime HAL driver."
  ${IREE_HAL_DRIVER_WEBGPU_DEFAULT})
option(IREE_HAL_DRIVER_HIP_RCCL
  "Enables RCCL collective support in the HIP HAL driver."
  OFF)
if(IREE_HAL_DRIVER_HIP_RCCL AND NOT IREE_HAL_DRIVER_HIP)
  message(FATAL_ERROR
    "IREE_HAL_DRIVER_HIP_RCCL=ON requires IREE_HAL_DRIVER_HIP=ON.")
endif()

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
if(IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE)
  set(IREE_MODULE_VMVX ON CACHE BOOL
    "Builds the VMVX runtime module boundary." FORCE)
endif()

option(IREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS
  "Sets the default value for all runtime HAL executable plugin mechanisms." ON)
option(IREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF
  "Enables the embedded dynamic library plugin mechanism for local HAL drivers."
  ${IREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS})
option(IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY
  "Enables the system dynamic library plugin mechanism for local HAL drivers."
  ${IREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS})

set(IREE_HAL_AMDGPU_TARGETS
  "gfx9-generic;gfx90a;gfx9-4-generic;gfx10-1-generic;gfx10-3-generic;gfx11-generic;gfx12-generic"
  CACHE STRING
  "AMDGPU target selectors supported by this runtime build.")

include(flatbuffer_c_library)
include(binary)
include(selectors)
include(iree_execution_test_suite)
include(iree_runtime_amdgpu_toolchain)
include(iree_vmasm_module)
include(iree_hal_cts_test_suite)
include(iree_wasm_library)

function(iree_runtime_configure_project)
  iree_runtime_configure_amdgpu_toolchain()
endfunction()
