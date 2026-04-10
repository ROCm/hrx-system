# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0
#
# HRX Conformance Test Suite (CTS).
# Can be built in-tree (add_subdirectory from hrx-runtime) or standalone
# (find_package(hrx) for header, dlopen at runtime for library).

cmake_minimum_required(VERSION 3.20)
project(hrx-cts LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Standalone vs in-tree detection.
if(NOT TARGET hrx)
  find_package(hrx REQUIRED)
endif()

# Catch2.
find_package(Catch2 3 REQUIRED)
include(CTest)

#===------------------------------------------------------------------------===#
# Core infrastructure
#===------------------------------------------------------------------------===#

add_library(hrx_cts_core STATIC
    core/hrx_loader.cpp
    core/main.cpp
)
target_include_directories(hrx_cts_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    $<TARGET_PROPERTY:hrx,INTERFACE_INCLUDE_DIRECTORIES>
)
target_compile_definitions(hrx_cts_core PUBLIC
    HRX_CTS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(hrx_cts_core PUBLIC Catch2::Catch2 ${CMAKE_DL_LIBS})

#===------------------------------------------------------------------------===#
# Helper macro: add a CTS test category
#===------------------------------------------------------------------------===#

function(add_hrx_cts_test name)
  cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})
  add_executable(hrx_cts_${name} ${ARG_SOURCES})
  target_link_libraries(hrx_cts_${name} PRIVATE hrx_cts_core Catch2::Catch2)
  if(TARGET hrx)
    add_dependencies(hrx_cts_${name} hrx)
  endif()
  add_test(NAME hrx_cts_${name} COMMAND hrx_cts_${name})
  # Pass library path for dlopen.
  set_tests_properties(hrx_cts_${name} PROPERTIES
      ENVIRONMENT "HRX_LIBRARY=$<TARGET_FILE:hrx>")
endfunction()

#===------------------------------------------------------------------------===#
# Test categories
#===------------------------------------------------------------------------===#

add_hrx_cts_test(host_allocator SOURCES tests/host_allocator/host_allocator_test.cpp)
add_hrx_cts_test(lifecycle SOURCES tests/lifecycle/lifecycle_test.cpp)
add_hrx_cts_test(device SOURCES tests/device/device_test.cpp)
add_hrx_cts_test(allocator SOURCES tests/allocator/allocator_test.cpp)
add_hrx_cts_test(semaphore SOURCES tests/semaphore/semaphore_test.cpp)
add_hrx_cts_test(stream SOURCES tests/stream/stream_test.cpp)
add_hrx_cts_test(memory SOURCES tests/memory/memory_test.cpp)
add_hrx_cts_test(transfer SOURCES tests/transfer/transfer_test.cpp)
add_hrx_cts_test(stream_ops SOURCES tests/stream_ops/stream_ops_test.cpp)
add_hrx_cts_test(executable SOURCES tests/executable/executable_test.cpp)
add_hrx_cts_test(queue_ops SOURCES tests/queue_ops/queue_ops_test.cpp)
add_hrx_cts_test(status SOURCES tests/status/status_test.cpp)
add_hrx_cts_test(refcount SOURCES tests/refcount/refcount_test.cpp)
add_hrx_cts_test(virtual_memory SOURCES tests/virtual_memory/virtual_memory_test.cpp)
add_hrx_cts_test(vm SOURCES tests/vm/vm_test.cpp)

add_executable(hrx_cts_cxx_api tests/cxx_api/cxx_api_test.cpp)
target_link_libraries(hrx_cts_cxx_api PRIVATE hrx hrx_cts_core Catch2::Catch2)
add_test(NAME hrx_cts_cxx_api COMMAND hrx_cts_cxx_api)

add_executable(hrx_cts_compiler tests/compiler/compiler_test.cpp)
target_link_libraries(
    hrx_cts_compiler PRIVATE hrx hrx_cts_core Catch2::Catch2)
add_test(NAME hrx_cts_compiler COMMAND hrx_cts_compiler)
set_tests_properties(hrx_cts_compiler PROPERTIES
    ENVIRONMENT "HRX_LIBRARY=$<TARGET_FILE:hrx>")
