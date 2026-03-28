# Copyright 2026 The Pyre Authors
# SPDX-License-Identifier: Apache-2.0
#
# Pyre Conformance Test Suite (CTS).
# Can be built in-tree (add_subdirectory from pyre-runtime) or standalone
# (find_package(pyre) for header, dlopen at runtime for library).

cmake_minimum_required(VERSION 3.20)
project(pyre-cts LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Standalone vs in-tree detection.
if(NOT TARGET pyre)
  find_package(pyre REQUIRED)
endif()

# Catch2.
find_package(Catch2 3 REQUIRED)
include(CTest)

#===------------------------------------------------------------------------===#
# Core infrastructure
#===------------------------------------------------------------------------===#

add_library(pyre_cts_core STATIC
    core/pyre_loader.cpp
    core/main.cpp
)
target_include_directories(pyre_cts_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    $<TARGET_PROPERTY:pyre,INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(pyre_cts_core PUBLIC Catch2::Catch2 ${CMAKE_DL_LIBS})

#===------------------------------------------------------------------------===#
# Helper macro: add a CTS test category
#===------------------------------------------------------------------------===#

function(add_pyre_cts_test name)
  cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})
  add_executable(pyre_cts_${name} ${ARG_SOURCES})
  target_link_libraries(pyre_cts_${name} PRIVATE pyre_cts_core Catch2::Catch2)
  add_test(NAME pyre_cts_${name} COMMAND pyre_cts_${name})
  # Pass library path for dlopen.
  set_tests_properties(pyre_cts_${name} PROPERTIES
      ENVIRONMENT "PYRE_LIBRARY=$<TARGET_FILE:pyre>")
endfunction()

#===------------------------------------------------------------------------===#
# Test categories
#===------------------------------------------------------------------------===#

add_pyre_cts_test(host_allocator SOURCES tests/host_allocator/host_allocator_test.cpp)
add_pyre_cts_test(lifecycle SOURCES tests/lifecycle/lifecycle_test.cpp)
add_pyre_cts_test(device SOURCES tests/device/device_test.cpp)
add_pyre_cts_test(allocator SOURCES tests/allocator/allocator_test.cpp)
add_pyre_cts_test(semaphore SOURCES tests/semaphore/semaphore_test.cpp)
add_pyre_cts_test(stream SOURCES tests/stream/stream_test.cpp)
add_pyre_cts_test(memory SOURCES tests/memory/memory_test.cpp)
add_pyre_cts_test(transfer SOURCES tests/transfer/transfer_test.cpp)
add_pyre_cts_test(stream_ops SOURCES tests/stream_ops/stream_ops_test.cpp)
add_pyre_cts_test(status SOURCES tests/status/status_test.cpp)
add_pyre_cts_test(refcount SOURCES tests/refcount/refcount_test.cpp)
add_pyre_cts_test(virtual_memory SOURCES tests/virtual_memory/virtual_memory_test.cpp)
