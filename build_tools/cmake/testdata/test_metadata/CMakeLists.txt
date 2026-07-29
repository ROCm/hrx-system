# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26)
project(iree_cmake_test_metadata_test C)

if(NOT IREE_REPO_ROOT)
  message(FATAL_ERROR "IREE_REPO_ROOT is required")
endif()

set(IREE_BINARY_DIR "${CMAKE_BINARY_DIR}")
set(IREE_IDE_FOLDER IREE)
set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_macros.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_configure_testing.cmake")

function(add_metadata_test TEST_NAME BUILD_TARGET)
  add_custom_target("${BUILD_TARGET}")
  add_test(NAME "${TEST_NAME}" COMMAND "${CMAKE_COMMAND}" -E true)
  iree_register_test_build_targets(
    "${TEST_NAME}"
    TARGETS "${BUILD_TARGET}"
  )
endfunction()

add_metadata_test(host host_root)

add_metadata_test(benchmark benchmark_root)
set_property(TEST benchmark PROPERTY LABELS benchmark)

add_metadata_test(runtime-resource runtime_resource_root)
set_property(
  TEST runtime-resource
  PROPERTY LABELS runtime-resource=amd-gpu
)

add_metadata_test(manual manual_root)
set_property(TEST manual PROPERTY LABELS manual)

add_test(NAME source-only COMMAND "${CMAKE_COMMAND}" -E true)
iree_register_test_build_targets(source-only)

add_custom_target(tool)
add_custom_target(tool_backed_root)
add_dependencies(tool_backed_root tool)
add_test(NAME tool-backed COMMAND "${CMAKE_COMMAND}" -E true)
iree_register_test_build_targets(
  tool-backed
  TARGETS tool_backed_root
)

option(IREE_TEST_ADD_UNMANAGED_TEST "Add a test without build metadata" OFF)
if(IREE_TEST_ADD_UNMANAGED_TEST)
  add_test(NAME unmanaged COMMAND "${CMAKE_COMMAND}" -E true)
endif()

option(IREE_TEST_ADD_MISSING_ROOT "Add a test with a missing build root" OFF)
if(IREE_TEST_ADD_MISSING_ROOT)
  add_test(NAME missing-root COMMAND "${CMAKE_COMMAND}" -E true)
  iree_register_test_build_targets(
    missing-root
    TARGETS missing_root
  )
endif()

option(IREE_TEST_ADD_ALIAS_ROOT "Add a test with an alias build root" OFF)
if(IREE_TEST_ADD_ALIAS_ROOT)
  add_library(alias_root_actual INTERFACE)
  add_library(alias_root ALIAS alias_root_actual)
  add_test(NAME alias-root COMMAND "${CMAKE_COMMAND}" -E true)
  iree_register_test_build_targets(alias-root TARGETS alias_root)
endif()

option(
  IREE_TEST_ADD_IMPORTED_ROOT
  "Add a test with an imported build root"
  OFF
)
if(IREE_TEST_ADD_IMPORTED_ROOT)
  add_executable(imported_root IMPORTED)
  add_test(NAME imported-root COMMAND "${CMAKE_COMMAND}" -E true)
  iree_register_test_build_targets(imported-root TARGETS imported_root)
endif()

option(
  IREE_TEST_ADD_INTERFACE_ROOT
  "Add a test with an interface-library build root"
  OFF
)
if(IREE_TEST_ADD_INTERFACE_ROOT)
  add_library(interface_root INTERFACE)
  add_test(NAME interface-root COMMAND "${CMAKE_COMMAND}" -E true)
  iree_register_test_build_targets(interface-root TARGETS interface_root)
endif()

iree_finalize_test_build_targets()
