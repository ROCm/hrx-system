# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26)
project(iree_cmake_test_results NONE)
set(IREE_BUILD_TESTS ON)
set(IREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_INSTALL_LIBDIR lib)
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_macros.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_configure_testing.cmake")

# These invocations share one executable. Punctuation and package boundaries
# must retain their identity in scratch and report paths.
foreach(_NAME suite/a-b suite/a_b suite/a.b suite/a suite/a/child suite_a_b)
  add_test(NAME "${_NAME}" COMMAND "${CMAKE_COMMAND}"
    "-DRESULT_NAME=${_NAME}" -P "${CMAKE_CURRENT_SOURCE_DIR}/result.cmake")
  iree_configure_test("${_NAME}")
endforeach()

iree_create_ctest_customization()
