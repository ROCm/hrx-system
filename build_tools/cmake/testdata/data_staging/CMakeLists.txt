# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26)
project(iree_cmake_data_staging_test CXX)
enable_testing()

set(IREE_BUILD_TESTS ON)
set(IREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(IREE_TEST_TMPDIR_ROOT "${CMAKE_CURRENT_BINARY_DIR}/tmp")
set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX fixture)
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_macros.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_configure_testing.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_native_test.cmake")

add_executable(reader reader.cc)
iree_native_test(NAME first SRC reader ARGS "{{shared.txt}}" DATA shared.txt)
iree_native_test(NAME second SRC reader ARGS "{{shared.txt}}" DATA shared.txt)

# These distinct files have the same name after replacing slashes with
# underscores. They must have independent producers.
iree_native_test(NAME distinct_paths SRC reader
  ARGS "{{a/b.txt}}" "{{a_b.txt}}"
  DATA a/b.txt a_b.txt)

iree_finalize_target_dependencies()
