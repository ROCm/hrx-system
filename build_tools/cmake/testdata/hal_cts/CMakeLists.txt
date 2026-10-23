# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26)
project(iree_cmake_hal_cts_test C CXX)

set(CMAKE_CXX_STANDARD 20)
set(IREE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree/fixture")
set(IREE_BUILD_TESTS ON)
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_macros.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_copts.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_install_support.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_cc_library.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_cc_binary.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_c_embed_data.cmake")
include("${IREE_REPO_ROOT}/runtime/build_tools/cmake/iree_runtime_hal_cts_testdata.cmake")
set(IREE_SOURCE_DIR "${IREE_REPO_ROOT}")

add_executable(iree-c-embed-data "${IREE_REPO_ROOT}/build_tools/embed_data/embed_data.cc")
add_library(registry INTERFACE)
target_include_directories(registry INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
add_library(iree::hal::cts::util::registry ALIAS registry)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/products")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/products/kernel.bin" "kernel bytes")
iree_runtime_hal_cts_testdata(
  NAME fixture-data
  SRCS "${CMAKE_CURRENT_BINARY_DIR}/products/kernel.bin"
  BACKEND_NAMES primary alias
  TARGET_NAME target
  TARGET_FAMILY family
  TARGET_KEY family:target
)
iree_cc_binary(NAME probe SRCS main.cc DEPS ::fixture-data)
iree_finalize_target_dependencies()
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/probe-$<CONFIG>.txt"
  CONTENT "$<TARGET_FILE:iree_fixture_probe>\n")
