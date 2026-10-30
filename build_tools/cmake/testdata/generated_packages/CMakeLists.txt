# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26...4.2)
project(generated_packages C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(IREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(IREE_PACKAGE_ROOT_DIR "${IREE_SOURCE_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "fixture")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_macros.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_copts.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_install_support.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_cc_library.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_cc_binary.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_python.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_add_all_subdirs.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_generated_cmake.cmake")
iree_generate_cmake()
iree_add_subdirectory(package)
if(TARGET authored_usage OR NOT PROJECT_NAME STREQUAL "generated_packages")
  message(FATAL_ERROR "Authored package scope leaked into its parent")
endif()
iree_finalize_target_dependencies()
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/paths-$<CONFIG>.txt"
  CONTENT "$<TARGET_FILE:fixture_package_probe>\n$<TARGET_FILE:fixture_package_authored_child_probe>\n")
