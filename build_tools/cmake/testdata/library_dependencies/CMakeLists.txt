# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26)
project(iree_cmake_library_dependencies_test C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OPTIMIZE_DEPENDENCIES ON)
set(IREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree/fixture")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_macros.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_copts.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_install_support.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_cc_library.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_cc_binary.cmake")

# An existing source wins over a same-named file in the binary directory.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/library.c" "#error Binary shadow source\n")
foreach(_EXTENSION hpp inc inl)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/private.${_EXTENSION}" "#pragma once\n")
endforeach()
iree_cc_library(NAME headers
  SRCS inactive/generated.h
    "${CMAKE_CURRENT_BINARY_DIR}/private.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/private.inc"
    "${CMAKE_CURRENT_BINARY_DIR}/private.inl"
  TEXTUAL_HDRS "${CMAKE_CURRENT_BINARY_DIR}/producer/generated.h")
iree_cc_library(NAME leaf
  SRCS "${CMAKE_CURRENT_SOURCE_DIR}/leaf.hpp.c" DEFINES TRANSITIVE_VALUE=3)
# A false condition excludes the entire list; active system includes keep their
# order. The inactive header catches either a leaked path or reordered includes.
iree_cc_library(NAME library
  SRCS library.c producer/generated.c DEPS ::leaf ::headers
  INCLUDES "$<0:${CMAKE_CURRENT_BINARY_DIR}/unused;${CMAKE_CURRENT_SOURCE_DIR}/inactive>"
  SYSTEM_INCLUDES "$<1:${CMAKE_CURRENT_BINARY_DIR}/producer;${CMAKE_CURRENT_SOURCE_DIR}/inactive>")
iree_cc_binary(NAME probe SRCS main.c DEPS ::library)
iree_cc_unified_library(NAME unified ROOT ::library)
iree_cc_binary(NAME unified_probe SRCS main.c DEPS ::unified)

# Private link requirements stay link-only when forwarded to the unified target.
add_library(private_usage INTERFACE)
target_compile_definitions(private_usage INTERFACE PRIVATE_VALUE=7)
target_link_libraries(iree_fixture_library PRIVATE private_usage)
target_compile_definitions(iree_fixture_library PRIVATE LATE_VALUE=11)

# Independent forward producers in another directory must reach both the
# interface header's consumer and the library's generated source.
add_subdirectory(producer)
iree_finalize_target_dependencies()

file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/paths-$<CONFIG>.txt"
  CONTENT "$<TARGET_FILE:iree_fixture_leaf>\n$<TARGET_FILE:iree_fixture_probe>\n$<TARGET_FILE:iree_fixture_unified_probe>\n")
