# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26)
project(iree_cmake_selected_build_test C)

set(IREE_BINARY_DIR "${CMAKE_BINARY_DIR}")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_configure_testing.cmake")

# This always-run prerequisite makes duplicated native traversals observable.
add_custom_target(shared_visit
  COMMAND "${CMAKE_COMMAND}"
    "-DLOG=${CMAKE_BINARY_DIR}/visits.txt"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/record_visit.cmake"
  VERBATIM
)
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated.h"
  COMMAND "${CMAKE_COMMAND}"
    "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/value.txt"
    "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/generated.h"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/generate_header.cmake"
  DEPENDS value.txt generate_header.cmake
  VERBATIM
)
add_library(shared STATIC shared.c "${CMAKE_CURRENT_BINARY_DIR}/generated.h")
target_include_directories(shared PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
add_dependencies(shared shared_visit)

foreach(_NAME left right)
  add_executable(${_NAME} main.c)
  target_link_libraries(${_NAME} PRIVATE shared)
endforeach()

add_subdirectory("nested package")

# An unselected target must never enter the requested build closure.
add_executable(unselected unselected.c)
add_test(NAME unselected COMMAND unselected)
iree_configure_test(unselected)
iree_register_test_build_targets(unselected TARGETS unselected)

iree_finalize_test_build_targets()
iree_create_ctest_customization()
