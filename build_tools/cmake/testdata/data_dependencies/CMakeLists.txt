# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21)
project(iree_cmake_data_dependencies_test C)

if(NOT IREE_REPO_ROOT)
  message(FATAL_ERROR "IREE_REPO_ROOT is required")
endif()

set(IREE_BINARY_DIR "${CMAKE_BINARY_DIR}")
set(IREE_IDE_FOLDER IREE)
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_macros.cmake")

add_custom_target(data_consumer)
set(_GENERATED_DATA "${CMAKE_BINARY_DIR}/generated.txt")
set(_PROPERTY_DATA
  "$<TARGET_PROPERTY:property_data,IREE_GENERATED_FILE>")
iree_add_data_dependencies(
  NAME
    data_consumer
  DATA
    fixture.txt
    "${_GENERATED_DATA}"
    "${_PROPERTY_DATA}"
    fixture::tool
  OUT_FILE_DATA
    _FILE_DATA
  OUT_TARGET_DATA
    _TARGET_DATA
)

if(NOT "${_FILE_DATA}" STREQUAL
   "fixture.txt;${_GENERATED_DATA};${_PROPERTY_DATA}")
  message(FATAL_ERROR "Unexpected file DATA: ${_FILE_DATA}")
endif()
if(NOT "${_TARGET_DATA}" STREQUAL "fixture::tool")
  message(FATAL_ERROR "Unexpected target DATA: ${_TARGET_DATA}")
endif()

option(IREE_TEST_DECLARE_TOOL "Declare the later DATA tool target" ON)
if(IREE_TEST_DECLARE_TOOL)
  add_executable(data_tool tool.c)
  add_executable(fixture::tool ALIAS data_tool)
  add_custom_command(
    TARGET data_tool
    POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E touch
      "${CMAKE_BINARY_DIR}/tool-built.marker"
  )
endif()

add_custom_command(
  OUTPUT
    "${_GENERATED_DATA}"
  COMMAND
    "${CMAKE_COMMAND}" -E copy
    "${CMAKE_CURRENT_SOURCE_DIR}/fixture.txt"
    "${_GENERATED_DATA}"
  DEPENDS
    fixture.txt
)
add_custom_target(generated_data DEPENDS "${_GENERATED_DATA}")
iree_register_generated_output_producer(generated_data
  OUTPUTS "${_GENERATED_DATA}"
)

set(_PROPERTY_DATA_OUTPUT "${CMAKE_BINARY_DIR}/property-generated.txt")
add_custom_command(
  OUTPUT
    "${_PROPERTY_DATA_OUTPUT}"
  COMMAND
    "${CMAKE_COMMAND}" -E copy
    "${CMAKE_CURRENT_SOURCE_DIR}/fixture.txt"
    "${_PROPERTY_DATA_OUTPUT}"
  DEPENDS
    fixture.txt
)
add_custom_target(property_data DEPENDS "${_PROPERTY_DATA_OUTPUT}")
set_property(TARGET property_data PROPERTY
  IREE_GENERATED_FILE "${_PROPERTY_DATA_OUTPUT}")

iree_finalize_target_dependencies()
