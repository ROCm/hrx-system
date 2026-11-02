# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26)
project(iree_generated_file_identity NONE)
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_macros.cmake")

set(_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Generated")
set(_OUTPUT "${_OUTPUT_DIRECTORY}/Result.txt")
set(_FILE_INPUT "${_OUTPUT_DIRECTORY}/nested/../Result.txt")
set(_DIRECTORY_INPUT "${_OUTPUT_DIRECTORY}/./")
if(CMAKE_HOST_WIN32)
  string(TOLOWER "${_FILE_INPUT}" _FILE_INPUT)
  string(TOLOWER "${_DIRECTORY_INPUT}" _DIRECTORY_INPUT)
  string(REPLACE "/" "\\" _DIRECTORY_INPUT "${_DIRECTORY_INPUT}")
endif()

add_custom_target(before)
iree_generated_output_add_consumer("${_FILE_INPUT}" before)
add_custom_command(OUTPUT "${_OUTPUT}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${_OUTPUT_DIRECTORY}"
  COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_SOURCE_DIR}/payload.txt" "${_OUTPUT}"
  DEPENDS "${CMAKE_SOURCE_DIR}/payload.txt"
  VERBATIM)
add_custom_target(producer DEPENDS "${_OUTPUT}")
iree_register_generated_output_producer(producer
  OUTPUTS "${_OUTPUT}" "${_OUTPUT_DIRECTORY}")

add_custom_target(after)
iree_generated_output_add_consumer("${_DIRECTORY_INPUT}" after)

if(CASE_SENSITIVE_FILESYSTEM)
  add_custom_command(OUTPUT "${_OUTPUT_DIRECTORY}/result.txt"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_OUTPUT_DIRECTORY}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${_OUTPUT_DIRECTORY}/result.txt"
    VERBATIM)
  add_custom_target(distinct_producer DEPENDS "${_OUTPUT_DIRECTORY}/result.txt")
  iree_register_generated_output_producer(distinct_producer
    OUTPUTS "${_OUTPUT_DIRECTORY}/result.txt")
  add_custom_target(distinct_consumer)
  iree_generated_output_add_consumer("${_OUTPUT_DIRECTORY}/result.txt" distinct_consumer)
endif()

if(DUPLICATE_PRODUCER)
  add_custom_target(duplicate)
  iree_register_generated_output_producer(duplicate OUTPUTS "${_FILE_INPUT}")
endif()
