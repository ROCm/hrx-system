# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# CMake owns generation so direct configure, dev.py, and native regeneration
# consume the same declarations. Each HRX binary tree owns its package tree.
function(iree_generate_cmake)
  if(CMAKE_HOST_WIN32 AND CMAKE_VERSION VERSION_LESS 4.2)
    message(FATAL_ERROR
      "Windows HRX builds require CMake 4.2 or newer for short object names.")
  endif()
  if(NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "Configuring HRX requires a Python interpreter.")
  endif()
  set(IREE_GENERATED_CMAKE_DIR "${IREE_BINARY_DIR}/.iree/cmake")

  # Capture effective options, including parent and toolchain normal variables,
  # after native bootstrap. BUILD selection consumes values directly; native
  # target interfaces and compiler generator expressions stay owned by CMake.
  get_cmake_property(_VARIABLES VARIABLES)
  list(FILTER _VARIABLES INCLUDE REGEX
    "^(IREE_|LOOM_|AMDF_|LIBHRX_|HRX_|CMAKE_(SYSTEM_NAME|C_COMPILER_ID|CXX_COMPILER_ID|BINARY_DIR|GENERATOR|CONFIGURATION_TYPES)$|CMAKE_(ARCHIVE_OUTPUT_DIRECTORY|STATIC_LIBRARY_PREFIX|STATIC_LIBRARY_SUFFIX)|CMAKE_.*_POSTFIX$|BUILD_SHARED_LIBS$|WIN32$|MSVC$|APPLE$|UNIX$)")
  set(_CONFIGURATION "{\n")
  set(_SEPARATOR "")
  foreach(_VARIABLE IN LISTS _VARIABLES)
    set(_VALUE "${${_VARIABLE}}")
    string(REPLACE "\\" "\\\\" _VALUE "${_VALUE}")
    string(REPLACE "\"" "\\\"" _VALUE "${_VALUE}")
    string(REPLACE "\n" "\\n" _VALUE "${_VALUE}")
    string(REPLACE "\r" "\\r" _VALUE "${_VALUE}")
    string(REPLACE "\t" "\\t" _VALUE "${_VALUE}")
    string(APPEND _CONFIGURATION "${_SEPARATOR}\"${_VARIABLE}\":\"${_VALUE}\"")
    set(_SEPARATOR ",\n")
  endforeach()
  string(APPEND _CONFIGURATION "\n}\n")
  set(_CONFIGURATION_FILE "${IREE_BINARY_DIR}/.iree/configuration.json")
  file(WRITE "${_CONFIGURATION_FILE}" "${_CONFIGURATION}")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../bazel_to_cmake/bazel_to_cmake.py"
      --output-dir "${IREE_GENERATED_CMAKE_DIR}"
      --configuration "${_CONFIGURATION_FILE}"
    WORKING_DIRECTORY "${IREE_SOURCE_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
  )
  include("${IREE_GENERATED_CMAKE_DIR}/inputs.cmake")
  set(IREE_GENERATED_CMAKE_DIR "${IREE_GENERATED_CMAKE_DIR}" PARENT_SCOPE)
endfunction()
