# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26)
project(iree_cmake_test_arguments C CXX)
enable_testing()
set(IREE_BUILD_TESTS ON)
set(CMAKE_CXX_STANDARD 17)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(IREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(IREE_TEST_TMPDIR_ROOT "${CMAKE_CURRENT_BINARY_DIR}/tmp")
set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX fixture)
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_macros.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_copts.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_configure_testing.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_install_support.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_native_test.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_cc_test.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_python.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_execution_test_suite.cmake")
include("${IREE_REPO_ROOT}/build_tools/sanitizer/iree_sanitizer_suppressions.cmake")
set(IREE_SANITIZER_SUPPRESSION_FIXTURE_LSAN "${CMAKE_CURRENT_SOURCE_DIR}/suppression.txt")
if(DEFINED IREE_TEST_INVALID_ARGUMENT)
  iree_native_test(NAME invalid SRC reader ARGS "${IREE_TEST_INVALID_ARGUMENT}")
endif()

add_executable(reader reader.cc)
add_executable(iree::tools::runner ALIAS reader)
iree_target_sanitizer_suppressions(reader SUPPRESSIONS lsan fixture)
iree_execution_test_suite(NAME execution_reader
  MANIFESTS suppression.test.json TOOLS reader=reader)
iree_py_test(NAME python_reader MAIN reader.py SRCS reader.py reader_helper.py
  PACKAGE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
  ARGS "{{$<TARGET_FILE:late_reader>}}" "{{${CMAKE_CURRENT_SOURCE_DIR}/fixture.txt}}")
iree_py_test(NAME generated_python_reader MAIN generated.py
  SRCS generated.py reader_helper.py
  GENERATED_IN_BINARY_DIR
  PACKAGE_DIRS "${CMAKE_CURRENT_BINARY_DIR}"
  ARGS "{{$<TARGET_FILE:late_reader>}}" "{{${CMAKE_CURRENT_SOURCE_DIR}/fixture.txt}}")
add_executable(late_reader EXCLUDE_FROM_ALL reader.cc)
add_custom_command(OUTPUT
    "${CMAKE_CURRENT_BINARY_DIR}/generated.py"
    "${CMAKE_CURRENT_BINARY_DIR}/reader_helper.py"
  COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_SOURCE_DIR}/reader.py"
    "${CMAKE_CURRENT_BINARY_DIR}/generated.py"
  COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_SOURCE_DIR}/reader_helper.py"
    "${CMAKE_CURRENT_BINARY_DIR}/reader_helper.py"
  DEPENDS reader.py reader_helper.py VERBATIM)
add_custom_target(generate_python DEPENDS
  "${CMAKE_CURRENT_BINARY_DIR}/generated.py"
  "${CMAKE_CURRENT_BINARY_DIR}/reader_helper.py")
iree_register_generated_output_producer(generate_python
  OUTPUTS generated.py reader_helper.py)

add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated.txt"
  COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_SOURCE_DIR}/fixture.txt"
    "${CMAKE_CURRENT_BINARY_DIR}/generated.txt"
  DEPENDS fixture.txt VERBATIM)
add_custom_target(generate_data DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/generated.txt")
iree_register_generated_output_producer(generate_data OUTPUTS generated.txt)
set(_PATH_ARGS
  "{{${CMAKE_CURRENT_SOURCE_DIR}/fixture.txt}}"
  "--input={{${CMAKE_CURRENT_SOURCE_DIR}/fixture.txt}}"
  "--wrapped=[{{${CMAKE_CURRENT_SOURCE_DIR}/fixture.txt}}]"
  "--pair={{${CMAKE_CURRENT_SOURCE_DIR}/fixture.txt}}|{{${CMAKE_CURRENT_SOURCE_DIR}/fixture.txt.more}}"
  "--literal=fixture.txt"
  "{{${CMAKE_CURRENT_SOURCE_DIR}/fixture.txt.more}}"
  "{{first input.txt}}" "{{second input.txt}}"
  "{{${CMAKE_CURRENT_BINARY_DIR}/generated.txt}}")
if(UNIX)
  foreach(_NAME "a|b.txt" "a::b.txt")
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/${_NAME}" "payload\n")
    list(APPEND _PATH_ARGS "{{${CMAKE_CURRENT_SOURCE_DIR}/${_NAME}}}")
  endforeach()
endif()
iree_native_test(NAME file_arguments SRC reader
  ARGS ${_PATH_ARGS} DATA fixture.txt fixture.txt.more)
iree_native_test(NAME source_working_directory SRC reader
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" ARGS fixture.txt DATA fixture.txt)

add_subdirectory(nested)
add_subdirectory(converted)
iree_finalize_target_dependencies()
iree_finalize_sanitizer_suppressions()
iree_finalize_execution_tool_environments()
iree_create_ctest_customization()
