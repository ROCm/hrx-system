# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_BUILD_TESTS OR ANDROID OR EMSCRIPTEN)
  return()
endif()

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "build_tools/cmake")

# These tests exercise the CMake/CTest integration itself. They belong to the
# CTest graph and must use the exact tools selected by the outer configure.
# Developer-only build-system tests are not part of the installed test suite.
set(IREE_SKIP_TEST_REGISTRATION ON)

function(_iree_cmake_integration_test)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "PACKAGE_DIRS;SRCS"
    ${ARGN}
  )

  list(GET _RULE_SRCS 0 _MAIN)
  iree_py_test(
    NAME
      "${_RULE_NAME}"
    MAIN
      "${_MAIN}"
    SRCS
      ${_RULE_SRCS}
      test_environment.py
    PACKAGE_DIRS
      "${PROJECT_SOURCE_DIR}"
      ${_RULE_PACKAGE_DIRS}
  )

  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH "${_PACKAGE_NS}")
  set(_TEST_NAME "${_PACKAGE_PATH}/${_RULE_NAME}")
  set(_TEST_ENVIRONMENT
    "IREE_TEST_CMAKE_COMMAND=${CMAKE_COMMAND}"
    "IREE_TEST_CTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
    "IREE_TEST_CMAKE_GENERATOR=${CMAKE_GENERATOR}"
    "IREE_TEST_CMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
    "IREE_TEST_CMAKE_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
    "IREE_TEST_CMAKE_GENERATOR_INSTANCE=${CMAKE_GENERATOR_INSTANCE}"
    "IREE_TEST_CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
    "IREE_TEST_CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    "IREE_TEST_CMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    "IREE_TEST_CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "IREE_TEST_CMAKE_AR=${CMAKE_AR}"
    "IREE_TEST_CMAKE_LINKER=${CMAKE_LINKER}"
    "IREE_TEST_CMAKE_RC_COMPILER=${CMAKE_RC_COMPILER}"
    "IREE_TEST_CMAKE_MT=${CMAKE_MT}"
  )
  set_property(TEST "${_TEST_NAME}" APPEND PROPERTY ENVIRONMENT
    ${_TEST_ENVIRONMENT})

  set(_TEST_ENVIRONMENT_MODIFICATIONS)
  foreach(_ENVIRONMENT_NAME PATH INCLUDE LIB LIBPATH)
    if(DEFINED ENV{${_ENVIRONMENT_NAME}})
      list(APPEND _TEST_ENVIRONMENT_MODIFICATIONS
        "${_ENVIRONMENT_NAME}=unset:")
      cmake_path(CONVERT "$ENV{${_ENVIRONMENT_NAME}}"
        TO_CMAKE_PATH_LIST _ENVIRONMENT_PATH_LIST NORMALIZE)
      foreach(_ENVIRONMENT_PATH IN LISTS _ENVIRONMENT_PATH_LIST)
        list(APPEND _TEST_ENVIRONMENT_MODIFICATIONS
          "${_ENVIRONMENT_NAME}=path_list_append:${_ENVIRONMENT_PATH}")
      endforeach()
    endif()
  endforeach()
  set_property(TEST "${_TEST_NAME}" APPEND PROPERTY ENVIRONMENT_MODIFICATION
    ${_TEST_ENVIRONMENT_MODIFICATIONS})
endfunction()

_iree_cmake_integration_test(
  NAME
    cmake_data_dependencies_test
  SRCS
    cmake_data_dependencies_test.py
)

_iree_cmake_integration_test(
  NAME
    cmake_test_metadata_test
  SRCS
    cmake_test_metadata_test.py
)

_iree_cmake_integration_test(
  NAME
    ctest_integration_test
  SRCS
    ../devtools/ctest_integration_test.py
  PACKAGE_DIRS
    "${PROJECT_SOURCE_DIR}"
)
