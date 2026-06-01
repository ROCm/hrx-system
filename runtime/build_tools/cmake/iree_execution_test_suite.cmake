# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_execution_test_suite()
#
# Creates a CTest test that runs YAML execution manifests for command-line
# tools. Mirrors the Bazel rule in build_tools/testing/build_defs.bzl.
#
# Parameters:
# NAME: name of the test.
# MANIFESTS: YAML manifest files, relative to the current source directory.
# TOOLS: list of tool_name=cmake_target entries.
# DATA: additional data files required by the manifests.
# ARGS: additional arguments passed to the runner.
# LABELS: additional labels to apply to the test.
# TIMEOUT: test timeout in seconds.
function(iree_execution_test_suite)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  if(NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "iree_execution_test_suite requires Python3")
  endif()
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import yaml"
    RESULT_VARIABLE _IREE_PYTHON_YAML_RESULT
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(NOT _IREE_PYTHON_YAML_RESULT EQUAL 0)
    message(FATAL_ERROR
      "iree_execution_test_suite requires PyYAML for ${Python3_EXECUTABLE}; "
      "install the repository development Python requirements with: "
      "uv pip install --python ${Python3_EXECUTABLE} -r "
      "${PROJECT_SOURCE_DIR}/requirements-dev.lock.txt")
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;TIMEOUT"
    "MANIFESTS;TOOLS;DATA;ARGS;LABELS"
    ${ARGN}
  )

  if(NOT _RULE_NAME)
    message(FATAL_ERROR "iree_execution_test_suite requires NAME")
  endif()
  if(NOT _RULE_MANIFESTS)
    message(FATAL_ERROR "iree_execution_test_suite requires MANIFESTS")
  endif()
  if(NOT _RULE_TOOLS)
    message(FATAL_ERROR "iree_execution_test_suite requires TOOLS")
  endif()

  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH "${_PACKAGE_NS}")
  set(_TEST_NAME "${_PACKAGE_PATH}/${_RULE_NAME}")

  set(_RUNNER "${PROJECT_SOURCE_DIR}/build_tools/testing/execution_main.py")
  set(_REQUIRED_FILES
    "${PROJECT_SOURCE_DIR}/build_tools/testing/execution.py"
    "${_RUNNER}"
  )

  set(_TEST_ARGS)
  foreach(_MANIFEST IN LISTS _RULE_MANIFESTS)
    if(IS_ABSOLUTE "${_MANIFEST}")
      set(_MANIFEST_PATH "${_MANIFEST}")
    else()
      set(_MANIFEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${_MANIFEST}")
    endif()
    list(APPEND _TEST_ARGS "--manifest=${_MANIFEST_PATH}")
    list(APPEND _REQUIRED_FILES "${_MANIFEST_PATH}")
  endforeach()

  foreach(_TOOL IN LISTS _RULE_TOOLS)
    if(NOT _TOOL MATCHES "^([^=]+)=(.+)$")
      message(FATAL_ERROR
        "iree_execution_test_suite TOOLS entries must be tool_name=cmake_target: '${_TOOL}'")
    endif()
    set(_TOOL_NAME "${CMAKE_MATCH_1}")
    set(_TOOL_TARGET "${CMAKE_MATCH_2}")
    string(REGEX REPLACE "^::" "${_PACKAGE_NS}::" _TOOL_TARGET "${_TOOL_TARGET}")
    list(APPEND _TEST_ARGS "--tool=${_TOOL_NAME}=$<TARGET_FILE:${_TOOL_TARGET}>")
  endforeach()

  foreach(_DATA IN LISTS _RULE_DATA)
    if(IS_ABSOLUTE "${_DATA}")
      list(APPEND _REQUIRED_FILES "${_DATA}")
    else()
      list(APPEND _REQUIRED_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${_DATA}")
    endif()
  endforeach()

  list(APPEND _TEST_ARGS ${_RULE_ARGS})

  add_test(
    NAME
      ${_TEST_NAME}
    COMMAND
      "${Python3_EXECUTABLE}"
      "${_RUNNER}"
      ${_TEST_ARGS}
  )
  iree_configure_test(${_TEST_NAME})

  if(NOT DEFINED _RULE_TIMEOUT)
    set(_RULE_TIMEOUT 60)
  endif()
  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")
  set_property(TEST ${_TEST_NAME} PROPERTY LABELS "${_RULE_LABELS}")
  set_property(TEST ${_TEST_NAME} PROPERTY TIMEOUT "${_RULE_TIMEOUT}")
  set_property(TEST ${_TEST_NAME} PROPERTY REQUIRED_FILES "${_REQUIRED_FILES}")
  set_property(TEST ${_TEST_NAME} APPEND PROPERTY ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
endfunction()
