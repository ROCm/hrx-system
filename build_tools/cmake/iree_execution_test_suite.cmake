# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_execution_test_suite()
#
# Creates a CTest test that runs JSON execution manifests for command-line
# tools. Mirrors the Bazel rule in build_tools/testing/build_defs.bzl.
#
# Parameters:
# NAME: name of the test.
# MANIFESTS: JSON manifest files, relative to the current source directory.
# TOOLS: list of tool_name=cmake_target entries.
# DATA: additional data files required by the manifests.
# ARGS: additional arguments passed to the runner.
# LABELS: additional labels to apply to the test.
# SANITIZER_SUPPRESSIONS: Sanitizer/name pairs selecting suppression files.
#     For example: lsan vulkan.
# TIMEOUT: test timeout in seconds.

function(iree_execution_test_suite)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  if(NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "iree_execution_test_suite requires Python3")
  endif()
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;TIMEOUT"
    "MANIFESTS;TOOLS;DATA;ARGS;LABELS;SANITIZER_SUPPRESSIONS"
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

  set(_TOOL_TARGETS)
  foreach(_TOOL IN LISTS _RULE_TOOLS)
    if(NOT _TOOL MATCHES "^([^=]+)=(.+)$")
      message(FATAL_ERROR
        "iree_execution_test_suite TOOLS entries must be tool_name=cmake_target: '${_TOOL}'")
    endif()
    set(_TOOL_NAME "${CMAKE_MATCH_1}")
    set(_TOOL_TARGET "${CMAKE_MATCH_2}")
    string(REGEX REPLACE "^::" "${_PACKAGE_NS}::" _TOOL_TARGET "${_TOOL_TARGET}")
    list(APPEND _TOOL_TARGETS "${_TOOL_TARGET}")
    list(APPEND _TEST_ARGS "--tool=${_TOOL_NAME}=$<TARGET_FILE:${_TOOL_TARGET}>")
  endforeach()

  iree_package_name(_PACKAGE_NAME)
  if(_PACKAGE_NAME)
    set(_TEST_TARGET_NAME "${_PACKAGE_NAME}_${_RULE_NAME}_test_deps")
  else()
    set(_TEST_TARGET_NAME "${_RULE_NAME}_test_deps")
  endif()
  # Build declared tools before any CTest selection attempts to run the suite.
  add_custom_target(${_TEST_TARGET_NAME} ALL)
  foreach(_TOOL_TARGET IN LISTS _TOOL_TARGETS)
    iree_register_test_target_dependency(
      TARGET
        "${_TEST_TARGET_NAME}"
      DEPENDENCY
        "${_TOOL_TARGET}"
    )
  endforeach()
  set_property(TARGET ${_TEST_TARGET_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/test)

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
  iree_register_test_resource_build_target(
    TEST_BUILD_TARGET
      "${_TEST_TARGET_NAME}"
    LABELS
      ${_RULE_LABELS}
  )
  set(_ENVIRONMENT_VARS "PYTHONDONTWRITEBYTECODE=1")
  if(_RULE_SANITIZER_SUPPRESSIONS)
    iree_append_sanitizer_suppression_environment(
      _ENVIRONMENT_VARS
      ${_RULE_SANITIZER_SUPPRESSIONS}
    )
  endif()
  set_property(TEST ${_TEST_NAME} APPEND PROPERTY ENVIRONMENT ${_ENVIRONMENT_VARS})
endfunction()
