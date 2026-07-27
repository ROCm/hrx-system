# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/licenses/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Loom check build helpers.
#
# These helpers mirror loom/build_tools/bazel/loom_check.bzl for generated
# CMake. The generated CMake should preserve the suite declaration; this file
# owns expansion into individual CTest entries.

function(_loom_check_test_base_name OUTPUT_BASE_NAME SRC)
  if(NOT SRC MATCHES "\\.loom-test$")
    message(FATAL_ERROR
      "loom_check_test source must use the .loom-test extension: ${SRC}")
  endif()
  string(REGEX REPLACE "\\.loom-test$" "" _BASE_NAME "${SRC}")
  set(${OUTPUT_BASE_NAME} "${_BASE_NAME}" PARENT_SCOPE)
endfunction()

function(_loom_check_test_name OUTPUT_TEST_NAME SRC TEST_NAME_PREFIX_TO_STRIP)
  _loom_check_test_base_name(_TEST_NAME_SRC "${SRC}")
  if(TEST_NAME_PREFIX_TO_STRIP)
    string(FIND "${_TEST_NAME_SRC}" "${TEST_NAME_PREFIX_TO_STRIP}" _PREFIX_INDEX)
    if(_PREFIX_INDEX EQUAL 0)
      string(LENGTH "${TEST_NAME_PREFIX_TO_STRIP}" _PREFIX_LENGTH)
      string(LENGTH "${_TEST_NAME_SRC}" _TEST_NAME_SRC_LENGTH)
      math(EXPR _SUFFIX_LENGTH "${_TEST_NAME_SRC_LENGTH} - ${_PREFIX_LENGTH}")
      string(
        SUBSTRING "${_TEST_NAME_SRC}" ${_PREFIX_LENGTH} ${_SUFFIX_LENGTH}
        _TEST_NAME_SRC
      )
    endif()
  endif()
  string(REPLACE "/" "_" _TEST_NAME "${_TEST_NAME_SRC}")
  set(${OUTPUT_TEST_NAME} "${_TEST_NAME}" PARENT_SCOPE)
endfunction()

function(loom_check_test_suite)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;RUNNER;TEST_NAME_PREFIX_TO_STRIP;RESOURCE_GROUP;TIMEOUT"
    "SRCS;DATA;ENV;LABELS"
    ${ARGN}
  )

  if(NOT _RULE_NAME)
    message(FATAL_ERROR "loom_check_test_suite requires NAME")
  endif()
  if(NOT _RULE_SRCS)
    message(FATAL_ERROR "loom_check_test_suite requires SRCS")
  endif()
  if(NOT _RULE_RUNNER)
    set(_RULE_RUNNER loom::tools::loom-check::loom-check)
  endif()

  list(FIND _RULE_LABELS "loom-check" _LOOM_CHECK_LABEL_INDEX)
  if(_LOOM_CHECK_LABEL_INDEX EQUAL -1)
    list(APPEND _RULE_LABELS "loom-check")
  endif()
  set(_DATA_ARG)
  if(_RULE_DATA)
    set(_DATA_ARG DATA ${_RULE_DATA})
  endif()
  set(_ENV_ARG)
  if(_RULE_ENV)
    set(_ENV_ARG ENV ${_RULE_ENV})
  endif()
  set(_LABELS_ARG)
  if(_RULE_LABELS)
    set(_LABELS_ARG LABELS ${_RULE_LABELS})
  endif()
  set(_RESOURCE_GROUP_ARG)
  if(_RULE_RESOURCE_GROUP)
    set(_RESOURCE_GROUP_ARG RESOURCE_GROUP ${_RULE_RESOURCE_GROUP})
  endif()
  set(_TIMEOUT_ARG)
  if(_RULE_TIMEOUT)
    set(_TIMEOUT_ARG TIMEOUT ${_RULE_TIMEOUT})
  endif()

  foreach(_SRC IN LISTS _RULE_SRCS)
    set(_ABS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${_SRC}")
    cmake_path(RELATIVE_PATH _ABS_SRC
      BASE_DIRECTORY "${IREE_ROOT_DIR}"
      OUTPUT_VARIABLE _LOGICAL_SRC)
    _loom_check_test_name(
      _TEST_NAME
      "${_SRC}"
      "${_RULE_TEST_NAME_PREFIX_TO_STRIP}"
    )
    iree_native_test(
      NAME
        "${_TEST_NAME}"
      ARGS
        "--source-prefix-map=${IREE_ROOT_DIR}/="
        "--source-prefix-map=${_ABS_SRC}=${_LOGICAL_SRC}"
        "${_ABS_SRC}"
      SRC
        ${_RULE_RUNNER}
      ${_DATA_ARG}
      ${_ENV_ARG}
      ${_LABELS_ARG}
      ${_RESOURCE_GROUP_ARG}
      ${_TIMEOUT_ARG}
    )
  endforeach()
endfunction()
