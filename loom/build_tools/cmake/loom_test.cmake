# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Sources and direct libraries contribute checks; transitive libraries supply
# reachable code. Independent execution/compiler consumers share the result.
function(loom_test_module)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()
  cmake_parse_arguments(
    _RULE "" "NAME" "SRCS;LIBRARIES;DATA" ${ARGN}
  )
  if(_RULE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown loom_test_module arguments: ${_RULE_UNPARSED_ARGUMENTS}")
  endif()
  iree_package_path(_PACKAGE_PATH)
  set(_COMPONENT "//${_PACKAGE_PATH}:${_RULE_NAME}")
  set(_LIBRARIES ${_RULE_LIBRARIES})
  if(_RULE_SRCS)
    loom_module(
      NAME "${_RULE_NAME}_library"
      COMPONENT "${_COMPONENT}_library"
      SRCS ${_RULE_SRCS}
      LIBRARIES ${_RULE_LIBRARIES}
      DATA ${_RULE_DATA}
      MODE merge
      OUTPUT_FORMAT bc
      STRICT_DEPS
    )
    list(PREPEND _LIBRARIES "::${_RULE_NAME}_library")
  endif()
  loom_module(
    NAME "${_RULE_NAME}"
    COMPONENT "${_COMPONENT}"
    DATA ${_RULE_DATA}
    LIBRARIES ${_LIBRARIES}
    MODE link
    OUTPUT_FORMAT bc
    INCLUDE_LIBRARY_TESTS
  )
endfunction()

# RUNNER_ARGS reaches every selected runner; ARGS is correctness-only.
function(loom_execution_test)
  iree_package_source_dir(IREE_PACKAGE_SOURCE_DIR)
  iree_package_binary_dir(IREE_PACKAGE_BINARY_DIR)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()
  cmake_parse_arguments(_RULE "" "NAME" "" ${ARGN})
  iree_package_name(_PACKAGE_NAME)
  set(_ID "${_PACKAGE_NAME}_${_RULE_NAME}")
  set_property(GLOBAL APPEND PROPERTY LOOM_EXECUTION_TESTS "${_ID}")
  set_property(GLOBAL PROPERTY "${_ID}_ARGUMENTS" "${ARGN}")
  iree_package_path(_PACKAGE_PATH)
  set_property(GLOBAL PROPERTY "${_ID}_PACKAGE_PATH" "${_PACKAGE_PATH}")
  foreach(_VARIABLE
      IREE_PACKAGE_SOURCE_DIR IREE_PACKAGE_BINARY_DIR
      IREE_IDE_FOLDER)
    set_property(GLOBAL PROPERTY "${_ID}_${_VARIABLE}" "${${_VARIABLE}}")
  endforeach()
endfunction()

# All module owners must exist before runtime fixture paths can be passed to
# native test registration, including its installed-test relocation boundary.
# Retaining the declaring package keeps test names and relative data unchanged
# even when the module is owned by a later package.
function(_loom_declare_execution_test ID)
  foreach(_VARIABLE
      IREE_PACKAGE_SOURCE_DIR IREE_PACKAGE_BINARY_DIR
      IREE_IDE_FOLDER)
    get_property("${_VARIABLE}" GLOBAL PROPERTY "${ID}_${_VARIABLE}")
  endforeach()
  set(IREE_PACKAGE_ROOT_DIR "${IREE_PACKAGE_SOURCE_DIR}")
  get_property(IREE_PACKAGE_ROOT_PREFIX GLOBAL PROPERTY "${ID}_PACKAGE_PATH")
  get_property(_ARGUMENTS GLOBAL PROPERTY "${ID}_ARGUMENTS")
  cmake_parse_arguments(
    _RULE "CORRECTNESS_ONLY" "NAME;MODULE;RESOURCE_GROUP" "ARGS;RUNNER_ARGS;LABELS" ${_ARGUMENTS}
  )
  if(_RULE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown loom_execution_test arguments: ${_RULE_UNPARSED_ARGUMENTS}")
  endif()
  iree_package_target_name(_MODULE_TARGET "${_RULE_MODULE}")
  get_target_property(_MODULE "${_MODULE_TARGET}" LOOM_MODULE_FILE)
  get_property(_DATA TARGET "${_MODULE_TARGET}" PROPERTY LOOM_MODULE_DATA)
  iree_native_test(
    NAME "${_RULE_NAME}"
    SRC loom::tools::loom-test
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    ARGS "{{${_MODULE}}}" ${_RULE_RUNNER_ARGS} ${_RULE_ARGS}
    DATA ${_DATA}
    LABELS ${_RULE_LABELS}
    RESOURCE_GROUP "${_RULE_RESOURCE_GROUP}"
  )
  if(NOT _RULE_CORRECTNESS_ONLY)
    iree_native_test(
      NAME "${_RULE_NAME}_benchmark"
      SRC loom::tools::loom-benchmark
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
      ARGS "{{${_MODULE}}}" ${_RULE_RUNNER_ARGS} --iterations=1 --warmup-iterations=0
        --output-format=jsonl --compile-report=none
      DATA ${_DATA}
      LABELS ${_RULE_LABELS}
      RESOURCE_GROUP "${_RULE_RESOURCE_GROUP}"
    )
  endif()
  iree_package_name(_PACKAGE_NAME)
  iree_register_target_dependency(
    TARGET "${_PACKAGE_NAME}_${_RULE_NAME}_test_deps"
    DEPENDENCY "${_MODULE_TARGET}"
  )
  if(NOT _RULE_CORRECTNESS_ONLY)
    iree_register_target_dependency(
      TARGET "${_PACKAGE_NAME}_${_RULE_NAME}_benchmark_test_deps"
      DEPENDENCY "${_MODULE_TARGET}"
    )
  endif()
endfunction()

function(loom_finalize_execution_tests)
  get_property(_TESTS GLOBAL PROPERTY LOOM_EXECUTION_TESTS)
  foreach(_TEST IN LISTS _TESTS)
    _loom_declare_execution_test("${_TEST}")
  endforeach()
endfunction()
