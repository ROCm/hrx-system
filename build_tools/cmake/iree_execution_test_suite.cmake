# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include("${CMAKE_CURRENT_LIST_DIR}/iree_test_arguments.cmake")

# iree_execution_test_suite()
#
# Creates a CTest test that runs JSON execution manifests for command-line
# tools. Mirrors the Bazel rule in build_tools/testing/build_defs.bzl.
#
# Parameters:
# NAME: name of the test.
# MANIFESTS: JSON manifest files, relative to the current source directory.
# TOOLS: list of tool_name=cmake_target entries. Targets may be native
#     executables or iree_py_library targets that declare MAIN.
# DATA: additional data files required by the manifests.
# ARGS: additional arguments passed to the runner.
# LABELS: additional labels to apply to the test.
# RESOURCE_GROUP: If set, tests sharing the same RESOURCE_GROUP name will not
#     run concurrently.
# TIMEOUT: test timeout in seconds.

function(iree_execution_test_suite)
  iree_package_source_dir(_IREE_PACKAGE_SOURCE_DIR)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  if(NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "iree_execution_test_suite requires Python3")
  endif()
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;RESOURCE_GROUP;TIMEOUT"
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
      set(_MANIFEST_PATH "${_IREE_PACKAGE_SOURCE_DIR}/${_MANIFEST}")
    endif()
    list(APPEND _TEST_ARGS "--manifest=${_MANIFEST_PATH}")
    list(APPEND _REQUIRED_FILES "${_MANIFEST_PATH}")
  endforeach()

  set(_TOOL_TARGETS)
  set(_CONFIGURED_TOOLS)
  set(_TOOL_PYTHON_PACKAGE_DIRS)
  foreach(_TOOL IN LISTS _RULE_TOOLS)
    if(NOT _TOOL MATCHES "^([^=]+)=(.+)$")
      message(FATAL_ERROR
        "iree_execution_test_suite TOOLS entries must be tool_name=cmake_target: '${_TOOL}'")
    endif()
    set(_TOOL_NAME "${CMAKE_MATCH_1}")
    set(_TOOL_TARGET "${CMAKE_MATCH_2}")
    string(REGEX REPLACE "^::" "${_PACKAGE_NS}::" _TOOL_TARGET "${_TOOL_TARGET}")
    list(APPEND _TOOL_TARGETS "${_TOOL_TARGET}")
    list(APPEND _CONFIGURED_TOOLS "${_TOOL_NAME}=${_TOOL_TARGET}")
    iree_package_target_name(_TOOL_CMAKE_TARGET "${_TOOL_TARGET}")
    set(_TOOL_IS_PYTHON FALSE)
    if(TARGET "${_TOOL_CMAKE_TARGET}")
      get_property(
        _TOOL_IS_PYTHON
        TARGET "${_TOOL_CMAKE_TARGET}"
        PROPERTY IREE_PY_ENTRYPOINT_ARGUMENTS
        SET
      )
    endif()
    if(_TOOL_IS_PYTHON)
      iree_py_library_entrypoint(_TOOL_ENTRYPOINT_ARGUMENTS "${_TOOL_TARGET}")
      iree_py_library_collect_sources(_TOOL_SOURCE_FILES "${_TOOL_TARGET}")
      iree_py_library_collect_package_dirs(
        _TOOL_PACKAGE_DIRS "${_TOOL_TARGET}"
      )
      list(APPEND _TEST_ARGS
        "--tool=${_TOOL_NAME}=${Python3_EXECUTABLE}"
      )
      foreach(_TOOL_ARGUMENT IN LISTS _TOOL_ENTRYPOINT_ARGUMENTS)
        list(APPEND _TEST_ARGS
          "--tool-arg=${_TOOL_NAME}=${_TOOL_ARGUMENT}"
        )
      endforeach()
      list(APPEND _REQUIRED_FILES ${_TOOL_SOURCE_FILES})
      list(APPEND _TOOL_PYTHON_PACKAGE_DIRS ${_TOOL_PACKAGE_DIRS})
    else()
      list(APPEND _TEST_ARGS
        "--tool=${_TOOL_NAME}=$<TARGET_FILE:${_TOOL_TARGET}>"
      )
    endif()
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
    iree_register_target_dependency(
      TARGET
        "${_TEST_TARGET_NAME}"
      DEPENDENCY
        "${_TOOL_TARGET}"
    )
  endforeach()
  set(_ENVIRONMENT_VARS "PYTHONDONTWRITEBYTECODE=1")
  set(_DATA_DEPENDENCIES)
  foreach(_DATA IN LISTS _RULE_DATA)
    if(IS_ABSOLUTE "${_DATA}" OR
       TARGET "${_DATA}" OR
       "${_DATA}" MATCHES "::")
      list(APPEND _DATA_DEPENDENCIES "${_DATA}")
    else()
      list(APPEND _DATA_DEPENDENCIES "${_IREE_PACKAGE_SOURCE_DIR}/${_DATA}")
    endif()
  endforeach()
  iree_add_data_dependencies(
    NAME
      "${_TEST_TARGET_NAME}"
    DATA
      ${_DATA_DEPENDENCIES}
    OUT_FILE_DATA
      _TEST_FILE_DATA
    OUT_TARGET_DATA
      _TEST_TARGET_DATA
  )
  set_property(TARGET ${_TEST_TARGET_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/test)

  list(APPEND _REQUIRED_FILES ${_TEST_FILE_DATA})
  foreach(_DATA_TARGET IN LISTS _TEST_TARGET_DATA)
    list(APPEND _REQUIRED_FILES "$<TARGET_FILE:${_DATA_TARGET}>")
  endforeach()

  iree_package_binary_dir(_IREE_PACKAGE_BINARY_DIR)
  set(_TOOL_ENVIRONMENT_FILE "${_IREE_PACKAGE_BINARY_DIR}/${_RULE_NAME}.tool-environment.json")
  list(APPEND _TEST_ARGS "--tool-environment=${_TOOL_ENVIRONMENT_FILE}")
  set_property(TARGET "${_TEST_TARGET_NAME}" PROPERTY IREE_EXECUTION_TOOL_ENVIRONMENT_FILE "${_TOOL_ENVIRONMENT_FILE}")
  set_property(TARGET "${_TEST_TARGET_NAME}" PROPERTY IREE_EXECUTION_TOOLS "${_CONFIGURED_TOOLS}")
  set_property(GLOBAL APPEND PROPERTY IREE_EXECUTION_TOOL_ENVIRONMENT_TARGETS "${_TEST_TARGET_NAME}")

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
  iree_register_test_build_targets(
    "${_TEST_NAME}"
    TARGETS "${_TEST_TARGET_NAME}"
  )

  if(NOT DEFINED _RULE_TIMEOUT)
    set(_RULE_TIMEOUT 60)
  endif()
  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")
  set_property(TEST ${_TEST_NAME} PROPERTY LABELS "${_RULE_LABELS}")
  set_property(TEST ${_TEST_NAME} PROPERTY TIMEOUT "${_RULE_TIMEOUT}")
  set_property(TEST ${_TEST_NAME} PROPERTY REQUIRED_FILES "${_REQUIRED_FILES}")
  if(_RULE_RESOURCE_GROUP)
    set_property(TEST ${_TEST_NAME} PROPERTY RESOURCE_LOCK "${_RULE_RESOURCE_GROUP}")
  endif()
  iree_register_test_resource_build_target(
    TEST_BUILD_TARGET
      "${_TEST_TARGET_NAME}"
    LABELS
      ${_RULE_LABELS}
  )
  iree_python_test_add_package_dirs(
    "${_TEST_NAME}" ${_TOOL_PYTHON_PACKAGE_DIRS}
  )
  set_property(TEST ${_TEST_NAME} APPEND PROPERTY ENVIRONMENT ${_ENVIRONMENT_VARS})
endfunction()

# Tool runtime metadata is complete after dependency policy has been resolved.
# Each subprocess receives only its own environment, including merged sanitizer
# policy for executables that contain several external runtime integrations.
function(iree_finalize_execution_tool_environments)
  get_property(_TARGETS GLOBAL PROPERTY IREE_EXECUTION_TOOL_ENVIRONMENT_TARGETS)
  foreach(_TARGET IN LISTS _TARGETS)
    get_property(_OUTPUT TARGET "${_TARGET}" PROPERTY IREE_EXECUTION_TOOL_ENVIRONMENT_FILE)
    get_property(_TOOLS TARGET "${_TARGET}" PROPERTY IREE_EXECUTION_TOOLS)
    set(_JSON "{}")
    foreach(_TOOL IN LISTS _TOOLS)
      string(REGEX MATCH "^([^=]+)=(.+)$" _MATCH "${_TOOL}")
      set(_NAME "${CMAKE_MATCH_1}")
      _iree_resolve_target(_TOOL_TARGET "${CMAKE_MATCH_2}")
      get_property(_ENVIRONMENT TARGET "${_TOOL_TARGET}" PROPERTY IREE_SANITIZER_ENVIRONMENT)
      set(_ENVIRONMENT_JSON "{}")
      foreach(_ENTRY IN LISTS _ENVIRONMENT)
        string(REGEX MATCH "^([^=]+)=(.*)$" _MATCH "${_ENTRY}")
        set(_KEY "${CMAKE_MATCH_1}")
        set(_VALUE "${CMAKE_MATCH_2}")
        string(REPLACE "\\" "\\\\" _VALUE "${_VALUE}")
        string(REPLACE "\"" "\\\"" _VALUE "${_VALUE}")
        string(JSON _ENVIRONMENT_JSON SET "${_ENVIRONMENT_JSON}" "${_KEY}" "\"${_VALUE}\"")
      endforeach()
      string(JSON _JSON SET "${_JSON}" "${_NAME}" "${_ENVIRONMENT_JSON}")
    endforeach()
    file(CONFIGURE OUTPUT "${_OUTPUT}" CONTENT "@_JSON@" @ONLY)
  endforeach()
endfunction()
