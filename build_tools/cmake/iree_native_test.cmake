# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include("${CMAKE_CURRENT_LIST_DIR}/iree_test_arguments.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../sanitizer/iree_sanitizer_suppressions.cmake")

# iree_native_test()
#
# Creates a test that runs the specified binary with the specified arguments.
#
# Mirrors the bzl function of the same name.
#
# Parameters:
# NAME: name of target
# DRIVER: If specified, will pass --device=DRIVER to the test binary.
# DATA: Additional input files needed by the test binary.
# ARGS: Additional arguments passed to the test binary. --device=DRIVER is
#     automatically added if specified.
#     File-related arguments can be passed with `{{}}` locator,
#     e.g., --input=@{{foo.npy}}. The locator is used to portably
#     pass file arguments to tests and add each file to DATA. Relative file
#     locators use the project source root; generated files use absolute paths
#     or target-file expressions. Ordinary argument text remains literal.
# ENV: Additional KEY=VALUE environment variables set while the test runs.
#     Values may contain file locators with the same semantics as ARGS.
# SRC: Binary target to run as the test. CMake applies the target's
#     CROSSCOMPILING_EMULATOR and TEST_LAUNCHER execution properties.
# WORKING_DIRECTORY: Source or build directory to run the test from. Installed
#     tests use the corresponding directory in their testdata tree.
# WILL_FAIL: The target will run, but its pass/fail status will be inverted.
# DISABLED: The target will be skipped and its status will be 'Not Run'.
# RESOURCE_GROUP: If set, tests sharing the same RESOURCE_GROUP name will not
#     run concurrently under CTest.
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
# SANITIZER_SUPPRESSIONS: Sanitizer/name pairs selecting suppression files.
#     For example: lsan vulkan.
# TIMEOUT: Test target timeout in seconds.
#
# Usage:
# iree_cc_binary(
#   NAME
#     requires_args_to_run
#   ...
# )
# iree_native_test(
#   NAME
#     requires_args_to_run_test
#   ARGS
#    --do-the-right-thing
#   SRC
#     ::requires_args_to_run
# )

function(iree_native_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRC;DRIVER;WILL_FAIL;DISABLED;RESOURCE_GROUP;WORKING_DIRECTORY"
    "ARGS;ENV;LABELS;DATA;TIMEOUT;SANITIZER_SUPPRESSIONS"
    ${ARGN}
  )

  # Prefix the test with the package name, so we get: iree_package_name
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  iree_package_ns(_PACKAGE_NS)
  iree_package_path(_PACKAGE_PATH)
  set(_TEST_NAME "${_PACKAGE_PATH}/${_RULE_NAME}")

  # If driver was specified, add the corresponding test arg.
  if(DEFINED _RULE_DRIVER)
    list(APPEND _RULE_ARGS "--device=${_RULE_DRIVER}")
  endif()

  set(_TEST_ENVIRONMENT_VARS ${_RULE_ENV})

  iree_resolve_test_arguments(_TEST_ARGS _ARG_DATA
    iree_build_test_file_argument ${_RULE_ARGS})
  iree_resolve_test_arguments(_TEST_ENVIRONMENT _ENV_DATA
    iree_build_test_file_argument ${_TEST_ENVIRONMENT_VARS})
  list(APPEND _RULE_DATA ${_ARG_DATA} ${_ENV_DATA})
  list(REMOVE_DUPLICATES _RULE_DATA)

  # Replace binary passed by relative ::name with iree::package::name
  string(REGEX REPLACE "^::" "${_PACKAGE_NS}::" _SRC_TARGET ${_RULE_SRC})

  set(_TEST_BUILD_TARGET "${_NAME}_test_deps")
  add_custom_target(${_TEST_BUILD_TARGET} ALL)
  iree_register_target_dependency(
    TARGET
      "${_TEST_BUILD_TARGET}"
    DEPENDENCY
      "${_SRC_TARGET}"
  )
  set(_TEST_FILE_DATA)
  set(_TEST_TARGET_DATA)
  iree_add_data_dependencies(
    NAME
      "${_TEST_BUILD_TARGET}"
    DATA
      ${_RULE_DATA}
    OUT_FILE_DATA
      _TEST_FILE_DATA
    OUT_TARGET_DATA
      _TEST_TARGET_DATA
  )
  set_property(
    TARGET ${_TEST_BUILD_TARGET}
    PROPERTY FOLDER ${IREE_IDE_FOLDER}/test
  )

  iree_target_sanitizer_suppressions("${_TEST_BUILD_TARGET}"
    DEPS "${_SRC_TARGET}" ${_TEST_TARGET_DATA}
    SUPPRESSIONS ${_RULE_SANITIZER_SUPPRESSIONS}
    ENV ${_TEST_ENVIRONMENT})

  set(_TEST_RUNTIME_DATA ${_TEST_FILE_DATA})
  foreach(_DATA_TARGET IN LISTS _TEST_TARGET_DATA)
    list(APPEND _TEST_RUNTIME_DATA "$<TARGET_FILE:${_DATA_TARGET}>")
  endforeach()

  add_test(
    NAME
      ${_TEST_NAME}
    COMMAND
      "${_SRC_TARGET}"
      ${_TEST_ARGS}
  )
  iree_configure_test(${_TEST_NAME})
  if(_RULE_WORKING_DIRECTORY)
    set_property(TEST "${_TEST_NAME}" PROPERTY WORKING_DIRECTORY
      "${_RULE_WORKING_DIRECTORY}")
  endif()
  iree_register_test_build_targets(
    "${_TEST_NAME}"
    TARGETS "${_TEST_BUILD_TARGET}"
  )

  # Apply accumulated test environment variables after the test exists.
  if(_TEST_ENVIRONMENT_VARS)
    set_property(TEST ${_TEST_NAME} APPEND PROPERTY ENVIRONMENT
      ${_TEST_ENVIRONMENT})
  endif()

  if (NOT DEFINED _RULE_TIMEOUT OR "${_RULE_TIMEOUT}" STREQUAL "")
    set(_RULE_TIMEOUT 60)
  endif()

  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")
  set_property(TEST ${_TEST_NAME} PROPERTY LABELS "${_RULE_LABELS}")
  set_property(TEST "${_TEST_NAME}" PROPERTY REQUIRED_FILES "${_TEST_RUNTIME_DATA}")
  set_property(TEST ${_TEST_NAME} PROPERTY TIMEOUT ${_RULE_TIMEOUT})
  iree_register_test_resource_build_target(
    TEST_BUILD_TARGET
      "${_TEST_BUILD_TARGET}"
    LABELS
      ${_RULE_LABELS}
  )
  if(_RULE_RESOURCE_GROUP)
    set_property(TEST ${_TEST_NAME} PROPERTY RESOURCE_LOCK "${_RULE_RESOURCE_GROUP}")
  endif()
  if(_RULE_WILL_FAIL)
    set_property(TEST ${_TEST_NAME} PROPERTY WILL_FAIL ${_RULE_WILL_FAIL})
  endif()
  if(_RULE_DISABLED)
    set_property(TEST ${_TEST_NAME} PROPERTY DISABLED ${_RULE_DISABLED})
  endif()

  set_property(TEST ${_TEST_NAME} APPEND PROPERTY ENVIRONMENT
    "$<TARGET_PROPERTY:${_TEST_BUILD_TARGET},IREE_SANITIZER_ENVIRONMENT>")
endfunction()
