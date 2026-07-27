# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_native_test()
#
# Creates a test that runs the specified binary with the specified arguments.
#
# Mirrors the bzl function of the same name.
#
# Parameters:
# NAME: name of target
# DRIVER: If specified, will pass --device=DRIVER to the test binary.
# DATA: Additional input files needed by the test binary. When running tests on
#     a separate device (e.g. Android), these files will be pushed to the
#     device. TEST_INPUT_FILE_ARG is automatically added if specified.
# ARGS: additional arguments passed to the test binary. TEST_INPUT_FILE_ARG and
#     --device=DRIVER are automatically added if specified.
#     File-related arguments can be passed with `{{}}` locator,
#     e.g., --input=@{{foo.npy}}. The locator is used to portably
#     pass the file arguments to tests and add the file to DATA.
# ENV: Additional KEY=VALUE environment variables set while the test runs.
# SRC: binary target to run as the test.
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
# Note: the DATA argument is not actually adding dependencies because CMake
# doesn't have a good way to specify a data dependency for a test.
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
    "NAME;SRC;DRIVER;WILL_FAIL;DISABLED;RESOURCE_GROUP"
    "ARGS;ENV;LABELS;DATA;TIMEOUT;SANITIZER_SUPPRESSIONS"
    ${ARGN}
  )

  # Prefix the test with the package name, so we get: iree_package_name
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  iree_package_ns(_PACKAGE_NS)
  iree_package_path(_PACKAGE_PATH)
  set(_TEST_NAME "${_PACKAGE_PATH}/${_RULE_NAME}")
  set(_IREE_TEST_CAN_REGISTER OFF)

  # If driver was specified, add the corresponding test arg.
  if(DEFINED _RULE_DRIVER)
    list(APPEND _RULE_ARGS "--device=${_RULE_DRIVER}")
  endif()

  set(_TEST_ENVIRONMENT_VARS)
  if(_RULE_SANITIZER_SUPPRESSIONS)
    iree_append_sanitizer_suppression_environment(
      _TEST_ENVIRONMENT_VARS
      ${_RULE_SANITIZER_SUPPRESSIONS}
    )
  endif()
  list(APPEND _TEST_ENVIRONMENT_VARS ${_RULE_ENV})

  if(ANDROID)
    set(_ANDROID_ABS_DIR "/data/local/tmp/${_PACKAGE_PATH}/${_RULE_NAME}")
  endif()

  # Detect file location with `{{}}` and handle its portability for all entries
  # in `_RULE_ARGS`.
  foreach(_ARG ${_RULE_ARGS})
    string(REGEX MATCH ".*{{(.+)}}" _FILE_ARG "${_ARG}")
    if(_FILE_ARG)
      set(_FILE_PATH ${CMAKE_MATCH_1})
      list(APPEND _RULE_DATA "${_FILE_PATH}")
      if (ANDROID)
        cmake_path(GET _FILE_PATH FILENAME _FILE_BASENAME)
        set(_FILE_PATH "${_ANDROID_ABS_DIR}/${_FILE_BASENAME}")
      endif()
      # remove the `{{}}` from `_ARG` and append it to `_TEST_ARGS`.
      string(REGEX REPLACE "{{.+}}" "" _FILE_FLAG_PREFIX "${_ARG}")
      list(APPEND _TEST_ARGS "${_FILE_FLAG_PREFIX}${_FILE_PATH}")
    else()  # naive append
      list(APPEND _TEST_ARGS "${_ARG}")
    endif(_FILE_ARG)
  endforeach(_ARG)

  # Replace binary passed by relative ::name with iree::package::name
  string(REGEX REPLACE "^::" "${_PACKAGE_NS}::" _SRC_TARGET ${_RULE_SRC})

  if(ANDROID)
    # Define a custom target for pushing and running the test on Android device.
    set(_TEST_NAME ${_TEST_NAME}_on_android_device)
    add_test(
      NAME
        ${_TEST_NAME}
      COMMAND
        "${CMAKE_SOURCE_DIR}/build_tools/cmake/run_android_test.${IREE_HOST_SCRIPT_EXT}"
        "${_ANDROID_ABS_DIR}/$<TARGET_FILE_NAME:${_SRC_TARGET}>"
        ${_TEST_ARGS}
    )
    # Use environment variables to instruct the script to push artifacts
    # onto the Android device before running the test. This needs to match
    # with the expectation of the run_android_test.{sh|bat|ps1} script.
    string(REPLACE ";" " " _DATA_SPACE_SEPARATED "${_RULE_DATA}")
    set(
      _ENVIRONMENT_VARS
        "TEST_ANDROID_ABS_DIR=${_ANDROID_ABS_DIR}"
        "TEST_EXECUTABLE=$<TARGET_FILE:${_SRC_TARGET}>"
        "TEST_DATA=${_DATA_SPACE_SEPARATED}"
        "TEST_TMPDIR=${_ANDROID_ABS_DIR}/test_tmpdir"
    )
    set_property(TEST ${_TEST_NAME} PROPERTY ENVIRONMENT ${_ENVIRONMENT_VARS})
  elseif((IREE_ARCH STREQUAL "riscv_64" OR
          IREE_ARCH STREQUAL "riscv_32") AND
         CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # The test target needs to run within the QEMU emulator for RV64 Linux
    # crosscompile build or on-device.
    add_test(
      NAME
        ${_TEST_NAME}
      COMMAND
        "${IREE_ROOT_DIR}/build_tools/cmake/run_riscv_test.sh"
        -L "${RISCV_TOOLCHAIN_ROOT}/sysroot"
        "$<TARGET_FILE:${_SRC_TARGET}>"
        ${_TEST_ARGS}
    )
    iree_configure_test(${_TEST_NAME})
    set_property(TEST ${_TEST_NAME} APPEND PROPERTY ENVIRONMENT
      "QEMU_CPU_FLAGS=${RISCV_QEMU_CPU_FLAGS}")
  elseif(IREE_ARCH STREQUAL "arm_64" AND "requires-arm-sme" IN_LIST _RULE_LABELS)
    add_test(
      NAME
        ${_TEST_NAME}
      COMMAND
        "${IREE_ROOT_DIR}/build_tools/cmake/run_arm_sme_test.sh"
        "$<TARGET_FILE:${_SRC_TARGET}>"
        ${_TEST_ARGS}
    )
    iree_configure_test(${_TEST_NAME})
  else()
    add_test(
      NAME
        ${_TEST_NAME}
      COMMAND
        "$<TARGET_FILE:${_SRC_TARGET}>"
        ${_TEST_ARGS}
    )
    iree_configure_test(${_TEST_NAME})
    set(_IREE_TEST_CAN_REGISTER ON)
  endif()

  # Apply accumulated test environment variables after the test exists.
  if(_TEST_ENVIRONMENT_VARS)
    set_property(TEST ${_TEST_NAME} APPEND PROPERTY ENVIRONMENT
      ${_TEST_ENVIRONMENT_VARS})
  endif()

  if (NOT DEFINED _RULE_TIMEOUT OR "${_RULE_TIMEOUT}" STREQUAL "")
    set(_RULE_TIMEOUT 60)
  endif()

  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")
  set_property(TEST ${_TEST_NAME} PROPERTY LABELS "${_RULE_LABELS}")
  set_property(TEST "${_TEST_NAME}" PROPERTY REQUIRED_FILES "${_RULE_DATA}")
  set_property(TEST ${_TEST_NAME} PROPERTY TIMEOUT ${_RULE_TIMEOUT})
  iree_register_test_resource_build_target(
    TEST_BUILD_TARGET
      "${_SRC_TARGET}"
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

  if(_IREE_TEST_CAN_REGISTER AND
     IREE_TEST_REGISTRATION_FUNCTION AND
     NOT IREE_SKIP_TEST_REGISTRATION)
    set(_IREE_REGISTERED_WILL_FAIL)
    if(_RULE_WILL_FAIL)
      set(_IREE_REGISTERED_WILL_FAIL WILL_FAIL)
    endif()
    set(_IREE_REGISTERED_DISABLED)
    if(_RULE_DISABLED)
      set(_IREE_REGISTERED_DISABLED DISABLED)
    endif()
    set(_IREE_REGISTERED_RESOURCE_GROUP)
    if(_RULE_RESOURCE_GROUP)
      set(_IREE_REGISTERED_RESOURCE_GROUP RESOURCE_GROUP "${_RULE_RESOURCE_GROUP}")
    endif()
    if(COMMAND ${IREE_TEST_REGISTRATION_FUNCTION})
      cmake_language(CALL ${IREE_TEST_REGISTRATION_FUNCTION}
        NAME
          "${_TEST_NAME}"
        TARGET
          "${_SRC_TARGET}"
        ARGS
          ${_TEST_ARGS}
        DATA
          ${_RULE_DATA}
        ENVIRONMENT
          ${_TEST_ENVIRONMENT_VARS}
        LABELS
          ${_RULE_LABELS}
        TIMEOUT
          ${_RULE_TIMEOUT}
        ${_IREE_REGISTERED_RESOURCE_GROUP}
        ${_IREE_REGISTERED_WILL_FAIL}
        ${_IREE_REGISTERED_DISABLED}
      )
    endif()
  endif()
endfunction()
