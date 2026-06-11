# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Functions for setting up testing in the IREE project. Configures some specific
# environment variables and sets up the creation of test-specific temporary
# directories.

enable_testing(iree)
# A property is apparently the only way to get an uncached global variable.
set_property(GLOBAL PROPERTY IREE_TEST_TMPDIRS "")
set(IREE_TEST_TMPDIR_ROOT "${IREE_BINARY_DIR}/test_tmpdir")
set(IREE_RUNTIME_RESOURCE_LABEL_PREFIX "runtime-resource=")

# iree_register_test_resource_build_target
#
# Adds TEST_BUILD_TARGET to aggregate build targets keyed by runtime resource
# labels. This lets CI build all tests for a resource class before selecting
# them with CTest labels without keeping a central inventory of test packages.
#
# Parameters:
#   TEST_BUILD_TARGET: CMake target that must be built before running the test.
#   LABELS: labels assigned to the test.
function(iree_register_test_resource_build_target)
  cmake_parse_arguments(
    _RULE
    ""
    "TEST_BUILD_TARGET"
    "LABELS"
    ${ARGN}
  )

  if(NOT _RULE_TEST_BUILD_TARGET OR NOT TARGET "${_RULE_TEST_BUILD_TARGET}")
    return()
  endif()

  get_target_property(
    _IREE_TEST_RESOURCE_IMPORTED
    "${_RULE_TEST_BUILD_TARGET}"
    IMPORTED
  )
  if(_IREE_TEST_RESOURCE_IMPORTED)
    return()
  endif()

  foreach(_LABEL IN LISTS _RULE_LABELS)
    if(NOT _LABEL MATCHES "^${IREE_RUNTIME_RESOURCE_LABEL_PREFIX}(.+)$")
      continue()
    endif()
    set(_RESOURCE_NAME "${CMAKE_MATCH_1}")
    string(
      REGEX REPLACE "[^A-Za-z0-9_.+-]" "-"
      _RESOURCE_TARGET_SUFFIX "${_RESOURCE_NAME}"
    )
    set(_RESOURCE_TARGET "iree-test-resource-${_RESOURCE_TARGET_SUFFIX}")
    if(NOT TARGET "${_RESOURCE_TARGET}")
      add_custom_target("${_RESOURCE_TARGET}"
        COMMENT
          "Building IREE tests requiring ${IREE_RUNTIME_RESOURCE_LABEL_PREFIX}${_RESOURCE_NAME}"
      )
      set_property(
        TARGET "${_RESOURCE_TARGET}"
        PROPERTY FOLDER ${IREE_IDE_FOLDER}/test
      )
    endif()
    add_dependencies("${_RESOURCE_TARGET}" "${_RULE_TEST_BUILD_TARGET}")
  endforeach()
endfunction()

# iree_configure_test
#
# Registers test for temporary directory creation and adds properties common to
# all IREE tests. This should be invoked with each test added with `add_test`.
#
# Parameters:
#   TEST_NAME: the test name, e.g. iree/base/math_test
function(iree_configure_test TEST_NAME)
  set(_TEST_TMPDIR "${IREE_TEST_TMPDIR_ROOT}/${TEST_NAME}_test_tmpdir")
  set_property(GLOBAL APPEND PROPERTY IREE_TEST_TMPDIRS ${_TEST_TMPDIR})
  set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "TEST_TMPDIR=${_TEST_TMPDIR}")
  set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "IREE_BINARY_DIR=${IREE_BINARY_DIR}")

  # File extension cmake uses for the target platform.
  set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "IREE_DYLIB_EXT=${CMAKE_SHARED_LIBRARY_SUFFIX}")

  # IREE_*_DISABLE environment variables may be used to skip test cases which
  # require a compatible runtime HAL driver.
  #
  # These variables may be set by the test environment, typically as a property
  # of some continuous execution test runner or by an individual developer, or
  # here by the build system.
  if(NOT IREE_HAL_DRIVER_VULKAN)
    set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "IREE_VULKAN_DISABLE=1")
  endif()

  if(NOT IREE_HAL_DRIVER_METAL)
    set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "IREE_METAL_DISABLE=1")
  endif()

endfunction()

# iree_create_ctest_customization
#
# Constructs a CTestCustom.cmake file with custom commands run before ctest
# runs all tests. These commands create new temporary directories for each test
# that was properly configured with `iree_configure_test`.
#
# Note that this must be called after all tests are registered as it depends on
# a global variable (gross, I know).
#
# Takes no arguments
function(iree_create_ctest_customization)
  get_property(IREE_TEST_TMPDIRS GLOBAL PROPERTY IREE_TEST_TMPDIRS)
  set(IREE_CREATE_TEST_TMPDIRS_COMMANDS "")
  set(_CMD_PREFIX "\"cmake -E make_directory")
  set(_CUR_CMD "${_CMD_PREFIX}")
  set(_CMD_LEN_LIMIT 8191)
  foreach(_DIR IN LISTS IREE_TEST_TMPDIRS)
    string(LENGTH "${_CUR_CMD}" _CUR_CMD_LEN)
    if(_CUR_CMD_LEN GREATER _CMD_LEN_LIMIT)
      message(SEND_ERROR
          "Make directory command for single test directory is longer than"
          " maximum command length ${_CMD_LEN_LIMIT}: '${_CUR_CMD}'")
    endif()
    string(LENGTH "${_DIR}" _DIR_LEN)
    math(EXPR _NEW_CMD_LEN "${_CUR_CMD_LEN} + ${_DIR_LEN} + 1")
    if(_NEW_CMD_LEN GREATER _CMD_LEN_LIMIT)
      string(APPEND _CUR_CMD "\"\n")
      string(APPEND IREE_CREATE_TEST_TMPDIRS_COMMANDS "${_CUR_CMD}")
      set(_CUR_CMD "${_CMD_PREFIX} ${_DIR}")
    else()
      string(APPEND _CUR_CMD " ${_DIR}")
    endif()
  endforeach()
  if(NOT _CUR_CMD STREQUAL _CMD_PREFIX)
    string(APPEND _CUR_CMD "\"\n")
    string(APPEND IREE_CREATE_TEST_TMPDIRS_COMMANDS "${_CUR_CMD}")
  endif()

  configure_file("build_tools/cmake/CTestCustom.cmake.in" "${IREE_BINARY_DIR}/CTestCustom.cmake" @ONLY)
endfunction()
