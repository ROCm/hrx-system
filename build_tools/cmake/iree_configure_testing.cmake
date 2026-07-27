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
set_property(GLOBAL PROPERTY IREE_TEST_TARGET_DEPENDENCIES "")
set_property(GLOBAL PROPERTY IREE_TEST_RESOURCE_BUILD_TARGETS "")
set(IREE_TEST_TMPDIR_ROOT "${IREE_BINARY_DIR}/test_tmpdir")
set(IREE_RUNTIME_RESOURCE_LABEL_PREFIX "runtime-resource=")

# iree_register_test_target_dependency
#
# Records a build dependency to resolve after all repository targets have been
# declared. Test helpers use this when a test in an early package depends on a
# tool target declared by a later package.
#
# Parameters:
#   TARGET: CMake target that should receive the dependency.
#   DEPENDENCY: CMake target required before TARGET is complete.
function(iree_register_test_target_dependency)
  cmake_parse_arguments(
    _RULE
    ""
    "TARGET;DEPENDENCY"
    ""
    ${ARGN}
  )

  if(NOT _RULE_TARGET OR NOT _RULE_DEPENDENCY)
    message(FATAL_ERROR
      "iree_register_test_target_dependency requires TARGET and DEPENDENCY")
  endif()

  set_property(GLOBAL APPEND PROPERTY IREE_TEST_TARGET_DEPENDENCIES
    "${_RULE_TARGET}|${_RULE_DEPENDENCY}")
endfunction()

# iree_register_test_resource_build_target
#
# Records TEST_BUILD_TARGET for aggregate build targets keyed by runtime
# resource labels. Targets are resolved after all repository targets have been
# declared, letting CI build all tests for a resource class before selecting
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

  if(NOT _RULE_TEST_BUILD_TARGET)
    message(FATAL_ERROR
      "iree_register_test_resource_build_target requires TEST_BUILD_TARGET")
  endif()

  foreach(_LABEL IN LISTS _RULE_LABELS)
    if(NOT _LABEL MATCHES "^${IREE_RUNTIME_RESOURCE_LABEL_PREFIX}(.+)$")
      continue()
    endif()
    set(_RESOURCE_NAME "${CMAKE_MATCH_1}")
    set_property(GLOBAL APPEND PROPERTY IREE_TEST_RESOURCE_BUILD_TARGETS
      "${_RESOURCE_NAME}|${_RULE_TEST_BUILD_TARGET}")
  endforeach()
endfunction()

function(_iree_resolve_test_build_target OUTPUT_TARGET_NAME TARGET_NAME)
  set(_TARGET_NAME "${TARGET_NAME}")
  if(TARGET "${_TARGET_NAME}")
    get_target_property(_ALIASED_TARGET "${_TARGET_NAME}" ALIASED_TARGET)
    if(_ALIASED_TARGET)
      set(_TARGET_NAME "${_ALIASED_TARGET}")
    endif()
  elseif("${_TARGET_NAME}" MATCHES "::")
    string(REPLACE "::" "_" _CANDIDATE_TARGET_NAME "${_TARGET_NAME}")
    if(TARGET "${_CANDIDATE_TARGET_NAME}")
      set(_TARGET_NAME "${_CANDIDATE_TARGET_NAME}")
    endif()
  endif()

  if(NOT TARGET "${_TARGET_NAME}")
    message(FATAL_ERROR
      "IREE test build target does not exist: ${TARGET_NAME}")
  endif()
  set(${OUTPUT_TARGET_NAME} "${_TARGET_NAME}" PARENT_SCOPE)
endfunction()

function(iree_finalize_test_build_targets)
  get_property(_TARGET_DEPENDENCIES
    GLOBAL PROPERTY IREE_TEST_TARGET_DEPENDENCIES)
  foreach(_ENTRY IN LISTS _TARGET_DEPENDENCIES)
    if(NOT _ENTRY MATCHES "^([^|]+)[|](.+)$")
      message(FATAL_ERROR
        "IREE test target dependency entry is malformed: ${_ENTRY}")
    endif()
    _iree_resolve_test_build_target(_TARGET_NAME "${CMAKE_MATCH_1}")
    _iree_resolve_test_build_target(_DEPENDENCY_TARGET_NAME "${CMAKE_MATCH_2}")
    get_target_property(_DEPENDENCY_IMPORTED
      "${_DEPENDENCY_TARGET_NAME}"
      IMPORTED)
    if(NOT _DEPENDENCY_IMPORTED)
      add_dependencies("${_TARGET_NAME}" "${_DEPENDENCY_TARGET_NAME}")
    endif()
  endforeach()

  get_property(_RESOURCE_BUILD_TARGETS
    GLOBAL PROPERTY IREE_TEST_RESOURCE_BUILD_TARGETS)
  foreach(_ENTRY IN LISTS _RESOURCE_BUILD_TARGETS)
    if(NOT _ENTRY MATCHES "^([^|]+)[|](.+)$")
      message(FATAL_ERROR
        "IREE test resource build target entry is malformed: ${_ENTRY}")
    endif()
    set(_RESOURCE_NAME "${CMAKE_MATCH_1}")
    _iree_resolve_test_build_target(_TEST_BUILD_TARGET "${CMAKE_MATCH_2}")
    get_target_property(_TEST_BUILD_TARGET_IMPORTED
      "${_TEST_BUILD_TARGET}"
      IMPORTED)
    if(_TEST_BUILD_TARGET_IMPORTED)
      continue()
    endif()

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
    add_dependencies("${_RESOURCE_TARGET}" "${_TEST_BUILD_TARGET}")
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
