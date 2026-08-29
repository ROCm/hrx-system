# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Functions for setting up testing in the IREE project. Configures some specific
# environment variables and sets up the creation of test-specific temporary
# directories.

enable_testing(iree)

# Empty root used to refresh the generated build graph before selecting CTests.
# Every generated target participates in CMake's normal regeneration check, so
# this is a no-op when the graph is current and regenerates stale CTest metadata
# before the selective test runner reads it.
add_custom_target(iree-ctest-refresh)

# A property is apparently the only way to get an uncached global variable.
set_property(GLOBAL PROPERTY IREE_TEST_TMPDIRS "")
set_property(GLOBAL PROPERTY IREE_TEST_RESOURCE_BUILD_TARGETS "")
set_property(GLOBAL PROPERTY IREE_TEST_BUILD_METADATA_KEYS "")
set(IREE_TEST_TMPDIR_ROOT "${IREE_BINARY_DIR}/test_tmpdir")
set(IREE_RUNTIME_RESOURCE_LABEL_PREFIX "runtime-resource=")
set(IREE_CTEST_BUILD_TARGETS_FILE
  "${CMAKE_BINARY_DIR}/iree_ctest_build_targets.json")

# iree_register_test_build_targets
#
# Records concrete build roots for a CTest record. An empty TARGETS list is an
# explicit source-only closure and remains distinct from a test that never
# joined this contract. Finalization writes the validated catalog beside the
# generated CTest files so test runners can join CTest's selected names to the
# corresponding build roots without relying on custom-property serialization.
#
# Rule owners must provide concrete, buildable target names. Targets are
# validated after all repository directories have been processed.
#
# Parameters:
#   TEST_NAME: CTest test receiving the metadata.
#   TARGETS: CMake targets whose transitive closure makes TEST_NAME runnable.
function(iree_register_test_build_targets TEST_NAME)
  cmake_parse_arguments(_RULE "" "" "TARGETS" ${ARGN})

  if(NOT TEST "${TEST_NAME}")
    message(FATAL_ERROR
      "cannot register build targets for missing CTest test: ${TEST_NAME}")
  endif()

  string(SHA256 _TEST_KEY "${TEST_NAME}")
  get_property(_EXISTING_TEST_NAME
    GLOBAL PROPERTY "IREE_TEST_BUILD_METADATA_NAME_${_TEST_KEY}")
  if(_EXISTING_TEST_NAME)
    message(FATAL_ERROR
      "CTest test has duplicate IREE_BUILD_TARGETS metadata: ${TEST_NAME}")
  endif()

  set(_BUILD_TARGETS ${_RULE_TARGETS})
  list(REMOVE_DUPLICATES _BUILD_TARGETS)
  set_property(GLOBAL APPEND PROPERTY IREE_TEST_BUILD_METADATA_KEYS
    "${_TEST_KEY}")
  set_property(GLOBAL PROPERTY "IREE_TEST_BUILD_METADATA_NAME_${_TEST_KEY}"
    "${TEST_NAME}")
  set_property(GLOBAL PROPERTY "IREE_TEST_BUILD_METADATA_TARGETS_${_TEST_KEY}"
    "${_BUILD_TARGETS}")
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

function(_iree_collect_repository_ctests SOURCE_DIRECTORY OUTPUT_TESTS)
  get_property(_TESTS DIRECTORY "${SOURCE_DIRECTORY}" PROPERTY TESTS)
  get_property(_SUBDIRECTORIES
    DIRECTORY "${SOURCE_DIRECTORY}"
    PROPERTY SUBDIRECTORIES)
  foreach(_SUBDIRECTORY IN LISTS _SUBDIRECTORIES)
    string(FIND
      "${_SUBDIRECTORY}/"
      "${PROJECT_SOURCE_DIR}/"
      _PROJECT_SOURCE_PREFIX_INDEX)
    if(NOT _PROJECT_SOURCE_PREFIX_INDEX EQUAL 0)
      continue()
    endif()
    _iree_collect_repository_ctests("${_SUBDIRECTORY}" _SUBDIRECTORY_TESTS)
    list(APPEND _TESTS ${_SUBDIRECTORY_TESTS})
  endforeach()
  set(${OUTPUT_TESTS} "${_TESTS}" PARENT_SCOPE)
endfunction()

function(iree_finalize_test_build_targets)
  get_property(_TEST_METADATA_KEYS
    GLOBAL PROPERTY IREE_TEST_BUILD_METADATA_KEYS)
  set(_TESTS_WITH_BUILD_METADATA)
  foreach(_TEST_KEY IN LISTS _TEST_METADATA_KEYS)
    get_property(_TEST_NAME
      GLOBAL PROPERTY "IREE_TEST_BUILD_METADATA_NAME_${_TEST_KEY}")
    get_property(_BUILD_TARGETS
      GLOBAL PROPERTY "IREE_TEST_BUILD_METADATA_TARGETS_${_TEST_KEY}")
    list(APPEND _TESTS_WITH_BUILD_METADATA "${_TEST_NAME}")
    foreach(_BUILD_TARGET IN LISTS _BUILD_TARGETS)
      if(NOT TARGET "${_BUILD_TARGET}")
        message(FATAL_ERROR
          "CTest test ${_TEST_NAME} has missing IREE_BUILD_TARGETS target: "
          "${_BUILD_TARGET}")
      endif()
      get_target_property(_BUILD_TARGET_ALIASED
        "${_BUILD_TARGET}"
        ALIASED_TARGET)
      if(_BUILD_TARGET_ALIASED)
        message(FATAL_ERROR
          "CTest test ${_TEST_NAME} has non-buildable alias "
          "IREE_BUILD_TARGETS target: ${_BUILD_TARGET}")
      endif()
      get_target_property(_BUILD_TARGET_IMPORTED
        "${_BUILD_TARGET}"
        IMPORTED)
      if(_BUILD_TARGET_IMPORTED)
        message(FATAL_ERROR
          "CTest test ${_TEST_NAME} has non-buildable imported "
          "IREE_BUILD_TARGETS target: ${_BUILD_TARGET}")
      endif()
      get_target_property(_BUILD_TARGET_TYPE "${_BUILD_TARGET}" TYPE)
      if(_BUILD_TARGET_TYPE STREQUAL "INTERFACE_LIBRARY")
        message(FATAL_ERROR
          "CTest test ${_TEST_NAME} has non-buildable interface library "
          "IREE_BUILD_TARGETS target: ${_BUILD_TARGET}")
      endif()
    endforeach()
  endforeach()

  _iree_collect_repository_ctests("${PROJECT_SOURCE_DIR}" _REPOSITORY_TESTS)
  foreach(_TEST_NAME IN LISTS _REPOSITORY_TESTS)
    if(NOT _TEST_NAME IN_LIST _TESTS_WITH_BUILD_METADATA)
      message(FATAL_ERROR
        "repository CTest test is missing IREE_BUILD_TARGETS metadata: "
        "${_TEST_NAME}")
    endif()
  endforeach()

  set(_BUILD_TARGET_CATALOG
    "{\"kind\":\"ireeCtestBuildTargets\",\"version\":1,\"tests\":{}}")
  foreach(_TEST_KEY IN LISTS _TEST_METADATA_KEYS)
    get_property(_TEST_NAME
      GLOBAL PROPERTY "IREE_TEST_BUILD_METADATA_NAME_${_TEST_KEY}")
    get_property(_BUILD_TARGETS
      GLOBAL PROPERTY "IREE_TEST_BUILD_METADATA_TARGETS_${_TEST_KEY}")
    set(_BUILD_TARGETS_JSON "[]")
    set(_BUILD_TARGET_INDEX 0)
    foreach(_BUILD_TARGET IN LISTS _BUILD_TARGETS)
      string(JSON _BUILD_TARGETS_JSON
        SET "${_BUILD_TARGETS_JSON}"
        ${_BUILD_TARGET_INDEX}
        "\"${_BUILD_TARGET}\"")
      math(EXPR _BUILD_TARGET_INDEX "${_BUILD_TARGET_INDEX} + 1")
    endforeach()
    string(JSON _BUILD_TARGET_CATALOG
      SET "${_BUILD_TARGET_CATALOG}"
      tests "${_TEST_NAME}" "${_BUILD_TARGETS_JSON}")
  endforeach()
  file(WRITE
    "${IREE_CTEST_BUILD_TARGETS_FILE}"
    "${_BUILD_TARGET_CATALOG}\n")

  get_property(_RESOURCE_BUILD_TARGETS
    GLOBAL PROPERTY IREE_TEST_RESOURCE_BUILD_TARGETS)
  foreach(_ENTRY IN LISTS _RESOURCE_BUILD_TARGETS)
    if(NOT _ENTRY MATCHES "^([^|]+)[|](.+)$")
      message(FATAL_ERROR
        "IREE test resource build target entry is malformed: ${_ENTRY}")
    endif()
    set(_RESOURCE_NAME "${CMAKE_MATCH_1}")
    _iree_resolve_target(_TEST_BUILD_TARGET "${CMAKE_MATCH_2}")
    if(NOT _TEST_BUILD_TARGET)
      message(FATAL_ERROR
        "IREE test build target does not exist: ${CMAKE_MATCH_2}")
    endif()
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
