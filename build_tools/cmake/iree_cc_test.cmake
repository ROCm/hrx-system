# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include("${CMAKE_CURRENT_LIST_DIR}/iree_test_arguments.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../sanitizer/iree_sanitizer_suppressions.cmake")

# iree_cc_test()
#
# CMake function to imitate Bazel's cc_test rule.
#
# Parameters:
# NAME: name of target. This name is used for the generated executable and
#     CTest target.
# ARGS: List of command line arguments to pass to the test binary.
#     Note: flag passing is only enforced through CTest, so manually running
#     the test binaries (such as under a debugger) will _not_ pass any
#     arguments without extra setup.
#     File paths use the same {{file}} locators as iree_native_test.
# ENV: KEY=VALUE environment variables, with optional {{file}} locators.
# SRCS: List of source files for the binary.
# DATA: List of other targets and files required for this binary.
# DEPS: List of other libraries to be linked in to the binary targets.
# COPTS: List of private compile options.
# CXX_STANDARD: Optional C++ standard version, such as 23.
# CXX_FEATURES: Required private C++ runtime features: exceptions and rtti.
# DEFINES: List of public defines.
# LINKOPTS: List of link options.
# GROUP: Optional test group to add the target to.
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
# SANITIZER_SUPPRESSIONS: Sanitizer/name pairs selecting suppression files.
#     For example: lsan vulkan.
# RESOURCE_GROUP: If set, tests sharing the same RESOURCE_GROUP name will not
#     run concurrently under CTest. Maps to CTest's RESOURCE_LOCK property.
#     Use for tests that compete for shared system resources (e.g.,
#     RLIMIT_MEMLOCK for io_uring, GPU device access for HIP).
#
# Note:
# iree_cc_test will create a binary called ${PACKAGE_NAME}_${NAME}, e.g.
# iree_base_foo_test. Native test commands honor the executable target's
# CROSSCOMPILING_EMULATOR and TEST_LAUNCHER properties.
#
#
# Usage:
# iree_cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
#   PUBLIC
# )
#
# iree_cc_test(
#   NAME
#     awesome_test
#   SRCS
#     "awesome_test.cc"
#   DEPS
#     gtest_main
#     iree::awesome
# )
function(iree_cc_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;RESOURCE_GROUP;CXX_STANDARD"
    "ARGS;SRCS;COPTS;CXX_FEATURES;DEFINES;LINKOPTS;DATA;DEPS;LABELS;GROUP;TIMEOUT;ENV;SANITIZER_SUPPRESSIONS"
    ${ARGN}
  )

  # Prefix the library with the package name, so we get: iree_package_name
  iree_package_name(_PACKAGE_NAME)
  iree_package_ns(_PACKAGE_NS)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  if(_DEBUG_IREE_PACKAGE_NAME)
    message(STATUS "  : iree_cc_test(${_NAME})")
    message(STATUS "  + alias ${_PACKAGE_NS}::${_RULE_NAME}")
  endif()
  add_executable(${_NAME} "")
  # Alias the iree_package_name test binary to iree::package::name.
  # This lets us more clearly map to Bazel and makes it possible to
  # disambiguate the underscores in paths vs. the separators.
  iree_add_alias_executable(${_PACKAGE_NS}::${_RULE_NAME} ${_NAME})

  # If the test binary name matches the package then treat it as a default.
  # For example, foo/bar/ library 'bar' would end up as 'foo::bar'. This isn't
  # likely to be common for tests, but is consistent with the behavior for
  # libraries.
  iree_package_dir(_PACKAGE_DIR)
  if(${_RULE_NAME} STREQUAL ${_PACKAGE_DIR})
    iree_add_alias_executable(${_PACKAGE_NS} ${_NAME})
  endif()

  set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "${_RULE_NAME}")
  if(_RULE_SRCS)
    iree_resolve_source_files(_RULE_SRCS ${_RULE_SRCS})
    iree_set_source_object_names(${_RULE_SRCS})
    target_sources(${_NAME}
      PRIVATE
        ${_RULE_SRCS}
    )
    foreach(_SOURCE IN LISTS _RULE_SRCS)
      iree_generated_output_add_consumer("${_SOURCE}" "${_NAME}")
    endforeach()
  else()
    iree_add_empty_object(${_NAME})
  endif()
  target_include_directories(${_NAME} SYSTEM
    PUBLIC
      "$<BUILD_INTERFACE:${IREE_SOURCE_DIR};${IREE_BINARY_DIR}>"
  )
  target_compile_definitions(${_NAME}
    PUBLIC
      ${_RULE_DEFINES}
  )
  target_compile_options(${_NAME}
    PRIVATE
      ${IREE_DEFAULT_COPTS}
      ${_RULE_COPTS}
  )
  target_link_options(${_NAME}
    PRIVATE
      ${IREE_DEFAULT_LINKOPTS}
      ${_RULE_LINKOPTS}
  )

  # Replace dependencies passed by ::name with iree::package::name

  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  target_link_libraries(${_NAME}
    PUBLIC
      ${_RULE_DEPS}
      ${IREE_DEFAULT_LINK_LIBRARIES}
      ${IREE_DEFAULT_EXECUTABLE_LINK_LIBRARIES}
  )

  # Add all IREE targets to a folder in the IDE for organization.
  set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/test)

  iree_set_cxx_options(${_NAME} "${_RULE_CXX_STANDARD}" ${_RULE_CXX_FEATURES})

  list(APPEND _RULE_DEPS "gmock")

  # Implicit deps.
  if(IREE_IMPLICIT_DEFS_CC_DEPS)
    list(APPEND _RULE_DEPS ${IREE_IMPLICIT_DEFS_CC_DEPS})
  endif()

  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(_NAME_PATH "${_PACKAGE_PATH}/${_RULE_NAME}")
  set(_TEST_BUILD_TARGET "${_NAME}")

  set(_ENVIRONMENT_VARS)
  list(APPEND _ENVIRONMENT_VARS ${_RULE_ENV})

  iree_resolve_test_arguments(_TEST_ARGS _ARG_DATA
    iree_build_test_file_argument ${_RULE_ARGS})
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "wasm32")
    # WASI: bundle the .wasm binary with JS companions and run via Node.js.
    # Uses _iree_wasm_setup_bundler for order-independent collection of JS
    # companion metadata and entry point discovery via genex chains.
    set(_OUTPUT_MJS "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.mjs")
    _iree_wasm_setup_bundler(
      WASM_TARGET ${_NAME}
      OUTPUT "${_OUTPUT_MJS}"
      DEPS ${_RULE_DEPS}
    )

    # Custom target to drive the bundling.
    add_custom_target(${_NAME}_bundle ALL DEPENDS "${_OUTPUT_MJS}")
    set(_TEST_BUILD_TARGET "${_NAME}_bundle")

    add_test(
      NAME
        ${_NAME_PATH}
      COMMAND
        "${NODE_EXECUTABLE}" "${_OUTPUT_MJS}"
        ${_TEST_ARGS}
    )
    iree_configure_test(${_NAME_PATH})
  else()
    add_test(
      NAME
        ${_NAME_PATH}
      COMMAND
        "${_NAME}"
        ${_TEST_ARGS}
      )

    iree_configure_test(${_NAME_PATH})
  endif()
  iree_register_test_build_targets(
    "${_NAME_PATH}"
    TARGETS "${_TEST_BUILD_TARGET}"
  )

  # TODO(benvanik): add an iree_runtime_cc_test that wraps this and adds the
  # runtime-specific options. Today all tests use iree_cc_test, but since this
  # is harmless for non-instrumented tests it's safe to apply to a bit more than
  # we intend.
  if(IREE_ENABLE_RUNTIME_COVERAGE)
    list(APPEND _ENVIRONMENT_VARS LLVM_PROFILE_FILE=${IREE_BINARY_DIR}/coverage/runtime/${_NAME}.profraw)
    set_property(GLOBAL APPEND PROPERTY IREE_RUNTIME_COVERAGE_TARGETS "${_NAME}=$<TARGET_FILE:${_NAME}>")
  endif()

  iree_resolve_test_arguments(_TEST_ENVIRONMENT _ENV_DATA
    iree_build_test_file_argument ${_ENVIRONMENT_VARS})
  list(APPEND _RULE_DATA ${_ARG_DATA} ${_ENV_DATA})
  iree_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA}
    OUT_TARGET_DATA _DATA_TARGETS)
  iree_target_sanitizer_suppressions("${_NAME}"
    DEPS ${_RULE_DEPS} ${_DATA_TARGETS}
    SUPPRESSIONS ${_RULE_SANITIZER_SUPPRESSIONS}
    ENV ${_TEST_ENVIRONMENT})
  set_property(TEST ${_NAME_PATH} APPEND PROPERTY ENVIRONMENT
    ${_TEST_ENVIRONMENT}
    "$<TARGET_PROPERTY:${_NAME},IREE_SANITIZER_ENVIRONMENT>")

  if(NOT DEFINED _RULE_TIMEOUT)
    set(_RULE_TIMEOUT 60)
  endif()

  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")
  set_property(TEST ${_NAME_PATH} PROPERTY LABELS "${_RULE_LABELS}")
  set_property(TEST ${_NAME_PATH} PROPERTY TIMEOUT ${_RULE_TIMEOUT})
  iree_register_test_resource_build_target(
    TEST_BUILD_TARGET
      "${_NAME}"
    LABELS
      ${_RULE_LABELS}
  )

  if(_RULE_RESOURCE_GROUP)
    set_property(TEST ${_NAME_PATH} PROPERTY RESOURCE_LOCK "${_RULE_RESOURCE_GROUP}")
  endif()

  if(_RULE_GROUP)
    if(NOT TARGET ${_RULE_GROUP})
      add_custom_target(${_RULE_GROUP}
        COMMENT
          "Building IREE unit tests for ${_RULE_GROUP}"
      )
    endif()
    add_dependencies(${_RULE_GROUP} "${_NAME}")
  endif()
endfunction()
