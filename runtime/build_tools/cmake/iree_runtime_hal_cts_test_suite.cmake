# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# Creates CTS test binaries for a HAL driver.
#
# Non-executable CTS binaries are always emitted. Executable-dependent CTS
# binaries are emitted only when TESTDATA_LIBS supplies libraries that register
# real executable artifacts with the CTS registry.
#
# Parameters:
#   BACKENDS: Driver-specific backend registration libraries.
#   TESTDATA_LIBS: Prebuilt executable artifact registration libraries.
#   NAME: Optional prefix for generated test binary names.
#   ARGS: Runtime arguments passed to all test binaries.
#   LABELS: Test labels used for filtering.
#   RESOURCE_GROUP: Optional shared resource group. Tests sharing the same
#     resource group do not run concurrently under CTest.
function(iree_runtime_hal_cts_test_suite)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;RESOURCE_GROUP"
    "BACKENDS;TESTDATA_LIBS;ARGS;LABELS"
    ${ARGN}
  )

  if(NOT IREE_BUILD_TESTS)
    return()
  endif()
  if(NOT _RULE_BACKENDS)
    message(FATAL_ERROR
      "iree_runtime_hal_cts_test_suite requires BACKENDS"
    )
  endif()

  set(_PREFIX "")
  if(_RULE_NAME)
    set(_PREFIX "${_RULE_NAME}_")
  endif()

  set(_COMMON_DEPS
    ${_RULE_BACKENDS}
    iree::base::tooling::flags
    iree::hal::cts::util::registry
    iree::hal::cts::util::test_base
    iree::testing::gtest
  )
  set(_TEST_MAIN
    "${PROJECT_SOURCE_DIR}/runtime/src/iree/hal/cts/util/test_main.cc"
  )

  set(_ARGS_BLOCK "")
  if(_RULE_ARGS)
    set(_ARGS_BLOCK ARGS ${_RULE_ARGS})
  endif()
  set(_LABELS_BLOCK "")
  if(_RULE_LABELS)
    set(_LABELS_BLOCK LABELS ${_RULE_LABELS})
  endif()
  set(_RESOURCE_GROUP_BLOCK "")
  if(_RULE_RESOURCE_GROUP)
    set(_RESOURCE_GROUP_BLOCK RESOURCE_GROUP "${_RULE_RESOURCE_GROUP}")
  endif()

  foreach(_CATEGORY buffer command_buffer core file queue)
    iree_cc_test(
      NAME "${_PREFIX}${_CATEGORY}_tests"
      SRCS "${_TEST_MAIN}"
      DEPS
        ${_COMMON_DEPS}
        "iree::hal::cts::${_CATEGORY}::all_tests"
      ${_ARGS_BLOCK}
      ${_LABELS_BLOCK}
      ${_RESOURCE_GROUP_BLOCK}
    )
  endforeach()

  if(_RULE_TESTDATA_LIBS)
    set(_EXECUTABLE_SUITES
      "dispatch_tests\;iree::hal::cts::command_buffer::all_dispatch_tests"
      "executable_tests\;iree::hal::cts::core::all_executable_tests"
      "queue_dispatch_tests\;iree::hal::cts::queue::queue_dispatch_test"
      "sanitizer_tests\;iree::hal::cts::sanitizer::all_tests"
    )
    foreach(_PAIR ${_EXECUTABLE_SUITES})
      list(GET _PAIR 0 _SUFFIX)
      list(GET _PAIR 1 _TEST_LIB)
      iree_cc_test(
        NAME "${_PREFIX}${_SUFFIX}"
        SRCS "${_TEST_MAIN}"
        DEPS
          ${_COMMON_DEPS}
          ${_RULE_TESTDATA_LIBS}
          ${_TEST_LIB}
        ${_ARGS_BLOCK}
        ${_LABELS_BLOCK}
        ${_RESOURCE_GROUP_BLOCK}
      )
    endforeach()
  endif()
endfunction()
