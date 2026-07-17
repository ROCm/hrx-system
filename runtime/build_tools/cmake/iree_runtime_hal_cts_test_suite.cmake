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

# Creates CTS test binaries for a concrete HAL backend wrapped by the remote
# loopback adapter. Calls made before the adapter target is declared are
# retained and emitted from the remote CTS package once that target exists.
#
# Parameters:
#   SOURCE_BACKEND_NAME: Concrete backend name registered with the CTS.
#   SOURCE_BACKENDS: Concrete backend registration library targets.
#   TESTDATA_LIBS: Prebuilt executable artifact registration libraries.
#   NAME: Non-empty prefix for generated remote test binary names.
#   ARGS: Runtime arguments passed to all generated test binaries.
#   LABELS: Test labels used for filtering.
#   RESOURCE_GROUP: Optional shared resource group.
function(_iree_runtime_hal_remote_cts_defer_test_suite)
  cmake_parse_arguments(
    _RULE
    ""
    "SOURCE_BACKEND_NAME;NAME;RESOURCE_GROUP"
    "SOURCE_BACKENDS;TESTDATA_LIBS;ARGS;LABELS"
    ${ARGN}
  )

  get_property(
    _DEFERRED_COUNT GLOBAL
    PROPERTY IREE_RUNTIME_HAL_REMOTE_CTS_DEFERRED_COUNT
  )
  if(NOT _DEFERRED_COUNT)
    set(_DEFERRED_COUNT 0)
  endif()
  math(EXPR _DEFERRED_INDEX "${_DEFERRED_COUNT} + 1")
  set_property(
    GLOBAL PROPERTY IREE_RUNTIME_HAL_REMOTE_CTS_DEFERRED_COUNT
    "${_DEFERRED_INDEX}"
  )

  iree_package_ns(_DEFERRED_PACKAGE_NS)
  set(_SOURCE_BACKENDS)
  foreach(_BACKEND ${_RULE_SOURCE_BACKENDS})
    string(
      REGEX REPLACE "^::" "${_DEFERRED_PACKAGE_NS}::" _FULL_BACKEND
      "${_BACKEND}"
    )
    list(APPEND _SOURCE_BACKENDS "${_FULL_BACKEND}")
  endforeach()
  set(_TESTDATA_LIBS)
  foreach(_LIB ${_RULE_TESTDATA_LIBS})
    string(
      REGEX REPLACE "^::" "${_DEFERRED_PACKAGE_NS}::" _FULL_LIB
      "${_LIB}"
    )
    list(APPEND _TESTDATA_LIBS "${_FULL_LIB}")
  endforeach()

  set(_PROPERTY_PREFIX
    "IREE_RUNTIME_HAL_REMOTE_CTS_DEFERRED_${_DEFERRED_INDEX}"
  )
  set_property(
    GLOBAL PROPERTY "${_PROPERTY_PREFIX}_SOURCE_BACKEND_NAME"
    "${_RULE_SOURCE_BACKEND_NAME}"
  )
  set_property(
    GLOBAL PROPERTY "${_PROPERTY_PREFIX}_SOURCE_BACKENDS"
    ${_SOURCE_BACKENDS}
  )
  set_property(
    GLOBAL PROPERTY "${_PROPERTY_PREFIX}_TESTDATA_LIBS"
    ${_TESTDATA_LIBS}
  )
  set_property(
    GLOBAL PROPERTY "${_PROPERTY_PREFIX}_NAME" "${_RULE_NAME}"
  )
  set_property(
    GLOBAL PROPERTY "${_PROPERTY_PREFIX}_ARGS" ${_RULE_ARGS}
  )
  set_property(
    GLOBAL PROPERTY "${_PROPERTY_PREFIX}_LABELS" ${_RULE_LABELS}
  )
  set_property(
    GLOBAL PROPERTY "${_PROPERTY_PREFIX}_RESOURCE_GROUP"
    "${_RULE_RESOURCE_GROUP}"
  )
endfunction()

function(_iree_runtime_hal_remote_cts_emit_test_suite)
  cmake_parse_arguments(
    _RULE
    ""
    "SOURCE_BACKEND_NAME;NAME;RESOURCE_GROUP"
    "SOURCE_BACKENDS;TESTDATA_LIBS;ARGS;LABELS"
    ${ARGN}
  )

  if(NOT _RULE_SOURCE_BACKEND_NAME)
    message(SEND_ERROR
      "iree_runtime_hal_remote_cts_test_suite requires SOURCE_BACKEND_NAME"
    )
  endif()
  if(NOT _RULE_SOURCE_BACKENDS)
    message(SEND_ERROR
      "iree_runtime_hal_remote_cts_test_suite requires SOURCE_BACKENDS"
    )
  endif()
  if(NOT _RULE_NAME)
    message(SEND_ERROR
      "iree_runtime_hal_remote_cts_test_suite requires a non-empty NAME"
    )
  endif()

  iree_package_ns(_PACKAGE_NS)
  set(_SOURCE_BACKENDS)
  foreach(_BACKEND ${_RULE_SOURCE_BACKENDS})
    string(
      REGEX REPLACE "^::" "${_PACKAGE_NS}::" _FULL_BACKEND "${_BACKEND}"
    )
    list(APPEND _SOURCE_BACKENDS "${_FULL_BACKEND}")
  endforeach()

  set(_ADAPTER_BACKENDS "${_RULE_NAME}_backends")
  iree_cc_library(
    NAME
      "${_ADAPTER_BACKENDS}"
    DEPS
      ${_SOURCE_BACKENDS}
      iree::hal::remote::cts::loopback_adapter
    TESTONLY
    ALWAYSLINK
    PUBLIC
  )

  iree_runtime_hal_cts_test_suite(
    BACKENDS
      "::${_ADAPTER_BACKENDS}"
    TESTDATA_LIBS
      ${_RULE_TESTDATA_LIBS}
    NAME
      "${_RULE_NAME}"
    ARGS
      ${_RULE_ARGS}
      "--cts_backend_filter=remote_${_RULE_SOURCE_BACKEND_NAME}"
    LABELS
      ${_RULE_LABELS}
      "driver=remote"
      "wrapped-driver=${_RULE_SOURCE_BACKEND_NAME}"
    RESOURCE_GROUP
      "${_RULE_RESOURCE_GROUP}"
  )
endfunction()

function(iree_runtime_hal_remote_cts_test_suite)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  if(NOT TARGET iree::hal::remote::cts::loopback_adapter)
    _iree_runtime_hal_remote_cts_defer_test_suite(${ARGN})
    return()
  endif()

  _iree_runtime_hal_remote_cts_emit_test_suite(${ARGN})
endfunction()

function(iree_runtime_hal_remote_cts_flush_deferred_suites)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  if(NOT TARGET iree::hal::remote::cts::loopback_adapter)
    message(SEND_ERROR
      "iree_runtime_hal_remote_cts_flush_deferred_suites requires "
      "iree::hal::remote::cts::loopback_adapter"
    )
  endif()

  get_property(
    _DEFERRED_COUNT GLOBAL
    PROPERTY IREE_RUNTIME_HAL_REMOTE_CTS_DEFERRED_COUNT
  )
  if(NOT _DEFERRED_COUNT)
    return()
  endif()

  foreach(_DEFERRED_INDEX RANGE 1 ${_DEFERRED_COUNT})
    set(_PROPERTY_PREFIX
      "IREE_RUNTIME_HAL_REMOTE_CTS_DEFERRED_${_DEFERRED_INDEX}"
    )
    foreach(
      _PROPERTY
      SOURCE_BACKEND_NAME
      SOURCE_BACKENDS
      TESTDATA_LIBS
      NAME
      ARGS
      LABELS
      RESOURCE_GROUP
    )
      get_property(
        _${_PROPERTY} GLOBAL PROPERTY "${_PROPERTY_PREFIX}_${_PROPERTY}"
      )
    endforeach()

    _iree_runtime_hal_remote_cts_emit_test_suite(
      SOURCE_BACKEND_NAME
        "${_SOURCE_BACKEND_NAME}"
      SOURCE_BACKENDS
        ${_SOURCE_BACKENDS}
      TESTDATA_LIBS
        ${_TESTDATA_LIBS}
      NAME
        "${_NAME}_${_SOURCE_BACKEND_NAME}"
      ARGS
        ${_ARGS}
      LABELS
        ${_LABELS}
      RESOURCE_GROUP
        "${_RESOURCE_GROUP}"
    )
  endforeach()

  set_property(GLOBAL PROPERTY IREE_RUNTIME_HAL_REMOTE_CTS_DEFERRED_COUNT 0)
endfunction()
