# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# CTS test suite infrastructure for HAL drivers.
#
# Two functions:
#
# 1. iree_hal_cts_testdata() - Accepts compiler-era executable testdata calls.
#    Runtime-only builds do not compile MLIR testdata, so the function emits no
#    targets. Consumers use target existence to decide whether executable
#    payload tests can be compiled.
#
# 2. iree_hal_cts_test_suite() - Creates CTS test binaries for a driver,
#    linking against pre-built testdata libraries from iree_hal_cts_testdata().

# iree_hal_cts_testdata()
#
# Accepts an executable testdata declaration from Bazel-generated CMake.
# Runtime-only builds do not have the compiler pipeline needed to lower MLIR
# test inputs into backend executable payloads, so this function intentionally
# creates no targets. Generated CMake uses target existence checks around
# executable-dependent tests and benchmarks; creating an empty table would make
# those tests compile and then fail at runtime.
#
# Parameters:
#   FORMAT_NAME: Short name (e.g., "vmvx", "llvm_cpu", "cuda").
#   FORMAT_VARIANT_TOKEN: Optional token in FORMAT_STRING to replace per variant.
#   FORMAT_VARIANTS: Optional variant values. Each value registers
#     ${FORMAT_NAME}_${variant} while preserving the base target names.
#   TARGET_DEVICE: Accepted for compatibility with old compiler-backed calls.
#   IDENTIFIER: C identifier for the embedded data (e.g., "iree_cts_testdata_vmvx").
#   BACKEND_NAME: Backend name for CtsRegistry (e.g., "local_task").
#   FORMAT_STRING: C expression for the format (e.g., "vmvx-bytecode-fb").
#   TESTDATA_DIR: Accepted for compatibility with old compiler-backed calls.
#   FLAGS: Accepted for compatibility with old compiler-backed calls.
function(iree_hal_cts_testdata)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()
endfunction()

# iree_hal_cts_test_suite()
#
# Creates CTS test binaries for a HAL driver. Non-executable tests (buffer,
# command_buffer, core, file, queue) are always created. Executable-dependent
# tests (dispatch, executable) are created only if TESTDATA_LIBS are provided.
#
# Parameters:
#   BACKENDS_LIB: CMake target for the backends registration library.
#   TESTDATA_LIBS: Testdata library targets from iree_hal_cts_testdata().
#   NAME: Optional prefix for test binary names (e.g., "graph", "stream").
#   ARGS: Runtime arguments passed to all test binaries.
#   LABELS: Test labels for filtering.
#   RESOURCE_GROUP: Optional shared resource group. Tests sharing the same
#     resource group do not run concurrently under CTest.
function(iree_hal_cts_test_suite)
  cmake_parse_arguments(
    _RULE
    ""
    "BACKENDS_LIB;NAME;RESOURCE_GROUP"
    "TESTDATA_LIBS;ARGS;LABELS"
    ${ARGN}
  )

  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  if(NOT DEFINED _RULE_BACKENDS_LIB)
    message(SEND_ERROR "iree_hal_cts_test_suite requires BACKENDS_LIB")
  endif()

  # Build the name prefix: "name_" if set, "" otherwise.
  set(_PREFIX "")
  if(_RULE_NAME)
    set(_PREFIX "${_RULE_NAME}_")
  endif()

  set(_COMMON_DEPS
    ${_RULE_BACKENDS_LIB}
    iree::hal::cts::util::registry
    iree::hal::cts::util::test_base
    iree::testing::gtest
  )

  set(_TEST_MAIN
    "${PROJECT_SOURCE_DIR}/runtime/src/iree/hal/cts/util/test_main.cc"
  )

  # Build ARGS block for iree_cc_test.
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

  # Non-executable test categories.
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

  # Executable-dependent test categories require explicit registration
  # libraries. Runtime-only builds must not assume an in-tree compiler can
  # synthesize those payloads from MLIR inputs.
  if(_RULE_TESTDATA_LIBS)
    # Verify all testdata targets exist before wiring the executable suites.
    set(_TESTDATA_AVAILABLE TRUE)
    iree_package_ns(_TESTDATA_PACKAGE_NS)
    foreach(_LIB ${_RULE_TESTDATA_LIBS})
      string(REGEX REPLACE "^::" "${_TESTDATA_PACKAGE_NS}::" _FULL_LIB "${_LIB}")
      string(REPLACE "::" "_" _TARGET_NAME "${_FULL_LIB}")
      if(NOT TARGET "${_TARGET_NAME}")
        set(_TESTDATA_AVAILABLE FALSE)
        break()
      endif()
    endforeach()
  endif()

  if(_RULE_TESTDATA_LIBS AND _TESTDATA_AVAILABLE)
    set(_EXECUTABLE_SUITES
      "dispatch_tests\;iree::hal::cts::command_buffer::all_dispatch_tests"
      "executable_tests\;iree::hal::cts::core::all_executable_tests"
      "queue_dispatch_tests\;iree::hal::cts::queue::queue_dispatch_test"
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
