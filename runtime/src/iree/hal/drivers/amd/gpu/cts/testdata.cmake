# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include_guard(GLOBAL)

# Matches the source-once, configured-product contract in testdata.bzl. TARGETS
# contains selector=capability-variable pairs generated from Loom's target table.
function(iree_hal_amd_gpu_cts_testdata)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()
  cmake_parse_arguments(_RULE "" "NAME" "SRCS;BACKEND_NAMES;TARGETS" ${ARGN})
  if(_RULE_UNPARSED_ARGUMENTS OR NOT _RULE_NAME OR NOT _RULE_SRCS OR
     NOT _RULE_BACKEND_NAMES OR NOT _RULE_TARGETS)
    message(FATAL_ERROR "Incomplete AMD GPU CTS corpus declaration")
  endif()

  set(_TARGETS)
  foreach(_ENTRY IN LISTS _RULE_TARGETS)
    string(REPLACE "=" ";" _FIELDS "${_ENTRY}")
    list(GET _FIELDS 0 _SELECTOR)
    list(GET _FIELDS 1 _CAPABILITY)
    if(${_CAPABILITY})
      list(APPEND _TARGETS "${_SELECTOR}")
    endif()
  endforeach()
  if(NOT _TARGETS)
    return()
  endif()

  set(_PROGRAMS)
  foreach(_SOURCE IN LISTS _RULE_SRCS)
    get_filename_component(_PROGRAM "${_SOURCE}" NAME_WLE)
    if(_PROGRAM IN_LIST _PROGRAMS)
      message(FATAL_ERROR "CTS sources must have distinct basenames: ${_PROGRAM}")
    endif()
    list(APPEND _PROGRAMS "${_PROGRAM}")
    loom_module(
      NAME "${_RULE_NAME}_source_${_PROGRAM}"
      SRCS "${_SOURCE}"
      MODE merge
      OUTPUT_FORMAT bc
      STRICT_DEPS
    )
  endforeach()

  set(_REGISTRATIONS)
  foreach(_SELECTOR IN LISTS _TARGETS)
    set(_PROFILE "cts_profile_${_SELECTOR}")
    iree_package_target_name(_PROFILE_TARGET "::${_PROFILE}")
    if(NOT TARGET "${_PROFILE_TARGET}")
      loom_target_profile(NAME "${_PROFILE}" FAMILY amdgpu SELECTOR "${_SELECTOR}")
    endif()
    set(_PRODUCTS)
    foreach(_PROGRAM IN LISTS _PROGRAMS)
      set(_OUTPUT "${_RULE_NAME}_${_SELECTOR}/${_PROGRAM}.bin")
      loom_kernel_binary(
        NAME "${_RULE_NAME}_${_SELECTOR}_${_PROGRAM}"
        TARGET "::${_PROFILE}"
        OUTPUT "${_OUTPUT}"
        LIBRARIES "::${_RULE_NAME}_source_${_PROGRAM}"
        TESTONLY
      )
      list(APPEND _PRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/${_OUTPUT}")
    endforeach()
    set(_REGISTRATION "${_RULE_NAME}_${_SELECTOR}")
    string(REPLACE "-" "_" _TARGET_NAME "${_SELECTOR}")
    iree_runtime_hal_cts_testdata(
      NAME "${_REGISTRATION}"
      SRCS ${_PRODUCTS}
      BACKEND_NAMES ${_RULE_BACKEND_NAMES}
      TARGET_NAME "${_TARGET_NAME}"
      TARGET_FAMILY amdgpu
      TARGET_KEY "${_SELECTOR}"
    )
    list(APPEND _REGISTRATIONS "::${_REGISTRATION}")
  endforeach()
  iree_cc_library(NAME "${_RULE_NAME}" DEPS ${_REGISTRATIONS}
    ALWAYSLINK TESTONLY PUBLIC)
endfunction()
