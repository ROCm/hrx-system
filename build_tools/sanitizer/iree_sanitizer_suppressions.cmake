# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Shared sanitizer suppression helpers for CTest rules.

set(IREE_SANITIZER_SUPPRESSION_ROCM_LSAN
  "${CMAKE_SOURCE_DIR}/build_tools/sanitizer/lsan_suppressions_rocm.txt")
set(IREE_SANITIZER_SUPPRESSION_VULKAN_LSAN
  "${CMAKE_SOURCE_DIR}/build_tools/sanitizer/lsan_suppressions_vulkan.txt")

function(iree_sanitizer_suppression_file OUTPUT_VARIABLE SANITIZER NAME)
  string(TOUPPER "${SANITIZER}" _SANITIZER_UPPER)
  string(TOUPPER "${NAME}" _NAME_UPPER)
  set(_VARIABLE_NAME "IREE_SANITIZER_SUPPRESSION_${_NAME_UPPER}_${_SANITIZER_UPPER}")
  if(NOT DEFINED ${_VARIABLE_NAME})
    message(FATAL_ERROR
      "unknown sanitizer suppression '${SANITIZER}=${NAME}'")
  endif()
  set(${OUTPUT_VARIABLE} "${${_VARIABLE_NAME}}" PARENT_SCOPE)
endfunction()

function(iree_append_sanitizer_suppression_environment OUTPUT_VARIABLE)
  set(_ENVIRONMENT ${${OUTPUT_VARIABLE}})
  set(_ARGS ${ARGN})
  list(LENGTH _ARGS _ARG_COUNT)
  math(EXPR _ARG_COUNT_IS_ODD "${_ARG_COUNT} % 2")
  if(_ARG_COUNT_IS_ODD)
    message(FATAL_ERROR
      "sanitizer suppressions must be SANITIZER NAME pairs")
  endif()

  while(_ARGS)
    list(POP_FRONT _ARGS _SANITIZER)
    list(POP_FRONT _ARGS _NAME)
    string(TOUPPER "${_SANITIZER}" _SANITIZER_UPPER)
    iree_sanitizer_suppression_file(_SUPPRESSION_FILE "${_SANITIZER}" "${_NAME}")
    set(_SANITIZER_OPTIONS "suppressions=${_SUPPRESSION_FILE}")
    if(_SANITIZER_UPPER STREQUAL "LSAN")
      list(APPEND _SANITIZER_OPTIONS "allow_addr2line=1")
    endif()
    string(JOIN ":" _SANITIZER_OPTIONS_VALUE ${_SANITIZER_OPTIONS})
    list(APPEND _ENVIRONMENT
      "${_SANITIZER_UPPER}_OPTIONS=${_SANITIZER_OPTIONS_VALUE}")
  endwhile()

  set(${OUTPUT_VARIABLE} ${_ENVIRONMENT} PARENT_SCOPE)
endfunction()
