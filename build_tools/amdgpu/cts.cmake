# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)
include("${CMAKE_CURRENT_LIST_DIR}/binary.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/selectors.cmake")

# Builds and registers AMDGPU HAL CTS executable testdata.
#
# Parameters:
# NAME: Aggregate registration library target.
# TARGET: LLVM `-target` flag.
# TARGETS: AMDGPU target selectors to expand to exact targets.
# FORMAT_NAME: CTS executable format prefix.
# FORMAT_STRING: HAL executable format string with `{AMDGPU_TARGET}`.
# IDENTIFIER: C identifier prefix for generated TOC functions.
# BACKEND_NAME: CTS backend name. Defaults to "amdgpu".
# SRCS: C sources. Each source basename maps to `<basename>.bin` in the CTS
#       executable-data table of contents.
# INTERNAL_HDRS: Headers that should invalidate device compilation.
# COPTS: Additional flags to pass to clang.
# LINKOPTS: Additional flags to pass to lld.
# TESTONLY: Only build generated targets when tests are enabled.
function(iree_amdgpu_hal_cts_testdata)
  cmake_parse_arguments(
    _RULE
    "TESTONLY"
    "NAME;TARGET;FORMAT_NAME;FORMAT_STRING;IDENTIFIER;BACKEND_NAME"
    "TARGETS;SRCS;INTERNAL_HDRS;COPTS;LINKOPTS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()
  if(NOT _RULE_NAME)
    message(FATAL_ERROR "iree_amdgpu_hal_cts_testdata requires NAME")
  endif()
  if(NOT _RULE_TARGET)
    message(FATAL_ERROR "iree_amdgpu_hal_cts_testdata requires TARGET")
  endif()
  if(NOT _RULE_FORMAT_NAME)
    message(FATAL_ERROR "iree_amdgpu_hal_cts_testdata requires FORMAT_NAME")
  endif()
  if(NOT _RULE_FORMAT_STRING)
    message(FATAL_ERROR "iree_amdgpu_hal_cts_testdata requires FORMAT_STRING")
  endif()
  if(NOT _RULE_IDENTIFIER)
    message(FATAL_ERROR "iree_amdgpu_hal_cts_testdata requires IDENTIFIER")
  endif()
  if(NOT _RULE_BACKEND_NAME)
    set(_RULE_BACKEND_NAME "amdgpu")
  endif()

  iree_amdgpu_expand_target_selectors(
    _EXACT_TARGETS
    "${IREE_AMDGPU_TARGET_EXPANSION_EXACT}"
    ${_RULE_TARGETS}
  )

  set(_TESTONLY_ARG)
  if(_RULE_TESTONLY)
    set(_TESTONLY_ARG TESTONLY)
  endif()

  set(_TARGET_LIBS)
  foreach(_EXACT_TARGET ${_EXACT_TARGETS})
    iree_amdgpu_target_label_fragment(_TARGET_FRAGMENT "${_EXACT_TARGET}")
    iree_amdgpu_target_code_object(_CODE_OBJECT_TARGET "${_EXACT_TARGET}")
    set(_TARGET_IDENTIFIER "${_RULE_IDENTIFIER}_${_TARGET_FRAGMENT}")
    set(_TARGET_SRCS)
    foreach(_SRC ${_RULE_SRCS})
      get_filename_component(_SRC_STEM "${_SRC}" NAME_WE)
      set(_BINARY_NAME "${_RULE_NAME}_${_TARGET_FRAGMENT}_${_SRC_STEM}")
      set(_BINARY_OUT "${_RULE_NAME}_${_TARGET_FRAGMENT}/${_SRC_STEM}.bin")
      iree_amdgpu_binary(
        NAME
          "${_BINARY_NAME}"
        OUT
          "${_BINARY_OUT}"
        TARGET
          "${_RULE_TARGET}"
        ARCH
          "${_CODE_OBJECT_TARGET}"
        SRCS
          "${_SRC}"
        INTERNAL_HDRS
          ${_RULE_INTERNAL_HDRS}
        COPTS
          ${_RULE_COPTS}
        LINKOPTS
          ${_RULE_LINKOPTS}
      )
      list(APPEND _TARGET_SRCS "${_BINARY_OUT}")
    endforeach()

    set(_DATA_NAME "${_RULE_NAME}_${_TARGET_FRAGMENT}_data")
    set(_DATA_HEADER "${_DATA_NAME}.h")
    iree_c_embed_data(
      NAME
        "${_DATA_NAME}"
      SRCS
        ${_TARGET_SRCS}
      C_FILE_OUTPUT
        "${_DATA_NAME}.c"
      H_FILE_OUTPUT
        "${_DATA_HEADER}"
      IDENTIFIER
        "${_TARGET_IDENTIFIER}"
      FLATTEN
      ${_TESTONLY_ARG}
      PUBLIC
    )

    set(_TARGET_FORMAT_NAME "${_RULE_FORMAT_NAME}_${_TARGET_FRAGMENT}")
    string(REPLACE "{AMDGPU_TARGET}" "${_EXACT_TARGET}" _FORMAT_STRING
      "${_RULE_FORMAT_STRING}"
    )
    set(_REGISTRATION "${_RULE_NAME}_${_TARGET_FRAGMENT}_registration.cc")
    set(_REGISTRATION_TEMPLATE
      "${IREE_SOURCE_DIR}/runtime/src/iree/hal/cts/util/testdata_format.cc.tpl"
    )
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      "${_REGISTRATION_TEMPLATE}"
    )
    file(READ
      "${_REGISTRATION_TEMPLATE}"
      _REGISTRATION_CONTENT
    )
    string(REPLACE "{BACKEND_NAME}" "${_RULE_BACKEND_NAME}"
      _REGISTRATION_CONTENT "${_REGISTRATION_CONTENT}"
    )
    string(REPLACE "{FORMAT_FUNC_NAME}" "${_TARGET_IDENTIFIER}"
      _REGISTRATION_CONTENT "${_REGISTRATION_CONTENT}"
    )
    string(REPLACE "{FORMAT_NAME}" "${_TARGET_FORMAT_NAME}"
      _REGISTRATION_CONTENT "${_REGISTRATION_CONTENT}"
    )
    string(REPLACE "{FORMAT_STRING}" "${_FORMAT_STRING}"
      _REGISTRATION_CONTENT "${_REGISTRATION_CONTENT}"
    )
    string(REPLACE "{FORMAT_VAR_NAME}" "${_TARGET_IDENTIFIER}_format"
      _REGISTRATION_CONTENT "${_REGISTRATION_CONTENT}"
    )
    string(REPLACE "{HEADER_PATH}" "${_DATA_HEADER}"
      _REGISTRATION_CONTENT "${_REGISTRATION_CONTENT}"
    )
    string(REPLACE "{IDENTIFIER}" "${_TARGET_IDENTIFIER}"
      _REGISTRATION_CONTENT "${_REGISTRATION_CONTENT}"
    )
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${_REGISTRATION}"
      "${_REGISTRATION_CONTENT}"
    )

    set(_TARGET_LIB_NAME "${_RULE_NAME}_${_TARGET_FRAGMENT}_lib")
    iree_cc_library(
      NAME
        "${_TARGET_LIB_NAME}"
      SRCS
        "${_REGISTRATION}"
      DEPS
        "::${_DATA_NAME}"
        iree::hal::cts::util::registry
      ALWAYSLINK
      ${_TESTONLY_ARG}
    )
    list(APPEND _TARGET_LIBS "::${_TARGET_LIB_NAME}")
  endforeach()

  iree_cc_library(
    NAME
      "${_RULE_NAME}"
    DEPS
      ${_TARGET_LIBS}
    ${_TESTONLY_ARG}
  )
endfunction()
