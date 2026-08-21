# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)
include("${CMAKE_CURRENT_LIST_DIR}/binary.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/selectors.cmake")

function(_iree_amdgpu_cts_target_deps out_var code_object_target)
  iree_amdgpu_target_label_fragment(
    _CODE_OBJECT_TARGET_FRAGMENT
    "${code_object_target}"
  )
  set(_TARGET_DEPS)
  foreach(_DEP ${ARGN})
    set(_TARGET_DEP "${_DEP}")
    string(REPLACE "{AMDGPU_CODE_OBJECT_TARGET}" "${code_object_target}"
      _TARGET_DEP "${_TARGET_DEP}"
    )
    string(REPLACE
      "{AMDGPU_CODE_OBJECT_TARGET_FRAGMENT}" "${_CODE_OBJECT_TARGET_FRAGMENT}"
      _TARGET_DEP "${_TARGET_DEP}"
    )
    list(APPEND _TARGET_DEPS "${_TARGET_DEP}")
  endforeach()
  set(${out_var} "${_TARGET_DEPS}" PARENT_SCOPE)
endfunction()

# Builds and registers AMDGPU HAL CTS executable testdata.
#
# Parameters:
# NAME: Aggregate registration library target.
# TARGET: LLVM `-target` flag.
# TARGETS: AMDGPU target selectors to expand to exact targets.
# TARGET_NAME: CTS executable target prefix.
# IDENTIFIER: C identifier prefix for generated TOC functions.
# BACKEND_NAME: CTS backend name. Defaults to "amdgpu".
# SRCS: C sources. Each source basename maps to `<basename>.bin` in the CTS
#       executable-data table of contents.
# DEPS: Bitcode archives passed to each generated executable. Entries may use
#       `{AMDGPU_CODE_OBJECT_TARGET}` and
#       `{AMDGPU_CODE_OBJECT_TARGET_FRAGMENT}` placeholders.
# INTERNAL_HDRS: Headers that should invalidate device compilation.
# COPTS: Additional flags to pass to clang.
# LINKOPTS: Additional flags to pass to lld.
# INTERNALIZE: Whether to internalize linked dependency symbols after lazy
#              archive extraction. Defaults ON.
# TESTONLY: Only build generated targets when tests are enabled.
function(iree_amdgpu_hal_cts_testdata)
  cmake_parse_arguments(
    _RULE
    "TESTONLY"
    "NAME;TARGET;TARGET_NAME;IDENTIFIER;BACKEND_NAME;TARGET_FAMILY;INTERNALIZE"
    "TARGETS;SRCS;DEPS;INTERNAL_HDRS;COPTS;LINKOPTS"
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
  if(NOT _RULE_TARGET_NAME)
    message(FATAL_ERROR "iree_amdgpu_hal_cts_testdata requires TARGET_NAME")
  endif()
  if(NOT _RULE_IDENTIFIER)
    message(FATAL_ERROR "iree_amdgpu_hal_cts_testdata requires IDENTIFIER")
  endif()
  if(NOT _RULE_BACKEND_NAME)
    set(_RULE_BACKEND_NAME "amdgpu")
  endif()
  if(NOT _RULE_TARGET_FAMILY)
    set(_RULE_TARGET_FAMILY "amdgpu")
  endif()

  if(NOT IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_AVAILABLE)
    set(_TESTONLY_ARG)
    if(_RULE_TESTONLY)
      set(_TESTONLY_ARG TESTONLY)
    endif()
    iree_cc_library(
      NAME
        "${_RULE_NAME}"
      ${_TESTONLY_ARG}
    )
    return()
  endif()

  iree_amdgpu_expand_target_selectors(
    _CODE_OBJECT_TARGETS
    "${IREE_AMDGPU_TARGET_EXPANSION_CODE_OBJECT}"
    ${_RULE_TARGETS}
  )

  set(_TESTONLY_ARG)
  if(_RULE_TESTONLY)
    set(_TESTONLY_ARG TESTONLY)
  endif()
  set(_INTERNALIZE_ARG)
  if(DEFINED _RULE_INTERNALIZE)
    set(_INTERNALIZE_ARG INTERNALIZE "${_RULE_INTERNALIZE}")
  endif()

  set(_TARGET_LIBS)
  foreach(_CODE_OBJECT_TARGET ${_CODE_OBJECT_TARGETS})
    iree_amdgpu_target_label_fragment(
      _TARGET_FRAGMENT
      "${_CODE_OBJECT_TARGET}"
    )
    _iree_amdgpu_cts_target_deps(
      _TARGET_DEPS
      "${_CODE_OBJECT_TARGET}"
      ${_RULE_DEPS}
    )
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
        DEPS
          ${_TARGET_DEPS}
        INTERNAL_HDRS
          ${_RULE_INTERNAL_HDRS}
        COPTS
          ${_RULE_COPTS}
        LINKOPTS
          ${_RULE_LINKOPTS}
        ${_INTERNALIZE_ARG}
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

    set(_REGISTRATION_TARGET_NAME
      "${_RULE_TARGET_NAME}_${_TARGET_FRAGMENT}"
    )
    set(_REGISTRATION "${_RULE_NAME}_${_TARGET_FRAGMENT}_registration.cc")
    set(_REGISTRATION_TEMPLATE
      "${IREE_SOURCE_DIR}/runtime/src/iree/hal/cts/util/testdata_target.cc.tpl"
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
    string(REPLACE "{TARGET_FUNC_NAME}" "${_TARGET_IDENTIFIER}"
      _REGISTRATION_CONTENT "${_REGISTRATION_CONTENT}"
    )
    string(REPLACE "{TARGET_NAME}" "${_REGISTRATION_TARGET_NAME}"
      _REGISTRATION_CONTENT "${_REGISTRATION_CONTENT}"
    )
    string(REPLACE "{TARGET_FAMILY}" "${_RULE_TARGET_FAMILY}"
      _REGISTRATION_CONTENT "${_REGISTRATION_CONTENT}"
    )
    string(REPLACE "{TARGET_KEY}" "${_CODE_OBJECT_TARGET}"
      _REGISTRATION_CONTENT "${_REGISTRATION_CONTENT}"
    )
    string(REPLACE "{TARGET_VAR_NAME}" "${_TARGET_IDENTIFIER}_target"
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
