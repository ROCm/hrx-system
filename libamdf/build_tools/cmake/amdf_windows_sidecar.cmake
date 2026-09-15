# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Builds a private Windows DLL whose complete C++ and CRT ABI stays behind a C
# entry point. Binary-only dependencies use their release static CRT in every
# parent build configuration.
function(amdf_windows_sidecar_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;DEPS;LINKOPTS"
    ${ARGN}
  )
  if(NOT _RULE_NAME)
    message(FATAL_ERROR "amdf_windows_sidecar_library requires NAME.")
  endif()
  if(NOT _RULE_SRCS)
    message(FATAL_ERROR
      "amdf_windows_sidecar_library requires at least one source file.")
  endif()

  iree_cc_library(
    NAME
      "${_RULE_NAME}"
    SRCS
      ${_RULE_SRCS}
    DEPS
      ${_RULE_DEPS}
    LINKOPTS
      ${_RULE_LINKOPTS}
    SHARED
  )

  iree_package_name(_PACKAGE_NAME)
  set(_TARGET_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  set_target_properties(
    ${_TARGET_NAME}
    ${_TARGET_NAME}.objects
    PROPERTIES
      MSVC_RUNTIME_LIBRARY "MultiThreaded"
  )
  target_compile_definitions(${_TARGET_NAME}.objects
    PRIVATE
      _DISABLE_STRING_ANNOTATION
      _DISABLE_VECTOR_ANNOTATION
      _ITERATOR_DEBUG_LEVEL=0
  )
  set_target_properties(${_TARGET_NAME} PROPERTIES
    OUTPUT_NAME "${_RULE_NAME}"
    RUNTIME_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:amdf>"
  )
  install(
    TARGETS ${_TARGET_NAME}
    COMPONENT AMDF
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  )
endfunction()
