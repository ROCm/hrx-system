# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Builds an unloadable private Linux DSO whose binary-only C++ ABI stays behind
# one versioned C entry point.
function(amdf_linux_sidecar_library)
  cmake_parse_arguments(
    _RULE
    "TESTONLY"
    "NAME"
    "SRCS;DEPS;COPTS;LINKOPTS"
    ${ARGN}
  )
  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()
  if(NOT _RULE_NAME)
    message(FATAL_ERROR "amdf_linux_sidecar_library requires NAME.")
  endif()
  if(NOT _RULE_SRCS)
    message(FATAL_ERROR
      "amdf_linux_sidecar_library requires at least one source file.")
  endif()

  set(_LINKOPTS ${_RULE_LINKOPTS})
  if(NOT IREE_ENABLE_ASAN AND NOT IREE_ENABLE_MSAN AND
     NOT IREE_ENABLE_TSAN AND NOT IREE_ENABLE_UBSAN)
    list(APPEND _LINKOPTS "-Wl,--no-undefined")
  endif()
  iree_cc_library(
    NAME
      "${_RULE_NAME}"
    SRCS
      ${_RULE_SRCS}
    DEPS
      ${_RULE_DEPS}
    COPTS
      ${_RULE_COPTS}
    LINKOPTS
      ${_LINKOPTS}
    SHARED
  )

  iree_package_name(_PACKAGE_NAME)
  set(_TARGET_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  set_target_properties(${_TARGET_NAME} PROPERTIES
    OUTPUT_NAME "${_RULE_NAME}"
    LIBRARY_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:amdf>"
  )
  if(NOT AMDF_LINUX_GPU_PROVIDER STREQUAL "dxg")
    set_target_properties(${_TARGET_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  elseif(NOT _RULE_TESTONLY)
    install(
      TARGETS ${_TARGET_NAME}
      COMPONENT AMDF
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )
  endif()
endfunction()
