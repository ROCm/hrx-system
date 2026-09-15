# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Defines one libamdf CTS executable in the public artifact directory. The
# installed library and its private runtime companions share this layout, so
# every linkage mode exercises the same sibling-library discovery contract.
function(amdf_cts_binary)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    ""
    ${ARGN}
  )
  if(NOT _RULE_NAME)
    message(FATAL_ERROR "amdf_cts_binary requires NAME.")
  endif()

  iree_cc_binary(${ARGN})

  iree_package_name(_PACKAGE_NAME)
  set(_TARGET_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  set_target_properties(${_TARGET_NAME} PROPERTIES
    OUTPUT_NAME "${_TARGET_NAME}"
    RUNTIME_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:amdf>"
  )
endfunction()
