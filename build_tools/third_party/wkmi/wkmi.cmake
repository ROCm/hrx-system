# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

function(iree_configure_amdf_wkmi)
  if(TARGET iree::third_party::amdf_wkmi)
    return()
  endif()

  iree_populate_locked_file(amdf_wkmi_header _wkmi_header)
  if(WIN32)
    iree_populate_locked_file(amdf_wkmi_library _wkmi_library)
  elseif(
    CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
    CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$"
  )
    iree_populate_locked_file(amdf_wkmi_linux_library _wkmi_library)
  else()
    message(FATAL_ERROR "WKMI is available only for x86-64 Windows and Linux")
  endif()
  cmake_path(GET _wkmi_header PARENT_PATH _wkmi_file_dir)

  add_library(iree_amdf_wkmi STATIC IMPORTED GLOBAL)
  set_target_properties(iree_amdf_wkmi PROPERTIES
    IMPORTED_LOCATION "${_wkmi_library}"
  )
  target_include_directories(iree_amdf_wkmi SYSTEM INTERFACE
    "${_wkmi_file_dir}"
  )
  add_library(iree::third_party::amdf_wkmi ALIAS iree_amdf_wkmi)
endfunction()
