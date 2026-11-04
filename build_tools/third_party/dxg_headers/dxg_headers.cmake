# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

# Declares the pinned public headers used by the Linux DXG transport.
function(iree_configure_amdf_dxg_headers)
  if(TARGET iree::third_party::amdf_dxg_headers)
    return()
  endif()
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
     NOT IREE_ARCH STREQUAL "x86_64")
    message(FATAL_ERROR
      "Linux DXG headers are available only for x86-64 Linux")
  endif()

  foreach(_DEPENDENCY directx_headers libdxg_headers)
    iree_dependency_require_pinned_source_allowed("${_DEPENDENCY}")
    iree_populate_locked_fetch_content(${_DEPENDENCY}
      _${_DEPENDENCY}_SOURCE_DIR)
  endforeach()

  add_library(iree_amdf_directx_wsl_stubs INTERFACE)
  target_include_directories(iree_amdf_directx_wsl_stubs SYSTEM INTERFACE
    "$<BUILD_INTERFACE:${_directx_headers_SOURCE_DIR}/include/wsl/stubs>")
  add_library(iree::third_party::amdf_directx_wsl_stubs ALIAS
    iree_amdf_directx_wsl_stubs)

  add_library(iree_amdf_dxg_headers INTERFACE)
  target_include_directories(iree_amdf_dxg_headers SYSTEM INTERFACE
    "$<BUILD_INTERFACE:${_libdxg_headers_SOURCE_DIR}/include>")
  target_link_libraries(iree_amdf_dxg_headers INTERFACE
    iree::third_party::amdf_directx_wsl_stubs)
  add_library(iree::third_party::amdf_dxg_headers ALIAS
    iree_amdf_dxg_headers)

endfunction()
