# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

# Declares the pinned Linux DXG protocol headers and private thunk archive.
# These dependencies expose binary-only interfaces and must remain behind the
# unloadable libamdf sidecar instead of entering the installed SDK interface.
function(iree_configure_amdf_rocdxg_thunk_proxy)
  if(TARGET iree::third_party::amdf_rocdxg_thunk_proxy)
    return()
  endif()
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
     NOT IREE_ARCH STREQUAL "x86_64")
    message(FATAL_ERROR
      "ROCDXG thunk support is available only for x86-64 Linux")
  endif()

  foreach(_DEPENDENCY directx_headers libdxg_headers rocdxg_thunk_proxy)
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

  add_library(iree_amdf_rocdxg_headers INTERFACE)
  target_include_directories(iree_amdf_rocdxg_headers SYSTEM INTERFACE
    "$<BUILD_INTERFACE:${_rocdxg_thunk_proxy_SOURCE_DIR}>")
  target_link_libraries(iree_amdf_rocdxg_headers INTERFACE
    iree::third_party::amdf_directx_wsl_stubs
    iree::third_party::amdf_dxg_headers)
  add_library(iree::third_party::amdf_rocdxg_headers ALIAS
    iree_amdf_rocdxg_headers)

  add_library(iree_amdf_rocdxg_thunk_proxy STATIC IMPORTED GLOBAL)
  set_target_properties(iree_amdf_rocdxg_thunk_proxy PROPERTIES
    IMPORTED_LOCATION
      "${_rocdxg_thunk_proxy_SOURCE_DIR}/shared/src/thunk_proxy/libthunk_proxy.a"
  )
  add_library(iree::third_party::amdf_rocdxg_thunk_proxy ALIAS
    iree_amdf_rocdxg_thunk_proxy)
endfunction()
