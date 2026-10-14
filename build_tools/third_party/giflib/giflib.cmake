# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

iree_add_alias_interface(iree::third_party::giflib)

function(_iree_configure_giflib_facade target_name)
  get_target_property(_facade_target
    iree::third_party::giflib ALIASED_TARGET)
  target_link_libraries(${_facade_target} INTERFACE ${target_name})
  set_property(TARGET ${_facade_target} PROPERTY IREE_GIFLIB_CONFIGURED TRUE)
endfunction()

function(iree_configure_giflib)
  get_target_property(_facade_target
    iree::third_party::giflib ALIASED_TARGET)
  get_target_property(_already_configured
    ${_facade_target} IREE_GIFLIB_CONFIGURED)
  if(_already_configured)
    return()
  endif()

  iree_dependency_package_discovery_allowed(_package_discovery_allowed)
  if(_package_discovery_allowed)
    find_package(GIF QUIET)
    if(TARGET GIF::GIF)
      _iree_configure_giflib_facade(GIF::GIF)
      return()
    endif()
  endif()

  iree_dependency_require_pinned_source_allowed("giflib")
  iree_declare_locked_fetch_content(giflib)
  FetchContent_MakeAvailable(giflib)
  add_library(iree_third_party_giflib_source STATIC
    ${giflib_SOURCE_DIR}/dgif_lib.c
    ${giflib_SOURCE_DIR}/gifalloc.c
    ${giflib_SOURCE_DIR}/gif_err.c
    ${giflib_SOURCE_DIR}/openbsd-reallocarray.c
  )
  target_include_directories(iree_third_party_giflib_source
    PUBLIC ${giflib_SOURCE_DIR})
  if(WIN32)
    target_compile_definitions(iree_third_party_giflib_source PRIVATE _WIN32)
  endif()
  _iree_configure_giflib_facade(iree_third_party_giflib_source)
endfunction()
