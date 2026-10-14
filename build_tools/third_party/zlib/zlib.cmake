# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

iree_add_alias_interface(iree::third_party::zlib)

function(_iree_configure_zlib_facade target_name)
  get_target_property(_facade_target
    iree::third_party::zlib ALIASED_TARGET)
  target_link_libraries(${_facade_target} INTERFACE ${target_name})
  set_property(TARGET ${_facade_target} PROPERTY IREE_ZLIB_CONFIGURED TRUE)
endfunction()

function(iree_configure_zlib)
  get_target_property(_facade_target
    iree::third_party::zlib ALIASED_TARGET)
  get_target_property(_already_configured
    ${_facade_target} IREE_ZLIB_CONFIGURED)
  if(_already_configured)
    return()
  endif()

  iree_dependency_package_discovery_allowed(_package_discovery_allowed)
  if(_package_discovery_allowed)
    find_package(ZLIB QUIET)
    if(TARGET ZLIB::ZLIB)
      _iree_configure_zlib_facade(ZLIB::ZLIB)
      return()
    endif()
  endif()

  iree_dependency_require_pinned_source_allowed("zlib")
  set(ZLIB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(ZLIB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
  set(ZLIB_BUILD_STATIC ON CACHE BOOL "" FORCE)
  set(ZLIB_INSTALL OFF CACHE BOOL "" FORCE)
  iree_declare_locked_fetch_content(zlib)
  FetchContent_MakeAvailable(zlib)
  if(NOT TARGET zlibstatic)
    message(FATAL_ERROR "zlib did not provide its static library target")
  endif()
  if(NOT TARGET ZLIB::ZLIB)
    add_library(ZLIB::ZLIB ALIAS zlibstatic)
  endif()
  _iree_configure_zlib_facade(zlibstatic)
endfunction()
