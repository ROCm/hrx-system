# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

# The facade always exists so generated CMake targets can name the Bazel-owned
# dependency in configurations that do not enable the HIP binding. It remains
# inert until iree_configure_zstd() supplies an implementation.
iree_add_alias_interface(iree::third_party::zstd)

function(_iree_zstd_facade_target out_var)
  get_target_property(_facade_target
    iree::third_party::zstd ALIASED_TARGET)
  if(NOT _facade_target)
    set(_facade_target iree::third_party::zstd)
  endif()
  set(${out_var} ${_facade_target} PARENT_SCOPE)
endfunction()

function(_iree_configure_zstd_facade)
  foreach(_target_name
      zstd::libzstd
      zstd::libzstd_shared
      zstd::libzstd_static
      libzstd
      libzstd_shared
      libzstd_static)
    if(TARGET ${_target_name})
      _iree_zstd_facade_target(_facade_target)
      target_link_libraries(${_facade_target} INTERFACE ${_target_name})
      target_compile_definitions(${_facade_target} INTERFACE IREE_HAVE_ZSTD=1)
      set_property(TARGET ${_facade_target} PROPERTY IREE_ZSTD_CONFIGURED TRUE)
      return()
    endif()
  endforeach()
  message(FATAL_ERROR "zstd did not provide a supported library target")
endfunction()

function(iree_configure_zstd)
  _iree_zstd_facade_target(_facade_target)
  get_target_property(_already_configured
    ${_facade_target} IREE_ZSTD_CONFIGURED)
  if(_already_configured)
    return()
  endif()

  iree_dependency_package_discovery_allowed(_package_discovery_allowed)
  if(_package_discovery_allowed)
    find_package(zstd CONFIG QUIET)
    foreach(_target_name
        zstd::libzstd
        zstd::libzstd_shared
      zstd::libzstd_static)
      if(TARGET ${_target_name})
        _iree_configure_zstd_facade()
        return()
      endif()
    endforeach()
  endif()

  iree_dependency_require_pinned_source_allowed("zstd")
  set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
  set(ZSTD_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
  set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
  set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
  set(ZSTD_BUILD_DICTBUILDER OFF CACHE BOOL "" FORCE)
  set(ZSTD_BUILD_DEPRECATED OFF CACHE BOOL "" FORCE)
  set(ZSTD_LEGACY_SUPPORT OFF CACHE BOOL "" FORCE)
  set(ZSTD_MULTITHREAD_SUPPORT OFF CACHE BOOL "" FORCE)
  iree_declare_locked_fetch_content(zstd SOURCE_SUBDIR build/cmake)
  FetchContent_MakeAvailable(zstd)
  _iree_configure_zstd_facade()
endfunction()
