# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

set(_IREE_ROCM_DEPENDENCY_MODES pinned package auto)

function(_iree_rocm_dependency_mode out_var)
  if(DEFINED IREE_ROCM_DEPENDENCY_MODE
      AND NOT "${IREE_ROCM_DEPENDENCY_MODE}" STREQUAL "")
    string(TOLOWER "${IREE_ROCM_DEPENDENCY_MODE}" _mode)
  elseif(DEFINED IREE_ROCM_PATH AND NOT "${IREE_ROCM_PATH}" STREQUAL "")
    set(_mode "package")
  else()
    iree_dependency_mode(_mode)
  endif()

  if(NOT _mode IN_LIST _IREE_ROCM_DEPENDENCY_MODES)
    message(FATAL_ERROR
      "ROCm dependency mode must resolve to pinned, package, or auto; got "
      "'${_mode}' from IREE_ROCM_DEPENDENCY_MODE.")
  endif()

  set(${out_var} "${_mode}" PARENT_SCOPE)
endfunction()

function(_iree_rocm_package_discovery_allowed out_var)
  _iree_rocm_dependency_mode(_mode)
  if(_mode STREQUAL "package" OR _mode STREQUAL "auto")
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(_iree_rocm_require_pinned_source_allowed dep_name)
  _iree_rocm_dependency_mode(_mode)
  if(NOT _mode STREQUAL "pinned" AND NOT _mode STREQUAL "auto")
    message(FATAL_ERROR
      "${dep_name} was not found through ROCm package discovery and "
      "resolved ROCm dependency mode '${_mode}' forbids pinned FetchContent "
      "sources. Provide the headers through IREE_ROCM_PATH/CMAKE_PREFIX_PATH "
      "or set IREE_ROCM_DEPENDENCY_MODE=pinned or auto.")
  endif()
endfunction()

function(_iree_rocm_collect_system_include_dirs out_var)
  set(_include_dirs "")

  if(DEFINED IREE_ROCM_PATH AND NOT "${IREE_ROCM_PATH}" STREQUAL "")
    list(APPEND _include_dirs "${IREE_ROCM_PATH}/include")
  endif()
  foreach(_prefix IN LISTS CMAKE_PREFIX_PATH)
    if(NOT "${_prefix}" STREQUAL "")
      list(APPEND _include_dirs "${_prefix}/include")
    endif()
  endforeach()

  if(TARGET hsa-runtime64::hsa-runtime64)
    get_target_property(_hsa_runtime_include_dirs
      hsa-runtime64::hsa-runtime64 INTERFACE_INCLUDE_DIRECTORIES)
    if(_hsa_runtime_include_dirs)
      list(APPEND _include_dirs ${_hsa_runtime_include_dirs})
    endif()
  endif()

  if(_include_dirs)
    list(REMOVE_DUPLICATES _include_dirs)
  endif()
  set(${out_var} ${_include_dirs} PARENT_SCOPE)
endfunction()

function(_iree_rocm_find_system_header_include_dir
    out_var out_found_var dependency_name required_header)
  _iree_rocm_collect_system_include_dirs(_include_dirs)
  foreach(_include_dir IN LISTS _include_dirs)
    if(EXISTS "${_include_dir}/${required_header}")
      set(${out_var} "${_include_dir}" PARENT_SCOPE)
      set(${out_found_var} TRUE PARENT_SCOPE)
      return()
    endif()
  endforeach()

  set(${out_var} "" PARENT_SCOPE)
  set(${out_found_var} FALSE PARENT_SCOPE)
endfunction()

function(_iree_rocm_fail_missing_system_header dependency_name required_header)
  _iree_rocm_collect_system_include_dirs(_include_dirs)
  if(_include_dirs)
    string(JOIN "\n  " _searched_include_dirs ${_include_dirs})
  else()
    set(_searched_include_dirs "(none)")
  endif()
  message(FATAL_ERROR
    "${dependency_name} headers were not found. Expected to find "
    "${required_header} under a ROCm/TheRock include directory.\n"
    "Set IREE_ROCM_PATH to the ROCm/TheRock root or provide a root through "
    "CMAKE_PREFIX_PATH.\n"
    "Searched include directories:\n"
    "  ${_searched_include_dirs}")
endfunction()

function(_iree_rocm_add_header_target target_name include_dir)
  if(TARGET ${target_name})
    return()
  endif()
  add_library(${target_name} INTERFACE)
  target_include_directories(${target_name} SYSTEM INTERFACE "${include_dir}")
endfunction()

function(_iree_rocm_try_system_hsa_runtime_headers out_found_var)
  find_package(hsa-runtime64 CONFIG QUIET)
  if(TARGET hsa-runtime64::hsa-runtime64)
    get_target_property(_hsa_runtime_include_dirs
      hsa-runtime64::hsa-runtime64 INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT _hsa_runtime_include_dirs)
      message(FATAL_ERROR
        "hsa-runtime64::hsa-runtime64 does not publish include directories")
    endif()
    add_library(iree_rocm_hsa_runtime_headers INTERFACE)
    target_include_directories(iree_rocm_hsa_runtime_headers SYSTEM INTERFACE
      ${_hsa_runtime_include_dirs})
    iree_add_alias_interface(
      iree::third_party::hsa_runtime hsa-runtime64::hsa-runtime64)
    set(${out_found_var} TRUE PARENT_SCOPE)
    return()
  endif()

  _iree_rocm_find_system_header_include_dir(
    _hsa_runtime_include_dir _found "HSA runtime" "hsa/hsa.h")
  if(_found)
    _iree_rocm_add_header_target(
      iree_rocm_hsa_runtime_headers "${_hsa_runtime_include_dir}")
    set(${out_found_var} TRUE PARENT_SCOPE)
    return()
  endif()
  set(${out_found_var} FALSE PARENT_SCOPE)
endfunction()

function(_iree_rocm_configure_pinned_hsa_runtime_headers)
  _iree_rocm_require_pinned_source_allowed("hsa_runtime_headers")
  iree_populate_locked_fetch_content(
    hsa_runtime_headers _hsa_runtime_headers_source_dir)
  _iree_rocm_add_header_target(
    iree_rocm_hsa_runtime_headers
    "${_hsa_runtime_headers_source_dir}/include")
endfunction()

function(iree_configure_rocm_hsa_runtime_headers)
  if(TARGET iree::third_party::hsa_runtime_headers)
    return()
  endif()

  _iree_rocm_package_discovery_allowed(_package_discovery_allowed)
  if(_package_discovery_allowed)
    _iree_rocm_try_system_hsa_runtime_headers(_found)
    if(NOT _found)
      _iree_rocm_dependency_mode(_mode)
      if(_mode STREQUAL "package")
        _iree_rocm_fail_missing_system_header("HSA runtime" "hsa/hsa.h")
      endif()
    endif()
  endif()
  if(NOT TARGET iree_rocm_hsa_runtime_headers)
    _iree_rocm_configure_pinned_hsa_runtime_headers()
  endif()

  iree_add_alias_interface(
    iree::third_party::hsa_runtime_headers iree_rocm_hsa_runtime_headers)
endfunction()

function(iree_configure_rocm_aqlprofile_sdk_headers)
  if(TARGET iree::third_party::aqlprofile_sdk_headers)
    return()
  endif()

  _iree_rocm_package_discovery_allowed(_package_discovery_allowed)
  if(_package_discovery_allowed)
    find_package(hsa-runtime64 CONFIG QUIET)
    _iree_rocm_find_system_header_include_dir(
      _aqlprofile_sdk_include_dir
      _found
      "AQL profile SDK"
      "aqlprofile-sdk/aql_profile_v2.h")
    if(_found)
      _iree_rocm_add_header_target(
        iree_rocm_aqlprofile_sdk_headers "${_aqlprofile_sdk_include_dir}")
    else()
      _iree_rocm_dependency_mode(_mode)
      if(_mode STREQUAL "package")
        _iree_rocm_fail_missing_system_header(
          "AQL profile SDK" "aqlprofile-sdk/aql_profile_v2.h")
      endif()
    endif()
  endif()
  if(NOT TARGET iree_rocm_aqlprofile_sdk_headers)
    _iree_rocm_require_pinned_source_allowed("hsa_runtime_headers")
    iree_populate_locked_fetch_content(
      hsa_runtime_headers _hsa_runtime_headers_source_dir)
    _iree_rocm_add_header_target(
      iree_rocm_aqlprofile_sdk_headers
      "${_hsa_runtime_headers_source_dir}/include")
  endif()

  iree_add_alias_interface(
    iree::third_party::aqlprofile_sdk_headers
    iree_rocm_aqlprofile_sdk_headers)
endfunction()

function(iree_configure_rocm_hip_api_headers)
  if(TARGET iree::third_party::hip_api_headers)
    return()
  endif()

  _iree_rocm_package_discovery_allowed(_package_discovery_allowed)
  if(_package_discovery_allowed)
    _iree_rocm_find_system_header_include_dir(
      _hip_api_include_dir _found "HIP API" "hip/hip_runtime_api.h")
    if(_found)
      _iree_rocm_add_header_target(
        iree_rocm_hip_api_headers "${_hip_api_include_dir}")
    else()
      _iree_rocm_dependency_mode(_mode)
      if(_mode STREQUAL "package")
        _iree_rocm_fail_missing_system_header(
          "HIP API" "hip/hip_runtime_api.h")
      endif()
    endif()
  endif()
  if(NOT TARGET iree_rocm_hip_api_headers)
    _iree_rocm_require_pinned_source_allowed("hip_api_headers")
    iree_populate_locked_fetch_content(
      hip_api_headers _hip_api_headers_source_dir)
    _iree_rocm_add_header_target(
      iree_rocm_hip_api_headers "${_hip_api_headers_source_dir}/include")
  endif()

  iree_add_alias_interface(
    iree::third_party::hip_api_headers iree_rocm_hip_api_headers)
endfunction()

function(iree_configure_rocm_rccl_headers)
  if(TARGET iree::third_party::rccl_headers)
    return()
  endif()

  _iree_rocm_package_discovery_allowed(_package_discovery_allowed)
  if(NOT _package_discovery_allowed)
    _iree_rocm_dependency_mode(_mode)
    message(FATAL_ERROR
      "RCCL headers require package discovery, but "
      "resolved ROCm dependency mode '${_mode}' forbids package dependencies. "
      "Set IREE_ROCM_PATH, set IREE_ROCM_DEPENDENCY_MODE=package or auto, or "
      "disable IREE_HAL_DRIVER_HIP_RCCL.")
  endif()

  _iree_rocm_find_system_header_include_dir(
    _rccl_include_dir _found "RCCL" "rccl/rccl.h")
  if(NOT _found)
    _iree_rocm_dependency_mode(_mode)
    if(_mode STREQUAL "auto")
      message(FATAL_ERROR
        "RCCL headers were not found through package discovery and no pinned "
        "RCCL header source is declared. Set IREE_ROCM_PATH to a ROCm/TheRock "
        "root that provides rccl/rccl.h, use IREE_ROCM_DEPENDENCY_MODE=package "
        "for TheRock validation, or disable IREE_HAL_DRIVER_HIP_RCCL.")
    endif()
    _iree_rocm_fail_missing_system_header("RCCL" "rccl/rccl.h")
  endif()
  _iree_rocm_add_header_target(
    iree_rocm_rccl_headers "${_rccl_include_dir}")
  iree_add_alias_interface(
    iree::third_party::rccl_headers iree_rocm_rccl_headers)
endfunction()
