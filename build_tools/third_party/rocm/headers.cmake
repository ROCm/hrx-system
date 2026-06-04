# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

function(_iree_rocm_append_system_include_dirs out_var)
  set(_include_dirs ${${out_var}})

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

function(_iree_rocm_find_header_include_dir out_var dependency_name required_header)
  set(_include_dirs "")
  _iree_rocm_append_system_include_dirs(_include_dirs)
  foreach(_include_dir IN LISTS _include_dirs)
    if(EXISTS "${_include_dir}/${required_header}")
      set(${out_var} "${_include_dir}" PARENT_SCOPE)
      return()
    endif()
  endforeach()

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

function(iree_configure_rocm_hsa_runtime_headers)
  if(TARGET iree::third_party::hsa_runtime_headers)
    return()
  endif()

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
  else()
    _iree_rocm_find_header_include_dir(
      _hsa_runtime_include_dir "HSA runtime" "hsa/hsa.h")
    _iree_rocm_add_header_target(
      iree_rocm_hsa_runtime_headers "${_hsa_runtime_include_dir}")
  endif()

  iree_add_alias_interface(
    iree::third_party::hsa_runtime_headers iree_rocm_hsa_runtime_headers)
endfunction()

function(iree_configure_rocm_aqlprofile_sdk_headers)
  if(TARGET iree::third_party::aqlprofile_sdk_headers)
    return()
  endif()

  find_package(hsa-runtime64 CONFIG QUIET)
  _iree_rocm_find_header_include_dir(
    _aqlprofile_sdk_include_dir
    "AQL profile SDK"
    "aqlprofile-sdk/aql_profile_v2.h")
  _iree_rocm_add_header_target(
    iree_rocm_aqlprofile_sdk_headers "${_aqlprofile_sdk_include_dir}")
  iree_add_alias_interface(
    iree::third_party::aqlprofile_sdk_headers
    iree_rocm_aqlprofile_sdk_headers)
endfunction()

function(iree_configure_rocm_hip_api_headers)
  if(TARGET iree::third_party::hip_api_headers)
    return()
  endif()

  _iree_rocm_find_header_include_dir(
    _hip_api_include_dir "HIP API" "hip/hip_runtime_api.h")
  _iree_rocm_add_header_target(
    iree_rocm_hip_api_headers "${_hip_api_include_dir}")
  iree_add_alias_interface(
    iree::third_party::hip_api_headers iree_rocm_hip_api_headers)
endfunction()

function(iree_configure_rocm_rccl_headers)
  if(TARGET iree::third_party::rccl_headers)
    return()
  endif()

  _iree_rocm_find_header_include_dir(
    _rccl_include_dir "RCCL" "rccl/rccl.h")
  _iree_rocm_add_header_target(
    iree_rocm_rccl_headers "${_rccl_include_dir}")
  iree_add_alias_interface(
    iree::third_party::rccl_headers iree_rocm_rccl_headers)
endfunction()
