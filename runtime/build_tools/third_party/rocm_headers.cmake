# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(iree_runtime_configure_rocm_headers)
  if(NOT IREE_HAL_DRIVER_AMDGPU)
    return()
  endif()

  find_package(hsa-runtime64 CONFIG REQUIRED)
  if(NOT TARGET hsa_runtime::headers)
    get_target_property(_hsa_runtime_include_dirs
      hsa-runtime64::hsa-runtime64 INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT _hsa_runtime_include_dirs)
      message(FATAL_ERROR
        "hsa-runtime64::hsa-runtime64 does not publish include directories")
    endif()
    add_library(hsa_runtime_headers INTERFACE)
    target_include_directories(hsa_runtime_headers SYSTEM INTERFACE
      ${_hsa_runtime_include_dirs})
    add_library(hsa_runtime::headers ALIAS hsa_runtime_headers)
  endif()

  # TODO(upstream ROCm packaging): replace this compatibility target with:
  #
  #   find_package(aqlprofile-sdk CONFIG REQUIRED)
  #
  # and link against the target exported by that package. As of the ROCm nightly
  # used for this bring-up, the SDK headers are installed under
  # include/aqlprofile-sdk (including the generated version.h), but there is no
  # CMake package config dedicated to the headers.
  #
  # The temporary assumption, confirmed for the local ROCm tree, is that the HSA
  # runtime package's include directories are broad enough to make
  # aqlprofile-sdk/aql_profile_v2.h and aqlprofile-sdk/version.h visible. This
  # is intentionally represented as a separate aqlprofile-sdk::headers target so
  # generated CMake can model the real dependency shape now. Once ROCm publishes
  # the package, only this shim should change.
  if(NOT TARGET aqlprofile-sdk::headers)
    add_library(aqlprofile_sdk_headers INTERFACE)
    target_link_libraries(aqlprofile_sdk_headers INTERFACE hsa_runtime::headers)
    add_library(aqlprofile-sdk::headers ALIAS aqlprofile_sdk_headers)
  endif()
endfunction()
