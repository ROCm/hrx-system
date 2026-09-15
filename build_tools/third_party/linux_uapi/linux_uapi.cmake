# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

# Assemble only the selected driver protocols. Fundamental Linux types and
# ioctl encoding remain supplied by the target platform's sysroot.
function(_iree_linux_uapi_copy_file dependency)
  iree_populate_locked_file(${dependency} _source_file)
  iree_get_locked_dependency_property(
    _relative_path ${dependency} DOWNLOADED_FILE_PATH)
  configure_file("${_source_file}"
    "${CMAKE_BINARY_DIR}/_deps/linux_uapi/include/${_relative_path}" COPYONLY)
endfunction()

function(_iree_linux_uapi_add_target name)
  add_library(iree_${name} INTERFACE)
  target_include_directories(iree_${name} SYSTEM INTERFACE
    "${CMAKE_BINARY_DIR}/_deps/linux_uapi/include")
  target_link_libraries(iree_${name} INTERFACE ${ARGN})
  add_library(iree::third_party::${name} ALIAS iree_${name})
endfunction()

function(iree_configure_linux_drm_uapi)
  if(TARGET iree::third_party::linux_drm_uapi)
    return()
  endif()
  iree_dependency_require_pinned_source_allowed("libdrm_headers")
  iree_populate_locked_fetch_content(libdrm_headers _source_dir)
  foreach(_header drm.h drm_mode.h)
    # Both spellings are part of the upstream driver header include contracts.
    foreach(_directory drm libdrm)
      configure_file("${_source_dir}/include/drm/${_header}"
        "${CMAKE_BINARY_DIR}/_deps/linux_uapi/include/${_directory}/${_header}"
        COPYONLY)
    endforeach()
  endforeach()
  _iree_linux_uapi_add_target(linux_drm_uapi)
endfunction()

function(iree_configure_linux_amdgpu_uapi)
  if(TARGET iree::third_party::linux_amdgpu_uapi)
    return()
  endif()
  iree_configure_linux_drm_uapi()
  iree_dependency_require_pinned_source_allowed("libdrm_headers")
  iree_populate_locked_fetch_content(libdrm_headers _source_dir)
  configure_file("${_source_dir}/include/drm/amdgpu_drm.h"
    "${CMAKE_BINARY_DIR}/_deps/linux_uapi/include/drm/amdgpu_drm.h" COPYONLY)
  _iree_linux_uapi_add_target(linux_amdgpu_uapi
    iree::third_party::linux_drm_uapi)
endfunction()

function(iree_configure_linux_kfd_uapi)
  if(TARGET iree::third_party::linux_kfd_uapi)
    return()
  endif()
  iree_configure_linux_drm_uapi()
  _iree_linux_uapi_copy_file(linux_kfd_ioctl_header)
  _iree_linux_uapi_copy_file(linux_kfd_sysfs_header)
  _iree_linux_uapi_add_target(linux_kfd_uapi
    iree::third_party::linux_drm_uapi)
endfunction()

function(iree_configure_linux_xdna_uapi)
  if(TARGET iree::third_party::linux_xdna_uapi)
    return()
  endif()
  iree_configure_linux_drm_uapi()
  _iree_linux_uapi_copy_file(linux_xdna_header)
  _iree_linux_uapi_copy_file(linux_const_header)
  _iree_linux_uapi_copy_file(linux_stddef_header)
  _iree_linux_uapi_add_target(linux_xdna_uapi
    iree::third_party::linux_drm_uapi)
endfunction()
