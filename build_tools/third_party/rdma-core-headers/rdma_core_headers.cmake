# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

function(_iree_rdma_core_headers_add_target include_dir)
  if(TARGET iree_rdma_core_headers)
    return()
  endif()

  set(_virtual_root
    "${CMAKE_BINARY_DIR}/third_party/rdma-core-headers/include-prefix")
  set(_virtual_include
    "${_virtual_root}/third_party/rdma-core-headers/include")
  file(MAKE_DIRECTORY "${_virtual_root}/third_party/rdma-core-headers")
  file(REMOVE_RECURSE "${_virtual_include}")
  file(CREATE_LINK "${include_dir}" "${_virtual_include}"
    SYMBOLIC COPY_ON_ERROR)

  add_library(iree_rdma_core_headers INTERFACE)
  target_include_directories(iree_rdma_core_headers SYSTEM
    INTERFACE
      "$<BUILD_INTERFACE:${_virtual_root}>"
      "$<BUILD_INTERFACE:${include_dir}>"
  )
  if(NOT TARGET rdma_core::headers)
    add_library(rdma_core::headers ALIAS iree_rdma_core_headers)
  endif()
endfunction()

function(_iree_rdma_core_headers_try_system out_found_var)
  find_path(_rdma_core_headers_include_dir
    NAMES
      infiniband/verbs.h
  )
  if(_rdma_core_headers_include_dir AND
      EXISTS "${_rdma_core_headers_include_dir}/rdma/rdma_cma.h")
    _iree_rdma_core_headers_add_target("${_rdma_core_headers_include_dir}")
    set(${out_found_var} TRUE PARENT_SCOPE)
    return()
  endif()
  set(${out_found_var} FALSE PARENT_SCOPE)
endfunction()

function(iree_configure_rdma_core_headers)
  if(TARGET iree::third_party::rdma_core_headers)
    return()
  endif()

  iree_dependency_package_discovery_allowed(_package_discovery_allowed)
  if(_package_discovery_allowed)
    _iree_rdma_core_headers_try_system(_found)
    if(NOT _found)
      iree_dependency_mode(_mode)
      if(_mode STREQUAL "package")
        message(FATAL_ERROR
          "RDMA core headers were not found. Provide infiniband/verbs.h and "
          "rdma/rdma_cma.h through CMAKE_PREFIX_PATH or use "
          "IREE_DEPENDENCY_MODE=pinned or auto.")
      endif()
    endif()
  endif()

  if(NOT TARGET iree_rdma_core_headers)
    iree_dependency_require_pinned_source_allowed("rdma_core_headers")
    iree_populate_locked_fetch_content(
      rdma_core_headers _rdma_core_headers_source_dir)
    _iree_rdma_core_headers_add_target(
      "${_rdma_core_headers_source_dir}/include")
  endif()

  iree_add_alias_interface(
    iree::third_party::rdma_core_headers iree_rdma_core_headers)
endfunction()
