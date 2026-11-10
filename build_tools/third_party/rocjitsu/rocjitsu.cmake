# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

function(iree_configure_rocjitsu_headers)
  if(TARGET iree::third_party::rocjitsu_headers)
    return()
  endif()
  foreach(_name rj_compiler rj_status rj_version rj_vm rj_vm_config)
    set(_dependency "rocjitsu_${_name}_header")
    iree_populate_locked_file(${_dependency} _source_file)
    iree_get_locked_dependency_property(
      _relative_path ${_dependency} DOWNLOADED_FILE_PATH)
    configure_file("${_source_file}"
      "${CMAKE_BINARY_DIR}/_deps/rocjitsu_headers/include/${_relative_path}"
      COPYONLY)
  endforeach()
  add_library(iree_rocjitsu_headers INTERFACE)
  target_include_directories(iree_rocjitsu_headers SYSTEM INTERFACE
    "${CMAKE_BINARY_DIR}/_deps/rocjitsu_headers/include")
  add_library(iree::third_party::rocjitsu_headers ALIAS iree_rocjitsu_headers)
endfunction()
