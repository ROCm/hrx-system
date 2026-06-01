# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(iree_fetch_content_assert_allowed dep_name)
  if(IREE_HERMETIC_BUILD)
    message(FATAL_ERROR
      "${dep_name} was not found via package discovery and IREE_HERMETIC_BUILD=ON "
      "forbids FetchContent. Provide the package via CMAKE_PREFIX_PATH or an "
      "equivalent CMake package path.")
  endif()
endfunction()

function(iree_add_alias_interface alias_name)
  set(_deps ${ARGN})
  if(TARGET ${alias_name})
    return()
  endif()
  string(MAKE_C_IDENTIFIER "${alias_name}" _target_name)
  set(_target_name "iree_${_target_name}")
  add_library(${_target_name} INTERFACE)
  target_link_libraries(${_target_name} INTERFACE ${_deps})
  add_library(${alias_name} ALIAS ${_target_name})
endfunction()
