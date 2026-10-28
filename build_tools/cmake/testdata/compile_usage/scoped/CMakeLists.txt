# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This name is visible only here, including when its usage expression runs.
project(scoped_provider NONE)
function(late_provider)
  if(FIXTURE_REQUIRE_CALLER)
    get_property(_CALLER DIRECTORY PROPERTY CALLER_PROJECT)
    if(NOT _CALLER STREQUAL PROJECT_NAME)
      message(FATAL_ERROR "The caller's project hook was not preserved")
    endif()
  endif()
  add_library(local_provider INTERFACE IMPORTED)
  set_property(TARGET local_provider PROPERTY INTERFACE_COMPILE_DEFINITIONS SCOPED_VALUE=53)
  if(FIXTURE_CUSTOM)
    set_property(TARGET local_provider PROPERTY TRANSITIVE_COMPILE_PROPERTIES TOKEN)
  endif()
endfunction()
cmake_language(DEFER CALL late_provider)
iree_cc_library(PACKAGE iree::fixture NAME scoped DEPS local_provider)
