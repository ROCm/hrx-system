# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Loom C/C++ build rule facades.
#
# Loom C code includes both the reduced IREE runtime C root and the Loom C root.
# These wrappers add those project roots while keeping the shared root
# iree_cc_* rules as the implementation.

function(_loom_cc_args_with_project_deps OUTPUT_ARGS)
  set(_ARGS ${ARGN})
  foreach(_DEP_TARGET iree_defs loom_defs)
    if(NOT TARGET ${_DEP_TARGET})
      message(FATAL_ERROR "Loom C target requires ${_DEP_TARGET}")
    endif()
  endforeach()

  list(FIND _ARGS "DEPS" _DEPS_INDEX)
  if(_DEPS_INDEX EQUAL -1)
    list(APPEND _ARGS DEPS iree_defs loom_defs)
  else()
    math(EXPR _INSERT_INDEX "${_DEPS_INDEX} + 1")
    list(INSERT _ARGS ${_INSERT_INDEX} iree_defs loom_defs)
  endif()
  set(${OUTPUT_ARGS} ${_ARGS} PARENT_SCOPE)
endfunction()

function(loom_cc_library)
  _loom_cc_args_with_project_deps(_ARGS ${ARGN})
  iree_cc_library(${_ARGS})
endfunction()

function(loom_cc_binary)
  _loom_cc_args_with_project_deps(_ARGS ${ARGN})
  iree_cc_binary(${_ARGS})
endfunction()

function(loom_cc_test)
  _loom_cc_args_with_project_deps(_ARGS ${ARGN})
  iree_cc_test(${_ARGS})
endfunction()

function(loom_cc_benchmark)
  _loom_cc_args_with_project_deps(_ARGS ${ARGN})
  iree_cc_binary_benchmark(${_ARGS})
endfunction()

function(loom_cc_fuzz)
  _loom_cc_args_with_project_deps(_ARGS ${ARGN})
  iree_cc_fuzz(${_ARGS})
endfunction()
