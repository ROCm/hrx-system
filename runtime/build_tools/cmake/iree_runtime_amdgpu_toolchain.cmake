# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(iree_runtime_configure_amdgpu_toolchain)
  if(NOT IREE_HAL_DRIVER_AMDGPU)
    return()
  endif()
  if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
      "IREE_HAL_DRIVER_AMDGPU=ON requires the configured C compiler to be "
      "Clang-family and capable of targeting amdgcn-amd-amdhsa. Configure "
      "with ROCm/LLVM clang instead of relying on a separate ROCm path knob.")
  endif()

  set(IREE_CLANG_BINARY "${CMAKE_C_COMPILER}" CACHE FILEPATH
    "Clang used by IREE runtime AMDGPU device binary builds." FORCE)

  set(_probe_src "${CMAKE_BINARY_DIR}/CMakeFiles/iree-runtime-amdgcn-probe.c")
  set(_probe_obj "${CMAKE_BINARY_DIR}/CMakeFiles/iree-runtime-amdgcn-probe.bc")
  file(WRITE "${_probe_src}" "void iree_runtime_amdgcn_probe(void) {}\n")
  execute_process(
    COMMAND "${IREE_CLANG_BINARY}"
      -target amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib
      -x c -std=c11 -c -emit-llvm "${_probe_src}" -o "${_probe_obj}"
    RESULT_VARIABLE _probe_result
    OUTPUT_VARIABLE _probe_stdout
    ERROR_VARIABLE _probe_stderr
  )
  if(NOT _probe_result EQUAL 0)
    message(FATAL_ERROR
      "Configured C compiler cannot compile a minimal amdgcn-amd-amdhsa object.\n"
      "Compiler: ${IREE_CLANG_BINARY}\n"
      "stderr:\n${_probe_stderr}")
  endif()

  execute_process(
    COMMAND "${IREE_CLANG_BINARY}" -print-prog-name=llvm-link
    OUTPUT_VARIABLE _llvm_link_from_clang
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(EXISTS "${_llvm_link_from_clang}")
    set(IREE_LLVM_LINK_BINARY "${_llvm_link_from_clang}" CACHE FILEPATH
      "llvm-link used by IREE runtime AMDGPU device binary builds." FORCE)
  else()
    find_program(_llvm_link_fallback llvm-link REQUIRED)
    set(IREE_LLVM_LINK_BINARY "${_llvm_link_fallback}" CACHE FILEPATH
      "llvm-link used by IREE runtime AMDGPU device binary builds." FORCE)
  endif()

  execute_process(
    COMMAND "${IREE_CLANG_BINARY}" -print-prog-name=ld.lld
    OUTPUT_VARIABLE _lld_from_clang
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(EXISTS "${_lld_from_clang}")
    set(IREE_LLD_BINARY "${_lld_from_clang}" CACHE FILEPATH
      "lld used by IREE runtime AMDGPU device binary builds." FORCE)
  else()
    find_program(_lld_fallback NAMES ld.lld lld REQUIRED)
    set(IREE_LLD_BINARY "${_lld_fallback}" CACHE FILEPATH
      "lld used by IREE runtime AMDGPU device binary builds." FORCE)
  endif()

  execute_process(
    COMMAND "${IREE_CLANG_BINARY}" -print-prog-name=llvm-objcopy
    OUTPUT_VARIABLE _objcopy_from_clang
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(EXISTS "${_objcopy_from_clang}")
    set(IREE_LLVM_OBJCOPY_BINARY "${_objcopy_from_clang}" CACHE FILEPATH
      "llvm-objcopy used by IREE runtime AMDGPU device binary builds." FORCE)
  else()
    find_program(_objcopy_fallback llvm-objcopy REQUIRED)
    set(IREE_LLVM_OBJCOPY_BINARY "${_objcopy_fallback}" CACHE FILEPATH
      "llvm-objcopy used by IREE runtime AMDGPU device binary builds." FORCE)
  endif()

  execute_process(
    COMMAND "${IREE_CLANG_BINARY}" -print-resource-dir
    OUTPUT_VARIABLE _clang_resource_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _clang_resource_dir OR NOT EXISTS "${_clang_resource_dir}/include")
    message(FATAL_ERROR
      "Could not determine clang resource include directory from ${IREE_CLANG_BINARY}")
  endif()
  set(IREE_CLANG_BUILTIN_HEADERS_PATH "${_clang_resource_dir}/include" CACHE PATH
    "Clang resource include directory used by AMDGPU device binary builds." FORCE)
endfunction()
