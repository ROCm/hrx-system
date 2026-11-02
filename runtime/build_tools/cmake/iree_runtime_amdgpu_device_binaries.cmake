# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


include_guard(GLOBAL)
include(selectors)

function(_iree_hal_amdgpu_source_generator_args out_args device_binary_targets_arg)
  iree_package_binary_dir(_PACKAGE_BINARY_DIR)
  if(NOT IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_AVAILABLE)
    message(FATAL_ERROR
      "AMDGPU builtin device binary source mode requires the configured "
      "AMDGPU device toolchain")
  endif()
  if(NOT Python3_EXECUTABLE)
    message(FATAL_ERROR
      "AMDGPU builtin device binary source mode requires Python 3")
  endif()
  set(_args
    "${Python3_EXECUTABLE}"
    "${IREE_SOURCE_DIR}/build_tools/scripts/amdgpu_device_binaries.py"
    "--repo-root"
    "${IREE_SOURCE_DIR}"
    "--binary-root"
    "${IREE_BINARY_DIR}"
    "--output-dir"
    "${_PACKAGE_BINARY_DIR}"
    "--device-binary-targets"
    "${device_binary_targets_arg}"
    "--clang"
    "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_CLANG_BINARY}"
    "--llvm-link"
    "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_LLVM_LINK_BINARY}"
    "--lld"
    "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_LLD_BINARY}"
    "--llvm-objcopy"
    "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_LLVM_OBJCOPY_BINARY}"
    "--clang-resource-include"
    "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_CLANG_RESOURCE_INCLUDE}"
  )
  set(${out_args} "${_args}" PARENT_SCOPE)
endfunction()

function(iree_hal_amdgpu_source_device_binaries)
  iree_package_binary_dir(_PACKAGE_BINARY_DIR)
  if(NOT IREE_HAL_DRIVER_AMDGPU OR
     NOT IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE STREQUAL "source")
    return()
  endif()
  cmake_parse_arguments(_RULE "" "NAME" "INPUTS" ${ARGN})
  iree_amdgpu_expand_target_selectors(_TARGETS
    "${IREE_AMDGPU_TARGET_EXPANSION_DEVICE_BINARY}" ${IREE_HAL_AMDGPU_TARGETS})
  string(JOIN "," _TARGETS_ARG ${_TARGETS})
  _iree_hal_amdgpu_source_generator_args(_COMMAND "${_TARGETS_ARG}")
  set(_OUTPUTS)
  foreach(_ARCH IN LISTS _TARGETS)
    list(APPEND _OUTPUTS "${_PACKAGE_BINARY_DIR}/amdgcn-amd-amdhsa--${_ARCH}.so")
  endforeach()
  add_custom_command(OUTPUT ${_OUTPUTS}
    COMMAND ${_COMMAND}
    DEPENDS ${_RULE_INPUTS}
    COMMENT "Generating AMDGPU builtin device binaries"
    VERBATIM)
  iree_package_target_name(_TARGET "::${_RULE_NAME}")
  add_custom_target(${_TARGET} DEPENDS ${_OUTPUTS})
endfunction()

function(iree_hal_amdgpu_device_binaries)
  if(NOT IREE_HAL_DRIVER_AMDGPU)
    return()
  endif()
  cmake_parse_arguments(_RULE "" "NAME;TARGET" "TARGETS" ${ARGN})
  iree_amdgpu_expand_target_selectors(_TARGETS
    "${IREE_AMDGPU_TARGET_EXPANSION_DEVICE_BINARY}" ${_RULE_TARGETS})
  iree_package_source_dir(_SOURCE_DIR)
  iree_package_binary_dir(_PACKAGE_BINARY_DIR)
  set(_BINARIES)
  foreach(_ARCH IN LISTS _TARGETS)
    set(_FILENAME "${_RULE_TARGET}--${_ARCH}.so")
    if(IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE STREQUAL "prebuilt")
      set(_BINARY "${_SOURCE_DIR}/prebuilt/${_FILENAME}")
      if(NOT EXISTS "${_BINARY}")
        message(FATAL_ERROR
          "AMDGPU device binary target '${_ARCH}' is unavailable in prebuilt mode. "
          "Use -DIREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE=source or regenerate the blob.")
      endif()
    else()
      set(_BINARY "${_PACKAGE_BINARY_DIR}/source/${_FILENAME}")
    endif()
    list(APPEND _BINARIES "${_BINARY}")
  endforeach()
  iree_c_embed_data(NAME "${_RULE_NAME}" SRCS ${_BINARIES}
    C_FILE_OUTPUT "toc.c" H_FILE_OUTPUT "toc.h"
    IDENTIFIER iree_hal_amdgpu_device_binaries FLATTEN PUBLIC)
endfunction()
