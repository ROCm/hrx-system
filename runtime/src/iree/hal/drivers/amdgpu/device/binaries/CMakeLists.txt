# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Builtin AMDGPU device binaries embedded into the runtime.

include(selectors)

if(NOT IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE STREQUAL "prebuilt" AND
   NOT IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE STREQUAL "source")
  message(FATAL_ERROR
    "Unknown IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE='${IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE}'. "
    "Expected 'prebuilt' or 'source'.")
endif()

function(_iree_hal_amdgpu_read_device_bitcode_sources out_var)
  set(_source_list_path "${CMAKE_CURRENT_SOURCE_DIR}/../device_bitcode_sources.bzl")
  set_property(DIRECTORY APPEND PROPERTY
    CMAKE_CONFIGURE_DEPENDS "${_source_list_path}")
  file(READ "${_source_list_path}" _source_list)
  string(REGEX MATCH
    "IREE_HAL_AMDGPU_DEVICE_BITCODE_SRCS[ \t\r\n]*=[ \t\r\n]*\\[([^]]*)\\]"
    _source_match "${_source_list}"
  )
  if(NOT _source_match)
    message(FATAL_ERROR
      "${_source_list_path} must assign IREE_HAL_AMDGPU_DEVICE_BITCODE_SRCS "
      "to a literal list")
  endif()

  set(_source_body "${CMAKE_MATCH_1}")
  string(REGEX REPLACE "#[^\r\n]*" "" _source_body "${_source_body}")
  string(REGEX MATCHALL "\"[^\"]+\"" _quoted_sources "${_source_body}")
  if(NOT _quoted_sources)
    message(FATAL_ERROR
      "${_source_list_path} must list at least one C source file")
  endif()

  set(_source_remainder "${_source_body}")
  set(_source_paths)
  foreach(_quoted_source ${_quoted_sources})
    string(REPLACE "${_quoted_source}" "" _source_remainder "${_source_remainder}")
    string(REGEX REPLACE "^\"(.*)\"$" "\\1" _source_name "${_quoted_source}")
    if(_source_name MATCHES "(^/|(^|/)\\.\\.(/|$))")
      message(FATAL_ERROR
        "${_source_list_path} entries must be package-relative: ${_source_name}")
    endif()
    if(NOT _source_name MATCHES "\\.c$")
      message(FATAL_ERROR
        "${_source_list_path} entry is not a C source file: ${_source_name}")
    endif()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../${_source_name}")
      message(FATAL_ERROR
        "${_source_list_path} entry does not exist: ${_source_name}")
    endif()
    list(APPEND _source_paths "../${_source_name}")
  endforeach()
  string(REGEX REPLACE "[, \t\r\n]" "" _source_remainder "${_source_remainder}")
  if(_source_remainder)
    message(FATAL_ERROR
      "${_source_list_path} must contain only quoted source file names")
  endif()
  set(${out_var} "${_source_paths}" PARENT_SCOPE)
endfunction()

function(_iree_hal_amdgpu_source_generator_args out_args device_binary_targets_arg)
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
    "${CMAKE_CURRENT_BINARY_DIR}"
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

_iree_hal_amdgpu_read_device_bitcode_sources(_BITCODE_SRCS)

set(_ABI_HDRS
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/command_buffer.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/common.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/feedback.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/kernel_args.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/profile.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/queue.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/signal.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/timestamp.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../abi/tsan.h"
)

set(_BITCODE_HDRS
  "${CMAKE_CURRENT_SOURCE_DIR}/../blit.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../dispatch.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../kernel_tables.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../kernels.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../support/asan.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../support/common.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../support/feedback.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../support/kernel.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../support/queue.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../support/signal.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../support/tsan.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../timestamp.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../tsan.h"
  ${_ABI_HDRS}
)

iree_amdgpu_expand_target_selectors(
  _DEVICE_BINARY_TARGETS
  "${IREE_AMDGPU_TARGET_EXPANSION_DEVICE_BINARY}"
  ${IREE_HAL_AMDGPU_TARGETS}
)

set(_ARCH_BINARIES)
if(IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE STREQUAL "prebuilt")
  foreach(_ARCH ${_DEVICE_BINARY_TARGETS})
    set(_BINARY_PATH
        "${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/amdgcn-amd-amdhsa--${_ARCH}.so")
    if(NOT EXISTS "${_BINARY_PATH}")
      message(FATAL_ERROR
        "AMDGPU device binary target '${_ARCH}' was requested in prebuilt "
        "mode, but ${_BINARY_PATH} does not exist. Use "
        "-DIREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE=source or regenerate and "
        "check in the prebuilt blob.")
    endif()
    list(APPEND _ARCH_BINARIES "${_BINARY_PATH}")
  endforeach()
else()
  string(JOIN "," _DEVICE_BINARY_TARGETS_ARG ${_DEVICE_BINARY_TARGETS})
  _iree_hal_amdgpu_source_generator_args(
    _SOURCE_GENERATOR_ARGS "${_DEVICE_BINARY_TARGETS_ARG}")
  foreach(_ARCH ${_DEVICE_BINARY_TARGETS})
    list(APPEND _ARCH_BINARIES
      "${CMAKE_CURRENT_BINARY_DIR}/amdgcn-amd-amdhsa--${_ARCH}.so"
    )
  endforeach()
  add_custom_command(
    OUTPUT
      ${_ARCH_BINARIES}
    COMMAND
      ${_SOURCE_GENERATOR_ARGS}
    DEPENDS
      "${IREE_SOURCE_DIR}/build_tools/scripts/amdgpu_device_binaries.py"
      "${IREE_SOURCE_DIR}/build_tools/amdgpu/target_map.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../device_bitcode_sources.bzl"
      ${_BITCODE_SRCS}
      ${_BITCODE_HDRS}
    COMMENT
      "Generating AMDGPU builtin device binaries"
    VERBATIM
  )
endif()

iree_c_embed_data(
  NAME
    toc
  SRCS
    "${_ARCH_BINARIES}"
  C_FILE_OUTPUT
    "toc.c"
  H_FILE_OUTPUT
    "toc.h"
  IDENTIFIER
    "iree_hal_amdgpu_device_binaries"
  FLATTEN
  PUBLIC
)
