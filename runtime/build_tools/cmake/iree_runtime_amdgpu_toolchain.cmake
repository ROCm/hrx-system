# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

set(IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE "prebuilt" CACHE STRING
  "AMDGPU builtin device binary producer: prebuilt or source.")
set_property(CACHE IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE
  PROPERTY STRINGS prebuilt source)

set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN "auto" CACHE STRING
  "AMDGPU device compiler provider: none, auto, rocm, or llvm-tools.")
set_property(CACHE IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN
  PROPERTY STRINGS none auto rocm llvm-tools)
set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH "" CACHE PATH
  "ROCm or TheRock SDK root used by AMDGPU device compilation.")
set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_TOOLS_DIR "" CACHE PATH
  "Directory containing an AMDGPU-capable clang and matching LLVM tools.")
set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_BINARY "" CACHE FILEPATH
  "Exact clang or amdclang executable used by AMDGPU device compilation.")
set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_AR_BINARY "" CACHE FILEPATH
  "Exact llvm-ar executable used by AMDGPU device compilation.")
set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_LINK_BINARY "" CACHE FILEPATH
  "Exact llvm-link executable used by AMDGPU device compilation.")
set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLD_BINARY "" CACHE FILEPATH
  "Exact lld executable used by AMDGPU device compilation.")
set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_OBJCOPY_BINARY "" CACHE FILEPATH
  "Exact llvm-objcopy executable used by AMDGPU device compilation.")
set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_RESOURCE_INCLUDE "" CACHE PATH
  "Exact clang resource include directory used by AMDGPU device compilation.")
set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_OFFLOAD_BUNDLER_BINARY ""
  CACHE FILEPATH
  "Exact clang-offload-bundler executable used by HIP device fixtures.")
set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_DEVICE_LIB_PATH "" CACHE PATH
  "Exact ROCm bitcode library directory used by HIP device fixtures.")

function(_iree_amdgpu_reset_resolved_device_toolchain)
  foreach(_VAR
      IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_PROVIDER
      IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_CLANG_BINARY
      IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_LLVM_AR_BINARY
      IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_LLVM_LINK_BINARY
      IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_LLD_BINARY
      IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_LLVM_OBJCOPY_BINARY
      IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_CLANG_RESOURCE_INCLUDE
      IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_CLANG_OFFLOAD_BUNDLER_BINARY
      IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_ROCM_DEVICE_LIB_PATH)
    set(${_VAR} "" CACHE INTERNAL "Resolved AMDGPU device toolchain state."
      FORCE)
  endforeach()
  set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_AVAILABLE OFF CACHE INTERNAL
    "Whether source-built AMDGPU device artifacts are available." FORCE)
  set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_HIP_AVAILABLE OFF CACHE INTERNAL
    "Whether source-built HIP device fixtures are available." FORCE)
endfunction()

function(_iree_amdgpu_resolve_explicit_path OUT_VAR PATH_VALUE)
  if(NOT PATH_VALUE)
    set(${OUT_VAR} "" PARENT_SCOPE)
    return()
  endif()
  if(EXISTS "${PATH_VALUE}" AND NOT IS_DIRECTORY "${PATH_VALUE}")
    get_filename_component(_PATH "${PATH_VALUE}" ABSOLUTE)
    set(${OUT_VAR} "${_PATH}" PARENT_SCOPE)
    return()
  endif()
  if(CMAKE_EXECUTABLE_SUFFIX AND
     EXISTS "${PATH_VALUE}${CMAKE_EXECUTABLE_SUFFIX}" AND
     NOT IS_DIRECTORY "${PATH_VALUE}${CMAKE_EXECUTABLE_SUFFIX}")
    get_filename_component(
      _PATH "${PATH_VALUE}${CMAKE_EXECUTABLE_SUFFIX}" ABSOLUTE)
    set(${OUT_VAR} "${_PATH}" PARENT_SCOPE)
    return()
  endif()
  set(${OUT_VAR} "" PARENT_SCOPE)
endfunction()

function(_iree_amdgpu_find_device_tool OUT_VAR)
  cmake_parse_arguments(
    _RULE
    ""
    "OVERRIDE"
    "NAMES;HINTS"
    ${ARGN}
  )
  if(_RULE_OVERRIDE)
    _iree_amdgpu_resolve_explicit_path(_TOOL "${_RULE_OVERRIDE}")
    set(${OUT_VAR} "${_TOOL}" PARENT_SCOPE)
    return()
  endif()
  find_program(_TOOL
    NAMES ${_RULE_NAMES}
    HINTS ${_RULE_HINTS}
    NO_DEFAULT_PATH
    NO_CACHE
  )
  set(${OUT_VAR} "${_TOOL}" PARENT_SCOPE)
endfunction()

function(_iree_amdgpu_device_toolchain_unavailable REASON)
  string(JOIN "" _REASON "${REASON}" ${ARGN})
  if(IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE STREQUAL "source")
    message(FATAL_ERROR
      "IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE=source requires a complete "
      "AMDGPU device toolchain: ${_REASON}")
  endif()
  if(NOT IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN STREQUAL "auto" AND
     NOT IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN STREQUAL "none")
    message(FATAL_ERROR
      "IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN} "
      "could not be configured: ${_REASON}")
  endif()
  message(STATUS
    "AMDGPU device source compilation unavailable: ${_REASON}; checked-in "
    "device binaries remain enabled")
endfunction()

function(_iree_amdgpu_resolve_device_toolchain_provider OUT_VAR)
  set(_MODE "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN}")
  if(NOT _MODE STREQUAL "none" AND
     NOT _MODE STREQUAL "auto" AND
     NOT _MODE STREQUAL "rocm" AND
     NOT _MODE STREQUAL "llvm-tools")
    message(FATAL_ERROR
      "Unknown IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN='${_MODE}'. Expected none, "
      "auto, rocm, or llvm-tools.")
  endif()
  if(NOT _MODE STREQUAL "auto")
    set(${OUT_VAR} "${_MODE}" PARENT_SCOPE)
    return()
  endif()

  if(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH OR IREE_ROCM_PATH)
    set(_MODE "rocm")
  elseif(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_TOOLS_DIR OR
         IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_BINARY OR
         IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_AR_BINARY OR
         IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_LINK_BINARY OR
         IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLD_BINARY OR
         IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_OBJCOPY_BINARY)
    set(_MODE "llvm-tools")
  else()
    set(_MODE "none")
  endif()
  set(${OUT_VAR} "${_MODE}" PARENT_SCOPE)
endfunction()

function(_iree_amdgpu_find_rocm_device_libraries OUT_VAR ROCM_ROOT)
  set(_CANDIDATES)
  if(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_DEVICE_LIB_PATH)
    list(APPEND _CANDIDATES
      "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_DEVICE_LIB_PATH}")
  endif()
  if(ROCM_ROOT)
    list(APPEND _CANDIDATES
      "${ROCM_ROOT}/amdgcn/bitcode"
      "${ROCM_ROOT}/lib/amdgcn/bitcode"
      "${ROCM_ROOT}/llvm/amdgcn/bitcode"
      "${ROCM_ROOT}/lib/llvm/amdgcn/bitcode"
    )
  endif()
  foreach(_CANDIDATE ${_CANDIDATES})
    if(EXISTS "${_CANDIDATE}/hip.bc" AND
       EXISTS "${_CANDIDATE}/ockl.bc" AND
       EXISTS "${_CANDIDATE}/ocml.bc")
      get_filename_component(_DEVICE_LIB_PATH "${_CANDIDATE}" ABSOLUTE)
      set(${OUT_VAR} "${_DEVICE_LIB_PATH}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${OUT_VAR} "" PARENT_SCOPE)
endfunction()

function(iree_runtime_configure_amdgpu_toolchain)
  _iree_amdgpu_reset_resolved_device_toolchain()
  if(NOT IREE_HAL_DRIVER_AMDGPU)
    return()
  endif()
  if(NOT IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE STREQUAL "prebuilt" AND
     NOT IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE STREQUAL "source")
    message(FATAL_ERROR
      "Unknown IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE="
      "'${IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE}'. Expected prebuilt or "
      "source.")
  endif()

  _iree_amdgpu_resolve_device_toolchain_provider(_PROVIDER)
  if(_PROVIDER STREQUAL "none")
    _iree_amdgpu_device_toolchain_unavailable(
      "no ROCm SDK or explicit LLVM device-tool provider was configured")
    return()
  endif()

  set(_ROCM_ROOT)
  set(_TOOL_HINTS)
  if(_PROVIDER STREQUAL "rocm")
    if(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH)
      set(_ROCM_ROOT "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH}")
    else()
      set(_ROCM_ROOT "${IREE_ROCM_PATH}")
    endif()
    if(NOT IS_DIRECTORY "${_ROCM_ROOT}")
      _iree_amdgpu_device_toolchain_unavailable(
        "configured ROCm SDK root does not exist: ${_ROCM_ROOT}")
      return()
    endif()
    list(APPEND _TOOL_HINTS
      "${_ROCM_ROOT}/lib/llvm/bin"
      "${_ROCM_ROOT}/llvm/bin"
      "${_ROCM_ROOT}/bin"
    )
  elseif(_PROVIDER STREQUAL "llvm-tools")
    if(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_TOOLS_DIR)
      set(_LLVM_TOOLS_DIR
        "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_TOOLS_DIR}")
      list(APPEND _TOOL_HINTS
        "${_LLVM_TOOLS_DIR}"
        "${_LLVM_TOOLS_DIR}/bin"
        "${_LLVM_TOOLS_DIR}/llvm/bin"
        "${_LLVM_TOOLS_DIR}/lib/llvm/bin"
      )
    endif()
  endif()

  _iree_amdgpu_find_device_tool(_CLANG
    OVERRIDE "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_BINARY}"
    NAMES clang amdclang
    HINTS ${_TOOL_HINTS}
  )
  _iree_amdgpu_find_device_tool(_LLVM_AR
    OVERRIDE "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_AR_BINARY}"
    NAMES llvm-ar
    HINTS ${_TOOL_HINTS}
  )
  _iree_amdgpu_find_device_tool(_LLVM_LINK
    OVERRIDE "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_LINK_BINARY}"
    NAMES llvm-link
    HINTS ${_TOOL_HINTS}
  )
  _iree_amdgpu_find_device_tool(_LLD
    OVERRIDE "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLD_BINARY}"
    NAMES lld ld.lld
    HINTS ${_TOOL_HINTS}
  )
  _iree_amdgpu_find_device_tool(_LLVM_OBJCOPY
    OVERRIDE "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_OBJCOPY_BINARY}"
    NAMES llvm-objcopy
    HINTS ${_TOOL_HINTS}
  )
  set(_CLANG_RESOURCE_INCLUDE
    "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_RESOURCE_INCLUDE}")

  set(_MISSING_TOOLS)
  foreach(_TOOL_VAR CLANG LLVM_AR LLVM_LINK LLD LLVM_OBJCOPY)
    if(NOT _${_TOOL_VAR} OR
       NOT EXISTS "${_${_TOOL_VAR}}" OR
       IS_DIRECTORY "${_${_TOOL_VAR}}")
      string(TOLOWER "${_TOOL_VAR}" _TOOL_NAME)
      string(REPLACE "_" "-" _TOOL_NAME "${_TOOL_NAME}")
      list(APPEND _MISSING_TOOLS "${_TOOL_NAME}")
    endif()
  endforeach()
  if(_MISSING_TOOLS)
    list(JOIN _MISSING_TOOLS ", " _MISSING_TOOL_LIST)
    _iree_amdgpu_device_toolchain_unavailable(
      "provider ${_PROVIDER} is missing ${_MISSING_TOOL_LIST}")
    return()
  endif()

  if(NOT _CLANG_RESOURCE_INCLUDE)
    execute_process(
      COMMAND "${_CLANG}" -print-resource-dir
      RESULT_VARIABLE _RESOURCE_RESULT
      OUTPUT_VARIABLE _CLANG_RESOURCE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_VARIABLE _RESOURCE_STDERR
    )
    if(NOT _RESOURCE_RESULT EQUAL 0)
      string(STRIP "${_RESOURCE_STDERR}" _RESOURCE_STDERR)
      _iree_amdgpu_device_toolchain_unavailable(
        "clang resource directory query failed for ${_CLANG}: "
        "${_RESOURCE_STDERR}")
      return()
    endif()
    set(_CLANG_RESOURCE_INCLUDE "${_CLANG_RESOURCE_DIR}/include")
  endif()
  if(NOT EXISTS "${_CLANG_RESOURCE_INCLUDE}/stddef.h")
    _iree_amdgpu_device_toolchain_unavailable(
      "clang resource header stddef.h was not found under "
      "${_CLANG_RESOURCE_INCLUDE}")
    return()
  endif()

  set(_PROBE_SOURCE
    "${CMAKE_BINARY_DIR}/CMakeFiles/iree-amdgpu-device-toolchain-probe.c")
  set(_PROBE_OBJECT
    "${CMAKE_BINARY_DIR}/CMakeFiles/iree-amdgpu-device-toolchain-probe.bc")
  file(WRITE "${_PROBE_SOURCE}" "void iree_amdgpu_probe(void) {}\n")
  execute_process(
    COMMAND "${_CLANG}"
      -target amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib
      -x c -std=c11 -c -emit-llvm "${_PROBE_SOURCE}" -o "${_PROBE_OBJECT}"
    RESULT_VARIABLE _PROBE_RESULT
    ERROR_VARIABLE _PROBE_STDERR
  )
  if(NOT _PROBE_RESULT EQUAL 0)
    _iree_amdgpu_device_toolchain_unavailable(
      "clang cannot compile amdgcn-amd-amdhsa code with ${_CLANG}: "
      "${_PROBE_STDERR}")
    return()
  endif()

  set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_PROVIDER "${_PROVIDER}"
    CACHE INTERNAL "Resolved AMDGPU device toolchain provider." FORCE)
  foreach(_TOOL_VAR CLANG LLVM_AR LLVM_LINK LLD LLVM_OBJCOPY)
    set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_${_TOOL_VAR}_BINARY
      "${_${_TOOL_VAR}}" CACHE INTERNAL
      "Resolved AMDGPU device tool." FORCE)
  endforeach()
  get_filename_component(
    _CLANG_RESOURCE_INCLUDE "${_CLANG_RESOURCE_INCLUDE}" ABSOLUTE)
  set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_CLANG_RESOURCE_INCLUDE
    "${_CLANG_RESOURCE_INCLUDE}" CACHE INTERNAL
    "Resolved clang resource include directory for AMDGPU device compilation."
    FORCE)
  set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_AVAILABLE ON CACHE INTERNAL
    "Whether source-built AMDGPU device artifacts are available." FORCE)

  _iree_amdgpu_find_rocm_device_libraries(_ROCM_DEVICE_LIB_PATH "${_ROCM_ROOT}")
  if(_ROCM_DEVICE_LIB_PATH)
    _iree_amdgpu_find_device_tool(_CLANG_OFFLOAD_BUNDLER
      OVERRIDE
        "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_OFFLOAD_BUNDLER_BINARY}"
      NAMES clang-offload-bundler
      HINTS ${_TOOL_HINTS}
    )
    if(_CLANG_OFFLOAD_BUNDLER)
      set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_CLANG_OFFLOAD_BUNDLER_BINARY
        "${_CLANG_OFFLOAD_BUNDLER}" CACHE INTERNAL
        "Resolved clang-offload-bundler used by HIP device fixtures." FORCE)
      set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_ROCM_DEVICE_LIB_PATH
        "${_ROCM_DEVICE_LIB_PATH}" CACHE INTERNAL
        "Resolved ROCm bitcode library directory used by HIP device fixtures."
        FORCE)
      set(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_HIP_AVAILABLE ON CACHE INTERNAL
        "Whether source-built HIP device fixtures are available." FORCE)
    endif()
  endif()

  message(STATUS
    "AMDGPU device source compilation: ${_PROVIDER} (${_CLANG})")
  if(IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_HIP_AVAILABLE)
    message(STATUS
      "AMDGPU HIP device fixtures: ${_ROCM_DEVICE_LIB_PATH}")
  else()
    message(STATUS
      "AMDGPU HIP device fixtures unavailable: ROCm device libraries or "
      "clang-offload-bundler were not found")
  endif()
endfunction()
