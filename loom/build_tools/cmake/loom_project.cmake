# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_LIST_DIR}/../amdgpu"
  "${CMAKE_CURRENT_LIST_DIR}"
)

include(loom_amdgpu_selectors)
include(loom_target_presets)

if(NOT DEFINED LOOM_BUILD)
  option(LOOM_BUILD
    "Build Loom compiler libraries, tools, bindings, and tests." ON)
endif()

option(LOOM_LINK_SHARED_TOOLS
  "Link LoomC tools to one shared compiler and its exact-build IREE runtime." OFF)
if(LOOM_LINK_SHARED_TOOLS AND CMAKE_VERSION VERSION_LESS 3.27)
  message(FATAL_ERROR "LOOM_LINK_SHARED_TOOLS requires CMake 3.27 or newer.")
endif()
if(LOOM_LINK_SHARED_TOOLS AND BUILD_SHARED_LIBS)
  message(FATAL_ERROR
    "LOOM_LINK_SHARED_TOOLS requires BUILD_SHARED_LIBS=OFF so the bundle owns its component implementations.")
endif()

loom_resolve_target_presets()

if(NOT DEFINED LOOM_ARCH_AMDGPU_TARGETS)
  set(LOOM_ARCH_AMDGPU_TARGETS
    ${_LOOM_AMDGPU_DEFAULT_TARGET_SELECTORS}
    CACHE STRING
    "AMDGPU architecture selectors compiled into Loom target support.")
endif()

if(LOOM_ARCH_AMDGPU)
  loom_amdgpu_configure_target_selectors()
endif()

set(LOOM_ARTIFACT_AIE2P_ELF OFF)
set(LOOM_ARTIFACT_AMDGPU_ELF OFF)
set(LOOM_ARTIFACT_SPIRV_BINARY OFF)
set(LOOM_ARTIFACT_VM_BINARY OFF)
set(LOOM_ARTIFACT_WASM_BINARY OFF)
set(LOOM_ARTIFACT_X86_ELF OFF)
if(LOOM_ARCH_AIE2P AND LOOM_EMIT_ELF)
  set(LOOM_ARTIFACT_AIE2P_ELF ON)
endif()
if(LOOM_ARCH_AMDGPU AND LOOM_EMIT_ELF)
  set(LOOM_ARTIFACT_AMDGPU_ELF ON)
endif()
if(LOOM_ARCH_SPIRV AND LOOM_EMIT_SPIRV)
  set(LOOM_ARTIFACT_SPIRV_BINARY ON)
endif()
if(LOOM_ARCH_VM AND LOOM_EMIT_VM)
  set(LOOM_ARTIFACT_VM_BINARY ON)
endif()
if(LOOM_ARCH_WASM AND LOOM_EMIT_WASM)
  set(LOOM_ARTIFACT_WASM_BINARY ON)
endif()
if(LOOM_ARCH_X86 AND LOOM_EMIT_ELF)
  set(LOOM_ARTIFACT_X86_ELF ON)
endif()

set(LOOM_ROUTE_AIE2P_XDNA_HAL OFF)
set(LOOM_ROUTE_AMDGPU_HAL OFF)
set(LOOM_ROUTE_SPIRV_VULKAN_HAL OFF)
set(LOOM_ROUTE_VM_IREE_VM OFF)
set(LOOM_ROUTE_WASM_HOSTED OFF)
set(LOOM_ROUTE_X86_TASK_HAL OFF)
if(LOOM_ARTIFACT_AIE2P_ELF AND LOOM_EXECUTE_IREE_HAL AND
    IREE_HAL_DRIVER_AMD_XDNA)
  set(LOOM_ROUTE_AIE2P_XDNA_HAL ON)
endif()
if(LOOM_ARTIFACT_AMDGPU_ELF AND LOOM_EXECUTE_IREE_HAL AND
    IREE_HAL_DRIVER_AMD_GPU)
  set(LOOM_ROUTE_AMDGPU_HAL ON)
endif()
if(LOOM_ARTIFACT_SPIRV_BINARY AND LOOM_EXECUTE_IREE_HAL AND
    IREE_HAL_DRIVER_VULKAN)
  set(LOOM_ROUTE_SPIRV_VULKAN_HAL ON)
endif()
if(LOOM_ARTIFACT_VM_BINARY AND LOOM_EXECUTE_IREE_VM)
  set(LOOM_ROUTE_VM_IREE_VM ON)
endif()
if(LOOM_ARTIFACT_WASM_BINARY AND LOOM_EXECUTE_WASM_HOST)
  set(LOOM_ROUTE_WASM_HOSTED ON)
endif()
if(LOOM_ARTIFACT_X86_ELF AND LOOM_EXECUTE_IREE_HAL AND
    IREE_HAL_DRIVER_TASK AND IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF)
  set(LOOM_ROUTE_X86_TASK_HAL ON)
endif()

option(LOOM_IMPORT_CXX
  "Enables the native Loom C/C++ importer, tools, and binding." OFF)
option(LOOM_IMPORT_CXX_EMBED_INCLUDES
  "Embeds Loom C++ and HIP source headers in the C++ importer." ON)

option(LOOM_MEDIA_IMAGE_INPUT
  "Enables image inputs to Loom services."
  ${IREE_CODEC_HOST})
if(LOOM_ARTIFACT_VM_BINARY AND LOOM_MEDIA_IMAGE_INPUT AND NOT IREE_CODEC_HOST)
  message(FATAL_ERROR
    "Loom image input requires Linux, macOS, or Windows. Set LOOM_MEDIA_IMAGE_INPUT=OFF to build other serving modalities.")
endif()
foreach(_codec png jpeg gif webp)
  string(TOUPPER "${_codec}" _codec_upper)
  set(LOOM_MEDIA_IMAGE_CODEC_${_codec_upper} OFF)
  if(LOOM_ARTIFACT_VM_BINARY AND LOOM_MEDIA_IMAGE_INPUT AND
      IREE_CODEC_IMAGE_CODEC_${_codec_upper})
    set(LOOM_MEDIA_IMAGE_CODEC_${_codec_upper} ON)
  endif()
endforeach()

function(loom_configure_project)
  if(LOOM_ARCH_AMDGPU)
    iree_configure_rocm_hsa_runtime_headers()
  endif()
endfunction()
